from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from brain_subspace_paper.config import pipeline_config, project_config, project_root
from brain_subspace_paper.logging_utils import append_markdown_log, bootstrap_logs, write_text


FEATURE_CONDITIONS = ("raw", "shared", "specific", "full", "mismatched_shared")


@dataclass(slots=True)
class FeatureBuildSummary:
    model_name: str
    feature_manifest_path: Path
    permutation_manifest_path: Path
    report_path: Path
    n_triplets: int
    n_layers: int
    hidden_size: int
    n_feature_arrays: int


def _embedding_manifest_path() -> Path:
    return project_root() / "data" / "interim" / "embeddings" / "embedding_manifest.parquet"


def _triplets_path() -> Path:
    return project_root() / "data" / "processed" / "alignment_triplets.parquet"


def _feature_root() -> Path:
    return project_root() / "data" / "processed" / "features"


def _feature_manifest_path() -> Path:
    return _feature_root() / "feature_manifest.parquet"


def _permutation_manifest_path(model_name: str) -> Path:
    return _feature_root() / model_name / "mismatched_shared_permutations.parquet"


def _report_path() -> Path:
    return project_root() / "outputs" / "logs" / "feature_decomposition_report.md"


def _epsilon() -> float:
    return float(pipeline_config().get("embedding", {}).get("epsilon", 1e-8))


def _n_shuffles() -> int:
    return int(pipeline_config().get("design_matrix", {}).get("mismatched_shared_shuffles", 5))


def _random_seed() -> int:
    return int(project_config().get("random_seed", 20260314))


def _load_embedding_manifest(model_name: str) -> pd.DataFrame:
    manifest = pd.read_parquet(_embedding_manifest_path()).copy()
    manifest = manifest.loc[manifest["model"] == model_name].copy()
    if manifest.empty:
        raise FileNotFoundError(f"No embedding manifest rows found for model={model_name}.")
    return manifest.sort_values(["language", "layer_index"]).reset_index(drop=True)


def _load_triplets() -> pd.DataFrame:
    triplets = pd.read_parquet(_triplets_path()).copy()
    return triplets.sort_values("triplet_id").reset_index(drop=True)


def _language_arrays_for_layer(
    manifest: pd.DataFrame,
    *,
    model_name: str,
    layer_index: int,
) -> dict[str, np.ndarray]:
    rows = manifest.loc[manifest["layer_index"].astype(int) == layer_index].copy()
    arrays: dict[str, np.ndarray] = {}
    for row in rows.itertuples(index=False):
        arrays[str(row.language)] = np.load(row.filepath).astype(np.float32, copy=False)
    expected_languages = {"en", "fr", "zh"}
    if set(arrays) != expected_languages:
        raise RuntimeError(
            f"Layer {layer_index} for model={model_name} is missing languages: "
            f"{sorted(expected_languages.difference(arrays))}"
        )
    return arrays


def _validate_layer_shapes(
    arrays_by_language: dict[str, np.ndarray],
    *,
    n_triplets: int,
) -> int:
    shapes = {language: array.shape for language, array in arrays_by_language.items()}
    hidden_sizes = {shape[1] for shape in shapes.values()}
    row_counts = {shape[0] for shape in shapes.values()}
    if row_counts != {n_triplets}:
        raise RuntimeError(f"Embedding row-count mismatch across languages: {shapes}")
    if len(hidden_sizes) != 1:
        raise RuntimeError(f"Embedding hidden-size mismatch across languages: {shapes}")
    return int(next(iter(hidden_sizes)))


def _specific_residual(raw: np.ndarray, shared: np.ndarray, eps: float) -> np.ndarray:
    residual = raw - shared
    denom = np.sum(shared * shared, axis=1, keepdims=True) + eps
    projection_scale = np.sum(shared * residual, axis=1, keepdims=True) / denom
    return residual - projection_scale * shared


def _derangement_for_indices(indices: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    if len(indices) < 2:
        raise RuntimeError("MISMATCHED_SHARED requires at least two triplets within each run/section.")
    perm = indices.copy()
    for _ in range(128):
        perm = rng.permutation(indices)
        if not np.any(perm == indices):
            return perm
    # Guaranteed fallback: rotate by one position if random derangement did not appear.
    return np.roll(indices, 1)


def _build_run_local_permutations(
    triplets: pd.DataFrame,
    *,
    model_name: str,
    n_shuffles: int,
) -> tuple[dict[tuple[str, int], np.ndarray], pd.DataFrame]:
    metadata_rows: list[dict[str, Any]] = []
    permutations: dict[tuple[str, int], np.ndarray] = {}
    base_index = np.arange(len(triplets), dtype=np.int64)
    sections = triplets["section_index"].astype(int).to_numpy()
    triplet_ids = triplets["triplet_id"].astype(int).to_numpy()

    for language_index, language in enumerate(("en", "fr", "zh")):
        for shuffle_index in range(n_shuffles):
            rng = np.random.default_rng(_random_seed() + 10_000 * language_index + shuffle_index)
            permuted_index = base_index.copy()
            for section_index in sorted(np.unique(sections).tolist()):
                mask = np.flatnonzero(sections == section_index)
                permuted_index[mask] = _derangement_for_indices(mask, rng=rng)
            permutations[(language, shuffle_index)] = permuted_index
            metadata_rows.extend(
                {
                    "model": model_name,
                    "language": language,
                    "shuffle_index": shuffle_index,
                    "section_index": int(section_index),
                    "triplet_id": int(triplet_ids[row_index]),
                    "permuted_triplet_id": int(triplet_ids[permuted_index[row_index]]),
                }
                for row_index, section_index in enumerate(sections)
            )
    metadata_df = pd.DataFrame(metadata_rows)
    return permutations, metadata_df


def _write_feature_array(
    *,
    model_name: str,
    language: str,
    condition: str,
    layer_index: int,
    array: np.ndarray,
    shuffle_index: int | None = None,
) -> dict[str, Any]:
    if shuffle_index is None:
        path = _feature_root() / model_name / language / condition / f"layer_{layer_index:02d}.npy"
    else:
        path = (
            _feature_root()
            / model_name
            / language
            / condition
            / f"shuffle_{shuffle_index:02d}"
            / f"layer_{layer_index:02d}.npy"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32, copy=False))
    return {
        "model": model_name,
        "language": language,
        "layer_index": layer_index,
        "condition": condition,
        "shuffle_index": shuffle_index,
        "filepath": path.as_posix(),
        "n_rows": int(array.shape[0]),
        "feature_dim": int(array.shape[1]),
    }


def build_decomposition_features(model_name: str = "xlmr") -> FeatureBuildSummary:
    bootstrap_logs()
    triplets = _load_triplets()
    manifest = _load_embedding_manifest(model_name)
    n_triplets = len(triplets)
    layer_indices = sorted(manifest["layer_index"].astype(int).unique().tolist())
    n_shuffles = _n_shuffles()
    permutations, permutation_df = _build_run_local_permutations(
        triplets,
        model_name=model_name,
        n_shuffles=n_shuffles,
    )
    permutation_manifest_path = _permutation_manifest_path(model_name)
    permutation_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    permutation_df.to_parquet(permutation_manifest_path, index=False)

    feature_rows: list[dict[str, Any]] = []
    orthogonality_rows: list[dict[str, Any]] = []
    hidden_size: int | None = None
    eps = _epsilon()

    for layer_index in layer_indices:
        arrays_by_language = _language_arrays_for_layer(manifest, model_name=model_name, layer_index=layer_index)
        hidden_size = _validate_layer_shapes(arrays_by_language, n_triplets=n_triplets)

        for target_language in ("en", "fr", "zh"):
            other_languages = [language for language in ("en", "fr", "zh") if language != target_language]
            raw = arrays_by_language[target_language]
            shared = (
                arrays_by_language[other_languages[0]] + arrays_by_language[other_languages[1]]
            ) / 2.0
            specific = _specific_residual(raw, shared, eps=eps).astype(np.float32, copy=False)
            full = np.concatenate([shared, specific], axis=1).astype(np.float32, copy=False)

            feature_rows.append(
                _write_feature_array(
                    model_name=model_name,
                    language=target_language,
                    condition="raw",
                    layer_index=layer_index,
                    array=raw,
                )
            )
            feature_rows.append(
                _write_feature_array(
                    model_name=model_name,
                    language=target_language,
                    condition="shared",
                    layer_index=layer_index,
                    array=shared,
                )
            )
            feature_rows.append(
                _write_feature_array(
                    model_name=model_name,
                    language=target_language,
                    condition="specific",
                    layer_index=layer_index,
                    array=specific,
                )
            )
            feature_rows.append(
                _write_feature_array(
                    model_name=model_name,
                    language=target_language,
                    condition="full",
                    layer_index=layer_index,
                    array=full,
                )
            )

            for shuffle_index in range(n_shuffles):
                permuted_index = permutations[(target_language, shuffle_index)]
                mismatched_shared = shared[permuted_index]
                feature_rows.append(
                    _write_feature_array(
                        model_name=model_name,
                        language=target_language,
                        condition="mismatched_shared",
                        layer_index=layer_index,
                        array=mismatched_shared,
                        shuffle_index=shuffle_index,
                    )
                )

            orthogonality_rows.append(
                {
                    "model": model_name,
                    "language": target_language,
                    "layer_index": layer_index,
                    "shared_specific_dot_mean": float(np.mean(np.sum(shared * specific, axis=1))),
                    "shared_specific_dot_abs_max": float(np.max(np.abs(np.sum(shared * specific, axis=1)))),
                    "shared_norm_mean": float(np.mean(np.linalg.norm(shared, axis=1))),
                    "specific_norm_mean": float(np.mean(np.linalg.norm(specific, axis=1))),
                }
            )

    feature_manifest = pd.DataFrame(feature_rows)
    feature_manifest_path = _feature_manifest_path()
    feature_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if feature_manifest_path.exists():
        existing = pd.read_parquet(feature_manifest_path)
        if "shuffle_index" not in existing.columns:
            existing["shuffle_index"] = pd.NA
        existing = existing.loc[existing["model"] != model_name].copy()
        feature_manifest = pd.concat([existing, feature_manifest], ignore_index=True)
    feature_manifest = feature_manifest.sort_values(
        ["model", "language", "condition", "shuffle_index", "layer_index"],
        na_position="first",
    ).reset_index(drop=True)
    feature_manifest.to_parquet(feature_manifest_path, index=False)

    orthogonality_df = pd.DataFrame(orthogonality_rows)
    report_lines = [
        "# Feature Decomposition Report",
        "",
        f"- model: `{model_name}`",
        f"- triplets: `{n_triplets}`",
        f"- layers: `{len(layer_indices)}`",
        f"- hidden size: `{hidden_size}`",
        f"- mismatched shuffles: `{n_shuffles}`",
        f"- feature manifest: `{feature_manifest_path.relative_to(project_root()).as_posix()}`",
        f"- permutation manifest: `{permutation_manifest_path.relative_to(project_root()).as_posix()}`",
        "",
        "## Condition Counts",
        "",
    ]
    for row in (
        feature_manifest.loc[feature_manifest["model"] == model_name]
        .groupby(["language", "condition"])["filepath"]
        .count()
        .reset_index(name="n_arrays")
        .itertuples(index=False)
    ):
        report_lines.append(f"- `{row.language}` / `{row.condition}`: `{int(row.n_arrays)}` arrays")
    report_lines.extend(["", "## Orthogonality Check", ""])
    orth_summary = orthogonality_df.groupby("language")[
        ["shared_specific_dot_mean", "shared_specific_dot_abs_max", "shared_norm_mean", "specific_norm_mean"]
    ].agg(["mean", "max"])
    report_lines.append(orth_summary.to_string())
    report_lines.extend(["", "## Permutation Check", ""])
    permutation_check_df = (
        permutation_df.assign(
            is_fixed_point=permutation_df["triplet_id"].astype(int) == permutation_df["permuted_triplet_id"].astype(int)
        )
        .groupby(["language", "shuffle_index"], as_index=False)["is_fixed_point"]
        .sum()
        .rename(columns={"is_fixed_point": "n_fixed_points"})
    )
    for row in permutation_check_df.itertuples(index=False):
        report_lines.append(
            f"- `{row.language}` shuffle `{int(row.shuffle_index)}` fixed points: `{int(row.n_fixed_points)}`"
        )
    report_path = _report_path()
    write_text(report_path, "\n".join(report_lines) + "\n")

    append_markdown_log(
        project_root() / "outputs" / "logs" / "progress_log.md",
        "Feature Decomposition",
        [
            f"model={model_name}",
            f"layers={len(layer_indices)}",
            f"feature_manifest={feature_manifest_path.as_posix()}",
            f"permutation_manifest={permutation_manifest_path.as_posix()}",
            f"report={report_path.as_posix()}",
        ],
    )

    return FeatureBuildSummary(
        model_name=model_name,
        feature_manifest_path=feature_manifest_path,
        permutation_manifest_path=permutation_manifest_path,
        report_path=report_path,
        n_triplets=n_triplets,
        n_layers=len(layer_indices),
        hidden_size=int(hidden_size or 0),
        n_feature_arrays=len(feature_rows),
    )
