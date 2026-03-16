from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from brain_subspace_paper.config import project_root
from brain_subspace_paper.encoding.english_prototype import _random_seed
from brain_subspace_paper.encoding.xlmr_roi_pipeline import (
    _confirmatory_effects,
    _group_results,
    _pipeline_value,
    _write_parquet_atomic,
)
from brain_subspace_paper.logging_utils import append_markdown_log, bootstrap_logs


LANGUAGES = ("en", "fr", "zh")
LANGUAGE_PAIRS = (("en", "fr"), ("en", "zh"), ("fr", "zh"))
SEMANTIC_ROIS = (
    "L_AG",
    "R_AG",
    "L_pMTG",
    "R_pMTG",
    "L_TemporalPole",
    "R_TemporalPole",
    "L_IFGtri",
    "R_IFGtri",
)


@dataclass(slots=True)
class PaperStatsSummary:
    subject_results_path: Path
    group_results_path: Path
    confirmatory_path: Path
    geometry_metrics_path: Path
    geometry_brain_coupling_path: Path
    n_subject_rows: int
    n_group_rows: int
    n_confirmatory_rows: int
    n_geometry_rows: int
    n_coupling_rows: int


def _stats_root() -> Path:
    return project_root() / "outputs" / "stats"


def _canonical_stats_paths() -> dict[str, Path]:
    root = _stats_root()
    return {
        "subject": root / "subject_level_roi_results.parquet",
        "group": root / "group_level_roi_results.parquet",
        "confirmatory": root / "confirmatory_effects.parquet",
        "geometry": root / "geometry_metrics.parquet",
        "coupling": root / "geometry_brain_coupling.parquet",
    }


def _embedding_manifest_path() -> Path:
    return project_root() / "data" / "interim" / "embeddings" / "embedding_manifest.parquet"


def _feature_manifest_path() -> Path:
    return project_root() / "data" / "processed" / "features" / "feature_manifest.parquet"


def _load_embedding_manifest(model_name: str) -> pd.DataFrame:
    manifest = pd.read_parquet(_embedding_manifest_path()).copy()
    subset = manifest.loc[manifest["model"] == model_name].copy()
    if subset.empty:
        raise FileNotFoundError(f"No embedding manifest rows found for model={model_name}.")
    return subset.sort_values(["language", "layer_index"]).reset_index(drop=True)


def _load_feature_manifest(model_name: str) -> pd.DataFrame:
    manifest = pd.read_parquet(_feature_manifest_path()).copy()
    subset = manifest.loc[manifest["model"] == model_name].copy()
    if subset.empty:
        raise FileNotFoundError(f"No feature manifest rows found for model={model_name}.")
    return subset.sort_values(["language", "layer_index", "condition", "shuffle_index"]).reset_index(drop=True)


def _load_layer_array(manifest: pd.DataFrame, *, language: str, layer_index: int) -> np.ndarray:
    rows = manifest.loc[
        (manifest["language"] == language) & (manifest["layer_index"].astype(int) == int(layer_index))
    ].copy()
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one embedding array for model={manifest['model'].iloc[0]}, language={language}, "
            f"layer={layer_index}; got {len(rows)}"
        )
    return np.load(rows.iloc[0]["filepath"]).astype(np.float32, copy=False)


def _load_feature_array(
    manifest: pd.DataFrame,
    *,
    language: str,
    layer_index: int,
    condition: str,
) -> np.ndarray:
    rows = manifest.loc[
        (manifest["language"] == language)
        & (manifest["layer_index"].astype(int) == int(layer_index))
        & (manifest["condition"] == condition)
        & (manifest["shuffle_index"].isna())
    ].copy()
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one feature array for model={manifest['model'].iloc[0]}, language={language}, "
            f"layer={layer_index}, condition={condition}; got {len(rows)}"
        )
    return np.load(rows.iloc[0]["filepath"]).astype(np.float32, copy=False)


def _retrieval_r1(left: np.ndarray, right: np.ndarray) -> float:
    similarity = left @ right.T
    predicted = np.argmax(similarity, axis=1)
    target = np.arange(left.shape[0], dtype=np.int64)
    return float(np.mean(predicted == target))


def _geometry_metrics_for_model(model_name: str) -> pd.DataFrame:
    embedding_manifest = _load_embedding_manifest(model_name)
    feature_manifest = _load_feature_manifest(model_name)
    layer_indices = sorted(embedding_manifest["layer_index"].astype(int).unique().tolist())
    max_layer_index = max(layer_indices)
    rows: list[dict[str, float | int | str]] = []

    for layer_index in layer_indices:
        arrays = {
            language: _load_layer_array(embedding_manifest, language=language, layer_index=layer_index)
            for language in LANGUAGES
        }
        same_values: list[float] = []
        mismatch_values: list[float] = []
        retrieval_values: list[float] = []
        for left_language, right_language in LANGUAGE_PAIRS:
            left = arrays[left_language]
            right = arrays[right_language]
            same_values.append(float(np.mean(np.sum(left * right, axis=1))))
            mismatch_values.append(float(np.mean(np.sum(left * np.roll(right, shift=1, axis=0), axis=1))))
            retrieval_values.append(_retrieval_r1(left, right))
            retrieval_values.append(_retrieval_r1(right, left))

        specificity_ratios: list[float] = []
        for language in LANGUAGES:
            shared = _load_feature_array(
                feature_manifest,
                language=language,
                layer_index=layer_index,
                condition="shared",
            )
            specific = _load_feature_array(
                feature_manifest,
                language=language,
                layer_index=layer_index,
                condition="specific",
            )
            shared_energy = np.sum(shared * shared, axis=1)
            specific_energy = np.sum(specific * specific, axis=1)
            ratio = specific_energy / np.clip(shared_energy + specific_energy, 1e-12, None)
            specificity_ratios.append(float(np.mean(ratio)))

        align_mean = float(np.mean(same_values))
        cas = float(np.mean(np.asarray(same_values) - np.asarray(mismatch_values)))
        rows.append(
            {
                "model": model_name,
                "layer_index": int(layer_index),
                "layer_depth": float(layer_index / max(1, max_layer_index)),
                "align_mean": align_mean,
                "cas": cas,
                "retrieval_r1_mean": float(np.mean(retrieval_values)),
                "specificity_energy": float(np.mean(specificity_ratios)),
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "layer_index"]).reset_index(drop=True)


def _brain_delta_by_layer(group_df: pd.DataFrame) -> pd.DataFrame:
    subset = group_df.loc[
        (group_df["roi_family"] == "semantic") & (group_df["condition"].isin(["shared", "specific"]))
    ].copy()
    pivot = subset.pivot_table(
        index=["model", "language", "layer_index", "layer_depth", "roi_name"],
        columns="condition",
        values="mean_z",
    ).reset_index()
    pivot["delta_shared_specific"] = pivot["shared"] - pivot["specific"]
    return (
        pivot.groupby(["model", "language", "layer_index", "layer_depth"], as_index=False)["delta_shared_specific"]
        .mean()
        .sort_values(["model", "language", "layer_index"])
        .reset_index(drop=True)
    )


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    rho = spearmanr(x, y).statistic
    return float(rho) if rho is not None and np.isfinite(rho) else float("nan")


def _layer_order_permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    observed = _spearman_correlation(x, y)
    if not np.isfinite(observed):
        return float("nan")
    n_layers = len(x)
    if n_layers <= 8:
        permuted_indices = list(itertools.permutations(range(n_layers)))
        permuted = np.array([_spearman_correlation(x[list(order)], y) for order in permuted_indices], dtype=np.float64)
        return float((1 + np.sum(np.abs(permuted) >= abs(observed) - 1e-12)) / (len(permuted) + 1))

    permuted = np.empty(n_permutations, dtype=np.float64)
    for idx in range(n_permutations):
        order = rng.permutation(n_layers)
        permuted[idx] = _spearman_correlation(x[order], y)
    return float((1 + np.sum(np.abs(permuted) >= abs(observed) - 1e-12)) / (n_permutations + 1))


def _geometry_brain_coupling(
    *,
    geometry_df: pd.DataFrame,
    group_df: pd.DataFrame,
    n_permutations: int,
) -> pd.DataFrame:
    brain_df = _brain_delta_by_layer(group_df)
    rng = np.random.default_rng(_random_seed())
    rows: list[dict[str, float | int | str]] = []

    for (model, language), subset in brain_df.groupby(["model", "language"], sort=True):
        geometry_subset = geometry_df.loc[geometry_df["model"] == model, ["layer_index", "cas"]].copy()
        merged = subset.merge(geometry_subset, on="layer_index", how="inner").sort_values("layer_index")
        if merged.empty:
            continue
        x = merged["cas"].to_numpy(dtype=np.float64)
        y = merged["delta_shared_specific"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "model": model,
                "language": language,
                "rho_spearman": _spearman_correlation(x, y),
                "p_perm": _layer_order_permutation_pvalue(x, y, n_permutations=n_permutations, rng=rng),
                "n_layers": int(len(merged)),
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "language"]).reset_index(drop=True)


def build_paper_level_stats(
    *,
    xlmr_subject_results_path: Path,
    nllb_subject_results_path: Path,
    n_permutations: int | None = None,
    n_bootstraps: int | None = None,
) -> PaperStatsSummary:
    bootstrap_logs()
    if n_permutations is None:
        n_permutations = int(_pipeline_value("statistics", "permutation_n", 10000))
    if n_bootstraps is None:
        n_bootstraps = int(_pipeline_value("statistics", "bootstrap_n", 10000))

    subject_frames = [
        pd.read_parquet(xlmr_subject_results_path).copy(),
        pd.read_parquet(nllb_subject_results_path).copy(),
    ]
    subject_df = pd.concat(subject_frames, ignore_index=True)
    subject_df = subject_df.sort_values(
        ["model", "language", "subject_id", "roi_name", "layer_index", "condition", "metric_name"]
    ).reset_index(drop=True)

    duplicate_cols = ["subject_id", "language", "model", "roi_name", "layer_index", "condition", "metric_name"]
    duplicate_mask = subject_df.duplicated(subset=duplicate_cols, keep=False)
    if duplicate_mask.any():
        raise RuntimeError(
            "Duplicate subject-level rows detected while building canonical paper stats. "
            f"Offending rows: {int(duplicate_mask.sum())}"
        )

    group_df = _group_results(subject_df)
    confirm_df = _confirmatory_effects(
        subject_df,
        n_permutations=n_permutations,
        n_bootstraps=n_bootstraps,
    )
    geometry_frames = [
        _geometry_metrics_for_model("xlmr"),
        _geometry_metrics_for_model("nllb_encoder"),
    ]
    geometry_df = pd.concat(geometry_frames, ignore_index=True).sort_values(["model", "layer_index"]).reset_index(drop=True)
    coupling_df = _geometry_brain_coupling(
        geometry_df=geometry_df,
        group_df=group_df,
        n_permutations=n_permutations,
    )

    paths = _canonical_stats_paths()
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    _write_parquet_atomic(subject_df, paths["subject"])
    _write_parquet_atomic(group_df, paths["group"])
    _write_parquet_atomic(confirm_df, paths["confirmatory"])
    _write_parquet_atomic(geometry_df, paths["geometry"])
    _write_parquet_atomic(coupling_df, paths["coupling"])

    append_markdown_log(
        project_root() / "outputs" / "logs" / "progress_log.md",
        "paper-level stats",
        [
            f"xlmr_subject_results={xlmr_subject_results_path.as_posix()}",
            f"nllb_subject_results={nllb_subject_results_path.as_posix()}",
            f"permutations={n_permutations}",
            f"bootstraps={n_bootstraps}",
            f"subject_results={paths['subject'].as_posix()}",
            f"group_results={paths['group'].as_posix()}",
            f"confirmatory={paths['confirmatory'].as_posix()}",
            f"geometry={paths['geometry'].as_posix()}",
            f"geometry_brain_coupling={paths['coupling'].as_posix()}",
        ],
    )

    return PaperStatsSummary(
        subject_results_path=paths["subject"],
        group_results_path=paths["group"],
        confirmatory_path=paths["confirmatory"],
        geometry_metrics_path=paths["geometry"],
        geometry_brain_coupling_path=paths["coupling"],
        n_subject_rows=len(subject_df),
        n_group_rows=len(group_df),
        n_confirmatory_rows=len(confirm_df),
        n_geometry_rows=len(geometry_df),
        n_coupling_rows=len(coupling_df),
    )
