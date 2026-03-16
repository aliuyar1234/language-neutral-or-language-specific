from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.glm.first_level.hemodynamic_models import glover_hrf
import numpy as np
import pandas as pd

from brain_subspace_paper.config import project_root
from brain_subspace_paper.encoding.english_prototype import _random_seed, _run_scan_counts, _run_single_condition
from brain_subspace_paper.encoding.xlmr_roi_pipeline import (
    _bootstrap_ci,
    _build_run_nuisance_and_acoustic,
    _dz,
    _language_spec,
    _load_annotation_tables,
    _load_feature_array,
    _load_feature_manifest,
    _load_roi_targets,
    _load_triplets,
    _pipeline_value,
    _se,
    _sentence_basis,
    _sign_flip_pvalue,
    _write_parquet_atomic,
)
from brain_subspace_paper.features.decomposition import _epsilon, _specific_residual
from brain_subspace_paper.logging_utils import append_markdown_log, bootstrap_logs
from brain_subspace_paper.roi.targets import _roi_indices_from_metadata


LANGUAGE_ORDER = ("en", "fr", "zh")
MODEL_ORDER = ("xlmr", "nllb_encoder")
MODEL_LABELS = {"xlmr": "XLM-R", "nllb_encoder": "NLLB encoder"}
LANGUAGE_LABELS = {"en": "EN", "fr": "FR", "zh": "ZH"}
LANGUAGE_TEXT_COLUMNS = {"en": "en_text", "fr": "fr_text", "zh": "zh_text"}
NLLB_LANGUAGE_CODES = {"en": "eng_Latn", "fr": "fra_Latn", "zh": "zho_Hans"}
ROBUSTNESS_ORDER = (
    "last_token_pooling",
    "fir_4lag",
    "no_acoustic_nuisance",
    "no_pitch_nuisance",
    "previous_2_sentence_context",
    "voxelwise_within_roi_mean_z",
)
ROBUSTNESS_LABELS = {
    "last_token_pooling": "Last-token pooling",
    "fir_4lag": "4-lag FIR",
    "no_acoustic_nuisance": "No acoustic nuisance",
    "no_pitch_nuisance": "No pitch nuisance",
    "previous_2_sentence_context": "Previous-2 context",
    "voxelwise_within_roi_mean_z": "Voxelwise within ROI",
}


@dataclass(slots=True)
class RobustnessBuildSummary:
    cell_results_path: Path
    summary_path: Path
    representative_layers_path: Path
    table05_path: Path
    figure09_path: Path
    n_conditions: int
    n_cells: int


@dataclass(slots=True)
class RobustnessLanguageSupport:
    language: str
    roi_manifest: pd.DataFrame
    semantic_roi_indices: tuple[int, ...]
    semantic_roi_names: tuple[str, ...]
    subject_ids: tuple[str, ...]
    run_order: tuple[int, ...]
    row_positions_by_run: dict[int, np.ndarray]
    sentence_basis_by_run: dict[int, np.ndarray]
    fir_event_basis_by_run: dict[int, np.ndarray]
    nuisance_by_mode: dict[str, dict[int, np.ndarray]]
    roi_path_by_subject_run: dict[tuple[str, int], Path]
    bold_path_by_subject_run: dict[tuple[str, int], Path]


_LANGUAGE_SUPPORT_CACHE: dict[str, RobustnessLanguageSupport] = {}
_SEMANTIC_VOXEL_INDICES_CACHE: list[np.ndarray] | None = None


def _normalize_conditions(conditions: tuple[str, ...] | None) -> tuple[str, ...]:
    if conditions is None:
        return ROBUSTNESS_ORDER
    normalized = tuple(str(condition).strip() for condition in conditions if str(condition).strip())
    invalid = [condition for condition in normalized if condition not in ROBUSTNESS_LABELS]
    if invalid:
        raise ValueError(
            f"Unsupported robustness conditions: {', '.join(invalid)}. "
            f"Expected a subset of {', '.join(ROBUSTNESS_ORDER)}."
        )
    return normalized


def _stats_root() -> Path:
    return project_root() / "outputs" / "stats"


def _tables_root() -> Path:
    return project_root() / "outputs" / "tables"


def _figures_root() -> Path:
    return project_root() / "outputs" / "figures"


def _tagged_path(path: Path, output_tag: str | None) -> Path:
    if not output_tag:
        return path
    return path.with_name(f"{path.stem}__{output_tag}{path.suffix}")


def _robustness_feature_cache_root() -> Path:
    return _stats_root() / "robustness_feature_cache"


def _robustness_voxel_cache_root() -> Path:
    return project_root() / "outputs" / "cache" / "robustness_voxels"


def _variant_feature_cache_path(
    *,
    model_name: str,
    layer_index: int,
    pooling_mode: str,
    previous_sentences: int,
    language: str,
) -> Path:
    filename = (
        f"{model_name}__layer{int(layer_index):02d}"
        f"__pool-{pooling_mode}__ctx-{int(previous_sentences)}"
        f"__lang-{language}.npy"
    )
    return _robustness_feature_cache_root() / filename


def _voxel_cache_path(*, language: str, subject_id: str, run_index: int, roi_name: str) -> Path:
    safe_roi = roi_name.replace("/", "_")
    return _robustness_voxel_cache_root() / language / subject_id / f"run_{int(run_index):02d}__{safe_roi}.npy"


def _write_npy_atomic(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("wb") as handle:
        np.save(handle, array)
    temp_path.replace(path)


def _as_absolute_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_root() / path


def _robustness_paths(output_tag: str | None = None) -> dict[str, Path]:
    root = _stats_root()
    return {
        "group": root / "group_level_roi_results.parquet",
        "subject": root / "subject_level_roi_results.parquet",
        "cell_results": _tagged_path(root / "robustness_cell_results.parquet", output_tag),
        "summary": _tagged_path(root / "robustness_summary.parquet", output_tag),
        "representative_layers": _tagged_path(root / "robustness_representative_layers.parquet", output_tag),
        "table05": _tagged_path(_tables_root() / "table05_robustness_summary.csv", output_tag),
        "fig09": _tagged_path(_figures_root() / "fig09_robustness_summary.png", output_tag),
    }


def _representative_layers(group_df: pd.DataFrame) -> pd.DataFrame:
    semantic = group_df.loc[
        (group_df["roi_family"] == "semantic")
        & (group_df["condition"].isin(["shared", "specific"]))
    ].copy()
    pivot = semantic.pivot_table(
        index=["model", "language", "layer_index", "layer_depth", "roi_name"],
        columns="condition",
        values="mean_z",
    ).reset_index()
    pivot["delta"] = pivot["shared"] - pivot["specific"]
    by_layer = (
        pivot.groupby(["model", "layer_index", "layer_depth"], as_index=False)["delta"]
        .mean()
        .sort_values(["model", "delta", "layer_index"], ascending=[True, False, True])
    )
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        model_rows = by_layer.loc[by_layer["model"] == model]
        if model_rows.empty:
            continue
        best = model_rows.iloc[0]
        rows.append(
            {
                "model": model,
                "layer_index": int(best["layer_index"]),
                "layer_depth": float(best["layer_depth"]),
                "mean_semantic_delta": float(best["delta"]),
            }
        )
    return pd.DataFrame(rows)


def _semantic_lookup(language: str) -> tuple[pd.DataFrame, list[int], list[str]]:
    _, roi_lookup = _load_roi_targets(language)
    semantic = roi_lookup.loc[roi_lookup["roi_family"] == "semantic"].copy()
    semantic = semantic.sort_values("roi_name").reset_index(drop=True)
    roi_indices = (semantic["roi_index"].astype(int) - 1).tolist()
    roi_names = semantic["roi_name"].astype(str).tolist()
    return semantic, roi_indices, roi_names


def _prepare_language_support(language: str) -> RobustnessLanguageSupport:
    cached = _LANGUAGE_SUPPORT_CACHE.get(language)
    if cached is not None:
        return cached

    triplets = _load_triplets().reset_index(names="triplet_row_index")
    roi_manifest, roi_lookup = _load_roi_targets(language)
    semantic = roi_lookup.loc[roi_lookup["roi_family"] == "semantic"].copy()
    semantic = semantic.sort_values("roi_name").reset_index(drop=True)

    spec = _language_spec(language)
    fine_hz = float(_pipeline_value("design_matrix", "fine_grid_hz", 10))
    tr = float(_pipeline_value("design_matrix", "tr_sec", 2.0))
    hrf = glover_hrf(t_r=fine_hz ** -1, oversampling=1)
    prosody, word_info = _load_annotation_tables(language)
    scan_counts = _run_scan_counts(roi_manifest)

    row_positions_by_run: dict[int, np.ndarray] = {}
    sentence_basis_by_run: dict[int, np.ndarray] = {}
    fir_event_basis_by_run: dict[int, np.ndarray] = {}
    nuisance_by_mode = {
        "full": {},
        "no_acoustic": {},
        "no_pitch": {},
    }

    for run_index, n_scans in scan_counts.items():
        run_triplets = (
            triplets.loc[triplets["section_index"].astype(int) == int(run_index)]
            .copy()
            .reset_index(drop=True)
        )
        row_positions_by_run[int(run_index)] = run_triplets["triplet_row_index"].to_numpy(dtype=np.int64)
        nuisance_df, acoustic_df = _build_run_nuisance_and_acoustic(
            language=language,
            run_index=int(run_index),
            n_scans=int(n_scans),
            tr=tr,
            fine_hz=fine_hz,
            hrf=hrf,
            prosody=prosody,
            word_info=word_info,
            sentence_onsets=run_triplets[spec["onset_col"]].to_numpy(dtype=np.float32),
        )
        nuisance_by_mode["full"][int(run_index)] = _selected_text_nuisance(
            nuisance_df=nuisance_df,
            acoustic_df=acoustic_df,
            nuisance_mode="full",
        )
        nuisance_by_mode["no_acoustic"][int(run_index)] = _selected_text_nuisance(
            nuisance_df=nuisance_df,
            acoustic_df=acoustic_df,
            nuisance_mode="no_acoustic",
        )
        nuisance_by_mode["no_pitch"][int(run_index)] = _selected_text_nuisance(
            nuisance_df=nuisance_df,
            acoustic_df=acoustic_df,
            nuisance_mode="no_pitch",
        )
        sentence_basis_by_run[int(run_index)] = _sentence_basis(
            run_triplets,
            onset_col=spec["onset_col"],
            offset_col=spec["offset_col"],
            n_scans=int(n_scans),
            tr=tr,
            fine_hz=fine_hz,
            hrf=hrf,
        ).astype(np.float32, copy=False)
        fir_event_basis_by_run[int(run_index)] = _sentence_onset_basis(
            run_triplets,
            onset_col=spec["onset_col"],
            n_scans=int(n_scans),
            tr=tr,
        ).astype(np.float32, copy=False)

    nuisance_by_mode = {
        mode: _align_nuisance_widths(matrices)
        for mode, matrices in nuisance_by_mode.items()
    }

    roi_path_by_subject_run: dict[tuple[str, int], Path] = {}
    bold_path_by_subject_run: dict[tuple[str, int], Path] = {}
    for row in roi_manifest.itertuples(index=False):
        key = (str(row.subject_id), int(row.canonical_run_index))
        roi_path_by_subject_run[key] = _as_absolute_path(row.filepath)
        bold_path_by_subject_run[key] = _as_absolute_path(row.bold_filepath)

    support = RobustnessLanguageSupport(
        language=language,
        roi_manifest=roi_manifest,
        semantic_roi_indices=tuple((semantic["roi_index"].astype(int) - 1).tolist()),
        semantic_roi_names=tuple(semantic["roi_name"].astype(str).tolist()),
        subject_ids=tuple(sorted(roi_manifest["subject_id"].astype(str).unique().tolist())),
        run_order=tuple(sorted(scan_counts)),
        row_positions_by_run=row_positions_by_run,
        sentence_basis_by_run=sentence_basis_by_run,
        fir_event_basis_by_run=fir_event_basis_by_run,
        nuisance_by_mode=nuisance_by_mode,
        roi_path_by_subject_run=roi_path_by_subject_run,
        bold_path_by_subject_run=bold_path_by_subject_run,
    )
    _LANGUAGE_SUPPORT_CACHE[language] = support
    return support


def _contextualized_texts(triplets: pd.DataFrame, *, language: str, previous_sentences: int) -> list[str]:
    if previous_sentences <= 0:
        return triplets[LANGUAGE_TEXT_COLUMNS[language]].astype(str).tolist()

    spec = _language_spec(language)
    onset_col = spec["onset_col"]
    ordered = (
        triplets.reset_index(names="triplet_row_index")
        .sort_values(["section_index", onset_col, "triplet_id"])
        .reset_index(drop=True)
    )
    texts = [""] * len(triplets)
    history_by_section: dict[int, list[str]] = {}
    for row in ordered.itertuples(index=False):
        section_index = int(row.section_index)
        current = str(getattr(row, LANGUAGE_TEXT_COLUMNS[language]))
        history = history_by_section.setdefault(section_index, [])
        prefix = " ".join(history[-previous_sentences:])
        texts[int(row.triplet_row_index)] = f"{prefix} {current}".strip() if prefix else current
        history.append(current)
    return texts


def _extract_variant_arrays(
    *,
    model_name: str,
    layer_index: int,
    pooling_mode: str,
    previous_sentences: int,
    max_tokens_per_batch: int,
) -> dict[str, np.ndarray]:
    import torch
    from brain_subspace_paper.models.extraction import (
        _batch_index_groups,
        _load_all_triplets,
        _load_model_bundle,
        _normalize_rows,
    )

    tokenizer, model, device = _load_model_bundle(model_name)
    model.eval()
    triplets = _load_all_triplets()
    arrays: dict[str, np.ndarray] = {}

    for language in LANGUAGE_ORDER:
        cache_path = _variant_feature_cache_path(
            model_name=model_name,
            layer_index=layer_index,
            pooling_mode=pooling_mode,
            previous_sentences=previous_sentences,
            language=language,
        )
        if cache_path.exists():
            arrays[language] = np.load(cache_path).astype(np.float32, copy=False)
            continue

        texts = _contextualized_texts(triplets, language=language, previous_sentences=previous_sentences)
        tokenizer_kwargs: dict[str, Any] = {"add_special_tokens": True, "truncation": True}
        if model_name == "nllb_encoder":
            tokenizer.src_lang = NLLB_LANGUAGE_CODES[language]
        token_lengths = [len(ids) for ids in tokenizer(texts, **tokenizer_kwargs)["input_ids"]]
        batches = _batch_index_groups(token_lengths, max_tokens_per_batch=max_tokens_per_batch)

        pooled_array: np.ndarray | None = None
        for batch_indices in batches:
            batch_texts = [texts[idx] for idx in batch_indices]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            pooling_mask = encoded["attention_mask"].bool() & ~encoded["special_tokens_mask"].bool()
            with torch.inference_mode():
                if model_name == "nllb_encoder":
                    outputs = model.get_encoder()(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        output_hidden_states=True,
                        return_dict=True,
                    )
                else:
                    outputs = model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        output_hidden_states=True,
                        return_dict=True,
                    )
            hidden = outputs.hidden_states[int(layer_index)]
            if pooled_array is None:
                pooled_array = np.zeros((len(triplets), int(hidden.shape[-1])), dtype=np.float32)
            if pooling_mode == "last_token":
                token_positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
                last_idx = (pooling_mask.to(torch.int64) * token_positions).amax(dim=1)
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx]
                pooled = pooled.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                token_counts = pooling_mask.sum(dim=1).clamp(min=1)
                masked = hidden * pooling_mask.unsqueeze(-1)
                pooled = (masked.sum(dim=1) / token_counts.unsqueeze(-1)).detach().cpu().numpy().astype(np.float32, copy=False)
            pooled = _normalize_rows(pooled)
            for row_offset, triplet_index in enumerate(batch_indices):
                pooled_array[triplet_index] = pooled[row_offset]

        if pooled_array is None:
            raise RuntimeError(
                f"Failed to extract variant embeddings for model={model_name}, language={language}, layer={layer_index}."
            )
        _write_npy_atomic(pooled_array, cache_path)
        arrays[language] = pooled_array
    return arrays


def _build_in_memory_features(arrays_by_language: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    eps = _epsilon()
    features: dict[str, dict[str, np.ndarray]] = {}
    for target_language in LANGUAGE_ORDER:
        other_languages = [language for language in LANGUAGE_ORDER if language != target_language]
        raw = arrays_by_language[target_language]
        shared = ((arrays_by_language[other_languages[0]] + arrays_by_language[other_languages[1]]) / 2.0).astype(
            np.float32,
            copy=False,
        )
        specific = _specific_residual(raw, shared, eps=eps).astype(np.float32, copy=False)
        features[target_language] = {
            "shared": shared,
            "specific": specific,
        }
    return features


def _sentence_onset_basis(
    run_triplets: pd.DataFrame,
    *,
    onset_col: str,
    n_scans: int,
    tr: float,
) -> np.ndarray:
    basis = np.zeros((n_scans, len(run_triplets)), dtype=np.float32)
    onset_scans = np.clip(np.round(run_triplets[onset_col].to_numpy(dtype=np.float32) / tr).astype(int), 0, n_scans - 1)
    for col, onset_scan in enumerate(onset_scans.tolist()):
        basis[onset_scan, col] += 1.0
    return basis


def _fir_design(
    run_triplets: pd.DataFrame,
    *,
    onset_col: str,
    n_scans: int,
    tr: float,
    feature_array: np.ndarray,
    n_lags: int = 4,
) -> np.ndarray:
    event_basis = _sentence_onset_basis(run_triplets, onset_col=onset_col, n_scans=n_scans, tr=tr)
    row_positions = run_triplets["triplet_row_index"].to_numpy(dtype=np.int64)
    event_features = (event_basis @ feature_array[row_positions, :]).astype(np.float32, copy=False)
    lagged: list[np.ndarray] = []
    for lag in range(n_lags):
        shifted = np.zeros_like(event_features)
        if lag == 0:
            shifted = event_features
        else:
            shifted[lag:, :] = event_features[:-lag, :]
        lagged.append(shifted)
    return np.concatenate(lagged, axis=1).astype(np.float32, copy=False)


def _selected_text_nuisance(
    *,
    nuisance_df: pd.DataFrame,
    acoustic_df: pd.DataFrame,
    nuisance_mode: str,
) -> np.ndarray:
    if nuisance_mode == "no_acoustic":
        return nuisance_df.to_numpy(dtype=np.float32, copy=False)
    if nuisance_mode == "no_pitch":
        acoustic_no_pitch = acoustic_df.drop(columns=["f0"], errors="ignore")
        return np.column_stack(
            [
                nuisance_df.to_numpy(dtype=np.float32, copy=False),
                acoustic_no_pitch.to_numpy(dtype=np.float32, copy=False),
            ]
        ).astype(np.float32, copy=False)
    return np.column_stack(
        [
            nuisance_df.to_numpy(dtype=np.float32, copy=False),
            acoustic_df.to_numpy(dtype=np.float32, copy=False),
        ]
    ).astype(np.float32, copy=False)


def _align_nuisance_widths(nuisance_by_run: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    max_cols = max(matrix.shape[1] for matrix in nuisance_by_run.values())
    aligned: dict[int, np.ndarray] = {}
    for run_index, matrix in nuisance_by_run.items():
        if matrix.shape[1] == max_cols:
            aligned[run_index] = matrix.astype(np.float32, copy=False)
            continue
        padded = np.zeros((matrix.shape[0], max_cols), dtype=np.float32)
        padded[:, : matrix.shape[1]] = matrix.astype(np.float32, copy=False)
        aligned[run_index] = padded
    return aligned


def _semantic_voxel_indices() -> list[np.ndarray]:
    global _SEMANTIC_VOXEL_INDICES_CACHE
    if _SEMANTIC_VOXEL_INDICES_CACHE is not None:
        return _SEMANTIC_VOXEL_INDICES_CACHE
    atlas_path = project_root() / "data" / "interim" / "roi" / "harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz"
    roi_metadata_path = project_root() / "data" / "interim" / "roi" / "roi_metadata.parquet"
    atlas_img = nib.load(str(atlas_path))
    roi_metadata = pd.read_parquet(roi_metadata_path).copy()
    semantic = roi_metadata.loc[roi_metadata["family"] == "semantic"].sort_values("roi_name").reset_index(drop=True)
    _SEMANTIC_VOXEL_INDICES_CACHE = _roi_indices_from_metadata(atlas_img, semantic)
    return _SEMANTIC_VOXEL_INDICES_CACHE


def _ensure_subject_run_voxel_cache(
    *,
    support: RobustnessLanguageSupport,
    subject_id: str,
    run_index: int,
    voxel_indices_by_roi: list[np.ndarray],
) -> None:
    missing: list[tuple[str, np.ndarray, Path]] = []
    for roi_name, voxel_indices in zip(support.semantic_roi_names, voxel_indices_by_roi, strict=True):
        cache_path = _voxel_cache_path(
            language=support.language,
            subject_id=subject_id,
            run_index=run_index,
            roi_name=roi_name,
        )
        if cache_path.exists():
            continue
        missing.append((roi_name, voxel_indices, cache_path))
    if not missing:
        return

    bold_path = support.bold_path_by_subject_run[(subject_id, run_index)]
    img = nib.load(str(bold_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    flat = data.reshape(-1, data.shape[-1])
    for _, voxel_indices, cache_path in missing:
        voxel_ts = flat[voxel_indices, :].T.astype(np.float32, copy=False)
        _write_npy_atomic(voxel_ts, cache_path)


def _load_cached_voxel_timeseries(
    *,
    support: RobustnessLanguageSupport,
    subject_id: str,
    run_index: int,
    roi_name: str,
) -> np.ndarray:
    cache_path = _voxel_cache_path(
        language=support.language,
        subject_id=subject_id,
        run_index=run_index,
        roi_name=roi_name,
    )
    return np.load(cache_path, mmap_mode="r")


def _voxelwise_roi_workers() -> int:
    raw_value = os.environ.get("BRAIN_SUBSPACE_VOXELWISE_ROI_WORKERS", "").strip()
    if raw_value:
        try:
            return max(1, int(raw_value))
        except ValueError:
            pass
    return 1


def _batched_roi_mean_values(
    *,
    support: RobustnessLanguageSupport,
    subject_ids: list[str],
    condition_designs: dict[str, dict[int, np.ndarray]],
    nuisance_by_run: dict[int, np.ndarray],
) -> np.ndarray:
    n_rois = len(support.semantic_roi_indices)
    roi_series_by_run: dict[int, np.ndarray] = {}
    for run_index in support.run_order:
        blocks: list[np.ndarray] = []
        for subject_id in subject_ids:
            roi_path = support.roi_path_by_subject_run[(subject_id, run_index)]
            roi_data = np.load(roi_path, mmap_mode="r")
            blocks.append(np.asarray(roi_data[:, support.semantic_roi_indices], dtype=np.float32))
        roi_series_by_run[run_index] = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)

    shared_result = _run_single_condition(
        run_designs={run_index: condition_designs["shared"][run_index] for run_index in support.run_order},
        z_by_run={run_index: nuisance_by_run[run_index] for run_index in support.run_order},
        roi_series_by_run=roi_series_by_run,
    )[1]
    specific_result = _run_single_condition(
        run_designs={run_index: condition_designs["specific"][run_index] for run_index in support.run_order},
        z_by_run={run_index: nuisance_by_run[run_index] for run_index in support.run_order},
        roi_series_by_run=roi_series_by_run,
    )[1]
    delta = (shared_result - specific_result).reshape(len(subject_ids), n_rois)
    return delta.mean(axis=1).astype(np.float64, copy=False)


def _voxelwise_roi_delta_by_subject(
    *,
    support: RobustnessLanguageSupport,
    subject_ids: list[str],
    condition_designs: dict[str, dict[int, np.ndarray]],
    nuisance_by_run: dict[int, np.ndarray],
    roi_name: str,
) -> np.ndarray:
    voxel_series_by_run: dict[int, np.ndarray] = {}
    voxel_count = 0
    for run_index in support.run_order:
        blocks: list[np.ndarray] = []
        for subject_id in subject_ids:
            voxel_ts = _load_cached_voxel_timeseries(
                support=support,
                subject_id=subject_id,
                run_index=run_index,
                roi_name=roi_name,
            )
            blocks.append(np.asarray(voxel_ts, dtype=np.float32))
        voxel_series_by_run[run_index] = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
        if voxel_count == 0:
            voxel_count = int(blocks[0].shape[1])

    shared_result = _run_single_condition(
        run_designs={run_index: condition_designs["shared"][run_index] for run_index in support.run_order},
        z_by_run={run_index: nuisance_by_run[run_index] for run_index in support.run_order},
        roi_series_by_run=voxel_series_by_run,
    )[1]
    specific_result = _run_single_condition(
        run_designs={run_index: condition_designs["specific"][run_index] for run_index in support.run_order},
        z_by_run={run_index: nuisance_by_run[run_index] for run_index in support.run_order},
        roi_series_by_run=voxel_series_by_run,
    )[1]
    delta = (shared_result - specific_result).reshape(len(subject_ids), voxel_count)
    return delta.mean(axis=1)


def _batched_voxelwise_mean_values(
    *,
    support: RobustnessLanguageSupport,
    subject_ids: list[str],
    condition_designs: dict[str, dict[int, np.ndarray]],
    nuisance_by_run: dict[int, np.ndarray],
    voxel_indices_by_roi: list[np.ndarray],
) -> np.ndarray:
    for subject_id in subject_ids:
        for run_index in support.run_order:
            _ensure_subject_run_voxel_cache(
                support=support,
                subject_id=subject_id,
                run_index=run_index,
                voxel_indices_by_roi=voxel_indices_by_roi,
            )

    roi_workers = min(_voxelwise_roi_workers(), len(support.semantic_roi_names))
    if roi_workers <= 1:
        roi_deltas = [
            _voxelwise_roi_delta_by_subject(
                support=support,
                subject_ids=subject_ids,
                condition_designs=condition_designs,
                nuisance_by_run=nuisance_by_run,
                roi_name=roi_name,
            )
            for roi_name in support.semantic_roi_names
        ]
    else:
        with ThreadPoolExecutor(max_workers=roi_workers) as executor:
            futures = [
                executor.submit(
                    _voxelwise_roi_delta_by_subject,
                    support=support,
                    subject_ids=subject_ids,
                    condition_designs=condition_designs,
                    nuisance_by_run=nuisance_by_run,
                    roi_name=roi_name,
                )
                for roi_name in support.semantic_roi_names
            ]
            roi_deltas = [future.result() for future in futures]

    return np.mean(np.column_stack(roi_deltas), axis=1).astype(np.float64, copy=False)


def _subject_semantic_delta(
    *,
    subject_runs: pd.DataFrame,
    semantic_roi_names: list[str],
    roi_indices: list[int],
    voxel_indices_by_roi: list[np.ndarray] | None,
    condition_designs: dict[str, dict[int, np.ndarray]],
    nuisance_by_run: dict[int, np.ndarray],
    target_mode: str,
) -> float:
    if target_mode == "roi_mean":
        roi_run_data = {
            int(row.canonical_run_index): np.load(_as_absolute_path(row.filepath)).astype(np.float32, copy=False)[:, roi_indices]
            for row in subject_runs.itertuples(index=False)
        }
        shared_result = _run_single_condition(
            run_designs={run_index: condition_designs["shared"][run_index] for run_index in roi_run_data},
            z_by_run={run_index: nuisance_by_run[run_index] for run_index in roi_run_data},
            roi_series_by_run=roi_run_data,
        )[1]
        specific_result = _run_single_condition(
            run_designs={run_index: condition_designs["specific"][run_index] for run_index in roi_run_data},
            z_by_run={run_index: nuisance_by_run[run_index] for run_index in roi_run_data},
            roi_series_by_run=roi_run_data,
        )[1]
        return float(np.mean(shared_result - specific_result))

    if voxel_indices_by_roi is None:
        raise RuntimeError("voxel_indices_by_roi is required for voxelwise target mode.")

    delta_by_roi: list[float] = []
    for roi_name, voxel_indices in zip(semantic_roi_names, voxel_indices_by_roi, strict=True):
        voxel_run_data: dict[int, np.ndarray] = {}
        for row in subject_runs.itertuples(index=False):
            bold_path = _as_absolute_path(row.bold_filepath)
            img = nib.load(str(bold_path))
            data = np.asarray(img.dataobj, dtype=np.float32)
            flat = data.reshape(-1, data.shape[-1])
            voxel_ts = flat[voxel_indices, :].T.astype(np.float32, copy=False)
            voxel_run_data[int(row.canonical_run_index)] = voxel_ts
        shared_result = _run_single_condition(
            run_designs={run_index: condition_designs["shared"][run_index] for run_index in voxel_run_data},
            z_by_run={run_index: nuisance_by_run[run_index] for run_index in voxel_run_data},
            roi_series_by_run=voxel_run_data,
        )[1]
        specific_result = _run_single_condition(
            run_designs={run_index: condition_designs["specific"][run_index] for run_index in voxel_run_data},
            z_by_run={run_index: nuisance_by_run[run_index] for run_index in voxel_run_data},
            roi_series_by_run=voxel_run_data,
        )[1]
        delta_by_roi.append(float(np.mean(shared_result - specific_result)))
    return float(np.mean(delta_by_roi))


def _condition_designs_for_language(
    *,
    support: RobustnessLanguageSupport,
    feature_arrays: dict[str, np.ndarray],
    design_mode: str,
    nuisance_mode: str,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[int, np.ndarray], pd.DataFrame]:
    if nuisance_mode not in support.nuisance_by_mode:
        raise ValueError(f"Unsupported nuisance_mode={nuisance_mode!r}.")
    designs = {"shared": {}, "specific": {}}
    for run_index in support.run_order:
        row_positions = support.row_positions_by_run[run_index]
        for condition in ("shared", "specific"):
            if design_mode == "fir_4lag":
                event_features = (
                    support.fir_event_basis_by_run[run_index] @ feature_arrays[condition][row_positions, :]
                ).astype(np.float32, copy=False)
                lagged: list[np.ndarray] = []
                for lag in range(4):
                    shifted = np.zeros_like(event_features)
                    if lag == 0:
                        shifted = event_features
                    else:
                        shifted[lag:, :] = event_features[:-lag, :]
                    lagged.append(shifted)
                design = np.concatenate(lagged, axis=1).astype(np.float32, copy=False)
            else:
                design = (
                    support.sentence_basis_by_run[run_index] @ feature_arrays[condition][row_positions, :]
                ).astype(np.float32, copy=False)
            designs[condition][run_index] = design

    return designs, support.nuisance_by_mode[nuisance_mode], support.roi_manifest


def _cell_result_rows_from_values(
    *,
    robustness_condition: str,
    model_name: str,
    language: str,
    layer_index: int,
    layer_depth: float,
    values: np.ndarray,
    n_permutations: int,
    n_bootstraps: int,
    note: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(_random_seed() + layer_index + len(values))
    ci_low, ci_high = _bootstrap_ci(values, n_bootstraps=n_bootstraps, rng=rng)
    mean_delta = float(np.mean(values))
    return {
        "robustness_condition": robustness_condition,
        "model": model_name,
        "language": language,
        "layer_index": int(layer_index),
        "layer_depth": float(layer_depth),
        "mean_delta": mean_delta,
        "se": _se(values),
        "dz": _dz(values),
        "p_perm": _sign_flip_pvalue(values, n_permutations=n_permutations, rng=rng),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_subjects": int(len(values)),
        "sign_positive": bool(mean_delta > 0.0),
        "note": note,
    }


def _base_condition_rows(
    *,
    subject_df: pd.DataFrame,
    representative_layers: pd.DataFrame,
    n_permutations: int,
    n_bootstraps: int,
) -> list[dict[str, Any]]:
    subset = subject_df.loc[
        (subject_df["metric_name"] == "z")
        & (subject_df["roi_family"] == "semantic")
        & (subject_df["condition"].isin(["shared", "specific"]))
    ].copy()
    rows: list[dict[str, Any]] = []
    for rep in representative_layers.itertuples(index=False):
        layer_subset = subset.loc[
            (subset["model"] == rep.model)
            & (subset["layer_index"].astype(int) == int(rep.layer_index))
        ].copy()
        pivot = layer_subset.pivot_table(
            index=["subject_id", "language", "model", "roi_name"],
            columns="condition",
            values="value",
        ).reset_index()
        pivot["delta"] = pivot["shared"] - pivot["specific"]
        semantic = (
            pivot.groupby(["subject_id", "language", "model"], as_index=False)["delta"]
            .mean()
            .rename(columns={"delta": "semantic_delta"})
        )
        for language in LANGUAGE_ORDER:
            values = semantic.loc[
                (semantic["model"] == rep.model) & (semantic["language"] == language),
                "semantic_delta",
            ].to_numpy(dtype=np.float64)
            rows.append(
                _cell_result_rows_from_values(
                    robustness_condition="canonical_representative_layer_base",
                    model_name=str(rep.model),
                    language=language,
                    layer_index=int(rep.layer_index),
                    layer_depth=float(rep.layer_depth),
                    values=values,
                    n_permutations=n_permutations,
                    n_bootstraps=n_bootstraps,
                    note="Reference semantic H1 at the canonical representative layer using the saved ROI-mean checkpoint.",
                )
            )
    return rows


def _variant_rows(
    *,
    robustness_condition: str,
    representative_layers: pd.DataFrame,
    n_permutations: int,
    n_bootstraps: int,
    max_subjects: int | None,
    max_tokens_per_batch: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    voxel_indices_by_roi = _semantic_voxel_indices() if robustness_condition == "voxelwise_within_roi_mean_z" else None
    in_memory_feature_cache: dict[tuple[str, int, str], dict[str, dict[str, np.ndarray]]] = {}
    condition_settings = {
        "last_token_pooling": {
            "pooling_mode": "last_token",
            "previous_sentences": 0,
            "design_mode": "hrf",
            "nuisance_mode": "full",
            "target_mode": "roi_mean",
            "note": "Representative-layer semantic H1 using last-token pooled embeddings.",
        },
        "fir_4lag": {
            "pooling_mode": "mean",
            "previous_sentences": 0,
            "design_mode": "fir_4lag",
            "nuisance_mode": "full",
            "target_mode": "roi_mean",
            "note": "Representative-layer semantic H1 using onset-based 4-lag FIR text regressors.",
        },
        "no_acoustic_nuisance": {
            "pooling_mode": "mean",
            "previous_sentences": 0,
            "design_mode": "hrf",
            "nuisance_mode": "no_acoustic",
            "target_mode": "roi_mean",
            "note": "Representative-layer semantic H1 after removing all acoustic nuisance regressors from text-model residualization.",
        },
        "no_pitch_nuisance": {
            "pooling_mode": "mean",
            "previous_sentences": 0,
            "design_mode": "hrf",
            "nuisance_mode": "no_pitch",
            "target_mode": "roi_mean",
            "note": "Representative-layer semantic H1 after removing only the pitch/F0 nuisance regressor.",
        },
        "previous_2_sentence_context": {
            "pooling_mode": "mean",
            "previous_sentences": 2,
            "design_mode": "hrf",
            "nuisance_mode": "full",
            "target_mode": "roi_mean",
            "note": "Representative-layer semantic H1 using previous-two-sentence context in the text encoder input.",
        },
        "voxelwise_within_roi_mean_z": {
            "pooling_mode": "mean",
            "previous_sentences": 0,
            "design_mode": "hrf",
            "nuisance_mode": "full",
            "target_mode": "voxelwise_mean_z",
            "note": "Representative-layer semantic H1 computed from within-ROI voxelwise predictions averaged as mean Fisher-z.",
        },
    }
    settings = condition_settings[robustness_condition]

    for rep in representative_layers.itertuples(index=False):
        model_name = str(rep.model)
        layer_index = int(rep.layer_index)
        layer_depth = float(rep.layer_depth)

        if settings["pooling_mode"] != "mean" or int(settings["previous_sentences"]) > 0:
            cache_key = (model_name, layer_index, robustness_condition)
            if cache_key not in in_memory_feature_cache:
                extracted_arrays = _extract_variant_arrays(
                    model_name=model_name,
                    layer_index=layer_index,
                    pooling_mode=str(settings["pooling_mode"]),
                    previous_sentences=int(settings["previous_sentences"]),
                    max_tokens_per_batch=max_tokens_per_batch,
                )
                in_memory_feature_cache[cache_key] = _build_in_memory_features(extracted_arrays)
            feature_by_language = in_memory_feature_cache[cache_key]
        else:
            feature_by_language = {}
            for language in LANGUAGE_ORDER:
                language_manifest = _load_feature_manifest(model_name, language)
                feature_by_language[language] = {
                    "shared": _load_feature_array(
                        language_manifest,
                        model_name=model_name,
                        language=language,
                        condition="shared",
                        layer_index=layer_index,
                    ),
                    "specific": _load_feature_array(
                        language_manifest,
                        model_name=model_name,
                        language=language,
                        condition="specific",
                        layer_index=layer_index,
                    ),
                }

        for language in LANGUAGE_ORDER:
            support = _prepare_language_support(language)
            designs, nuisance_by_run, roi_manifest = _condition_designs_for_language(
                support=support,
                feature_arrays=feature_by_language[language],
                design_mode=str(settings["design_mode"]),
                nuisance_mode=str(settings["nuisance_mode"]),
            )
            subject_ids = list(support.subject_ids)
            if max_subjects is not None:
                subject_ids = subject_ids[:max_subjects]
            if settings["target_mode"] == "roi_mean":
                values = _batched_roi_mean_values(
                    support=support,
                    subject_ids=subject_ids,
                    condition_designs=designs,
                    nuisance_by_run=nuisance_by_run,
                )
            elif settings["target_mode"] == "voxelwise_mean_z":
                if voxel_indices_by_roi is None:
                    raise RuntimeError("voxel_indices_by_roi is required for voxelwise target mode.")
                values = _batched_voxelwise_mean_values(
                    support=support,
                    subject_ids=subject_ids,
                    condition_designs=designs,
                    nuisance_by_run=nuisance_by_run,
                    voxel_indices_by_roi=voxel_indices_by_roi,
                )
            else:
                raise ValueError(f"Unsupported target_mode={settings['target_mode']!r}.")

            rows.append(
                _cell_result_rows_from_values(
                    robustness_condition=robustness_condition,
                    model_name=model_name,
                    language=language,
                    layer_index=layer_index,
                    layer_depth=layer_depth,
                    values=np.asarray(values, dtype=np.float64),
                    n_permutations=n_permutations,
                    n_bootstraps=n_bootstraps,
                    note=str(settings["note"]),
                )
            )
    return rows


def _summarize_robustness(cell_df: pd.DataFrame, *, conditions: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_df = cell_df.loc[cell_df["robustness_condition"] == "canonical_representative_layer_base"].copy()
    base_positive = int(base_df["sign_positive"].sum())
    for condition in conditions:
        subset = cell_df.loc[cell_df["robustness_condition"] == condition].copy()
        if subset.empty:
            continue
        positive = int(subset["sign_positive"].sum())
        failed = subset.loc[~subset["sign_positive"], ["model", "language", "mean_delta"]]
        if failed.empty:
            note = (
                f"{positive}/{len(subset)} model-language cells stayed positive. "
                f"Reference base was {base_positive}/{len(base_df)} positive."
            )
        else:
            failed_text = ", ".join(
                f"{MODEL_LABELS[str(row.model)]}/{LANGUAGE_LABELS[str(row.language)]} ({float(row.mean_delta):+.3f})"
                for row in failed.itertuples(index=False)
            )
            note = (
                f"{positive}/{len(subset)} model-language cells stayed positive; "
                f"failures: {failed_text}. Reference base was {base_positive}/{len(base_df)} positive."
            )
        rows.append(
            {
                "robustness_condition": condition,
                "core_effect_sign_preserved?": bool(positive == len(subset)),
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def _figure09(path: Path, cell_df: pd.DataFrame, *, conditions: tuple[str, ...]) -> None:
    panel = cell_df.loc[cell_df["robustness_condition"].isin(conditions)].copy()
    panel["column_label"] = panel["model"].map(MODEL_LABELS) + " " + panel["language"].map(LANGUAGE_LABELS)
    column_order = [f"{MODEL_LABELS[model]} {LANGUAGE_LABELS[language]}" for model in MODEL_ORDER for language in LANGUAGE_ORDER]
    pivot = panel.pivot_table(
        index="robustness_condition",
        columns="column_label",
        values="mean_delta",
    ).reindex(index=list(conditions), columns=column_order)
    values = pivot.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0.05
    vmax = max(vmax, 0.05)

    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(column_order)), labels=column_order, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(conditions)), labels=[ROBUSTNESS_LABELS[key] for key in conditions])
    ax.set_title("Robustness of the representative-layer semantic shared advantage", fontsize=13, fontweight="bold")
    ax.set_xlabel("Model / language")
    ax.set_ylabel("Robustness condition")
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            if np.isnan(value):
                continue
            text_color = "white" if abs(value) > (0.55 * vmax) else "#111111"
            ax.text(col_index, row_index, f"{value:+.03f}", ha="center", va="center", fontsize=9, color=text_color)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    colorbar.set_label("Mean semantic SHARED - SPECIFIC delta", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _append_provenance(summary: RobustnessBuildSummary) -> None:
    figure_prov_path = project_root() / "outputs" / "manuscript" / "figure_provenance.md"
    table_prov_path = project_root() / "outputs" / "manuscript" / "table_provenance.md"
    append_markdown_log(
        table_prov_path,
        "Generated robustness table",
        [
            "generating_script=src/brain_subspace_paper/stats/robustness.py",
            f"table05={summary.table05_path.relative_to(project_root()).as_posix()}",
            f"robustness_cell_results={summary.cell_results_path.relative_to(project_root()).as_posix()}",
            f"robustness_summary={summary.summary_path.relative_to(project_root()).as_posix()}",
            f"representative_layers={summary.representative_layers_path.relative_to(project_root()).as_posix()}",
            "source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/group_level_roi_results.parquet, data/processed/features/feature_manifest.parquet, data/interim/roi/*_roi_target_manifest.parquet, data/interim/roi/roi_metadata.parquet, data/interim/roi/harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz",
        ],
    )
    append_markdown_log(
        figure_prov_path,
        "Generated robustness figure",
        [
            "generating_script=src/brain_subspace_paper/stats/robustness.py",
            f"fig09={summary.figure09_path.relative_to(project_root()).as_posix()}",
            f"robustness_cell_results={summary.cell_results_path.relative_to(project_root()).as_posix()}",
            f"source_stats={summary.cell_results_path.relative_to(project_root()).as_posix()}",
        ],
    )


def _available_conditions(cell_df: pd.DataFrame) -> tuple[str, ...]:
    present = set(cell_df["robustness_condition"].astype(str).tolist())
    return tuple(condition for condition in ROBUSTNESS_ORDER if condition in present)


def _persist_outputs(
    *,
    paths: dict[str, Path],
    cell_df: pd.DataFrame,
    representative_layers: pd.DataFrame,
    render_figure: bool,
) -> pd.DataFrame:
    summary_df = _summarize_robustness(cell_df, conditions=_available_conditions(cell_df))
    for path in (paths["cell_results"], paths["summary"], paths["representative_layers"], paths["table05"], paths["fig09"]):
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(cell_df, paths["cell_results"])
    _write_parquet_atomic(summary_df, paths["summary"])
    _write_parquet_atomic(representative_layers, paths["representative_layers"])
    summary_df.to_csv(paths["table05"], index=False)
    if render_figure and not summary_df.empty:
        _figure09(paths["fig09"], cell_df, conditions=_available_conditions(cell_df))
    return summary_df


def _expected_subject_counts(subject_df: pd.DataFrame, *, max_subjects: int | None) -> dict[str, int]:
    available = (
        subject_df.loc[:, ["language", "subject_id"]]
        .drop_duplicates()
        .groupby("language")["subject_id"]
        .nunique()
        .to_dict()
    )
    return {
        language: int(min(int(count), max_subjects)) if max_subjects is not None else int(count)
        for language, count in available.items()
    }


def _condition_is_complete(
    existing_rows: pd.DataFrame,
    *,
    expected_rows_per_condition: int,
    expected_subject_counts: dict[str, int],
) -> bool:
    if len(existing_rows) != expected_rows_per_condition:
        return False
    if "n_subjects" not in existing_rows.columns:
        return False
    for row in existing_rows.itertuples(index=False):
        language = str(row.language)
        if int(row.n_subjects) != int(expected_subject_counts[language]):
            return False
    return True


def build_paper_robustness(
    *,
    n_permutations: int | None = None,
    n_bootstraps: int | None = None,
    conditions: tuple[str, ...] | None = None,
    max_subjects: int | None = None,
    max_tokens_per_batch: int = 2048,
    render_figure: bool = True,
    resume: bool = True,
    output_tag: str | None = None,
    include_base: bool = True,
) -> RobustnessBuildSummary:
    bootstrap_logs()
    if n_permutations is None:
        n_permutations = int(_pipeline_value("statistics", "permutation_n", 10000))
    if n_bootstraps is None:
        n_bootstraps = int(_pipeline_value("statistics", "bootstrap_n", 10000))

    paths = _robustness_paths(output_tag=output_tag)
    required = [paths["group"], paths["subject"]]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing canonical stats required for robustness: {missing_text}. "
            "Run `python -m brain_subspace_paper build-paper-stats` first."
        )

    group_df = pd.read_parquet(paths["group"]).copy()
    subject_df = pd.read_parquet(paths["subject"]).copy()
    representative_layers = _representative_layers(group_df)
    if representative_layers.empty:
        raise RuntimeError("Could not determine representative layers for robustness.")
    active_conditions = _normalize_conditions(conditions)
    expected_rows_per_condition = len(representative_layers) * len(LANGUAGE_ORDER)
    expected_subject_counts = _expected_subject_counts(subject_df, max_subjects=max_subjects)
    if resume and paths["cell_results"].exists():
        cell_df = pd.read_parquet(paths["cell_results"]).copy()
    else:
        cell_df = pd.DataFrame()
    base_condition = "canonical_representative_layer_base"
    if "robustness_condition" in cell_df.columns:
        base_rows = cell_df.loc[cell_df["robustness_condition"] == base_condition].copy()
    else:
        base_rows = pd.DataFrame()
    if include_base:
        if not _condition_is_complete(
            base_rows,
            expected_rows_per_condition=expected_rows_per_condition,
            expected_subject_counts=expected_subject_counts,
        ):
            if not base_rows.empty:
                cell_df = cell_df.loc[cell_df["robustness_condition"] != base_condition].copy()
            base_rows = pd.DataFrame(
                _base_condition_rows(
                    subject_df=subject_df,
                    representative_layers=representative_layers,
                    n_permutations=n_permutations,
                    n_bootstraps=n_bootstraps,
                )
            )
            cell_df = pd.concat([cell_df, base_rows], ignore_index=True)
            print(
                "[robustness] reference base ready at representative layers: "
                + ", ".join(
                    f"{MODEL_LABELS[str(row.model)]}=layer {int(row.layer_index):02d}"
                    for row in representative_layers.itertuples(index=False)
                ),
                flush=True,
            )
            _persist_outputs(
                paths=paths,
                cell_df=cell_df,
                representative_layers=representative_layers,
                render_figure=render_figure,
            )

    for condition in active_conditions:
        if "robustness_condition" in cell_df.columns:
            existing_rows = cell_df.loc[cell_df["robustness_condition"] == condition].copy()
        else:
            existing_rows = pd.DataFrame()
        if resume and _condition_is_complete(
            existing_rows,
            expected_rows_per_condition=expected_rows_per_condition,
            expected_subject_counts=expected_subject_counts,
        ):
            print(f"[robustness] skipping {condition} (already complete)", flush=True)
            continue
        if not existing_rows.empty:
            cell_df = cell_df.loc[cell_df["robustness_condition"] != condition].copy()
        print(f"[robustness] running {condition}", flush=True)
        new_rows = _variant_rows(
                robustness_condition=condition,
                representative_layers=representative_layers,
                n_permutations=n_permutations,
                n_bootstraps=n_bootstraps,
                max_subjects=max_subjects,
                max_tokens_per_batch=max_tokens_per_batch,
            )
        cell_df = pd.concat([cell_df, pd.DataFrame(new_rows)], ignore_index=True)
        summary_df = _persist_outputs(
            paths=paths,
            cell_df=cell_df,
            representative_layers=representative_layers,
            render_figure=render_figure,
        )

    summary_df = _persist_outputs(
        paths=paths,
        cell_df=cell_df,
        representative_layers=representative_layers,
        render_figure=render_figure,
    )

    summary = RobustnessBuildSummary(
        cell_results_path=paths["cell_results"],
        summary_path=paths["summary"],
        representative_layers_path=paths["representative_layers"],
        table05_path=paths["table05"],
        figure09_path=paths["fig09"],
        n_conditions=len(summary_df),
        n_cells=len(cell_df),
    )
    append_markdown_log(
        project_root() / "outputs" / "logs" / "progress_log.md",
        "paper robustness",
        [
            f"cell_results={summary.cell_results_path.as_posix()}",
            f"summary={summary.summary_path.as_posix()}",
            f"representative_layers={summary.representative_layers_path.as_posix()}",
            f"table05={summary.table05_path.as_posix()}",
            f"fig09={summary.figure09_path.as_posix()}",
            f"conditions={summary.n_conditions}",
            f"cells={summary.n_cells}",
            f"active_conditions={','.join(_available_conditions(cell_df))}",
            f"permutations={n_permutations}",
            f"bootstraps={n_bootstraps}",
            f"max_subjects={max_subjects if max_subjects is not None else 'all'}",
            f"resume={resume}",
            f"output_tag={output_tag if output_tag else 'default'}",
            f"include_base={include_base}",
        ],
    )
    _append_provenance(summary)
    return summary


def merge_paper_robustness(
    *,
    input_tags: tuple[str, ...],
    output_tag: str | None = None,
    render_figure: bool = True,
    max_subjects: int | None = None,
    n_permutations: int | None = None,
    n_bootstraps: int | None = None,
) -> RobustnessBuildSummary:
    bootstrap_logs()
    if n_permutations is None:
        n_permutations = int(_pipeline_value("statistics", "permutation_n", 10000))
    if n_bootstraps is None:
        n_bootstraps = int(_pipeline_value("statistics", "bootstrap_n", 10000))

    canonical_paths = _robustness_paths()
    required = [canonical_paths["group"], canonical_paths["subject"]]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing canonical stats required for robustness merge: {missing_text}. "
            "Run `python -m brain_subspace_paper build-paper-stats` first."
        )

    group_df = pd.read_parquet(canonical_paths["group"]).copy()
    subject_df = pd.read_parquet(canonical_paths["subject"]).copy()
    representative_layers = _representative_layers(group_df)
    if representative_layers.empty:
        raise RuntimeError("Could not determine representative layers for robustness merge.")

    parts: list[pd.DataFrame] = []
    for tag in input_tags:
        part_paths = _robustness_paths(output_tag=tag)
        part_path = part_paths["cell_results"]
        if not part_path.exists():
            raise FileNotFoundError(f"Missing robustness chunk output: {part_path}")
        part_df = pd.read_parquet(part_path).copy()
        part_df = part_df.loc[part_df["robustness_condition"] != "canonical_representative_layer_base"].copy()
        parts.append(part_df)

    if parts:
        cell_df = pd.concat(parts, ignore_index=True)
        dedupe_keys = ["robustness_condition", "model", "language", "layer_index"]
        cell_df = cell_df.sort_values(dedupe_keys).drop_duplicates(subset=dedupe_keys, keep="last").reset_index(drop=True)
    else:
        cell_df = pd.DataFrame()

    base_rows = pd.DataFrame(
        _base_condition_rows(
            subject_df=subject_df,
            representative_layers=representative_layers,
            n_permutations=n_permutations,
            n_bootstraps=n_bootstraps,
        )
    )
    cell_df = pd.concat([cell_df, base_rows], ignore_index=True)

    paths = _robustness_paths(output_tag=output_tag)
    summary_df = _persist_outputs(
        paths=paths,
        cell_df=cell_df,
        representative_layers=representative_layers,
        render_figure=render_figure,
    )

    summary = RobustnessBuildSummary(
        cell_results_path=paths["cell_results"],
        summary_path=paths["summary"],
        representative_layers_path=paths["representative_layers"],
        table05_path=paths["table05"],
        figure09_path=paths["fig09"],
        n_conditions=len(summary_df),
        n_cells=len(cell_df),
    )
    append_markdown_log(
        project_root() / "outputs" / "logs" / "progress_log.md",
        "paper robustness merge",
        [
            f"input_tags={','.join(input_tags)}",
            f"output_tag={output_tag if output_tag else 'default'}",
            f"cell_results={summary.cell_results_path.as_posix()}",
            f"summary={summary.summary_path.as_posix()}",
            f"representative_layers={summary.representative_layers_path.as_posix()}",
            f"table05={summary.table05_path.as_posix()}",
            f"fig09={summary.figure09_path.as_posix()}",
            f"conditions={summary.n_conditions}",
            f"cells={summary.n_cells}",
            f"render_figure={render_figure}",
            f"max_subjects={max_subjects if max_subjects is not None else 'all'}",
        ],
    )
    _append_provenance(summary)
    return summary
