from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import nibabel as nib
from nilearn.plotting import plot_glass_brain
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from brain_subspace_paper.config import project_root
from brain_subspace_paper.encoding.english_prototype import _run_single_condition
from brain_subspace_paper.encoding.xlmr_roi_pipeline import (
    _load_feature_manifest,
    _normalize_model_name,
    _pipeline_value,
    _prepare_language_data,
)
from brain_subspace_paper.logging_utils import append_markdown_log, bootstrap_logs, write_text


LANGUAGE_ORDER = ("en", "fr", "zh")
LANGUAGE_LABELS = {"en": "English", "fr": "French", "zh": "Chinese"}
MODEL_ORDER = ("xlmr", "nllb_encoder")
MODEL_LABELS = {"xlmr": "XLM-R", "nllb_encoder": "NLLB encoder"}
MAP_CONDITIONS = ("shared", "specific")


@dataclass(slots=True)
class WholeBrainSummary:
    representative_layers_path: Path
    figure_path: Path
    artifact_roots: tuple[Path, ...]
    n_subject_maps: int
    skipped_subject_maps: int


def _stats_root() -> Path:
    return project_root() / "outputs" / "stats"


def _whole_brain_root() -> Path:
    return _stats_root() / "whole_brain"


def _figure_path() -> Path:
    return project_root() / "outputs" / "figures" / "fig08_whole_brain_maps.png"


def _representative_layers_path() -> Path:
    return _whole_brain_root() / "representative_layers.json"


def _canonical_group_results_path() -> Path:
    return _stats_root() / "group_level_roi_results.parquet"


def _canonical_subject_results_path() -> Path:
    return _stats_root() / "subject_level_roi_results.parquet"


def _feature_manifest_path() -> Path:
    return project_root() / "data" / "processed" / "features" / "feature_manifest.parquet"


def _artifact_root(model_name: str, language: str, layer_index: int) -> Path:
    return _whole_brain_root() / model_name / language / f"layer_{int(layer_index):02d}"


def _subject_cache_dir(root: Path) -> Path:
    return root / "subjects"


def _subject_condition_map_path(root: Path, subject_id: str, condition: str) -> Path:
    return _subject_cache_dir(root) / f"{subject_id}__{condition}_z.npy"


def _subject_valid_mask_path(root: Path, subject_id: str) -> Path:
    return _subject_cache_dir(root) / f"{subject_id}__valid_mask.npy"


def _shared_map_path(root: Path) -> Path:
    return root / "shared_mean_z.nii.gz"


def _specific_map_path(root: Path) -> Path:
    return root / "specific_mean_z.nii.gz"


def _delta_map_path(root: Path) -> Path:
    return root / "shared_minus_specific_mean_z.nii.gz"


def _brain_mask_path(root: Path) -> Path:
    return root / "brain_mask.nii.gz"


def _coverage_count_path(root: Path) -> Path:
    return root / "coverage_count.nii.gz"


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def _included_subject_ids() -> dict[str, list[str]]:
    subject_df = pd.read_parquet(_canonical_subject_results_path(), columns=["language", "subject_id"]).drop_duplicates()
    return {
        language: sorted(subject_df.loc[subject_df["language"] == language, "subject_id"].astype(str).tolist())
        for language in LANGUAGE_ORDER
    }


def _representative_layers() -> list[dict[str, float | int | str]]:
    group_df = pd.read_parquet(_canonical_group_results_path()).copy()
    subset = group_df.loc[
        (group_df["roi_family"] == "semantic")
        & (group_df["condition"].isin(["shared", "specific"]))
    ].copy()
    pivot = subset.pivot_table(
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
    rows: list[dict[str, float | int | str]] = []
    for model in MODEL_ORDER:
        model_rows = by_layer.loc[by_layer["model"] == model].copy()
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
    return rows


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root() / path
    return path


def _subject_run_rows(prepared, subject_id: str) -> pd.DataFrame:
    rows = (
        prepared.roi_manifest.loc[prepared.roi_manifest["subject_id"] == subject_id]
        .sort_values("canonical_run_index")
        .reset_index(drop=True)
    )
    if rows.empty:
        raise RuntimeError(f"Missing ROI manifest rows for subject={subject_id}, language={prepared.language}.")
    return rows


def _subject_valid_mask(subject_runs: pd.DataFrame) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray, nib.Nifti1Header]:
    valid_mask: np.ndarray | None = None
    volume_shape: tuple[int, int, int] | None = None
    affine: np.ndarray | None = None
    header: nib.Nifti1Header | None = None

    for row in subject_runs.itertuples(index=False):
        bold_path = _resolve_path(str(row.bold_filepath))
        img = nib.load(str(bold_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        if volume_shape is None:
            volume_shape = tuple(int(value) for value in data.shape[:3])
            affine = np.asarray(img.affine, dtype=np.float64)
            header = img.header.copy()
        flat = data.reshape(-1, data.shape[-1])
        run_mask = np.isfinite(flat).all(axis=1) & (flat.std(axis=1) > 1e-6)
        valid_mask = run_mask if valid_mask is None else (valid_mask & run_mask)

    if valid_mask is None or volume_shape is None or affine is None or header is None:
        raise RuntimeError("Failed to derive a subject-level valid voxel mask.")
    if not np.any(valid_mask):
        raise RuntimeError(f"Subject produced an empty whole-brain mask: {subject_runs['subject_id'].iloc[0]}")
    return valid_mask, volume_shape, affine, header


def _subject_z_series_by_run(subject_runs: pd.DataFrame, valid_mask: np.ndarray) -> dict[int, np.ndarray]:
    run_series_by_run: dict[int, np.ndarray] = {}
    for row in subject_runs.itertuples(index=False):
        bold_path = _resolve_path(str(row.bold_filepath))
        img = nib.load(str(bold_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        flat = data.reshape(-1, data.shape[-1])[valid_mask, :]
        voxel_mean = flat.mean(axis=1, keepdims=True)
        voxel_std = flat.std(axis=1, keepdims=True)
        voxel_std = np.where(voxel_std < 1e-6, 1.0, voxel_std)
        z_voxels = ((flat - voxel_mean) / voxel_std).T.astype(np.float32, copy=False)
        run_series_by_run[int(row.canonical_run_index)] = z_voxels
    return run_series_by_run


def _run_condition_chunked(
    *,
    run_designs: dict[int, np.ndarray],
    z_by_run: dict[int, np.ndarray],
    run_series_by_run: dict[int, np.ndarray],
    chunk_size: int,
) -> np.ndarray:
    n_targets = next(iter(run_series_by_run.values())).shape[1]
    output = np.empty(n_targets, dtype=np.float32)
    for start in range(0, n_targets, chunk_size):
        stop = min(start + chunk_size, n_targets)
        chunk_series = {
            run_index: series[:, start:stop]
            for run_index, series in run_series_by_run.items()
        }
        _, z_vals, _ = _run_single_condition(
            run_designs=run_designs,
            z_by_run=z_by_run,
            roi_series_by_run=chunk_series,
        )
        output[start:stop] = z_vals.astype(np.float32, copy=False)
    return output


def _full_volume_from_masked(masked_values: np.ndarray, valid_mask: np.ndarray, n_voxels_total: int) -> np.ndarray:
    full = np.full(n_voxels_total, np.nan, dtype=np.float32)
    full[valid_mask] = masked_values.astype(np.float32, copy=False)
    return full


def _feature_input_files(model_name: str, language: str, layer_index: int) -> dict[str, str]:
    manifest = pd.read_parquet(_feature_manifest_path()).copy()
    subset = manifest.loc[
        (manifest["model"] == model_name)
        & (manifest["language"] == language)
        & (manifest["layer_index"].astype(int) == int(layer_index))
        & (manifest["condition"].isin(["shared", "specific"]))
        & (manifest["shuffle_index"].isna())
    ].copy()
    return {
        str(row.condition): _resolve_path(str(row.filepath)).relative_to(project_root()).as_posix()
        for row in subset.itertuples(index=False)
    }


def _write_nifti(data_1d: np.ndarray, volume_shape: tuple[int, int, int], affine: np.ndarray, header: nib.Nifti1Header, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_out = header.copy()
    header_out.set_data_shape(volume_shape)
    header_out.set_data_dtype(np.float32)
    image = nib.Nifti1Image(data_1d.reshape(volume_shape).astype(np.float32, copy=False), affine, header_out)
    nib.save(image, str(path))


def _coverage_threshold(n_subjects: int, min_fraction: float) -> int:
    return max(1, int(math.ceil(float(min_fraction) * n_subjects)))


def _aggregate_group_maps(
    *,
    root: Path,
    subject_ids: list[str],
    volume_shape: tuple[int, int, int],
    affine: np.ndarray,
    header: nib.Nifti1Header,
    min_subject_coverage_fraction: float,
) -> None:
    n_voxels_total = int(np.prod(volume_shape))
    coverage_count = np.zeros(n_voxels_total, dtype=np.int32)
    shared_sum = np.zeros(n_voxels_total, dtype=np.float64)
    specific_sum = np.zeros(n_voxels_total, dtype=np.float64)
    delta_sum = np.zeros(n_voxels_total, dtype=np.float64)

    for subject_id in subject_ids:
        shared = np.load(_subject_condition_map_path(root, subject_id, "shared")).astype(np.float32, copy=False)
        specific = np.load(_subject_condition_map_path(root, subject_id, "specific")).astype(np.float32, copy=False)
        valid = np.isfinite(shared) & np.isfinite(specific)
        coverage_count[valid] += 1
        shared_sum[valid] += shared[valid]
        specific_sum[valid] += specific[valid]
        delta_sum[valid] += (shared[valid] - specific[valid])

    min_subjects = _coverage_threshold(len(subject_ids), min_subject_coverage_fraction)
    group_mask = coverage_count >= min_subjects
    shared_mean = np.zeros(n_voxels_total, dtype=np.float32)
    specific_mean = np.zeros(n_voxels_total, dtype=np.float32)
    delta_mean = np.zeros(n_voxels_total, dtype=np.float32)
    shared_mean[group_mask] = (shared_sum[group_mask] / coverage_count[group_mask]).astype(np.float32, copy=False)
    specific_mean[group_mask] = (specific_sum[group_mask] / coverage_count[group_mask]).astype(np.float32, copy=False)
    delta_mean[group_mask] = (delta_sum[group_mask] / coverage_count[group_mask]).astype(np.float32, copy=False)

    _write_nifti(shared_mean, volume_shape, affine, header, _shared_map_path(root))
    _write_nifti(specific_mean, volume_shape, affine, header, _specific_map_path(root))
    _write_nifti(delta_mean, volume_shape, affine, header, _delta_map_path(root))
    _write_nifti(group_mask.astype(np.float32, copy=False), volume_shape, affine, header, _brain_mask_path(root))
    _write_nifti(coverage_count.astype(np.float32, copy=False), volume_shape, affine, header, _coverage_count_path(root))


def _manifest_payload(
    *,
    model_name: str,
    language: str,
    layer_index: int,
    layer_depth: float,
    subject_ids: list[str],
    min_subject_coverage_fraction: float,
    feature_input_files: dict[str, str],
) -> dict[str, object]:
    return {
        "model": model_name,
        "language": language,
        "selected_layer": int(layer_index),
        "layer_depth": float(layer_depth),
        "source_subject_ids": subject_ids,
        "generating_script": "src/brain_subspace_paper/stats/whole_brain.py",
        "input_feature_files": feature_input_files,
        "statistic_type": "mean_z",
        "conditions": ["shared", "specific", "shared_minus_specific"],
        "brain_mask_rule": {
            "type": "subject_coverage_threshold",
            "min_subject_coverage_fraction": float(min_subject_coverage_fraction),
            "min_subject_count": _coverage_threshold(len(subject_ids), min_subject_coverage_fraction),
        },
    }


def _update_figure_provenance(figure_path: Path, artifact_roots: list[Path]) -> None:
    append_markdown_log(
        project_root() / "outputs" / "manuscript" / "figure_provenance.md",
        "Generated whole-brain figure",
        [
            "generating_script=src/brain_subspace_paper/stats/whole_brain.py",
            f"fig08={figure_path.relative_to(project_root()).as_posix()}",
            *(f"artifact_root={root.relative_to(project_root()).as_posix()}" for root in artifact_roots),
        ],
    )


def _plot_colorbar(ax: plt.Axes, *, cmap_name: str, vmin: float, vmax: float, label: str) -> None:
    sm = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cm.get_cmap(cmap_name))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax)
    cbar.set_label(label, fontsize=9)


def _build_fig08(artifact_roots: list[Path], figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_roots = sorted(
        artifact_roots,
        key=lambda path: (
            MODEL_ORDER.index(path.parts[-3]),
            LANGUAGE_ORDER.index(path.parts[-2]),
        ),
    )
    shared_max = 0.0
    delta_max = 0.0
    for root in ordered_roots:
        shared_data = np.asarray(nib.load(str(_shared_map_path(root))).dataobj, dtype=np.float32)
        specific_data = np.asarray(nib.load(str(_specific_map_path(root))).dataobj, dtype=np.float32)
        delta_data = np.asarray(nib.load(str(_delta_map_path(root))).dataobj, dtype=np.float32)
        if np.any(shared_data > 0):
            shared_max = max(shared_max, float(np.percentile(shared_data[shared_data > 0], 99)))
        if np.any(specific_data > 0):
            shared_max = max(shared_max, float(np.percentile(specific_data[specific_data > 0], 99)))
        if np.any(np.abs(delta_data) > 0):
            delta_max = max(delta_max, float(np.percentile(np.abs(delta_data[np.abs(delta_data) > 0]), 99)))
    shared_max = max(shared_max, 0.05)
    delta_max = max(delta_max, 0.03)

    fig, axes = plt.subplots(len(ordered_roots), 3, figsize=(12, 2.4 * len(ordered_roots)))
    if len(ordered_roots) == 1:
        axes = np.asarray([axes])
    for row_index, root in enumerate(ordered_roots):
        manifest = json.loads(_manifest_path(root).read_text(encoding="utf-8"))
        model = str(manifest["model"])
        language = str(manifest["language"])
        layer_index = int(manifest["selected_layer"])
        for col_index, (path_fn, title, cmap_name, vmax, symmetric) in enumerate(
            [
                (_shared_map_path, "SHARED mean-z", "YlOrRd", shared_max, False),
                (_specific_map_path, "SPECIFIC mean-z", "YlOrRd", shared_max, False),
                (_delta_map_path, "SHARED - SPECIFIC", "RdBu_r", delta_max, True),
            ]
        ):
            axis = axes[row_index, col_index]
            axis.set_axis_off()
            plot_glass_brain(
                str(path_fn(root)),
                axes=axis,
                display_mode="ortho",
                colorbar=False,
                plot_abs=False,
                threshold=1e-6,
                cmap=cmap_name,
                vmax=vmax,
                symmetric_cbar=symmetric,
                annotate=False,
                black_bg=False,
            )
            if row_index == 0:
                axis.set_title(title, fontsize=11)
        axes[row_index, 0].text(
            -0.28,
            0.5,
            f"{MODEL_LABELS[model]}\n{LANGUAGE_LABELS[language]}\nlayer {layer_index:02d}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle("Whole-brain descriptive maps at representative layers", fontsize=14, fontweight="bold", y=0.995)
    shared_cax = fig.add_axes([0.92, 0.58, 0.015, 0.22])
    delta_cax = fig.add_axes([0.92, 0.20, 0.015, 0.22])
    _plot_colorbar(shared_cax, cmap_name="YlOrRd", vmin=0.0, vmax=shared_max, label="Mean z")
    _plot_colorbar(delta_cax, cmap_name="RdBu_r", vmin=-delta_max, vmax=delta_max, label="Delta z")
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_paper_whole_brain(
    *,
    models: tuple[str, ...] = MODEL_ORDER,
    languages: tuple[str, ...] = LANGUAGE_ORDER,
    chunk_size: int = 8192,
    max_subjects: int | None = None,
    min_subject_coverage_fraction: float = 0.9,
    resume: bool = True,
    render_figure: bool = True,
) -> WholeBrainSummary:
    bootstrap_logs()
    models = tuple(_normalize_model_name(model) for model in models)
    representative_layers = [row for row in _representative_layers() if row["model"] in models]
    representative_layers_path = _representative_layers_path()
    representative_layers_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(representative_layers_path, json.dumps(representative_layers, indent=2) + "\n")

    included_subjects = _included_subject_ids()
    artifact_roots: list[Path] = []
    computed_subject_maps = 0
    skipped_subject_maps = 0

    for row in representative_layers:
        model_name = str(row["model"])
        layer_index = int(row["layer_index"])
        layer_depth = float(row["layer_depth"])
        for language in languages:
            prepared = _prepare_language_data(
                language,
                model_name=model_name,
                layer_indices=(layer_index,),
                mismatch_shuffles=1,
            )
            subject_ids = [
                subject_id
                for subject_id in sorted(prepared.roi_manifest["subject_id"].unique().tolist())
                if subject_id in included_subjects[language]
            ]
            if max_subjects is not None:
                subject_ids = subject_ids[:max_subjects]
            if not subject_ids:
                continue

            root = _artifact_root(model_name, language, layer_index)
            _subject_cache_dir(root).mkdir(parents=True, exist_ok=True)
            artifact_roots.append(root)
            sample_volume_shape: tuple[int, int, int] | None = None
            sample_affine: np.ndarray | None = None
            sample_header: nib.Nifti1Header | None = None

            for subject_id in subject_ids:
                shared_map_path = _subject_condition_map_path(root, subject_id, "shared")
                specific_map_path = _subject_condition_map_path(root, subject_id, "specific")
                valid_mask_path = _subject_valid_mask_path(root, subject_id)
                if resume and shared_map_path.exists() and specific_map_path.exists() and valid_mask_path.exists():
                    skipped_subject_maps += 1
                    if sample_volume_shape is None:
                        subject_runs = _subject_run_rows(prepared, subject_id)
                        _, sample_volume_shape, sample_affine, sample_header = _subject_valid_mask(subject_runs)
                    print(f"[whole-brain] skip {model_name}/{language}/{subject_id}", flush=True)
                    continue

                subject_runs = _subject_run_rows(prepared, subject_id)
                valid_mask, volume_shape, affine, header = _subject_valid_mask(subject_runs)
                if sample_volume_shape is None:
                    sample_volume_shape = volume_shape
                    sample_affine = affine
                    sample_header = header
                run_series_by_run = _subject_z_series_by_run(subject_runs, valid_mask)
                n_voxels_total = int(np.prod(volume_shape))

                for condition in MAP_CONDITIONS:
                    z_map = _run_condition_chunked(
                        run_designs={
                            run_index: prepared.text_designs[(condition, layer_index)][run_index]
                            for run_index in run_series_by_run
                        },
                        z_by_run={run_index: prepared.text_nuisance_arrays[run_index] for run_index in run_series_by_run},
                        run_series_by_run=run_series_by_run,
                        chunk_size=chunk_size,
                    )
                    np.save(
                        _subject_condition_map_path(root, subject_id, condition),
                        _full_volume_from_masked(z_map, valid_mask, n_voxels_total),
                    )
                np.save(valid_mask_path, valid_mask.astype(np.uint8, copy=False))
                computed_subject_maps += 1
                print(f"[whole-brain] done {model_name}/{language}/{subject_id}", flush=True)
                del run_series_by_run

            if sample_volume_shape is None or sample_affine is None or sample_header is None:
                raise RuntimeError(f"Failed to establish template image metadata for {model_name} / {language}.")

            _aggregate_group_maps(
                root=root,
                subject_ids=subject_ids,
                volume_shape=sample_volume_shape,
                affine=sample_affine,
                header=sample_header,
                min_subject_coverage_fraction=min_subject_coverage_fraction,
            )
            payload = _manifest_payload(
                model_name=model_name,
                language=language,
                layer_index=layer_index,
                layer_depth=layer_depth,
                subject_ids=subject_ids,
                min_subject_coverage_fraction=min_subject_coverage_fraction,
                feature_input_files=_feature_input_files(model_name, language, layer_index),
            )
            write_text(_manifest_path(root), json.dumps(payload, indent=2, sort_keys=True) + "\n")
            append_markdown_log(
                project_root() / "outputs" / "logs" / "progress_log.md",
                "whole-brain artifacts",
                [
                    f"model={model_name}",
                    f"language={language}",
                    f"selected_layer={layer_index}",
                    f"layer_depth={layer_depth:.6f}",
                    f"subjects={len(subject_ids)}",
                    f"artifact_root={root.as_posix()}",
                    f"min_subject_coverage_fraction={min_subject_coverage_fraction}",
                    f"chunk_size={chunk_size}",
                ],
            )

    figure_path = _figure_path()
    if render_figure:
        _build_fig08(artifact_roots, figure_path)
        _update_figure_provenance(figure_path, artifact_roots)
        append_markdown_log(
            project_root() / "outputs" / "logs" / "progress_log.md",
            "whole-brain figure",
            [
                f"figure={figure_path.as_posix()}",
                f"representative_layers={representative_layers_path.as_posix()}",
                f"artifact_roots={len(artifact_roots)}",
            ],
        )
    return WholeBrainSummary(
        representative_layers_path=representative_layers_path,
        figure_path=figure_path,
        artifact_roots=tuple(artifact_roots),
        n_subject_maps=computed_subject_maps,
        skipped_subject_maps=skipped_subject_maps,
    )
