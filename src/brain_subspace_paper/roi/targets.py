from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
from nibabel.affines import apply_affine
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import load_img, resample_to_img
import numpy as np
import pandas as pd

from brain_subspace_paper.config import project_root
from brain_subspace_paper.logging_utils import append_markdown_log, bootstrap_logs, write_text


ROI_SPECS = (
    ("semantic", "pMTG", "Middle Temporal Gyrus, posterior division"),
    ("semantic", "AG", "Angular Gyrus"),
    ("semantic", "TemporalPole", "Temporal Pole"),
    ("semantic", "IFGtri", "Inferior Frontal Gyrus, pars triangularis"),
    ("auditory", "Heschl", "Heschl's Gyrus (includes H1 and H2)"),
    ("auditory", "pSTG", "Superior Temporal Gyrus, posterior division"),
    ("auditory", "aSTG", "Superior Temporal Gyrus, anterior division"),
    ("control", "Precentral", "Precentral Gyrus"),
    ("control", "OccipitalPole", "Occipital Pole"),
)


@dataclass(slots=True)
class RoiTargetSummary:
    language: str
    manifest_path: Path
    roi_metadata_path: Path
    atlas_path: Path
    report_path: Path
    n_subjects: int
    n_runs: int
    n_rois: int


VALID_LANGUAGES = ("en", "fr", "zh")


def _run_manifest_path() -> Path:
    return project_root() / "data" / "interim" / "lppc_run_manifest.parquet"


def _roi_root() -> Path:
    return project_root() / "data" / "interim" / "roi"


def _roi_manifest_path(language: str) -> Path:
    return _roi_root() / f"{language}_roi_target_manifest.parquet"


def _roi_metadata_path() -> Path:
    return _roi_root() / "roi_metadata.parquet"


def _atlas_output_path() -> Path:
    return _roi_root() / "harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz"


def _report_path(language: str) -> Path:
    return project_root() / "outputs" / "logs" / f"{language}_roi_target_report.md"


def _load_run_manifest(language: str) -> pd.DataFrame:
    if language not in VALID_LANGUAGES:
        raise ValueError(f"Unsupported language={language!r}. Expected one of {VALID_LANGUAGES}.")
    manifest = pd.read_parquet(_run_manifest_path()).copy()
    manifest = manifest.loc[manifest["language"] == language].copy()
    if manifest.empty:
        raise FileNotFoundError(f"No run manifest rows found for language={language}.")
    manifest = manifest.sort_values(["subject_id", "canonical_run_index"]).reset_index(drop=True)
    return manifest


def _is_fetched(path_str: str) -> bool:
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root() / path
    try:
        with path.open("rb") as handle:
            return handle.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _complete_subject_ids(manifest: pd.DataFrame) -> list[str]:
    fetched = manifest["filepath"].map(_is_fetched)
    summary = (
        manifest.assign(fetched=fetched)
        .groupby("subject_id", as_index=False)
        .agg(n_runs=("canonical_run_index", "count"), fetched_runs=("fetched", "sum"))
    )
    return summary.loc[(summary["n_runs"] == 9) & (summary["fetched_runs"] == 9), "subject_id"].astype(str).tolist()


def _sample_bold_path(manifest: pd.DataFrame, subject_ids: list[str]) -> Path:
    row = (
        manifest.loc[
            (manifest["subject_id"] == subject_ids[0]) & (manifest["canonical_run_index"].astype(int) == 1),
            "filepath",
        ]
        .iloc[0]
    )
    path = Path(row)
    if not path.is_absolute():
        path = project_root() / path
    return path


def _resampled_atlas_and_metadata(sample_bold_path: Path) -> tuple[nib.Nifti1Image, pd.DataFrame]:
    atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    atlas_img = load_img(atlas.maps)
    target_img = load_img(str(sample_bold_path))
    target_3d = nib.Nifti1Image(
        np.zeros(target_img.shape[:3], dtype=np.int16),
        target_img.affine,
        target_img.header,
    )
    resampled = resample_to_img(
        atlas_img,
        target_3d,
        interpolation="nearest",
        copy_header=True,
        force_resample=True,
    )
    atlas_data = np.asarray(resampled.dataobj).astype(np.int16, copy=False)
    output_data = np.zeros_like(atlas_data, dtype=np.int16)

    grid = np.indices(atlas_data.shape).reshape(3, -1).T
    xyz = apply_affine(resampled.affine, grid).reshape(*atlas_data.shape, 3)
    x_coords = xyz[..., 0]

    metadata_rows: list[dict[str, Any]] = []
    roi_index = 1
    for family, short_name, atlas_label in ROI_SPECS:
        label_index = atlas.labels.index(atlas_label)
        label_mask = atlas_data == label_index
        for hemisphere, hemi_mask in (("L", x_coords < 0), ("R", x_coords > 0)):
            mask = label_mask & hemi_mask
            n_voxels = int(mask.sum())
            if n_voxels == 0:
                raise RuntimeError(f"ROI mask empty after hemisphere split: {hemisphere}_{short_name}")
            metadata_rows.append(
                {
                    "roi_index": roi_index,
                    "roi_name": f"{hemisphere}_{short_name}",
                    "family": family,
                    "hemisphere": hemisphere,
                    "atlas_label": atlas_label,
                    "n_voxels": n_voxels,
                }
            )
            output_data[mask] = roi_index
            roi_index += 1

    out_img = nib.Nifti1Image(output_data.astype(np.int16, copy=False), resampled.affine, resampled.header)
    return out_img, pd.DataFrame(metadata_rows)


def _roi_indices_from_metadata(atlas_img: nib.Nifti1Image, roi_metadata: pd.DataFrame) -> list[np.ndarray]:
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int16, copy=False)
    indices: list[np.ndarray] = []
    for roi_index in roi_metadata["roi_index"].astype(int).tolist():
        voxel_indices = np.flatnonzero(atlas_data.reshape(-1) == roi_index)
        if len(voxel_indices) == 0:
            raise RuntimeError(f"ROI index {roi_index} produced an empty voxel set.")
        indices.append(voxel_indices)
    return indices


def _run_roi_timeseries(bold_path: Path, roi_linear_indices: list[np.ndarray]) -> np.ndarray:
    img = nib.load(str(bold_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    flat = data.reshape(-1, data.shape[-1])
    roi_series: list[np.ndarray] = []
    for voxel_indices in roi_linear_indices:
        voxel_ts = flat[voxel_indices, :]
        voxel_mean = voxel_ts.mean(axis=1, keepdims=True)
        voxel_std = voxel_ts.std(axis=1, keepdims=True)
        keep = voxel_std[:, 0] > 1e-6
        if not np.any(keep):
            raise RuntimeError(f"ROI has no non-constant voxels in run {bold_path.name}.")
        z_voxels = (voxel_ts[keep] - voxel_mean[keep]) / voxel_std[keep]
        roi_series.append(z_voxels.mean(axis=0).astype(np.float32, copy=False))
    return np.stack(roi_series, axis=1)


def extract_roi_targets(language: str = "en", *, max_subjects: int | None = None) -> RoiTargetSummary:
    bootstrap_logs()
    manifest = _load_run_manifest(language)
    subject_ids = _complete_subject_ids(manifest)
    if not subject_ids:
        raise RuntimeError(f"No complete fetched subjects found for language={language}.")
    if max_subjects is not None:
        subject_ids = subject_ids[:max_subjects]

    sample_bold_path = _sample_bold_path(manifest, subject_ids)
    atlas_img, roi_metadata = _resampled_atlas_and_metadata(sample_bold_path)
    atlas_path = _atlas_output_path()
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(atlas_img, str(atlas_path))

    roi_metadata_path = _roi_metadata_path()
    roi_metadata.to_parquet(roi_metadata_path, index=False)
    roi_linear_indices = _roi_indices_from_metadata(atlas_img, roi_metadata)

    rows: list[dict[str, Any]] = []
    for subject_id in subject_ids:
        subject_runs = manifest.loc[manifest["subject_id"] == subject_id].sort_values("canonical_run_index")
        for row in subject_runs.itertuples(index=False):
            bold_path = Path(row.filepath)
            if not bold_path.is_absolute():
                bold_path = project_root() / bold_path
            timeseries = _run_roi_timeseries(bold_path, roi_linear_indices)
            output_path = (
                _roi_root()
                / language
                / subject_id
                / f"run_{int(row.canonical_run_index):02d}_roi_timeseries.npy"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, timeseries.astype(np.float32, copy=False))
            rows.append(
                {
                    "subject_id": subject_id,
                    "language": language,
                    "canonical_run_index": int(row.canonical_run_index),
                    "filepath": output_path.as_posix(),
                    "bold_filepath": bold_path.as_posix(),
                    "n_scans": int(timeseries.shape[0]),
                    "n_rois": int(timeseries.shape[1]),
                }
            )

    roi_manifest = pd.DataFrame(rows).sort_values(["subject_id", "canonical_run_index"]).reset_index(drop=True)
    manifest_path = _roi_manifest_path(language)
    roi_manifest.to_parquet(manifest_path, index=False)

    report_lines = [
        f"# ROI Target Report ({language})",
        "",
        f"- subjects: `{len(subject_ids)}`",
        f"- runs: `{len(roi_manifest)}`",
        f"- rois: `{len(roi_metadata)}`",
        f"- atlas: `{atlas_path.relative_to(project_root()).as_posix()}`",
        f"- roi metadata: `{roi_metadata_path.relative_to(project_root()).as_posix()}`",
        f"- roi manifest: `{manifest_path.relative_to(project_root()).as_posix()}`",
        "",
        "## ROI Summary",
        "",
    ]
    for row in roi_metadata.itertuples(index=False):
        report_lines.append(
            f"- `{row.roi_name}` family `{row.family}` atlas `{row.atlas_label}` voxels `{int(row.n_voxels)}`"
        )
    report_lines.extend(["", "## Subject Summary", ""])
    for row in (
        roi_manifest.groupby("subject_id", as_index=False)
        .agg(n_runs=("canonical_run_index", "count"), n_scans_mean=("n_scans", "mean"))
        .itertuples(index=False)
    ):
        report_lines.append(
            f"- `{row.subject_id}` runs `{int(row.n_runs)}` mean scans `{row.n_scans_mean:.1f}`"
        )

    report_path = _report_path(language)
    write_text(report_path, "\n".join(report_lines) + "\n")

    append_markdown_log(
        project_root() / "outputs" / "logs" / "progress_log.md",
        "ROI target extraction",
        [
            f"language={language}",
            f"subjects={len(subject_ids)}",
            f"runs={len(roi_manifest)}",
            f"roi_manifest={manifest_path.as_posix()}",
            f"roi_metadata={roi_metadata_path.as_posix()}",
        ],
    )

    return RoiTargetSummary(
        language=language,
        manifest_path=manifest_path,
        roi_metadata_path=roi_metadata_path,
        atlas_path=atlas_path,
        report_path=report_path,
        n_subjects=len(subject_ids),
        n_runs=len(roi_manifest),
        n_rois=len(roi_metadata),
    )
