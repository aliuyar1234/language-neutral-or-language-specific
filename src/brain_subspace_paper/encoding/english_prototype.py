from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from nilearn.glm.first_level.hemodynamic_models import glover_hrf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from brain_subspace_paper.config import pipeline_config, project_config, project_root
from brain_subspace_paper.data.sentence_spans import _read_csv
from brain_subspace_paper.logging_utils import append_markdown_log, bootstrap_logs, write_text


PROTOTYPE_ROIS = ("L_pMTG", "R_pMTG", "L_AG", "L_Heschl", "L_pSTG", "L_IFGtri")
TEXT_CONDITIONS = ("raw", "shared", "specific")
PROTOTYPE_CONDITIONS = ("acoustic_only", "raw", "shared", "specific", "mismatched_shared")
PROTOTYPE_LAYER_INDICES = (0, 6, 12)


@dataclass(slots=True)
class EnglishPrototypeSummary:
    results_path: Path
    plot_path: Path
    report_path: Path
    n_subjects: int
    n_rois: int
    n_layers: int


def _feature_manifest_path() -> Path:
    return project_root() / "data" / "processed" / "features" / "feature_manifest.parquet"


def _roi_manifest_path() -> Path:
    return project_root() / "data" / "interim" / "roi" / "en_roi_target_manifest.parquet"


def _roi_metadata_path() -> Path:
    return project_root() / "data" / "interim" / "roi" / "roi_metadata.parquet"


def _triplets_path() -> Path:
    return project_root() / "data" / "processed" / "alignment_triplets.parquet"


def _results_path() -> Path:
    return project_root() / "outputs" / "stats" / "english_prototype_roi_results.parquet"


def _plot_path() -> Path:
    return project_root() / "outputs" / "figures" / "english_prototype_roi_summary.png"


def _report_path() -> Path:
    return project_root() / "outputs" / "logs" / "encoding_qc_report.md"


def _pipeline_value(section: str, key: str, default: Any) -> Any:
    return pipeline_config().get(section, {}).get(key, default)


def _alpha_grid() -> np.ndarray:
    return np.logspace(
        float(_pipeline_value("encoding", "ridge_alpha_min_log10", -2)),
        float(_pipeline_value("encoding", "ridge_alpha_max_log10", 6)),
        int(_pipeline_value("encoding", "ridge_alpha_n_values", 15)),
    )


def _random_seed() -> int:
    return int(project_config().get("random_seed", 20260314))


def _load_triplets() -> pd.DataFrame:
    return pd.read_parquet(_triplets_path()).sort_values("triplet_id").reset_index(drop=True)


def _load_features() -> pd.DataFrame:
    df = pd.read_parquet(_feature_manifest_path()).copy()
    return df.loc[df["model"] == "xlmr"].copy()


def _load_roi_targets(roi_names: tuple[str, ...] = PROTOTYPE_ROIS) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(_roi_manifest_path()).copy()
    metadata = pd.read_parquet(_roi_metadata_path()).copy()
    metadata = metadata.loc[metadata["roi_name"].isin(roi_names)].copy()
    return manifest.sort_values(["subject_id", "canonical_run_index"]).reset_index(drop=True), metadata


def _annotation_paths() -> tuple[Path, Path]:
    root = project_root() / "data" / "raw" / "ds003643" / "annotation" / "EN"
    return root / "lppEN_prosody.csv", root / "lppEN_word_information.csv"


def _load_annotation_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    prosody_path, word_info_path = _annotation_paths()
    prosody = _read_csv(prosody_path).copy()
    word_info = pd.read_csv(word_info_path)
    return prosody, word_info


def _run_scan_counts(roi_manifest: pd.DataFrame) -> dict[int, int]:
    counts = roi_manifest.groupby("canonical_run_index")["n_scans"].unique()
    out: dict[int, int] = {}
    for run_index, values in counts.items():
        unique = np.unique(values.astype(int))
        if len(unique) != 1:
            raise RuntimeError(f"Run {run_index} has inconsistent scan counts across subjects: {unique}")
        out[int(run_index)] = int(unique[0])
    return out


def _sentence_basis(
    run_triplets: pd.DataFrame,
    *,
    n_scans: int,
    tr: float,
    fine_hz: float,
    hrf: np.ndarray,
) -> np.ndarray:
    fine_dt = 1.0 / fine_hz
    total_duration = n_scans * tr
    grid_len = int(np.ceil(total_duration * fine_hz))
    basis = np.zeros((grid_len, len(run_triplets)), dtype=np.float32)
    for col, row in enumerate(run_triplets.itertuples(index=False)):
        onset_idx = max(0, int(np.floor(float(row.en_onset_sec) * fine_hz)))
        offset_idx = max(onset_idx + 1, int(np.ceil(float(row.en_offset_sec) * fine_hz)))
        offset_idx = min(offset_idx, grid_len)
        basis[onset_idx:offset_idx, col] = 1.0
    conv_len = grid_len + len(hrf) - 1
    convolved = np.empty((conv_len, basis.shape[1]), dtype=np.float32)
    for col in range(basis.shape[1]):
        convolved[:, col] = np.convolve(basis[:, col], hrf, mode="full").astype(np.float32, copy=False)
    scan_times = np.arange(n_scans, dtype=np.float32) * tr
    sample_idx = np.clip(np.round(scan_times * fine_hz).astype(int), 0, conv_len - 1)
    return convolved[sample_idx, :]


def _impulse_series(onsets: np.ndarray, *, n_scans: int, tr: float, fine_hz: float) -> np.ndarray:
    grid_len = int(np.ceil(n_scans * tr * fine_hz))
    series = np.zeros(grid_len, dtype=np.float32)
    for onset in onsets:
        idx = int(np.clip(np.round(float(onset) * fine_hz), 0, grid_len - 1))
        series[idx] += 1.0
    return series


def _boxcar_series(
    onsets: np.ndarray,
    durations: np.ndarray,
    *,
    n_scans: int,
    tr: float,
    fine_hz: float,
) -> np.ndarray:
    grid_len = int(np.ceil(n_scans * tr * fine_hz))
    series = np.zeros(grid_len, dtype=np.float32)
    for onset, duration in zip(onsets, durations, strict=False):
        start = int(np.clip(np.floor(float(onset) * fine_hz), 0, grid_len - 1))
        stop = int(np.clip(np.ceil((float(onset) + float(duration)) * fine_hz), start + 1, grid_len))
        series[start:stop] = 1.0
    return series


def _continuous_series(
    times: np.ndarray,
    values: np.ndarray,
    *,
    n_scans: int,
    tr: float,
    fine_hz: float,
) -> np.ndarray:
    grid_len = int(np.ceil(n_scans * tr * fine_hz))
    fine_times = np.arange(grid_len, dtype=np.float32) / fine_hz
    return np.interp(fine_times, times.astype(np.float32), values.astype(np.float32), left=0.0, right=0.0)


def _convolve_and_sample(series: np.ndarray, *, n_scans: int, tr: float, fine_hz: float, hrf: np.ndarray) -> np.ndarray:
    convolved = np.convolve(series, hrf, mode="full")
    scan_times = np.arange(n_scans, dtype=np.float32) * tr
    sample_idx = np.clip(np.round(scan_times * fine_hz).astype(int), 0, len(convolved) - 1)
    return convolved[sample_idx].astype(np.float32, copy=False)


def _highpass_basis(n_scans: int, tr: float, cutoff_sec: float) -> np.ndarray:
    total_duration = n_scans * tr
    n_harmonics = int(np.floor(2.0 * total_duration / cutoff_sec))
    if n_harmonics <= 0:
        return np.zeros((n_scans, 0), dtype=np.float32)
    n = np.arange(n_scans, dtype=np.float32)[:, None]
    k = np.arange(1, n_harmonics + 1, dtype=np.float32)[None, :]
    basis = np.cos(np.pi * (n + 0.5) * k / n_scans)
    return basis.astype(np.float32, copy=False)


def _build_run_nuisance_and_acoustic(
    *,
    run_index: int,
    n_scans: int,
    tr: float,
    fine_hz: float,
    hrf: np.ndarray,
    prosody: pd.DataFrame,
    word_info: pd.DataFrame,
    sentence_onsets: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prosody_run = prosody.loc[prosody["section"].astype(int) == run_index].copy()
    word_run = word_info.loc[word_info["section"].astype(int) == run_index].copy()

    rms = _convolve_and_sample(
        _continuous_series(prosody_run["time"].to_numpy(), prosody_run["intensity"].to_numpy(), n_scans=n_scans, tr=tr, fine_hz=fine_hz),
        n_scans=n_scans,
        tr=tr,
        fine_hz=fine_hz,
        hrf=hrf,
    )
    f0 = _convolve_and_sample(
        _continuous_series(prosody_run["time"].to_numpy(), prosody_run["f0"].to_numpy(), n_scans=n_scans, tr=tr, fine_hz=fine_hz),
        n_scans=n_scans,
        tr=tr,
        fine_hz=fine_hz,
        hrf=hrf,
    )
    word_rate = _convolve_and_sample(
        _impulse_series(word_run["onset"].to_numpy(), n_scans=n_scans, tr=tr, fine_hz=fine_hz),
        n_scans=n_scans,
        tr=tr,
        fine_hz=fine_hz,
        hrf=hrf,
    )
    sentence_impulse = _convolve_and_sample(
        _impulse_series(sentence_onsets, n_scans=n_scans, tr=tr, fine_hz=fine_hz),
        n_scans=n_scans,
        tr=tr,
        fine_hz=fine_hz,
        hrf=hrf,
    )

    if run_index == 1:
        picture_onsets = np.array([10.0, 35.0, 60.0], dtype=np.float32)
        picture_durations = np.array([15.0, 20.0, 15.0], dtype=np.float32)
        picture_event = _convolve_and_sample(
            _impulse_series(picture_onsets, n_scans=n_scans, tr=tr, fine_hz=fine_hz),
            n_scans=n_scans,
            tr=tr,
            fine_hz=fine_hz,
            hrf=hrf,
        )
        picture_block = _convolve_and_sample(
            _boxcar_series(picture_onsets, picture_durations, n_scans=n_scans, tr=tr, fine_hz=fine_hz),
            n_scans=n_scans,
            tr=tr,
            fine_hz=fine_hz,
            hrf=hrf,
        )
    else:
        picture_event = np.zeros(n_scans, dtype=np.float32)
        picture_block = np.zeros(n_scans, dtype=np.float32)

    acoustic_df = pd.DataFrame(
        {
            "rms": rms,
            "f0": f0,
            "word_rate": word_rate,
            "sentence_onset": sentence_impulse,
            "picture_event": picture_event,
            "picture_block": picture_block,
        }
    )

    trend = np.linspace(-1.0, 1.0, n_scans, dtype=np.float32)
    highpass = _highpass_basis(n_scans, tr=tr, cutoff_sec=float(_pipeline_value("nuisance", "highpass_cutoff_sec", 128)))
    nuisance_columns = {
        "intercept": np.ones(n_scans, dtype=np.float32),
        "linear_trend": trend,
    }
    for idx in range(highpass.shape[1]):
        nuisance_columns[f"highpass_{idx:02d}"] = highpass[:, idx]
    nuisance_df = pd.DataFrame(nuisance_columns)
    return nuisance_df, acoustic_df


def _load_feature_array(
    feature_manifest: pd.DataFrame,
    *,
    condition: str,
    layer_index: int,
    shuffle_index: int | None = None,
) -> np.ndarray:
    subset = feature_manifest.loc[
        (feature_manifest["model"] == "xlmr")
        & (feature_manifest["language"] == "en")
        & (feature_manifest["condition"] == condition)
        & (feature_manifest["layer_index"].astype(int) == layer_index)
    ].copy()
    if shuffle_index is None:
        subset = subset.loc[subset["shuffle_index"].isna()]
    else:
        subset = subset.loc[subset["shuffle_index"].astype(float) == float(shuffle_index)]
    if len(subset) != 1:
        raise RuntimeError(
            f"Expected one feature array for condition={condition}, layer={layer_index}, shuffle={shuffle_index}; got {len(subset)}"
        )
    return np.load(subset.iloc[0]["filepath"]).astype(np.float32, copy=False)


def _prepare_run_designs() -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    pd.DataFrame,
    pd.DataFrame,
    dict[int, pd.DataFrame],
]:
    triplets = _load_triplets().reset_index(names="triplet_row_index")
    feature_manifest = _load_features()
    feature_manifest = feature_manifest.loc[feature_manifest["model"] == "xlmr"].copy()
    fine_hz = float(_pipeline_value("design_matrix", "fine_grid_hz", 10))
    tr = float(_pipeline_value("design_matrix", "tr_sec", 2.0))
    hrf = glover_hrf(t_r=fine_hz ** -1, oversampling=1)
    prosody, word_info = _load_annotation_tables()
    roi_manifest, roi_metadata = _load_roi_targets()
    scan_counts = _run_scan_counts(roi_manifest)

    nuisance_by_run: dict[int, pd.DataFrame] = {}
    acoustic_by_run: dict[int, pd.DataFrame] = {}
    sentence_basis_by_run: dict[int, np.ndarray] = {}
    run_triplets_by_run: dict[int, pd.DataFrame] = {}

    for run_index, n_scans in scan_counts.items():
        run_triplets = triplets.loc[triplets["section_index"].astype(int) == int(run_index)].copy().reset_index(drop=True)
        run_triplets_by_run[int(run_index)] = run_triplets
        sentence_basis_by_run[int(run_index)] = _sentence_basis(
            run_triplets,
            n_scans=int(n_scans),
            tr=tr,
            fine_hz=fine_hz,
            hrf=hrf,
        )
        nuisance_df, acoustic_df = _build_run_nuisance_and_acoustic(
            run_index=int(run_index),
            n_scans=int(n_scans),
            tr=tr,
            fine_hz=fine_hz,
            hrf=hrf,
            prosody=prosody,
            word_info=word_info,
            sentence_onsets=run_triplets["en_onset_sec"].to_numpy(dtype=np.float32),
        )
        nuisance_by_run[int(run_index)] = nuisance_df
        acoustic_by_run[int(run_index)] = acoustic_df

    nuisance_columns = sorted({column for frame in nuisance_by_run.values() for column in frame.columns})
    nuisance_by_run = {
        run_index: frame.reindex(columns=nuisance_columns, fill_value=0.0)
        for run_index, frame in nuisance_by_run.items()
    }
    nuisance_arrays = {
        run_index: frame.to_numpy(dtype=np.float32, copy=False)
        for run_index, frame in nuisance_by_run.items()
    }
    acoustic_arrays = {
        run_index: frame.to_numpy(dtype=np.float32, copy=False)
        for run_index, frame in acoustic_by_run.items()
    }
    text_nuisance_arrays = {
        run_index: np.column_stack([nuisance_arrays[run_index], acoustic_arrays[run_index]]).astype(np.float32, copy=False)
        for run_index in nuisance_arrays
    }

    roi_lookup = roi_metadata.loc[:, ["roi_index", "roi_name", "family"]].rename(columns={"family": "roi_family"})
    return nuisance_arrays, text_nuisance_arrays, acoustic_arrays, sentence_basis_by_run, roi_lookup, feature_manifest, run_triplets_by_run


def _feature_design_for_run(
    *,
    run_basis: np.ndarray,
    run_triplets: pd.DataFrame,
    feature_array: np.ndarray,
) -> np.ndarray:
    row_positions = run_triplets["triplet_row_index"].to_numpy(dtype=np.int64)
    features_run = feature_array[row_positions, :]
    return (run_basis @ features_run).astype(np.float32, copy=False)


def _standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


def _as_2d_targets(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float32)
    if Y.ndim == 1:
        return Y[:, None]
    return Y


def _residualize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    Z_train: np.ndarray,
    Z_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Y_train = _as_2d_targets(Y_train)
    Y_test = _as_2d_targets(Y_test)
    beta_x, *_ = np.linalg.lstsq(Z_train, X_train, rcond=None)
    beta_y, *_ = np.linalg.lstsq(Z_train, Y_train, rcond=None)
    X_train_res = X_train - Z_train @ beta_x
    X_test_res = X_test - Z_test @ beta_x
    y_train_res = Y_train - (Z_train @ beta_y)
    y_test_res = Y_test - (Z_test @ beta_y)
    return X_train_res, X_test_res, y_train_res, y_test_res


def _fit_transform_pca(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_components = int(_pipeline_value("encoding", "pca_max_components", 128))
    variance_threshold = float(_pipeline_value("encoding", "pca_variance_threshold", 0.95))
    max_components = max(1, min(max_components, X_train.shape[0] - 1, X_train.shape[1]))
    solver = str(_pipeline_value("encoding", "pca_solver", "randomized"))
    pca_kwargs: dict[str, Any] = {
        "n_components": max_components,
        "svd_solver": solver,
        "random_state": _random_seed(),
    }
    if solver == "randomized":
        pca_kwargs["iterated_power"] = 3
    pca_full = PCA(**pca_kwargs)
    X_train_full = pca_full.fit_transform(X_train).astype(np.float32, copy=False)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    keep = int(np.searchsorted(cumulative, variance_threshold) + 1)
    keep = max(1, min(keep, max_components))
    return (
        X_train_full[:, :keep].astype(np.float32, copy=False),
        pca_full.transform(X_test)[:, :keep].astype(np.float32, copy=False),
    )


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = y_true - y_true.mean()
    y_pred = y_pred - y_pred.mean()
    denom = np.sqrt(np.sum(y_true * y_true) * np.sum(y_pred * y_pred))
    if denom <= 0:
        return 0.0
    return float(np.clip(np.sum(y_true * y_pred) / denom, -0.999999, 0.999999))


def _pearson_r_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = _as_2d_targets(y_true).astype(np.float64, copy=False)
    y_pred = _as_2d_targets(y_pred).astype(np.float64, copy=False)
    y_true = y_true - y_true.mean(axis=0, keepdims=True)
    y_pred = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(y_true * y_true, axis=0) * np.sum(y_pred * y_pred, axis=0))
    numer = np.sum(y_true * y_pred, axis=0)
    r = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    return np.clip(r, -0.999999, 0.999999)


def _ridge_svd_cache(X_train: np.ndarray, Y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, singular_values, Vt = np.linalg.svd(X_train, full_matrices=False)
    UtY = U.T @ Y_train
    return (
        singular_values.astype(np.float32, copy=False),
        Vt.astype(np.float32, copy=False),
        UtY.astype(np.float32, copy=False),
    )


def _ridge_predict_from_cache(
    cache: tuple[np.ndarray, np.ndarray, np.ndarray],
    X_test: np.ndarray,
    alpha: float,
    target_indices: np.ndarray | None = None,
) -> np.ndarray:
    singular_values, Vt, UtY = cache
    if target_indices is None:
        UtY_use = UtY
    else:
        UtY_use = UtY[:, target_indices]
    shrink = (singular_values / ((singular_values * singular_values) + float(alpha)))[:, None]
    coef = Vt.T @ (shrink * UtY_use)
    return (X_test @ coef).astype(np.float32, copy=False)


def _target_chunk_size() -> int:
    raw_value = os.environ.get("BRAIN_SUBSPACE_TARGET_CHUNK_SIZE", "").strip()
    if raw_value:
        try:
            return max(1, int(raw_value))
        except ValueError:
            pass
    return 512


def _target_column_slices(n_targets: int) -> tuple[slice, ...]:
    chunk_size = _target_chunk_size()
    return tuple(
        slice(start, min(start + chunk_size, n_targets))
        for start in range(0, n_targets, chunk_size)
    )


def _select_alpha(
    run_ids: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    alphas = _alpha_grid()
    Y = _as_2d_targets(Y)
    train_runs = sorted(np.unique(run_ids).tolist())
    target_slices = _target_column_slices(Y.shape[1])
    score_sum = np.zeros((len(alphas), Y.shape[1]), dtype=np.float64)
    for held_out_run in train_runs:
        train_mask = run_ids != held_out_run
        test_mask = run_ids == held_out_run
        X_train_res, X_test_res, y_train_res, y_test_res = _residualize_train_test(
            X[train_mask],
            X[test_mask],
            Y[train_mask],
            Y[test_mask],
            Z[train_mask],
            Z[test_mask],
        )
        X_train_std, X_test_std = _standardize_train_test(X_train_res, X_test_res)
        X_train_pca, X_test_pca = _fit_transform_pca(X_train_std, X_test_std)
        for target_slice in target_slices:
            y_train_chunk = y_train_res[:, target_slice]
            y_test_chunk = y_test_res[:, target_slice]
            cache = _ridge_svd_cache(X_train_pca, y_train_chunk)
            for alpha_index, alpha in enumerate(alphas):
                pred = _ridge_predict_from_cache(cache, X_test_pca, float(alpha))
                score_sum[alpha_index, target_slice] += np.arctanh(_pearson_r_per_target(y_test_chunk, pred))

    score_matrix = score_sum / len(train_runs)
    best_alphas = np.empty(Y.shape[1], dtype=np.float64)
    for target_index in range(Y.shape[1]):
        best_score = float(score_matrix[:, target_index].max())
        best_alphas[target_index] = max(
            alpha
            for alpha, score in zip(alphas, score_matrix[:, target_index], strict=False)
            if float(score) == best_score
        )
    return best_alphas


def _stack_condition_arrays(
    *,
    run_designs: dict[int, np.ndarray],
    z_by_run: dict[int, np.ndarray],
    roi_series_by_run: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    run_order = sorted(run_designs)
    run_ids = []
    X_blocks = []
    Y_blocks = []
    Z_blocks = []
    for run_index in run_order:
        run_ids.extend([run_index] * len(roi_series_by_run[run_index]))
        X_blocks.append(run_designs[run_index])
        Y_blocks.append(roi_series_by_run[run_index])
        Z_blocks.append(z_by_run[run_index])

    run_ids_array = np.asarray(run_ids, dtype=np.int64)
    X = np.vstack(X_blocks).astype(np.float32, copy=False)
    Y = _as_2d_targets(np.concatenate(Y_blocks, axis=0))
    Z = np.vstack(Z_blocks).astype(np.float32, copy=False)
    return run_ids_array, X, Y, Z


def _run_single_condition_arrays(
    *,
    run_ids_array: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    run_order = sorted(np.unique(run_ids_array).tolist())
    Y = _as_2d_targets(Y)
    target_slices = _target_column_slices(Y.shape[1])

    z_sum = np.zeros(Y.shape[1], dtype=np.float64)
    target_sum = np.zeros(Y.shape[1], dtype=np.float64)
    target_sq_sum = np.zeros(Y.shape[1], dtype=np.float64)
    sse_sum = np.zeros(Y.shape[1], dtype=np.float64)
    n_observations = 0
    for held_out_run in run_order:
        train_mask = run_ids_array != held_out_run
        test_mask = run_ids_array == held_out_run
        alpha_by_target = _select_alpha(run_ids_array[train_mask], X[train_mask], Y[train_mask], Z[train_mask])
        X_train_res, X_test_res, y_train_res, y_test_res = _residualize_train_test(
            X[train_mask],
            X[test_mask],
            Y[train_mask],
            Y[test_mask],
            Z[train_mask],
            Z[test_mask],
        )
        X_train_std, X_test_std = _standardize_train_test(X_train_res, X_test_res)
        X_train_pca, X_test_pca = _fit_transform_pca(X_train_std, X_test_std)
        n_observations += int(y_test_res.shape[0])
        for target_slice in target_slices:
            y_train_chunk = y_train_res[:, target_slice]
            y_test_chunk = y_test_res[:, target_slice]
            alpha_chunk = alpha_by_target[target_slice]
            cache = _ridge_svd_cache(X_train_pca, y_train_chunk)
            pred_chunk = np.empty_like(y_test_chunk)
            for alpha in sorted({float(value) for value in alpha_chunk.tolist()}):
                target_indices = np.flatnonzero(alpha_chunk == alpha)
                pred_chunk[:, target_indices] = _ridge_predict_from_cache(cache, X_test_pca, alpha, target_indices)
            z_sum[target_slice] += np.arctanh(_pearson_r_per_target(y_test_chunk, pred_chunk))
            y_test_chunk64 = y_test_chunk.astype(np.float64, copy=False)
            pred_chunk64 = pred_chunk.astype(np.float64, copy=False)
            target_sum[target_slice] += np.sum(y_test_chunk64, axis=0)
            target_sq_sum[target_slice] += np.sum(y_test_chunk64 * y_test_chunk64, axis=0)
            residual_chunk = y_test_chunk64 - pred_chunk64
            sse_sum[target_slice] += np.sum(residual_chunk * residual_chunk, axis=0)

    z_value = z_sum / len(run_order)
    r_value = np.tanh(z_value)
    denom = target_sq_sum - ((target_sum * target_sum) / float(n_observations))
    r2_value = np.where(denom <= 0, 0.0, 1.0 - (sse_sum / denom))
    return r_value.astype(np.float64), z_value.astype(np.float64), r2_value.astype(np.float64)


def _run_single_condition(
    *,
    run_designs: dict[int, np.ndarray],
    z_by_run: dict[int, np.ndarray],
    roi_series_by_run: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    run_ids_array, X, Y, Z = _stack_condition_arrays(
        run_designs=run_designs,
        z_by_run=z_by_run,
        roi_series_by_run=roi_series_by_run,
    )
    return _run_single_condition_arrays(
        run_ids_array=run_ids_array,
        X=X,
        Y=Y,
        Z=Z,
    )


def run_english_roi_prototype(
    *,
    max_subjects: int | None = None,
    layer_indices: tuple[int, ...] = PROTOTYPE_LAYER_INDICES,
) -> EnglishPrototypeSummary:
    bootstrap_logs()
    roi_manifest, _ = _load_roi_targets()
    nuisance_arrays, text_nuisance_arrays, acoustic_designs, sentence_basis_by_run, roi_lookup, feature_manifest, run_triplets_by_run = _prepare_run_designs()
    feature_manifest = feature_manifest.loc[feature_manifest["model"] == "xlmr"].copy()
    text_designs: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    mismatched_designs: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    for layer_index in layer_indices:
        for condition in TEXT_CONDITIONS:
            feature_array = _load_feature_array(feature_manifest, condition=condition, layer_index=layer_index)
            text_designs[(condition, layer_index)] = {
                run_index: _feature_design_for_run(
                    run_basis=sentence_basis_by_run[run_index],
                    run_triplets=run_triplets_by_run[run_index],
                    feature_array=feature_array,
                )
                for run_index in sentence_basis_by_run
            }
        for shuffle_index in range(int(_pipeline_value("design_matrix", "mismatched_shared_shuffles", 5))):
            feature_array = _load_feature_array(
                feature_manifest,
                condition="mismatched_shared",
                layer_index=layer_index,
                shuffle_index=shuffle_index,
            )
            mismatched_designs[(layer_index, shuffle_index)] = {
                run_index: _feature_design_for_run(
                    run_basis=sentence_basis_by_run[run_index],
                    run_triplets=run_triplets_by_run[run_index],
                    feature_array=feature_array,
                )
                for run_index in sentence_basis_by_run
            }
    roi_index_lookup = {
        str(row.roi_name): int(row.roi_index) - 1
        for row in roi_lookup.itertuples(index=False)
    }
    roi_family_lookup = {
        str(row.roi_name): str(row.roi_family)
        for row in roi_lookup.itertuples(index=False)
    }
    subject_ids = sorted(roi_manifest["subject_id"].unique().tolist())
    if max_subjects is not None:
        subject_ids = subject_ids[:max_subjects]

    results_rows: list[dict[str, Any]] = []
    for subject_id in subject_ids:
        subject_runs = roi_manifest.loc[roi_manifest["subject_id"] == subject_id].sort_values("canonical_run_index")
        roi_run_data = {
            int(row.canonical_run_index): np.load(row.filepath).astype(np.float32, copy=False)
            for row in subject_runs.itertuples(index=False)
        }

        roi_names = list(PROTOTYPE_ROIS)
        roi_indices = [roi_index_lookup[roi_name] for roi_name in roi_names]
        y_by_run = {run_index: roi_run_data[run_index][:, roi_indices] for run_index in roi_run_data}

        # acoustic-only baseline
        r_vals, z_vals, r2_vals = _run_single_condition(
            run_designs={run_index: acoustic_designs[run_index] for run_index in y_by_run},
            z_by_run={run_index: nuisance_arrays[run_index] for run_index in y_by_run},
            roi_series_by_run=y_by_run,
        )
        for roi_name, r_val, z_val, r2_val in zip(roi_names, r_vals, z_vals, r2_vals, strict=False):
            roi_family = roi_family_lookup[roi_name]
            for metric_name, value in (("r", r_val), ("z", z_val), ("r2", r2_val)):
                results_rows.append(
                    {
                        "subject_id": subject_id,
                        "language": "en",
                        "model": "xlmr",
                        "roi_name": roi_name,
                        "roi_family": roi_family,
                        "layer_index": -1,
                        "layer_depth": np.nan,
                        "condition": "acoustic_only",
                        "metric_name": metric_name,
                        "value": value,
                    }
                )

        for layer_index in layer_indices:
            layer_depth = layer_index / 12.0
            for condition in TEXT_CONDITIONS:
                r_vals, z_vals, r2_vals = _run_single_condition(
                    run_designs={run_index: text_designs[(condition, layer_index)][run_index] for run_index in y_by_run},
                    z_by_run={run_index: text_nuisance_arrays[run_index] for run_index in y_by_run},
                    roi_series_by_run=y_by_run,
                )
                for roi_name, r_val, z_val, r2_val in zip(roi_names, r_vals, z_vals, r2_vals, strict=False):
                    roi_family = roi_family_lookup[roi_name]
                    for metric_name, value in (("r", r_val), ("z", z_val), ("r2", r2_val)):
                        results_rows.append(
                            {
                                "subject_id": subject_id,
                                "language": "en",
                                "model": "xlmr",
                                "roi_name": roi_name,
                                "roi_family": roi_family,
                                "layer_index": layer_index,
                                "layer_depth": layer_depth,
                                "condition": condition,
                                "metric_name": metric_name,
                                "value": value,
                            }
                        )

            mismatch_metrics = []
            for shuffle_index in range(int(_pipeline_value("design_matrix", "mismatched_shared_shuffles", 5))):
                mismatch_metrics.append(
                    _run_single_condition(
                        run_designs={
                            run_index: mismatched_designs[(layer_index, shuffle_index)][run_index]
                            for run_index in y_by_run
                        },
                        z_by_run={run_index: text_nuisance_arrays[run_index] for run_index in y_by_run},
                        roi_series_by_run=y_by_run,
                    )
                )
            mismatch_r = np.tanh(
                np.mean(
                    np.arctanh(
                        np.clip(
                            np.vstack([metrics[0] for metrics in mismatch_metrics]),
                            -0.999999,
                            0.999999,
                        )
                    ),
                    axis=0,
                )
            )
            mismatch_z = np.mean(np.vstack([metrics[1] for metrics in mismatch_metrics]), axis=0)
            mismatch_r2 = np.mean(np.vstack([metrics[2] for metrics in mismatch_metrics]), axis=0)
            for roi_name, r_val, z_val, r2_val in zip(roi_names, mismatch_r, mismatch_z, mismatch_r2, strict=False):
                roi_family = roi_family_lookup[roi_name]
                for metric_name, value in (("r", r_val), ("z", z_val), ("r2", r2_val)):
                    results_rows.append(
                        {
                            "subject_id": subject_id,
                            "language": "en",
                            "model": "xlmr",
                            "roi_name": roi_name,
                            "roi_family": roi_family,
                            "layer_index": layer_index,
                            "layer_depth": layer_depth,
                            "condition": "mismatched_shared",
                            "metric_name": metric_name,
                            "value": value,
                        }
                    )

    results_df = pd.DataFrame(results_rows)
    results_path = _results_path()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(results_path, index=False)

    plot_df = (
        results_df.loc[(results_df["metric_name"] == "z") & (results_df["condition"].isin(PROTOTYPE_CONDITIONS))]
        .groupby(["condition"], as_index=False)["value"]
        .mean()
    )
    plot_path = _plot_path()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(plot_df["condition"], plot_df["value"], color=["#7a8", "#4c78a8", "#f58518", "#e45756", "#72b7b2"])
    ax.set_ylabel("Mean subject z")
    ax.set_title("English ROI Prototype Condition Summary")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    check_raw = (
        plot_df.loc[plot_df["condition"] == "raw", "value"].iloc[0]
        > plot_df.loc[plot_df["condition"] == "acoustic_only", "value"].iloc[0]
    )
    check_shared = (
        plot_df.loc[plot_df["condition"] == "shared", "value"].iloc[0]
        > plot_df.loc[plot_df["condition"] == "mismatched_shared", "value"].iloc[0]
    )
    report_lines = [
        "# Encoding QC Report",
        "",
        f"- prototype scope: `English / XLM-R / {len(PROTOTYPE_ROIS)} ROIs / layers {', '.join(str(layer) for layer in layer_indices)}`",
        f"- subjects: `{results_df['subject_id'].nunique()}`",
        f"- rois: `{len(PROTOTYPE_ROIS)}`",
        f"- layers: `{len(layer_indices)}`",
        f"- results: `{results_path.relative_to(project_root()).as_posix()}`",
        f"- plot: `{plot_path.relative_to(project_root()).as_posix()}`",
        "",
        "## Mean Z By Condition",
        "",
        plot_df.to_string(index=False),
        "",
        "## Stage Checks",
        "",
        f"- `RAW > acoustic_only` on mean prototype z: `{bool(check_raw)}`",
        f"- `SHARED > MISMATCHED_SHARED` on mean prototype z: `{bool(check_shared)}`",
        "",
    ]
    report_path = _report_path()
    write_text(report_path, "\n".join(report_lines) + "\n")

    append_markdown_log(
        project_root() / "outputs" / "logs" / "progress_log.md",
        "English ROI prototype",
        [
            f"subjects={results_df['subject_id'].nunique()}",
            f"results={results_path.as_posix()}",
            f"plot={plot_path.as_posix()}",
            f"raw_gt_acoustic={bool(check_raw)}",
            f"shared_gt_mismatched={bool(check_shared)}",
        ],
    )

    return EnglishPrototypeSummary(
        results_path=results_path,
        plot_path=plot_path,
        report_path=report_path,
        n_subjects=int(results_df["subject_id"].nunique()),
        n_rois=len(PROTOTYPE_ROIS),
        n_layers=len(layer_indices),
    )
