from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from nilearn.glm.first_level.hemodynamic_models import glover_hrf
import numpy as np
import pandas as pd

from brain_subspace_paper.config import project_root
from brain_subspace_paper.data.sentence_spans import _read_csv
from brain_subspace_paper.encoding.english_prototype import (
    _boxcar_series,
    _convolve_and_sample,
    _continuous_series,
    _highpass_basis,
    _impulse_series,
    _load_triplets as _prototype_load_triplets,
    _pipeline_value,
    _random_seed,
    _run_scan_counts,
    _run_single_condition,
)
from brain_subspace_paper.logging_utils import append_markdown_log, bootstrap_logs, write_text


LANGUAGE_SPECS = {
    "en": {
        "annotation_dir": "EN",
        "prefix": "lppEN",
        "onset_col": "en_onset_sec",
        "offset_col": "en_offset_sec",
        "picture_regressors": True,
    },
    "fr": {
        "annotation_dir": "FR",
        "prefix": "lppFR",
        "onset_col": "fr_onset_sec",
        "offset_col": "fr_offset_sec",
        "picture_regressors": False,
    },
    "zh": {
        "annotation_dir": "CN",
        "prefix": "lppCN",
        "onset_col": "zh_onset_sec",
        "offset_col": "zh_offset_sec",
        "picture_regressors": True,
    },
}

TEXT_CONDITIONS = ("raw", "shared", "specific", "full")


@dataclass(slots=True)
class ModelRoiPipelineSummary:
    subject_results_path: Path
    group_results_path: Path | None
    confirmatory_path: Path | None
    report_path: Path | None
    model_name: str
    languages: tuple[str, ...]
    n_subjects: int
    n_rows: int


@dataclass(slots=True)
class LanguagePreparedData:
    language: str
    roi_manifest: pd.DataFrame
    roi_lookup: pd.DataFrame
    nuisance_arrays: dict[int, np.ndarray]
    text_nuisance_arrays: dict[int, np.ndarray]
    run_triplets_by_run: dict[int, pd.DataFrame]
    acoustic_designs: dict[int, np.ndarray]
    text_designs: dict[tuple[str, int], dict[int, np.ndarray]]
    mismatched_designs: dict[tuple[int, int], dict[int, np.ndarray]]


def _roi_root() -> Path:
    return project_root() / "data" / "interim" / "roi"


def _roi_manifest_path(language: str) -> Path:
    return _roi_root() / f"{language}_roi_target_manifest.parquet"


def _roi_metadata_path() -> Path:
    return _roi_root() / "roi_metadata.parquet"


def _output_stem(model_name: str) -> str:
    return "nllb" if model_name == "nllb_encoder" else model_name


def _normalize_model_name(model_name: str) -> str:
    if model_name == "nllb":
        return "nllb_encoder"
    if model_name in {"xlmr", "nllb_encoder"}:
        return model_name
    raise ValueError(f"Unsupported model={model_name!r}. Expected one of ('xlmr', 'nllb_encoder', 'nllb').")


def _tagged_filename(filename: str, output_tag: str | None) -> str:
    if not output_tag:
        return filename
    path = Path(filename)
    return f"{path.stem}__{output_tag}{path.suffix}"


def _subject_results_path(model_name: str, output_tag: str | None = None) -> Path:
    filename = f"{_output_stem(model_name)}_subject_level_roi_results.parquet"
    return project_root() / "outputs" / "stats" / _tagged_filename(filename, output_tag)


def _subject_chunk_metadata_path(model_name: str, output_tag: str | None = None) -> Path:
    subject_path = _subject_results_path(model_name, output_tag=output_tag)
    return subject_path.with_name(f"{subject_path.stem}__metadata.json")


def _group_results_path(model_name: str, output_tag: str | None = None) -> Path:
    filename = f"{_output_stem(model_name)}_group_level_roi_results.parquet"
    return project_root() / "outputs" / "stats" / _tagged_filename(filename, output_tag)


def _confirmatory_path(model_name: str, output_tag: str | None = None) -> Path:
    filename = f"{_output_stem(model_name)}_confirmatory_effects.parquet"
    return project_root() / "outputs" / "stats" / _tagged_filename(filename, output_tag)


def _report_path(model_name: str, output_tag: str | None = None) -> Path:
    if output_tag is None:
        filename = "encoding_qc_report.md"
    else:
        filename = f"{_output_stem(model_name)}_encoding_qc_report.md"
    return project_root() / "outputs" / "logs" / _tagged_filename(filename, output_tag)


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    df.to_parquet(temp_path, index=False)
    temp_path.replace(path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    write_text(path.with_name(f"{path.name}.tmp"), json.dumps(payload, indent=2, sort_keys=True) + "\n").replace(path)


def _subject_chunk_metadata(
    *,
    model_name: str,
    languages: tuple[str, ...],
    layer_indices: tuple[int, ...],
    mismatch_shuffles: int,
    max_subjects: int | None,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "languages": list(languages),
        "layer_indices": list(layer_indices),
        "mismatch_shuffles": int(mismatch_shuffles),
        "max_subjects": None if max_subjects is None else int(max_subjects),
        "conditions": ["acoustic_only", *TEXT_CONDITIONS, "mismatched_shared"],
    }


def _load_existing_subject_chunk(
    *,
    subject_path: Path,
    metadata_path: Path,
    expected_metadata: dict[str, Any],
) -> pd.DataFrame:
    existing_df = pd.read_parquet(subject_path).copy()
    if not metadata_path.exists():
        return existing_df
    stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    comparable_keys = ("model_name", "languages", "layer_indices", "mismatch_shuffles", "max_subjects", "conditions")
    mismatches = [
        key
        for key in comparable_keys
        if stored_metadata.get(key) != expected_metadata.get(key)
    ]
    if mismatches:
        raise RuntimeError(
            "Cannot resume subject-only ROI chunk because the existing output tag was created with different settings. "
            f"Mismatched metadata keys: {', '.join(mismatches)}"
        )
    return existing_df


def _language_spec(language: str) -> dict[str, Any]:
    if language not in LANGUAGE_SPECS:
        raise ValueError(f"Unsupported language={language!r}. Expected one of {tuple(LANGUAGE_SPECS)}.")
    return LANGUAGE_SPECS[language]


def _annotation_paths(language: str) -> tuple[Path, Path]:
    spec = _language_spec(language)
    root = project_root() / "data" / "raw" / "ds003643" / "annotation" / spec["annotation_dir"]
    return root / f"{spec['prefix']}_prosody.csv", root / f"{spec['prefix']}_word_information.csv"


@lru_cache(maxsize=None)
def _cached_annotation_tables(language: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prosody_path, word_info_path = _annotation_paths(language)
    prosody = _read_csv(prosody_path).copy()
    word_info = _read_csv(word_info_path).copy()
    return prosody, word_info


def _load_annotation_tables(language: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prosody, word_info = _cached_annotation_tables(language)
    return prosody.copy(), word_info.copy()


@lru_cache(maxsize=None)
def _cached_roi_targets(language: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = _roi_manifest_path(language)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing ROI target manifest for language={language}: {manifest_path}. "
            "Run `python -m brain_subspace_paper extract-roi-targets --language "
            f"{language}` first."
        )
    roi_manifest = (
        pd.read_parquet(manifest_path)
        .copy()
        .sort_values(["subject_id", "canonical_run_index"])
        .reset_index(drop=True)
    )
    roi_metadata = pd.read_parquet(_roi_metadata_path()).copy()
    roi_lookup = roi_metadata.loc[:, ["roi_index", "roi_name", "family"]].rename(columns={"family": "roi_family"})
    return roi_manifest, roi_lookup


def _load_roi_targets(language: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    roi_manifest, roi_lookup = _cached_roi_targets(language)
    return roi_manifest.copy(), roi_lookup.copy()


@lru_cache(maxsize=None)
def _cached_feature_manifest(model_name: str, language: str) -> pd.DataFrame:
    path = project_root() / "data" / "processed" / "features" / "feature_manifest.parquet"
    df = pd.read_parquet(path).copy()
    subset = df.loc[(df["model"] == model_name) & (df["language"] == language)].copy()
    if subset.empty:
        raise FileNotFoundError(f"No feature manifest rows found for model={model_name}, language={language}.")
    return subset


def _load_feature_manifest(model_name: str, language: str) -> pd.DataFrame:
    return _cached_feature_manifest(model_name, language).copy()


@lru_cache(maxsize=None)
def _cached_triplets() -> pd.DataFrame:
    return _prototype_load_triplets()


def _load_triplets() -> pd.DataFrame:
    return _cached_triplets().copy()


def _load_feature_array(
    feature_manifest: pd.DataFrame,
    *,
    model_name: str,
    language: str,
    condition: str,
    layer_index: int,
    shuffle_index: int | None = None,
) -> np.ndarray:
    subset = feature_manifest.loc[
        (feature_manifest["model"] == model_name)
        & (feature_manifest["language"] == language)
        & (feature_manifest["condition"] == condition)
        & (feature_manifest["layer_index"].astype(int) == layer_index)
    ].copy()
    if shuffle_index is None:
        subset = subset.loc[subset["shuffle_index"].isna()]
    else:
        subset = subset.loc[subset["shuffle_index"].astype(float) == float(shuffle_index)]
    if len(subset) != 1:
        raise RuntimeError(
            f"Expected one feature array for model={model_name}, language={language}, condition={condition}, "
            f"layer={layer_index}, shuffle={shuffle_index}; got {len(subset)}"
        )
    return np.load(subset.iloc[0]["filepath"]).astype(np.float32, copy=False)


def _sentence_basis(
    run_triplets: pd.DataFrame,
    *,
    onset_col: str,
    offset_col: str,
    n_scans: int,
    tr: float,
    fine_hz: float,
    hrf: np.ndarray,
) -> np.ndarray:
    grid_len = int(np.ceil(n_scans * tr * fine_hz))
    basis = np.zeros((grid_len, len(run_triplets)), dtype=np.float32)
    for col, row in enumerate(run_triplets.itertuples(index=False)):
        onset_idx = max(0, int(np.floor(float(getattr(row, onset_col)) * fine_hz)))
        offset_idx = max(onset_idx + 1, int(np.ceil(float(getattr(row, offset_col)) * fine_hz)))
        offset_idx = min(offset_idx, grid_len)
        basis[onset_idx:offset_idx, col] = 1.0
    conv_len = grid_len + len(hrf) - 1
    convolved = np.empty((conv_len, basis.shape[1]), dtype=np.float32)
    for col in range(basis.shape[1]):
        convolved[:, col] = np.convolve(basis[:, col], hrf, mode="full").astype(np.float32, copy=False)
    scan_times = np.arange(n_scans, dtype=np.float32) * tr
    sample_idx = np.clip(np.round(scan_times * fine_hz).astype(int), 0, conv_len - 1)
    return convolved[sample_idx, :]


def _build_run_nuisance_and_acoustic(
    *,
    language: str,
    run_index: int,
    n_scans: int,
    tr: float,
    fine_hz: float,
    hrf: np.ndarray,
    prosody: pd.DataFrame,
    word_info: pd.DataFrame,
    sentence_onsets: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = _language_spec(language)
    prosody_run = prosody.loc[prosody["section"].astype(int) == run_index].copy()
    word_run = word_info.loc[word_info["section"].astype(int) == run_index].copy()

    rms = _convolve_and_sample(
        _continuous_series(
            prosody_run["time"].to_numpy(),
            prosody_run["intensity"].to_numpy(),
            n_scans=n_scans,
            tr=tr,
            fine_hz=fine_hz,
        ),
        n_scans=n_scans,
        tr=tr,
        fine_hz=fine_hz,
        hrf=hrf,
    )
    f0 = _convolve_and_sample(
        _continuous_series(
            prosody_run["time"].to_numpy(),
            prosody_run["f0"].to_numpy(),
            n_scans=n_scans,
            tr=tr,
            fine_hz=fine_hz,
        ),
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

    if spec["picture_regressors"] and run_index == 1:
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
    highpass = _highpass_basis(
        n_scans,
        tr=tr,
        cutoff_sec=float(_pipeline_value("nuisance", "highpass_cutoff_sec", 128)),
    )
    nuisance_columns = {
        "intercept": np.ones(n_scans, dtype=np.float32),
        "linear_trend": trend,
    }
    for idx in range(highpass.shape[1]):
        nuisance_columns[f"highpass_{idx:02d}"] = highpass[:, idx]
    nuisance_df = pd.DataFrame(nuisance_columns)
    return nuisance_df, acoustic_df


def _prepare_language_data(
    language: str,
    *,
    model_name: str,
    layer_indices: tuple[int, ...],
    mismatch_shuffles: int | None = None,
) -> LanguagePreparedData:
    triplets = _load_triplets().reset_index(names="triplet_row_index")
    feature_manifest = _load_feature_manifest(model_name, language)
    roi_manifest, roi_lookup = _load_roi_targets(language)
    fine_hz = float(_pipeline_value("design_matrix", "fine_grid_hz", 10))
    tr = float(_pipeline_value("design_matrix", "tr_sec", 2.0))
    hrf = glover_hrf(t_r=fine_hz ** -1, oversampling=1)
    prosody, word_info = _load_annotation_tables(language)
    spec = _language_spec(language)
    scan_counts = _run_scan_counts(roi_manifest)

    nuisance_by_run: dict[int, pd.DataFrame] = {}
    acoustic_by_run: dict[int, pd.DataFrame] = {}
    sentence_basis_by_run: dict[int, np.ndarray] = {}
    run_triplets_by_run: dict[int, pd.DataFrame] = {}

    for run_index, n_scans in scan_counts.items():
        run_triplets = (
            triplets.loc[triplets["section_index"].astype(int) == int(run_index)]
            .copy()
            .reset_index(drop=True)
        )
        run_triplets_by_run[int(run_index)] = run_triplets
        sentence_basis_by_run[int(run_index)] = _sentence_basis(
            run_triplets,
            onset_col=spec["onset_col"],
            offset_col=spec["offset_col"],
            n_scans=int(n_scans),
            tr=tr,
            fine_hz=fine_hz,
            hrf=hrf,
        )
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
    acoustic_designs = acoustic_arrays

    text_designs: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    mismatched_designs: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    n_shuffles = (
        int(mismatch_shuffles)
        if mismatch_shuffles is not None
        else int(_pipeline_value("design_matrix", "mismatched_shared_shuffles", 5))
    )
    for layer_index in layer_indices:
        for condition in TEXT_CONDITIONS:
            feature_array = _load_feature_array(
                feature_manifest,
                model_name=model_name,
                language=language,
                condition=condition,
                layer_index=layer_index,
            )
            text_designs[(condition, layer_index)] = {
                run_index: (
                    sentence_basis_by_run[run_index]
                    @ feature_array[
                        run_triplets_by_run[run_index]["triplet_row_index"].to_numpy(dtype=np.int64),
                        :,
                    ]
                ).astype(np.float32, copy=False)
                for run_index in sentence_basis_by_run
            }
        for shuffle_index in range(n_shuffles):
            feature_array = _load_feature_array(
                feature_manifest,
                model_name=model_name,
                language=language,
                condition="mismatched_shared",
                layer_index=layer_index,
                shuffle_index=shuffle_index,
            )
            mismatched_designs[(layer_index, shuffle_index)] = {
                run_index: (
                    sentence_basis_by_run[run_index]
                    @ feature_array[
                        run_triplets_by_run[run_index]["triplet_row_index"].to_numpy(dtype=np.int64),
                        :,
                    ]
                ).astype(np.float32, copy=False)
                for run_index in sentence_basis_by_run
            }

    return LanguagePreparedData(
        language=language,
        roi_manifest=roi_manifest,
        roi_lookup=roi_lookup,
        nuisance_arrays=nuisance_arrays,
        text_nuisance_arrays=text_nuisance_arrays,
        run_triplets_by_run=run_triplets_by_run,
        acoustic_designs=acoustic_designs,
        text_designs=text_designs,
        mismatched_designs=mismatched_designs,
    )


def _se(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _dz(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    sd = float(np.std(values, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(values) / sd)


def _sign_flip_pvalue(values: np.ndarray, *, n_permutations: int, rng: np.random.Generator) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = float(np.mean(values))
    flips = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_permutations, len(values)))
    permuted = (flips * values[None, :]).mean(axis=1)
    return float((1 + np.sum(permuted >= observed)) / (n_permutations + 1))


def _bootstrap_ci(values: np.ndarray, *, n_bootstraps: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(n_bootstraps, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running_max = 0.0
    n_tests = len(p_values)
    for rank, idx in enumerate(order):
        corrected = min((n_tests - rank) * float(p_values[idx]), 1.0)
        running_max = max(running_max, corrected)
        adjusted[idx] = running_max
    return adjusted.tolist()


def _group_results(subject_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["language", "model", "roi_name", "roi_family", "layer_index", "layer_depth", "condition"]
    r_df = (
        subject_df.loc[subject_df["metric_name"] == "r"]
        .groupby(key_cols, as_index=False)
        .agg(mean_r=("value", "mean"))
    )
    z_df = (
        subject_df.loc[subject_df["metric_name"] == "z"]
        .groupby(key_cols, as_index=False)
        .agg(
            mean_z=("value", "mean"),
            se_z=("value", lambda series: 0.0 if len(series) <= 1 else float(series.std(ddof=1) / np.sqrt(len(series)))),
            n_subjects=("subject_id", "nunique"),
        )
    )
    r2_df = (
        subject_df.loc[subject_df["metric_name"] == "r2"]
        .groupby(key_cols, as_index=False)
        .agg(mean_r2=("value", "mean"))
    )
    return r_df.merge(z_df, on=key_cols).merge(r2_df, on=key_cols)


def _confirmatory_effects(
    subject_df: pd.DataFrame,
    *,
    n_permutations: int,
    n_bootstraps: int,
) -> pd.DataFrame:
    z_df = subject_df.loc[
        (subject_df["metric_name"] == "z")
        & (subject_df["condition"].isin(["shared", "specific"]))
    ].copy()
    mid_min = float(_pipeline_value("statistics", "confirmatory_mid_layer_min_depth", 0.33))
    mid_max = float(_pipeline_value("statistics", "confirmatory_mid_layer_max_depth", 0.83))
    z_df = z_df.loc[(z_df["layer_depth"] >= mid_min) & (z_df["layer_depth"] <= mid_max)].copy()
    if z_df.empty:
        return pd.DataFrame(
            columns=[
                "hypothesis",
                "language",
                "model",
                "roi_family",
                "mean_delta_mid",
                "se",
                "dz",
                "p_perm",
                "p_holm_primary",
                "ci_low",
                "ci_high",
                "n_subjects",
            ]
        )

    pivot = z_df.pivot_table(
        index=["subject_id", "language", "model", "roi_name", "roi_family", "layer_index", "layer_depth"],
        columns="condition",
        values="value",
    ).reset_index()
    pivot["delta_shared_specific"] = pivot["shared"] - pivot["specific"]

    family_effects = (
        pivot.groupby(["subject_id", "language", "model", "roi_family"], as_index=False)["delta_shared_specific"]
        .mean()
        .rename(columns={"delta_shared_specific": "delta_mid"})
    )
    semantic = family_effects.loc[family_effects["roi_family"] == "semantic"].copy()
    auditory = family_effects.loc[family_effects["roi_family"] == "auditory"].copy()

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(_random_seed())
    for (language, model), semantic_group in semantic.groupby(["language", "model"], sort=True):
        semantic_values = semantic_group["delta_mid"].to_numpy(dtype=np.float64)
        low, high = _bootstrap_ci(semantic_values, n_bootstraps=n_bootstraps, rng=rng)
        rows.append(
            {
                "hypothesis": "H1_shared_gt_specific_semantic",
                "language": language,
                "model": model,
                "roi_family": "semantic",
                "mean_delta_mid": float(np.mean(semantic_values)),
                "se": _se(semantic_values),
                "dz": _dz(semantic_values),
                "p_perm": _sign_flip_pvalue(semantic_values, n_permutations=n_permutations, rng=rng),
                "p_holm_primary": np.nan,
                "ci_low": low,
                "ci_high": high,
                "n_subjects": int(len(semantic_values)),
            }
        )

        merged = semantic_group.merge(
            auditory.loc[(auditory["language"] == language) & (auditory["model"] == model)],
            on=["subject_id", "language", "model"],
            suffixes=("_semantic", "_auditory"),
        )
        if not merged.empty:
            h2_values = (merged["delta_mid_semantic"] - merged["delta_mid_auditory"]).to_numpy(dtype=np.float64)
            low, high = _bootstrap_ci(h2_values, n_bootstraps=n_bootstraps, rng=rng)
            rows.append(
                {
                    "hypothesis": "H2_semantic_minus_auditory",
                    "language": language,
                    "model": model,
                    "roi_family": "semantic_minus_auditory",
                    "mean_delta_mid": float(np.mean(h2_values)),
                    "se": _se(h2_values),
                    "dz": _dz(h2_values),
                    "p_perm": _sign_flip_pvalue(h2_values, n_permutations=n_permutations, rng=rng),
                    "p_holm_primary": np.nan,
                    "ci_low": low,
                    "ci_high": high,
                    "n_subjects": int(len(h2_values)),
                }
            )

    confirm_df = pd.DataFrame(rows).sort_values(["language", "model", "hypothesis"]).reset_index(drop=True)
    if not confirm_df.empty:
        confirm_df["p_holm_primary"] = _holm_adjust(confirm_df["p_perm"].astype(float).tolist())
    return confirm_df


def _report_lines(
    *,
    model_name: str,
    subject_df: pd.DataFrame,
    confirm_df: pd.DataFrame,
    languages: tuple[str, ...],
    layer_indices: tuple[int, ...],
    max_subjects: int | None,
) -> list[str]:
    z_df = subject_df.loc[subject_df["metric_name"] == "z"].copy()
    acoustic = (
        z_df.loc[z_df["condition"] == "acoustic_only", ["subject_id", "language", "roi_name", "value"]]
        .rename(columns={"value": "acoustic_only"})
    )
    merged = z_df.loc[z_df["condition"].isin(["raw", "shared", "mismatched_shared"])].merge(
        acoustic,
        on=["subject_id", "language", "roi_name"],
        how="left",
    )
    raw_rows = merged.loc[merged["condition"] == "raw"].copy()
    raw_gt_acoustic = bool((raw_rows["value"] > raw_rows["acoustic_only"]).any())
    shared_vs_mismatch = merged.pivot_table(
        index=["subject_id", "language", "roi_name", "layer_index"],
        columns="condition",
        values="value",
    )
    shared_gt_mismatched = bool((shared_vs_mismatch["shared"] > shared_vs_mismatch["mismatched_shared"]).any())

    lines = [
        "# Encoding QC Report",
        "",
        f"- model: `{model_name}`",
        f"- languages: `{', '.join(languages)}`",
        f"- max_subjects: `{max_subjects if max_subjects is not None else 'all available'}`",
        f"- layers: `{', '.join(str(layer) for layer in layer_indices)}`",
        f"- subject rows: `{len(subject_df)}`",
        f"- languages with results: `{', '.join(sorted(subject_df['language'].unique().tolist()))}`",
        "- motion regressors available in derivatives: `False`",
        "",
        "## Subject Counts",
        "",
    ]
    for row in (
        subject_df.groupby("language", as_index=False)["subject_id"].nunique().itertuples(index=False)
    ):
        lines.append(f"- `{row.language}` subjects: `{int(row.subject_id)}`")
    lines.extend(["", "## Confirmatory Effects", ""])
    if confirm_df.empty:
        lines.append("- no confirmatory rows were computed for the selected layer subset")
    else:
        lines.append(confirm_df.to_string(index=False))
    lines.extend(
        [
            "",
            "## Stage Checks",
            "",
            f"- `RAW > acoustic_only` somewhere: `{raw_gt_acoustic}`",
            f"- `SHARED > MISMATCHED_SHARED` somewhere: `{shared_gt_mismatched}`",
            "",
        ]
    )
    return lines


def _finalize_subject_df(
    *,
    subject_df: pd.DataFrame,
    model_name: str,
    languages: tuple[str, ...],
    layer_indices: tuple[int, ...],
    max_subjects: int | None,
    n_permutations: int,
    n_bootstraps: int,
    output_tag: str | None = None,
) -> ModelRoiPipelineSummary:
    subject_path = _subject_results_path(model_name, output_tag=output_tag)
    group_path = _group_results_path(model_name, output_tag=output_tag)
    confirm_path = _confirmatory_path(model_name, output_tag=output_tag)
    subject_path.parent.mkdir(parents=True, exist_ok=True)
    group_path.parent.mkdir(parents=True, exist_ok=True)
    confirm_path.parent.mkdir(parents=True, exist_ok=True)

    group_df = _group_results(subject_df)
    confirm_df = _confirmatory_effects(
        subject_df,
        n_permutations=n_permutations,
        n_bootstraps=n_bootstraps,
    )
    subject_df.to_parquet(subject_path, index=False)
    group_df.to_parquet(group_path, index=False)
    confirm_df.to_parquet(confirm_path, index=False)

    report_path = _report_path(model_name, output_tag=output_tag)
    write_text(
        report_path,
        "\n".join(
            _report_lines(
                model_name=model_name,
                subject_df=subject_df,
                confirm_df=confirm_df,
                languages=languages,
                layer_indices=layer_indices,
                max_subjects=max_subjects,
            )
        )
        + "\n",
    )

    append_markdown_log(
        project_root() / "outputs" / "logs" / "progress_log.md",
        f"{model_name} ROI pipeline",
        [
            f"languages={','.join(languages)}",
            f"max_subjects={max_subjects if max_subjects is not None else 'all'}",
            f"layers={','.join(str(layer) for layer in layer_indices)}",
            f"output_tag={output_tag if output_tag is not None else 'default'}",
            f"subject_results={subject_path.as_posix()}",
            f"group_results={group_path.as_posix()}",
            f"confirmatory={confirm_path.as_posix()}",
        ],
    )

    return ModelRoiPipelineSummary(
        subject_results_path=subject_path,
        group_results_path=group_path,
        confirmatory_path=confirm_path,
        report_path=report_path,
        model_name=model_name,
        languages=languages,
        n_subjects=int(subject_df["subject_id"].nunique()),
        n_rows=len(subject_df),
    )


def merge_subject_result_chunks(
    *,
    model_name: str,
    input_paths: tuple[Path, ...],
    n_permutations: int | None = None,
    n_bootstraps: int | None = None,
    output_tag: str | None = None,
) -> ModelRoiPipelineSummary:
    model_name = _normalize_model_name(model_name)
    bootstrap_logs()
    if n_permutations is None:
        n_permutations = int(_pipeline_value("statistics", "permutation_n", 10000))
    if n_bootstraps is None:
        n_bootstraps = int(_pipeline_value("statistics", "bootstrap_n", 10000))

    subject_frames = [pd.read_parquet(path).copy() for path in input_paths]
    subject_df = pd.concat(subject_frames, ignore_index=True)
    duplicate_cols = ["subject_id", "language", "model", "roi_name", "layer_index", "condition", "metric_name"]
    duplicate_mask = subject_df.duplicated(subset=duplicate_cols, keep=False)
    if duplicate_mask.any():
        raise RuntimeError(
            "Duplicate subject-level ROI rows detected while merging chunks. "
            f"Offending rows: {int(duplicate_mask.sum())}"
        )
    if subject_df["model"].nunique() != 1 or str(subject_df["model"].iloc[0]) != model_name:
        raise RuntimeError(
            f"Merged subject rows do not match model={model_name!r}. "
            f"Found models: {sorted(subject_df['model'].astype(str).unique().tolist())}"
        )

    subject_df = subject_df.sort_values(
        ["language", "subject_id", "roi_name", "layer_index", "condition", "metric_name"]
    ).reset_index(drop=True)
    layer_indices = tuple(
        sorted(subject_df.loc[subject_df["layer_index"].astype(int) >= 0, "layer_index"].astype(int).unique().tolist())
    )
    languages = tuple(sorted(subject_df["language"].astype(str).unique().tolist()))
    return _finalize_subject_df(
        subject_df=subject_df,
        model_name=model_name,
        languages=languages,
        layer_indices=layer_indices,
        max_subjects=None,
        n_permutations=n_permutations,
        n_bootstraps=n_bootstraps,
        output_tag=output_tag,
    )


def run_model_roi_pipeline(
    *,
    model_name: str,
    languages: tuple[str, ...] = ("en", "fr", "zh"),
    max_subjects: int | None = None,
    layer_indices: tuple[int, ...] | None = None,
    n_permutations: int | None = None,
    n_bootstraps: int | None = None,
    mismatch_shuffles: int | None = None,
    output_tag: str | None = None,
    subject_only: bool = False,
    resume: bool = False,
) -> ModelRoiPipelineSummary:
    bootstrap_logs()
    model_name = _normalize_model_name(model_name)
    available_layers = tuple(
        sorted(_load_feature_manifest(model_name, languages[0])["layer_index"].astype(int).unique().tolist())
    )
    if layer_indices is None:
        layer_indices = available_layers
    if n_permutations is None:
        n_permutations = int(_pipeline_value("statistics", "permutation_n", 10000))
    if n_bootstraps is None:
        n_bootstraps = int(_pipeline_value("statistics", "bootstrap_n", 10000))
    max_layer_index = max(available_layers)

    prepared_by_language = {
        language: _prepare_language_data(
            language,
            model_name=model_name,
            layer_indices=layer_indices,
            mismatch_shuffles=mismatch_shuffles,
        )
        for language in languages
    }
    n_shuffles = (
        int(mismatch_shuffles)
        if mismatch_shuffles is not None
        else int(_pipeline_value("design_matrix", "mismatched_shared_shuffles", 5))
    )

    existing_subject_df = pd.DataFrame()
    completed_subjects: set[tuple[str, str]] = set()
    subject_path = _subject_results_path(model_name, output_tag=output_tag) if subject_only else None
    metadata_path = _subject_chunk_metadata_path(model_name, output_tag=output_tag) if subject_only else None
    expected_metadata = _subject_chunk_metadata(
        model_name=model_name,
        languages=languages,
        layer_indices=layer_indices,
        mismatch_shuffles=n_shuffles,
        max_subjects=max_subjects,
    )
    if subject_only and resume:
        if subject_path is None or metadata_path is None:
            raise RuntimeError("Internal error: subject-only resume paths were not initialized.")
        if subject_path.exists():
            existing_subject_df = _load_existing_subject_chunk(
                subject_path=subject_path,
                metadata_path=metadata_path,
                expected_metadata=expected_metadata,
            )
            completed_subjects = {
                (str(language), str(subject_id))
                for language, subject_id in existing_subject_df.loc[:, ["language", "subject_id"]].itertuples(index=False)
            }
        elif metadata_path.exists():
            raise RuntimeError(
                f"Found resume metadata without subject results at {metadata_path}. "
                "Remove the stale metadata file or use a new output tag."
            )
    if subject_only and metadata_path is not None and not metadata_path.exists():
        _write_json_atomic(expected_metadata, metadata_path)

    results_rows: list[dict[str, Any]] = []
    for language in languages:
        prepared = prepared_by_language[language]
        roi_names = prepared.roi_lookup["roi_name"].astype(str).tolist()
        roi_indices = (prepared.roi_lookup["roi_index"].astype(int) - 1).tolist()
        roi_family_lookup = {
            str(row.roi_name): str(row.roi_family)
            for row in prepared.roi_lookup.itertuples(index=False)
        }
        subject_ids = sorted(prepared.roi_manifest["subject_id"].unique().tolist())
        if max_subjects is not None:
            subject_ids = subject_ids[:max_subjects]

        for subject_id in subject_ids:
            if subject_only and (language, str(subject_id)) in completed_subjects:
                continue

            subject_rows: list[dict[str, Any]] = []
            subject_runs = prepared.roi_manifest.loc[
                prepared.roi_manifest["subject_id"] == subject_id
            ].sort_values("canonical_run_index")
            roi_run_data = {
                int(row.canonical_run_index): np.load(row.filepath).astype(np.float32, copy=False)
                for row in subject_runs.itertuples(index=False)
            }
            y_by_run = {run_index: roi_run_data[run_index][:, roi_indices] for run_index in roi_run_data}

            acoustic_r, acoustic_z, acoustic_r2 = _run_single_condition(
                run_designs={run_index: prepared.acoustic_designs[run_index] for run_index in y_by_run},
                z_by_run={run_index: prepared.nuisance_arrays[run_index] for run_index in y_by_run},
                roi_series_by_run=y_by_run,
            )
            for roi_name, r_val, z_val, r2_val in zip(roi_names, acoustic_r, acoustic_z, acoustic_r2, strict=False):
                roi_family = roi_family_lookup[roi_name]
                for metric_name, value in (("r", r_val), ("z", z_val), ("r2", r2_val)):
                    subject_rows.append(
                        {
                            "subject_id": subject_id,
                            "language": language,
                            "model": model_name,
                            "roi_name": roi_name,
                            "roi_family": roi_family,
                            "layer_index": -1,
                            "layer_depth": np.nan,
                            "condition": "acoustic_only",
                            "metric_name": metric_name,
                            "value": float(value),
                        }
                    )

            for layer_index in layer_indices:
                layer_depth = layer_index / max(1, max_layer_index)
                for condition in TEXT_CONDITIONS:
                    r_vals, z_vals, r2_vals = _run_single_condition(
                        run_designs={
                            run_index: prepared.text_designs[(condition, layer_index)][run_index]
                            for run_index in y_by_run
                        },
                        z_by_run={run_index: prepared.text_nuisance_arrays[run_index] for run_index in y_by_run},
                        roi_series_by_run=y_by_run,
                    )
                    for roi_name, r_val, z_val, r2_val in zip(roi_names, r_vals, z_vals, r2_vals, strict=False):
                        roi_family = roi_family_lookup[roi_name]
                        for metric_name, value in (("r", r_val), ("z", z_val), ("r2", r2_val)):
                            subject_rows.append(
                                {
                                    "subject_id": subject_id,
                                    "language": language,
                                    "model": model_name,
                                    "roi_name": roi_name,
                                    "roi_family": roi_family,
                                    "layer_index": layer_index,
                                    "layer_depth": layer_depth,
                                    "condition": condition,
                                    "metric_name": metric_name,
                                    "value": float(value),
                                }
                            )

                mismatch_metrics = []
                for shuffle_index in range(n_shuffles):
                    mismatch_metrics.append(
                        _run_single_condition(
                            run_designs={
                                run_index: prepared.mismatched_designs[(layer_index, shuffle_index)][run_index]
                                for run_index in y_by_run
                            },
                            z_by_run={run_index: prepared.text_nuisance_arrays[run_index] for run_index in y_by_run},
                            roi_series_by_run=y_by_run,
                        )
                    )

                mismatch_r = np.tanh(
                    np.mean(
                        np.arctanh(
                            np.clip(np.vstack([metrics[0] for metrics in mismatch_metrics]), -0.999999, 0.999999)
                        ),
                        axis=0,
                    )
                )
                mismatch_z = np.mean(np.vstack([metrics[1] for metrics in mismatch_metrics]), axis=0)
                mismatch_r2 = np.mean(np.vstack([metrics[2] for metrics in mismatch_metrics]), axis=0)
                for roi_name, r_val, z_val, r2_val in zip(roi_names, mismatch_r, mismatch_z, mismatch_r2, strict=False):
                    roi_family = roi_family_lookup[roi_name]
                    for metric_name, value in (("r", r_val), ("z", z_val), ("r2", r2_val)):
                        subject_rows.append(
                            {
                                "subject_id": subject_id,
                                "language": language,
                                "model": model_name,
                                "roi_name": roi_name,
                                "roi_family": roi_family,
                                "layer_index": layer_index,
                                "layer_depth": layer_depth,
                                "condition": "mismatched_shared",
                                "metric_name": metric_name,
                                "value": float(value),
                            }
                        )

            results_rows.extend(subject_rows)

            if subject_only and subject_path is not None:
                new_subject_df = pd.DataFrame(subject_rows)
                if existing_subject_df.empty:
                    existing_subject_df = new_subject_df
                else:
                    existing_subject_df = pd.concat([existing_subject_df, new_subject_df], ignore_index=True)
                existing_subject_df = existing_subject_df.sort_values(
                    ["language", "subject_id", "roi_name", "layer_index", "condition", "metric_name"]
                ).reset_index(drop=True)
                _write_parquet_atomic(existing_subject_df, subject_path)
                completed_subjects.add((language, str(subject_id)))

    if subject_only and resume and not existing_subject_df.empty:
        subject_df = existing_subject_df.sort_values(
            ["language", "subject_id", "roi_name", "layer_index", "condition", "metric_name"]
        ).reset_index(drop=True)
    else:
        subject_df = pd.DataFrame(results_rows).sort_values(
        ["language", "subject_id", "roi_name", "layer_index", "condition", "metric_name"]
        ).reset_index(drop=True)

    if subject_only:
        if subject_path is None:
            raise RuntimeError("Internal error: missing subject path for subject-only run.")
        subject_path.parent.mkdir(parents=True, exist_ok=True)
        if not resume or not subject_path.exists():
            _write_parquet_atomic(subject_df, subject_path)
        append_markdown_log(
            project_root() / "outputs" / "logs" / "progress_log.md",
            f"{model_name} ROI subject chunk",
            [
                f"languages={','.join(languages)}",
                f"max_subjects={max_subjects if max_subjects is not None else 'all'}",
                f"layers={','.join(str(layer) for layer in layer_indices)}",
                f"mismatched_shuffles={n_shuffles}",
                f"output_tag={output_tag if output_tag is not None else 'default'}",
                f"resume={resume}",
                f"completed_subjects={int(subject_df['subject_id'].nunique())}",
                f"subject_results={subject_path.as_posix()}",
            ],
        )
        return ModelRoiPipelineSummary(
            subject_results_path=subject_path,
            group_results_path=None,
            confirmatory_path=None,
            report_path=None,
            model_name=model_name,
            languages=languages,
            n_subjects=int(subject_df["subject_id"].nunique()),
            n_rows=len(subject_df),
        )

    return _finalize_subject_df(
        subject_df=subject_df,
        model_name=model_name,
        languages=languages,
        layer_indices=layer_indices,
        max_subjects=max_subjects,
        n_permutations=n_permutations,
        n_bootstraps=n_bootstraps,
        output_tag=output_tag,
    )


def run_xlmr_roi_pipeline(
    *,
    languages: tuple[str, ...] = ("en", "fr", "zh"),
    max_subjects: int | None = None,
    layer_indices: tuple[int, ...] | None = None,
    n_permutations: int | None = None,
    n_bootstraps: int | None = None,
    mismatch_shuffles: int | None = None,
    output_tag: str | None = None,
    subject_only: bool = False,
    resume: bool = False,
) -> ModelRoiPipelineSummary:
    return run_model_roi_pipeline(
        model_name="xlmr",
        languages=languages,
        max_subjects=max_subjects,
        layer_indices=layer_indices,
        n_permutations=n_permutations,
        n_bootstraps=n_bootstraps,
        mismatch_shuffles=mismatch_shuffles,
        output_tag=output_tag,
        subject_only=subject_only,
        resume=resume,
    )


def run_nllb_roi_pipeline(
    *,
    languages: tuple[str, ...] = ("en", "fr", "zh"),
    max_subjects: int | None = None,
    layer_indices: tuple[int, ...] | None = None,
    n_permutations: int | None = None,
    n_bootstraps: int | None = None,
    mismatch_shuffles: int | None = None,
    output_tag: str | None = None,
    subject_only: bool = False,
    resume: bool = False,
) -> ModelRoiPipelineSummary:
    return run_model_roi_pipeline(
        model_name="nllb_encoder",
        languages=languages,
        max_subjects=max_subjects,
        layer_indices=layer_indices,
        n_permutations=n_permutations,
        n_bootstraps=n_bootstraps,
        mismatch_shuffles=mismatch_shuffles,
        output_tag=output_tag,
        subject_only=subject_only,
        resume=resume,
    )
