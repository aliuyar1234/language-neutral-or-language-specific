from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from brain_subspace_paper.config import project_root
from brain_subspace_paper.encoding.english_prototype import _random_seed
from brain_subspace_paper.encoding.xlmr_roi_pipeline import (
    _bootstrap_ci,
    _pipeline_value,
    _sign_flip_pvalue,
    _write_parquet_atomic,
)
from brain_subspace_paper.logging_utils import append_markdown_log, bootstrap_logs


LANGUAGE_ORDER = ("en", "fr", "zh")
MODEL_ORDER = ("xlmr", "nllb_encoder")
MODEL_DISPLAY = {
    "xlmr": "XLM-R-base",
    "nllb_encoder": "NLLB-200-distilled-600M encoder",
}
MODEL_METADATA = {
    "xlmr": {
        "hf_id": "FacebookAI/xlm-roberta-base",
        "architecture": "multilingual encoder",
        "main_role_in_paper": "strong multilingual encoder baseline",
        "used_in_core_paper": "yes",
    },
    "nllb_encoder": {
        "hf_id": "facebook/nllb-200-distilled-600M",
        "architecture": "seq2seq encoder",
        "main_role_in_paper": "explicitly multilingual shared-space model",
        "used_in_core_paper": "yes",
    },
}
TABLE_FILENAMES = {
    "table01": "table01_dataset_summary.csv",
    "table02": "table02_model_summary.csv",
    "table03": "table03_main_confirmatory_stats.csv",
    "table04": "table04_roi_condition_stats.csv",
}


@dataclass(slots=True)
class TableBuildSummary:
    table01_path: Path
    table02_path: Path
    table03_path: Path
    table04_path: Path
    roi_condition_stats_path: Path
    roi_family_panels_path: Path
    n_table01_rows: int
    n_table02_rows: int
    n_table03_rows: int
    n_table04_rows: int


def _stats_root() -> Path:
    return project_root() / "outputs" / "stats"


def _tables_root() -> Path:
    return project_root() / "outputs" / "tables"


def _canonical_input_paths() -> dict[str, Path]:
    root = _stats_root()
    return {
        "subject": root / "subject_level_roi_results.parquet",
        "group": root / "group_level_roi_results.parquet",
        "confirmatory": root / "confirmatory_effects.parquet",
        "geometry": root / "geometry_metrics.parquet",
    }


def _derived_output_paths() -> dict[str, Path]:
    root = _stats_root()
    return {
        "roi_condition_stats": root / "roi_condition_stats.parquet",
        "roi_family_panels": root / "roi_family_effect_panels.parquet",
    }


def _table_paths() -> dict[str, Path]:
    root = _tables_root()
    return {key: root / filename for key, filename in TABLE_FILENAMES.items()}


def _mid_layer_bounds() -> tuple[float, float]:
    return (
        float(_pipeline_value("statistics", "confirmatory_mid_layer_min_depth", 0.33)),
        float(_pipeline_value("statistics", "confirmatory_mid_layer_max_depth", 0.83)),
    )


def _bh_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running_min = 1.0
    n_tests = len(p_values)
    for offset, idx in enumerate(order[::-1], start=1):
        rank = n_tests - offset + 1
        corrected = min(float(p_values[idx]) * n_tests / rank, 1.0)
        running_min = min(running_min, corrected)
        adjusted[idx] = running_min
    return adjusted.tolist()


def _ordered_frame(df: pd.DataFrame, *, columns: list[str]) -> pd.DataFrame:
    return df.loc[:, columns].copy()


def _sort_language_model(df: pd.DataFrame) -> pd.DataFrame:
    language_dtype = pd.CategoricalDtype(categories=list(LANGUAGE_ORDER), ordered=True)
    model_dtype = pd.CategoricalDtype(categories=list(MODEL_ORDER), ordered=True)
    out = df.copy()
    if "language" in out.columns:
        out["language"] = out["language"].astype(language_dtype)
    if "model" in out.columns:
        out["model"] = out["model"].astype(model_dtype)
    sort_cols = [column for column in ("model", "language", "roi_family", "roi_name", "hypothesis") if column in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    if "language" in out.columns:
        out["language"] = out["language"].astype(str)
    if "model" in out.columns:
        out["model"] = out["model"].astype(str)
    return out


def _included_subjects(subject_df: pd.DataFrame) -> pd.DataFrame:
    return subject_df.loc[:, ["language", "subject_id"]].drop_duplicates().reset_index(drop=True)


def _build_table01(subject_df: pd.DataFrame) -> pd.DataFrame:
    run_manifest = pd.read_parquet(project_root() / "data" / "interim" / "lppc_run_manifest.parquet").copy()
    included = _included_subjects(subject_df)
    run_manifest = run_manifest.merge(included, on=["language", "subject_id"], how="inner")
    tr_sec = float(_pipeline_value("design_matrix", "tr_sec", 2.0))

    per_subject = (
        run_manifest.groupby(["language", "subject_id"], as_index=False)
        .agg(
            runs=("canonical_run_index", "nunique"),
            duration_minutes=("n_volumes", lambda series: float(series.sum()) * tr_sec / 60.0),
        )
    )
    table = (
        per_subject.groupby("language", as_index=False)
        .agg(
            subjects=("subject_id", "nunique"),
            runs=("runs", "median"),
            duration=("duration_minutes", "median"),
        )
    )
    table["dataset"] = "LPPC"
    table["TR"] = tr_sec
    table["preprocessed derivatives available?"] = True
    table["annotations available?"] = True
    table["free download?"] = True
    table["runs"] = table["runs"].round().astype(int)
    table["duration"] = table["duration"].round(2)
    table = table[[
        "dataset",
        "language",
        "subjects",
        "runs",
        "duration",
        "TR",
        "preprocessed derivatives available?",
        "annotations available?",
        "free download?",
    ]]
    return _sort_language_model(table)


def _build_table02() -> pd.DataFrame:
    embedding_manifest = pd.read_parquet(project_root() / "data" / "interim" / "embeddings" / "embedding_manifest.parquet")
    summary = (
        embedding_manifest.groupby("model", as_index=False)
        .agg(layers=("layer_index", "nunique"), hidden_size=("hidden_size", "max"))
    )
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        row = summary.loc[summary["model"] == model]
        if row.empty:
            continue
        metadata = MODEL_METADATA[model]
        rows.append(
            {
                "model": MODEL_DISPLAY[model],
                "hf_id": metadata["hf_id"],
                "architecture": metadata["architecture"],
                "layers": int(row.iloc[0]["layers"]),
                "hidden_size": int(row.iloc[0]["hidden_size"]),
                "main role in paper": metadata["main_role_in_paper"],
                "used in core paper?": metadata["used_in_core_paper"],
            }
        )
    return pd.DataFrame(rows)


def _build_table03(confirmatory_df: pd.DataFrame) -> pd.DataFrame:
    table = confirmatory_df.copy()
    return _ordered_frame(
        _sort_language_model(table),
        columns=[
            "model",
            "language",
            "hypothesis",
            "roi_family",
            "mean_delta_mid",
            "se",
            "dz",
            "p_perm",
            "p_holm_primary",
            "ci_low",
            "ci_high",
            "n_subjects",
        ],
    )


def _mid_layer_condition_means(subject_df: pd.DataFrame) -> pd.DataFrame:
    lower, upper = _mid_layer_bounds()
    subset = subject_df.loc[
        (subject_df["metric_name"] == "z")
        & (subject_df["condition"].isin(["raw", "shared", "specific", "mismatched_shared"]))
        & (subject_df["layer_depth"] >= lower)
        & (subject_df["layer_depth"] <= upper)
    ].copy()
    if subset.empty:
        raise RuntimeError("No mid-layer z rows available for Table 4 derivation.")
    return (
        subset.groupby(["subject_id", "language", "model", "roi_name", "roi_family", "condition"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "mean_z_mid"})
    )


def _build_table04(
    subject_df: pd.DataFrame,
    *,
    n_permutations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_means = _mid_layer_condition_means(subject_df)
    pivot = (
        subject_means.pivot_table(
            index=["subject_id", "language", "model", "roi_name", "roi_family"],
            columns="condition",
            values="mean_z_mid",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    pivot["delta_shared_specific"] = pivot["shared"] - pivot["specific"]

    rng = np.random.default_rng(_random_seed())
    rows: list[dict[str, object]] = []
    for (model, language, roi_name, roi_family), group in pivot.groupby(
        ["model", "language", "roi_name", "roi_family"],
        sort=True,
    ):
        delta = group["delta_shared_specific"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "model": model,
                "language": language,
                "roi_name": roi_name,
                "roi_family": roi_family,
                "mean_z_raw": float(group["raw"].mean()),
                "mean_z_shared": float(group["shared"].mean()),
                "mean_z_specific": float(group["specific"].mean()),
                "mean_z_mismatched": float(group["mismatched_shared"].mean()),
                "delta_shared_specific": float(delta.mean()),
                "p_perm": _sign_flip_pvalue(delta, n_permutations=n_permutations, rng=rng),
                "n_subjects": int(group["subject_id"].nunique()),
            }
        )

    table = pd.DataFrame(rows)
    adjusted: list[float] = []
    for (_, _), group in table.groupby(["model", "language"], sort=True):
        adjusted.extend(_bh_adjust(group["p_perm"].astype(float).tolist()))
    table["q_fdr"] = adjusted
    table = _ordered_frame(
        _sort_language_model(table),
        columns=[
            "model",
            "language",
            "roi_name",
            "roi_family",
            "mean_z_raw",
            "mean_z_shared",
            "mean_z_specific",
            "mean_z_mismatched",
            "delta_shared_specific",
            "p_perm",
            "q_fdr",
            "n_subjects",
        ],
    )
    return table, pivot


def _build_roi_family_panels(
    subject_df: pd.DataFrame,
    confirmatory_df: pd.DataFrame,
    *,
    n_bootstraps: int,
    n_permutations: int,
) -> pd.DataFrame:
    lower, upper = _mid_layer_bounds()
    z_df = subject_df.loc[
        (subject_df["metric_name"] == "z")
        & (subject_df["condition"].isin(["shared", "specific"]))
        & (subject_df["layer_depth"] >= lower)
        & (subject_df["layer_depth"] <= upper)
    ].copy()
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

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(_random_seed())
    for roi_family in ("semantic", "auditory"):
        family_df = family_effects.loc[family_effects["roi_family"] == roi_family].copy()
        for (model, language), group in family_df.groupby(["model", "language"], sort=True):
            values = group["delta_mid"].to_numpy(dtype=np.float64)
            low, high = _bootstrap_ci(values, n_bootstraps=n_bootstraps, rng=rng)
            rows.append(
                {
                    "panel": roi_family,
                    "model": model,
                    "language": language,
                    "roi_family": roi_family,
                    "mean_delta_mid": float(values.mean()),
                    "se": float(0.0 if len(values) <= 1 else values.std(ddof=1) / np.sqrt(len(values))),
                    "ci_low": low,
                    "ci_high": high,
                    "p_perm": _sign_flip_pvalue(values, n_permutations=n_permutations, rng=rng),
                    "n_subjects": int(len(values)),
                }
            )

    h2_rows = confirmatory_df.loc[confirmatory_df["hypothesis"] == "H2_semantic_minus_auditory"].copy()
    if not h2_rows.empty:
        h2_rows["panel"] = "semantic_minus_auditory"
        rows.extend(
            h2_rows.loc[
                :,
                [
                    "panel",
                    "model",
                    "language",
                    "roi_family",
                    "mean_delta_mid",
                    "se",
                    "ci_low",
                    "ci_high",
                    "p_perm",
                    "n_subjects",
                ],
            ].to_dict(orient="records")
        )

    return _ordered_frame(
        _sort_language_model(pd.DataFrame(rows)),
        columns=[
            "panel",
            "model",
            "language",
            "roi_family",
            "mean_delta_mid",
            "se",
            "ci_low",
            "ci_high",
            "p_perm",
            "n_subjects",
        ],
    )


def _append_table_provenance(summary: TableBuildSummary) -> None:
    path = project_root() / "outputs" / "manuscript" / "table_provenance.md"
    append_markdown_log(
        path,
        "Generated canonical paper tables",
        [
            "generating_script=src/brain_subspace_paper/viz/tables.py",
            f"table01={summary.table01_path.relative_to(project_root()).as_posix()}",
            f"table02={summary.table02_path.relative_to(project_root()).as_posix()}",
            f"table03={summary.table03_path.relative_to(project_root()).as_posix()}",
            f"table04={summary.table04_path.relative_to(project_root()).as_posix()}",
            f"derived_roi_condition_stats={summary.roi_condition_stats_path.relative_to(project_root()).as_posix()}",
            f"derived_roi_family_panels={summary.roi_family_panels_path.relative_to(project_root()).as_posix()}",
            "source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/confirmatory_effects.parquet, outputs/stats/geometry_metrics.parquet, data/interim/lppc_run_manifest.parquet, data/interim/embeddings/embedding_manifest.parquet",
        ],
    )


def build_paper_tables(
    *,
    n_permutations: int | None = None,
    n_bootstraps: int | None = None,
) -> TableBuildSummary:
    bootstrap_logs()
    if n_permutations is None:
        n_permutations = int(_pipeline_value("statistics", "permutation_n", 10000))
    if n_bootstraps is None:
        n_bootstraps = int(_pipeline_value("statistics", "bootstrap_n", 10000))

    input_paths = _canonical_input_paths()
    missing = [path for path in input_paths.values() if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing canonical stats required for table generation: {missing_text}. "
            "Run `python -m brain_subspace_paper build-paper-stats` first."
        )

    subject_df = pd.read_parquet(input_paths["subject"]).copy()
    confirmatory_df = pd.read_parquet(input_paths["confirmatory"]).copy()

    table01 = _build_table01(subject_df)
    table02 = _build_table02()
    table03 = _build_table03(confirmatory_df)
    table04, _ = _build_table04(subject_df, n_permutations=n_permutations)
    roi_family_panels = _build_roi_family_panels(
        subject_df,
        confirmatory_df,
        n_bootstraps=n_bootstraps,
        n_permutations=n_permutations,
    )

    table_paths = _table_paths()
    derived_paths = _derived_output_paths()
    for path in (*table_paths.values(), *derived_paths.values()):
        path.parent.mkdir(parents=True, exist_ok=True)

    table01.to_csv(table_paths["table01"], index=False)
    table02.to_csv(table_paths["table02"], index=False)
    table03.to_csv(table_paths["table03"], index=False)
    table04.to_csv(table_paths["table04"], index=False)
    _write_parquet_atomic(table04, derived_paths["roi_condition_stats"])
    _write_parquet_atomic(roi_family_panels, derived_paths["roi_family_panels"])

    summary = TableBuildSummary(
        table01_path=table_paths["table01"],
        table02_path=table_paths["table02"],
        table03_path=table_paths["table03"],
        table04_path=table_paths["table04"],
        roi_condition_stats_path=derived_paths["roi_condition_stats"],
        roi_family_panels_path=derived_paths["roi_family_panels"],
        n_table01_rows=len(table01),
        n_table02_rows=len(table02),
        n_table03_rows=len(table03),
        n_table04_rows=len(table04),
    )

    append_markdown_log(
        project_root() / "outputs" / "logs" / "progress_log.md",
        "paper tables",
        [
            f"table01={summary.table01_path.as_posix()}",
            f"table02={summary.table02_path.as_posix()}",
            f"table03={summary.table03_path.as_posix()}",
            f"table04={summary.table04_path.as_posix()}",
            f"roi_condition_stats={summary.roi_condition_stats_path.as_posix()}",
            f"roi_family_panels={summary.roi_family_panels_path.as_posix()}",
            f"rows_table01={summary.n_table01_rows}",
            f"rows_table02={summary.n_table02_rows}",
            f"rows_table03={summary.n_table03_rows}",
            f"rows_table04={summary.n_table04_rows}",
            f"permutations={n_permutations}",
            f"bootstraps={n_bootstraps}",
        ],
    )
    _append_table_provenance(summary)
    return summary
