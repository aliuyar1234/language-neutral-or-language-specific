from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from matplotlib import cm
from matplotlib import colors
import nibabel as nib
from matplotlib import patches
import matplotlib.pyplot as plt
from nilearn.plotting import plot_glass_brain
import numpy as np
import pandas as pd

from brain_subspace_paper.config import project_root
from brain_subspace_paper.logging_utils import append_markdown_log, bootstrap_logs
from brain_subspace_paper.stats.paper_level import _brain_delta_by_layer
from brain_subspace_paper.viz.tables import build_paper_tables


LANGUAGE_ORDER = ("en", "fr", "zh")
LANGUAGE_LABELS = {"en": "English", "fr": "French", "zh": "Chinese"}
LANGUAGE_MARKERS = {"en": "o", "fr": "s", "zh": "D"}
MODEL_ORDER = ("xlmr", "nllb_encoder")
MODEL_LABELS = {"xlmr": "XLM-R", "nllb_encoder": "NLLB encoder"}
MODEL_COLORS = {"xlmr": "#2f5d8a", "nllb_encoder": "#d1495b"}
CONDITION_ORDER = ("raw", "shared", "specific", "mismatched_shared")
CONDITION_LABELS = {
    "raw": "RAW",
    "shared": "SHARED",
    "specific": "SPECIFIC",
    "mismatched_shared": "MISMATCHED_SHARED",
}
CONDITION_COLORS = {
    "raw": "#1f9d8b",
    "shared": "#2364aa",
    "specific": "#f4a259",
    "mismatched_shared": "#7f8c8d",
}
FIGURE_FILENAMES = {
    "fig01": "fig01_pipeline.png",
    "fig02": "fig02_dataset_alignment_overview.png",
    "fig03": "fig03_text_geometry_by_layer.png",
    "fig04": "fig04_main_confirmatory_roi_families.png",
    "fig05": "fig05_roi_condition_comparison.png",
    "fig06": "fig06_layer_curves_key_rois.png",
    "fig07": "fig07_geometry_to_brain_coupling.png",
    "fig08": "fig08_whole_brain_maps.png",
}
FIGURE05_ROIS = ("L_pMTG", "R_pMTG", "L_AG", "L_Heschl", "L_pSTG", "L_IFGtri")
FIGURE06_ROIS = ("L_pMTG", "L_pSTG")


@dataclass(slots=True)
class FigureBuildSummary:
    figure_paths: dict[str, Path]
    coupling_points_path: Path
    fig08_note: str


def _figures_root() -> Path:
    return project_root() / "outputs" / "figures"


def _stats_root() -> Path:
    return project_root() / "outputs" / "stats"


def _figure_paths() -> dict[str, Path]:
    root = _figures_root()
    return {key: root / filename for key, filename in FIGURE_FILENAMES.items()}


def _canonical_stats_paths() -> dict[str, Path]:
    root = _stats_root()
    return {
        "subject": root / "subject_level_roi_results.parquet",
        "group": root / "group_level_roi_results.parquet",
        "geometry": root / "geometry_metrics.parquet",
        "coupling": root / "geometry_brain_coupling.parquet",
        "roi_condition_stats": root / "roi_condition_stats.parquet",
        "roi_family_panels": root / "roi_family_effect_panels.parquet",
        "coupling_points": root / "geometry_brain_coupling_points.parquet",
        "roi_projected_root": root / "roi_projected_maps",
    }


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.facecolor": "#fbfbfb",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(fig: plt.Figure, path: Path, *, dpi: int, tight_layout: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _load_required_frames() -> dict[str, pd.DataFrame]:
    paths = _canonical_stats_paths()
    required = ("subject", "group", "geometry", "coupling", "roi_condition_stats", "roi_family_panels")
    missing = [paths[name] for name in required if not paths[name].exists()]
    if missing:
        build_paper_tables()
    missing = [paths[name] for name in required if not paths[name].exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing figure input stats after table build: {missing_text}")
    return {
        "subject": pd.read_parquet(paths["subject"]).copy(),
        "group": pd.read_parquet(paths["group"]).copy(),
        "geometry": pd.read_parquet(paths["geometry"]).copy(),
        "coupling": pd.read_parquet(paths["coupling"]).copy(),
        "roi_condition_stats": pd.read_parquet(paths["roi_condition_stats"]).copy(),
        "roi_family_panels": pd.read_parquet(paths["roi_family_panels"]).copy(),
    }


def _roi_atlas_path() -> Path:
    return project_root() / "data" / "interim" / "roi" / "harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz"


def _roi_metadata_path() -> Path:
    return project_root() / "data" / "interim" / "roi" / "roi_metadata.parquet"


def _roi_projected_root() -> Path:
    return _canonical_stats_paths()["roi_projected_root"]


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
    rows = []
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


def _roi_projected_artifact_root(model: str, language: str, layer_index: int) -> Path:
    return _roi_projected_root() / model / language / f"layer_{layer_index:02d}"


def _roi_projected_map_path(root: Path, stem: str) -> Path:
    return root / f"{stem}.nii.gz"


def _write_roi_projected_maps(group_df: pd.DataFrame) -> list[Path]:
    atlas_img = nib.load(str(_roi_atlas_path()))
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int16, copy=False)
    roi_metadata = pd.read_parquet(_roi_metadata_path()).copy()
    representative = _representative_layers(group_df)
    artifact_roots: list[Path] = []

    for row in representative.itertuples(index=False):
        model = str(row.model)
        layer_index = int(row.layer_index)
        layer_depth = float(row.layer_depth)
        for language in LANGUAGE_ORDER:
            subset = group_df.loc[
                (group_df["model"] == model)
                & (group_df["language"] == language)
                & (group_df["layer_index"].astype(int) == layer_index)
                & (group_df["condition"].isin(["shared", "specific"]))
            ].copy()
            if subset.empty:
                continue
            pivot = subset.pivot_table(index="roi_name", columns="condition", values="mean_z").reset_index()
            merged = roi_metadata.merge(pivot, on="roi_name", how="left")
            if merged["shared"].isna().any() or merged["specific"].isna().any():
                missing = merged.loc[merged["shared"].isna() | merged["specific"].isna(), "roi_name"].tolist()
                raise RuntimeError(
                    f"Missing ROI-projected values for model={model}, language={language}, layer={layer_index}: {missing}"
                )

            shared_map = np.zeros_like(atlas_data, dtype=np.float32)
            specific_map = np.zeros_like(atlas_data, dtype=np.float32)
            delta_map = np.zeros_like(atlas_data, dtype=np.float32)
            for roi_row in merged.itertuples(index=False):
                mask = atlas_data == int(roi_row.roi_index)
                shared_value = float(roi_row.shared)
                specific_value = float(roi_row.specific)
                shared_map[mask] = shared_value
                specific_map[mask] = specific_value
                delta_map[mask] = shared_value - specific_value

            root = _roi_projected_artifact_root(model, language, layer_index)
            root.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(shared_map, atlas_img.affine, atlas_img.header), str(_roi_projected_map_path(root, "shared_mean_z")))
            nib.save(nib.Nifti1Image(specific_map, atlas_img.affine, atlas_img.header), str(_roi_projected_map_path(root, "specific_mean_z")))
            nib.save(
                nib.Nifti1Image(delta_map, atlas_img.affine, atlas_img.header),
                str(_roi_projected_map_path(root, "shared_minus_specific_mean_z")),
            )
            nib.save(
                nib.Nifti1Image((atlas_data > 0).astype(np.float32), atlas_img.affine, atlas_img.header),
                str(_roi_projected_map_path(root, "brain_mask")),
            )
            manifest = {
                "analysis_type": "roi_projected_descriptive",
                "model": model,
                "language": language,
                "selected_layer": layer_index,
                "layer_depth": layer_depth,
                "source_stats_file": "outputs/stats/group_level_roi_results.parquet",
                "source_atlas_file": "data/interim/roi/harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz",
                "note": "ROI means projected onto atlas parcels for descriptive visualization; not voxelwise encoding maps.",
            }
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            artifact_roots.append(root)

    return artifact_roots


def _figure08(path: Path, artifact_roots: list[Path], *, dpi: int) -> None:
    ordered_roots = sorted(
        artifact_roots,
        key=lambda artifact_root: (
            MODEL_ORDER.index(artifact_root.parts[-3]),
            LANGUAGE_ORDER.index(artifact_root.parts[-2]),
        ),
    )
    positive_max = 0.0
    delta_max = 0.0
    for root in ordered_roots:
        shared = np.asarray(nib.load(str(_roi_projected_map_path(root, "shared_mean_z"))).dataobj, dtype=np.float32)
        specific = np.asarray(nib.load(str(_roi_projected_map_path(root, "specific_mean_z"))).dataobj, dtype=np.float32)
        delta = np.asarray(nib.load(str(_roi_projected_map_path(root, "shared_minus_specific_mean_z"))).dataobj, dtype=np.float32)
        positive_max = max(positive_max, float(np.nanmax([shared.max(), specific.max()])))
        delta_max = max(delta_max, float(np.nanmax(np.abs(delta))))
    positive_max = max(positive_max, 0.05)
    delta_max = max(delta_max, 0.03)

    fig, axes = plt.subplots(len(ordered_roots), 3, figsize=(12.6, 2.32 * len(ordered_roots) + 1.2))
    if len(ordered_roots) == 1:
        axes = np.asarray([axes])
    for row_index, root in enumerate(ordered_roots):
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        model = str(manifest["model"])
        language = str(manifest["language"])
        layer_index = int(manifest["selected_layer"])
        label_color = MODEL_COLORS[model]
        for col_index, (stem, title, cmap_name, vmax, symmetric) in enumerate(
            [
                ("shared_mean_z", "SHARED ROI-projected", "YlOrRd", positive_max, False),
                ("specific_mean_z", "SPECIFIC ROI-projected", "YlOrRd", positive_max, False),
                ("shared_minus_specific_mean_z", "SHARED - SPECIFIC", "RdBu_r", delta_max, True),
            ]
        ):
            axis = axes[row_index, col_index]
            axis.set_axis_off()
            plot_glass_brain(
                str(_roi_projected_map_path(root, stem)),
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
                axis.set_title(title, fontsize=11, pad=10, fontweight="bold")
        axes[row_index, 0].text(
            -0.28,
            0.5,
            f"{MODEL_LABELS[model]}\n{LANGUAGE_LABELS[language]}\nlayer {layer_index:02d}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=label_color,
        )
        if row_index < len(ordered_roots) - 1:
            for col_index in range(3):
                axes[row_index, col_index].axhline(
                    -0.08,
                    color="#e5e7eb",
                    linewidth=0.8,
                    clip_on=False,
                    xmin=0.04,
                    xmax=0.96,
                )

    positive_norm = colors.Normalize(vmin=0.0, vmax=positive_max)
    delta_norm = colors.TwoSlopeNorm(vmin=-delta_max, vcenter=0.0, vmax=delta_max)
    cax_shared = fig.add_axes([0.17, 0.045, 0.23, 0.018])
    cax_specific = fig.add_axes([0.43, 0.045, 0.23, 0.018])
    cax_delta = fig.add_axes([0.69, 0.045, 0.23, 0.018])
    shared_cb = fig.colorbar(cm.ScalarMappable(norm=positive_norm, cmap="YlOrRd"), cax=cax_shared, orientation="horizontal")
    specific_cb = fig.colorbar(cm.ScalarMappable(norm=positive_norm, cmap="YlOrRd"), cax=cax_specific, orientation="horizontal")
    delta_cb = fig.colorbar(cm.ScalarMappable(norm=delta_norm, cmap="RdBu_r"), cax=cax_delta, orientation="horizontal")
    for colorbar, label in (
        (shared_cb, "Mean held-out Fisher-z"),
        (specific_cb, "Mean held-out Fisher-z"),
        (delta_cb, "Delta (SHARED - SPECIFIC)"),
    ):
        colorbar.outline.set_linewidth(0.6)
        colorbar.ax.tick_params(labelsize=8, length=2)
        colorbar.set_label(label, fontsize=8, labelpad=2)

    fig.suptitle("ROI-projected descriptive cortex maps at representative layers", fontsize=14, fontweight="bold", y=0.982)
    fig.text(
        0.5,
        0.958,
        "Atlas parcels are filled with ROI group means for visual context; these are descriptive projections, not voxelwise encoding maps.",
        ha="center",
        va="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.91, bottom=0.1, wspace=0.08, hspace=0.18)
    _save_figure(fig, path, dpi=dpi, tight_layout=False)


def _draw_box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=color,
        facecolor=color,
        alpha=0.15,
    )
    ax.add_patch(box)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=11, color="#222222")


def _draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = patches.FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#555555")
    ax.add_patch(arrow)


def _figure01(path: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    _draw_box(ax, 0.02, 0.63, 0.16, 0.2, "Parallel sentence spans\nEN / FR / ZH", "#3b7ea1")
    _draw_box(ax, 0.22, 0.63, 0.16, 0.2, "Multilingual model\nhidden states", "#2a9d8f")
    _draw_box(ax, 0.42, 0.63, 0.18, 0.2, "Factorization\nRAW / SHARED /\nSPECIFIC / FULL /\nMISMATCHED_SHARED", "#e76f51")
    _draw_box(ax, 0.64, 0.63, 0.15, 0.2, "Target-language\ntimeline placement", "#f4a261")
    _draw_box(ax, 0.82, 0.63, 0.14, 0.2, "HRF convolution\n+ ridge encoding", "#7a8f5f")
    _draw_box(ax, 0.68, 0.18, 0.16, 0.18, "ROI-family effects\nTables / Figures", "#6c5b7b")
    _draw_box(ax, 0.86, 0.18, 0.12, 0.18, "Whole-brain maps\nsecondary", "#8d99ae")

    _draw_arrow(ax, (0.18, 0.73), (0.22, 0.73))
    _draw_arrow(ax, (0.38, 0.73), (0.42, 0.73))
    _draw_arrow(ax, (0.60, 0.73), (0.64, 0.73))
    _draw_arrow(ax, (0.79, 0.73), (0.82, 0.73))
    _draw_arrow(ax, (0.89, 0.63), (0.76, 0.36))
    _draw_arrow(ax, (0.93, 0.63), (0.92, 0.36))

    ax.text(0.02, 0.94, "Conceptual pipeline", fontsize=14, fontweight="bold", ha="left", va="center")
    ax.text(
        0.02,
        0.08,
        "LPPC-only multilingual naturalistic fMRI pipeline: factorized text representations are placed on each target-language timeline, convolved, and tested with ROI-first encoding.",
        fontsize=10,
        ha="left",
        va="center",
        color="#444444",
    )
    _save_figure(fig, path, dpi=dpi)


def _figure02(path: Path, subject_df: pd.DataFrame, *, dpi: int) -> None:
    included = subject_df.loc[:, ["language", "subject_id"]].drop_duplicates()
    run_manifest = pd.read_parquet(project_root() / "data" / "interim" / "lppc_run_manifest.parquet").merge(
        included,
        on=["language", "subject_id"],
        how="inner",
    )
    triplets = pd.read_parquet(project_root() / "data" / "processed" / "alignment_triplets.parquet")
    triplets_qc = pd.read_parquet(project_root() / "data" / "processed" / "alignment_triplets_qc.parquet")
    sentence_spans = {
        language: pd.read_parquet(project_root() / "data" / "interim" / f"sentence_spans_{language}.parquet")
        for language in LANGUAGE_ORDER
    }

    participant_counts = included.groupby("language")["subject_id"].nunique().reindex(LANGUAGE_ORDER)
    run_structure = (
        run_manifest.groupby(["language", "canonical_run_index"], as_index=False)["n_volumes"].median().rename(
            columns={"n_volumes": "median_n_volumes"}
        )
    )
    span_counts = [len(sentence_spans[language]) for language in LANGUAGE_ORDER]
    merge_counts = triplets["merge_pattern"].value_counts().head(8)
    manual_status = triplets_qc["manual_status"].fillna("").replace("", "unreviewed")
    qc_counts = manual_status.value_counts().reindex(["approved", "unreviewed", "needs_fix"], fill_value=0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    axes[0].bar(
        [LANGUAGE_LABELS[language] for language in LANGUAGE_ORDER],
        participant_counts.tolist(),
        color=["#5b8e7d", "#3b7ea1", "#d77a61"],
    )
    axes[0].set_title("Included participants by language")
    axes[0].set_ylabel("Subjects")

    for language in LANGUAGE_ORDER:
        subset = run_structure.loc[run_structure["language"] == language]
        axes[1].plot(
            subset["canonical_run_index"],
            subset["median_n_volumes"],
            marker=LANGUAGE_MARKERS[language],
            linewidth=2,
            label=LANGUAGE_LABELS[language],
        )
    axes[1].set_title("Canonical run structure")
    axes[1].set_xlabel("Canonical run index")
    axes[1].set_ylabel("Median volumes")
    axes[1].legend(frameon=False, fontsize=8)

    count_labels = ["EN spans", "FR spans", "ZH spans", "Triplets"]
    count_values = span_counts + [int(len(triplets))]
    axes[2].bar(count_labels, count_values, color=["#3b7ea1", "#6c5b7b", "#2a9d8f", "#e76f51"])
    axes[2].set_title("Sentence spans and aligned triplets")
    axes[2].tick_params(axis="x", rotation=25)

    axes[3].bar(merge_counts.index.tolist(), merge_counts.values.tolist(), color="#5b8e7d")
    axes[3].set_title("Top merge-pattern frequencies")
    axes[3].tick_params(axis="x", rotation=35)
    axes[3].set_ylabel("Triplets")

    axes[4].bar(["approved", "unreviewed", "needs_fix"], qc_counts.tolist(), color=["#2a9d8f", "#8d99ae", "#d1495b"])
    axes[4].set_title("Alignment QC summary")
    axes[4].set_ylabel("Triplets")

    bins = np.linspace(0.0, 15.0, 31)
    for language in LANGUAGE_ORDER:
        axes[5].hist(
            sentence_spans[language]["duration_sec"],
            bins=bins,
            histtype="step",
            linewidth=2,
            label=LANGUAGE_LABELS[language],
        )
    axes[5].set_title("Sentence duration distribution")
    axes[5].set_xlabel("Duration (s)")
    axes[5].set_ylabel("Count")
    axes[5].legend(frameon=False, fontsize=8)

    fig.suptitle("Dataset and alignment overview", fontsize=14, fontweight="bold", y=1.02)
    _save_figure(fig, path, dpi=dpi)


def _figure03(path: Path, geometry_df: pd.DataFrame, *, dpi: int) -> None:
    metrics = [
        ("align_mean", "Same-sentence cosine"),
        ("cas", "CAS"),
        ("retrieval_r1_mean", "Retrieval R@1"),
        ("specificity_energy", "Specificity energy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    for axis, (column, label) in zip(axes, metrics, strict=True):
        for model in MODEL_ORDER:
            subset = geometry_df.loc[geometry_df["model"] == model]
            axis.plot(
                subset["layer_depth"],
                subset[column],
                marker="o",
                linewidth=2,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        axis.set_title(label)
        axis.set_xlabel("Normalized layer depth")
        axis.set_ylabel(label)
    axes[0].legend(frameon=False)
    fig.suptitle("Text-side multilingual geometry by layer", fontsize=14, fontweight="bold", y=1.02)
    _save_figure(fig, path, dpi=dpi)


def _figure04(path: Path, roi_family_panels: pd.DataFrame, *, dpi: int) -> None:
    panel_titles = {
        "semantic": "Semantic family Delta^mid",
        "auditory": "Auditory family Delta^mid",
        "semantic_minus_auditory": "Semantic minus auditory",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    language_positions = np.arange(len(LANGUAGE_ORDER))
    for axis, panel in zip(axes, ("semantic", "auditory", "semantic_minus_auditory"), strict=True):
        subset = roi_family_panels.loc[roi_family_panels["panel"] == panel]
        for model_offset, model in enumerate(MODEL_ORDER):
            model_subset = subset.loc[subset["model"] == model].set_index("language").reindex(LANGUAGE_ORDER).reset_index()
            x = language_positions + (-0.12 if model_offset == 0 else 0.12)
            y = model_subset["mean_delta_mid"].to_numpy(dtype=float)
            low = y - model_subset["ci_low"].to_numpy(dtype=float)
            high = model_subset["ci_high"].to_numpy(dtype=float) - y
            axis.errorbar(
                x,
                y,
                yerr=np.vstack([low, high]),
                fmt="o",
                capsize=4,
                linewidth=2,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        axis.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
        axis.set_xticks(language_positions, [LANGUAGE_LABELS[language] for language in LANGUAGE_ORDER])
        axis.set_title(panel_titles[panel])
        axis.set_ylabel("Mean Delta^mid")
    axes[0].legend(frameon=False)
    fig.suptitle("Main confirmatory ROI-family effects", fontsize=14, fontweight="bold", y=1.04)
    _save_figure(fig, path, dpi=dpi)


def _figure05(path: Path, roi_condition_stats: pd.DataFrame, *, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    axes = axes.ravel()
    x_positions = np.arange(len(CONDITION_ORDER))
    for axis, roi_name in zip(axes, FIGURE05_ROIS, strict=True):
        subset = roi_condition_stats.loc[roi_condition_stats["roi_name"] == roi_name].copy()
        for model in MODEL_ORDER:
            for language in LANGUAGE_ORDER:
                row = subset.loc[(subset["model"] == model) & (subset["language"] == language)]
                if row.empty:
                    continue
                row0 = row.iloc[0]
                values = [
                    float(row0["mean_z_raw"]),
                    float(row0["mean_z_shared"]),
                    float(row0["mean_z_specific"]),
                    float(row0["mean_z_mismatched"]),
                ]
                axis.plot(
                    x_positions,
                    values,
                    marker=LANGUAGE_MARKERS[language],
                    linewidth=1.8,
                    color=MODEL_COLORS[model],
                    alpha=0.9 if language == "en" else 0.75,
                    label=f"{MODEL_LABELS[model]} / {LANGUAGE_LABELS[language]}",
                )
        axis.set_title(roi_name)
        axis.set_xticks(x_positions, [CONDITION_LABELS[condition] for condition in CONDITION_ORDER], rotation=20)
        axis.set_ylabel("Mean held-out Fisher-z")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-ROI condition comparison", fontsize=14, fontweight="bold", y=1.02)
    _save_figure(fig, path, dpi=dpi)


def _figure06(path: Path, group_df: pd.DataFrame, *, dpi: int) -> None:
    subset = group_df.loc[
        (group_df["roi_name"].isin(FIGURE06_ROIS))
        & (group_df["condition"].isin(["raw", "shared", "specific"]))
    ].copy()

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True, sharey=True)
    for row_index, language in enumerate(LANGUAGE_ORDER):
        for col_index, model in enumerate(MODEL_ORDER):
            axis = axes[row_index, col_index]
            panel = subset.loc[(subset["language"] == language) & (subset["model"] == model)]
            for roi_name, linestyle in zip(FIGURE06_ROIS, ("solid", "dashed"), strict=True):
                roi_panel = panel.loc[panel["roi_name"] == roi_name]
                for condition in ("raw", "shared", "specific"):
                    curve = roi_panel.loc[roi_panel["condition"] == condition].sort_values("layer_depth")
                    if curve.empty:
                        continue
                    axis.plot(
                        curve["layer_depth"],
                        curve["mean_z"],
                        color=CONDITION_COLORS[condition],
                        linestyle=linestyle,
                        linewidth=2,
                        label=f"{roi_name} {CONDITION_LABELS[condition]}",
                    )
            axis.set_title(f"{MODEL_LABELS[model]} / {LANGUAGE_LABELS[language]}")
            axis.set_xlabel("Normalized layer depth")
            axis.set_ylabel("Mean held-out Fisher-z")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Layer curves in representative semantic and auditory ROIs\nsolid = L_pMTG, dashed = L_pSTG",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    _save_figure(fig, path, dpi=dpi)


def _build_coupling_points(group_df: pd.DataFrame, geometry_df: pd.DataFrame) -> pd.DataFrame:
    brain_delta = _brain_delta_by_layer(group_df).rename(columns={"delta_shared_specific": "brain_delta"})
    geometry_subset = geometry_df.loc[:, ["model", "layer_index", "layer_depth", "cas"]].copy()
    points = brain_delta.merge(
        geometry_subset,
        on=["model", "layer_index", "layer_depth"],
        how="inner",
    )
    return points.sort_values(["language", "model", "layer_index"]).reset_index(drop=True)


def _figure07(
    path: Path,
    coupling_points_path: Path,
    *,
    group_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    coupling_df: pd.DataFrame,
    dpi: int,
) -> None:
    points = _build_coupling_points(group_df, geometry_df)
    coupling_points_path.parent.mkdir(parents=True, exist_ok=True)
    points.to_parquet(coupling_points_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, language in zip(axes, LANGUAGE_ORDER, strict=True):
        panel = points.loc[points["language"] == language]
        stats_panel = coupling_df.loc[coupling_df["language"] == language]
        for model in MODEL_ORDER:
            subset = panel.loc[panel["model"] == model].sort_values("layer_index")
            if subset.empty:
                continue
            axis.scatter(
                subset["cas"],
                subset["brain_delta"],
                color=MODEL_COLORS[model],
                s=45,
                alpha=0.85,
                label=MODEL_LABELS[model],
            )
            if len(subset) >= 2:
                x = subset["cas"].to_numpy(dtype=float)
                y = subset["brain_delta"].to_numpy(dtype=float)
                slope, intercept = np.polyfit(x, y, deg=1)
                x_line = np.linspace(float(x.min()), float(x.max()), 50)
                axis.plot(x_line, slope * x_line + intercept, color=MODEL_COLORS[model], linewidth=2)
        axis.set_title(LANGUAGE_LABELS[language])
        axis.set_xlabel("CAS_l")
        axis.set_ylabel("Semantic shared advantage B_l")
        stat_lines = []
        for model in MODEL_ORDER:
            stat_row = stats_panel.loc[stats_panel["model"] == model]
            if stat_row.empty:
                continue
            stat_lines.append(
                f"{MODEL_LABELS[model]} rho={float(stat_row.iloc[0]['rho_spearman']):.2f}, p={float(stat_row.iloc[0]['p_perm']):.3f}"
            )
        if stat_lines:
            axis.text(
                0.03,
                0.97,
                "\n".join(stat_lines),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
            )
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Geometry-to-brain coupling", fontsize=14, fontweight="bold", y=1.05)
    _save_figure(fig, path, dpi=dpi)


def _append_figure_provenance(summary: FigureBuildSummary) -> None:
    path = project_root() / "outputs" / "manuscript" / "figure_provenance.md"
    figure_lines = [f"generating_script=src/brain_subspace_paper/viz/figures.py"]
    for key in sorted(summary.figure_paths):
        figure_lines.append(f"{key}={summary.figure_paths[key].relative_to(project_root()).as_posix()}")
    figure_lines.extend(
        [
            f"coupling_points={summary.coupling_points_path.relative_to(project_root()).as_posix()}",
            "source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/group_level_roi_results.parquet, outputs/stats/geometry_metrics.parquet, outputs/stats/geometry_brain_coupling.parquet, outputs/stats/roi_condition_stats.parquet, outputs/stats/roi_family_effect_panels.parquet, outputs/stats/roi_projected_maps/, data/interim/lppc_run_manifest.parquet, data/interim/sentence_spans_*.parquet, data/processed/alignment_triplets.parquet, data/processed/alignment_triplets_qc.parquet, data/interim/roi/harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz",
            f"fig08_note={summary.fig08_note}",
        ]
    )
    append_markdown_log(path, "Generated canonical paper figures", figure_lines)


def build_paper_figures(*, dpi: int = 200) -> FigureBuildSummary:
    bootstrap_logs()
    _apply_style()
    frames = _load_required_frames()
    figure_paths = _figure_paths()
    stats_paths = _canonical_stats_paths()
    for path in figure_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    _figure01(figure_paths["fig01"], dpi=dpi)
    _figure02(figure_paths["fig02"], frames["subject"], dpi=dpi)
    _figure03(figure_paths["fig03"], frames["geometry"], dpi=dpi)
    _figure04(figure_paths["fig04"], frames["roi_family_panels"], dpi=dpi)
    _figure05(figure_paths["fig05"], frames["roi_condition_stats"], dpi=dpi)
    _figure06(figure_paths["fig06"], frames["group"], dpi=dpi)
    _figure07(
        figure_paths["fig07"],
        stats_paths["coupling_points"],
        group_df=frames["group"],
        geometry_df=frames["geometry"],
        coupling_df=frames["coupling"],
        dpi=dpi,
    )
    roi_projected_roots = _write_roi_projected_maps(frames["group"])
    _figure08(figure_paths["fig08"], roi_projected_roots, dpi=dpi)

    summary = FigureBuildSummary(
        figure_paths=figure_paths,
        coupling_points_path=stats_paths["coupling_points"],
        fig08_note="generated as ROI-projected descriptive cortex maps from ROI group means; not voxelwise encoding maps",
    )
    append_markdown_log(
        project_root() / "outputs" / "logs" / "progress_log.md",
        "paper figures",
        [
            *(f"{name}={path.as_posix()}" for name, path in figure_paths.items()),
            f"coupling_points={summary.coupling_points_path.as_posix()}",
            f"fig08_note={summary.fig08_note}",
            f"dpi={dpi}",
        ],
    )
    _append_figure_provenance(summary)
    return summary
