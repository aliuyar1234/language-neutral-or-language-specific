# Figure Provenance

## Status As Of 2026-03-16

- Canonical figures `1-9` now exist for the current paper-complete checkpoint.
- A filled caption asset now exists at `paper/figure_captions.md` for Figures `1-9`.
- `Figure 08` is a descriptive ROI-projected cortex panel created from ROI group means at representative layers:
  - `outputs/figures/fig08_whole_brain_maps.png`
  - `outputs/stats/roi_projected_maps/`
- `Figure 09` is the completed robustness summary:
  - `outputs/figures/fig09_robustness_summary.png`
  - `outputs/stats/robustness_cell_results.parquet`
- Confirmatory interpretation should still be anchored by the ROI-family statistics bundle:
  - `outputs/stats/group_level_roi_results.parquet`
  - `outputs/stats/confirmatory_effects.parquet`
  - `outputs/stats/geometry_metrics.parquet`
  - `outputs/stats/geometry_brain_coupling.parquet`

## 2026-03-15 - Planned figure sources after T13

- Current canonical figure sources:
  - `Figure 03 text geometry by layer` -> `outputs/stats/geometry_metrics.parquet`
  - `Figure 04 main confirmatory ROI families` -> `outputs/stats/confirmatory_effects.parquet` and `outputs/stats/roi_family_effect_panels.parquet`
  - `Figure 05 ROI condition comparison` -> `outputs/stats/group_level_roi_results.parquet` and `outputs/stats/roi_condition_stats.parquet`
  - `Figure 06 layer curves key ROIs` -> `outputs/stats/group_level_roi_results.parquet`
  - `Figure 07 geometry to brain coupling` -> `outputs/stats/geometry_brain_coupling.parquet`
  - `Figure 08 ROI-projected cortex maps` -> `outputs/stats/roi_projected_maps/`

Current caveat:
- regenerate figure artifacts if the manuscript-final run definition replaces the current fast-paper checkpoint

## 2026-03-15T17:03:30+00:00 - Generated canonical paper figures

- generating_script=src/brain_subspace_paper/viz/figures.py
- fig01=outputs/figures/fig01_pipeline.png
- fig02=outputs/figures/fig02_dataset_alignment_overview.png
- fig03=outputs/figures/fig03_text_geometry_by_layer.png
- fig04=outputs/figures/fig04_main_confirmatory_roi_families.png
- fig05=outputs/figures/fig05_roi_condition_comparison.png
- fig06=outputs/figures/fig06_layer_curves_key_rois.png
- fig07=outputs/figures/fig07_geometry_to_brain_coupling.png
- coupling_points=outputs/stats/geometry_brain_coupling_points.parquet
- source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/group_level_roi_results.parquet, outputs/stats/geometry_metrics.parquet, outputs/stats/geometry_brain_coupling.parquet, outputs/stats/roi_condition_stats.parquet, outputs/stats/roi_family_effect_panels.parquet, data/interim/lppc_run_manifest.parquet, data/interim/sentence_spans_*.parquet, data/processed/alignment_triplets.parquet, data/processed/alignment_triplets_qc.parquet
- fig08_status=blocked (whole-brain voxelwise artifacts do not exist yet)
## 2026-03-15T17:05:46+00:00 - Generated canonical paper figures

- generating_script=src/brain_subspace_paper/viz/figures.py
- fig01=outputs/figures/fig01_pipeline.png
- fig02=outputs/figures/fig02_dataset_alignment_overview.png
- fig03=outputs/figures/fig03_text_geometry_by_layer.png
- fig04=outputs/figures/fig04_main_confirmatory_roi_families.png
- fig05=outputs/figures/fig05_roi_condition_comparison.png
- fig06=outputs/figures/fig06_layer_curves_key_rois.png
- fig07=outputs/figures/fig07_geometry_to_brain_coupling.png
- coupling_points=outputs/stats/geometry_brain_coupling_points.parquet
- source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/group_level_roi_results.parquet, outputs/stats/geometry_metrics.parquet, outputs/stats/geometry_brain_coupling.parquet, outputs/stats/roi_condition_stats.parquet, outputs/stats/roi_family_effect_panels.parquet, data/interim/lppc_run_manifest.parquet, data/interim/sentence_spans_*.parquet, data/processed/alignment_triplets.parquet, data/processed/alignment_triplets_qc.parquet
- fig08_status=blocked (whole-brain voxelwise artifacts do not exist yet)
## 2026-03-15T17:42:04+00:00 - Generated canonical paper figures

- generating_script=src/brain_subspace_paper/viz/figures.py
- fig01=outputs/figures/fig01_pipeline.png
- fig02=outputs/figures/fig02_dataset_alignment_overview.png
- fig03=outputs/figures/fig03_text_geometry_by_layer.png
- fig04=outputs/figures/fig04_main_confirmatory_roi_families.png
- fig05=outputs/figures/fig05_roi_condition_comparison.png
- fig06=outputs/figures/fig06_layer_curves_key_rois.png
- fig07=outputs/figures/fig07_geometry_to_brain_coupling.png
- fig08=outputs/figures/fig08_whole_brain_maps.png
- coupling_points=outputs/stats/geometry_brain_coupling_points.parquet
- source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/group_level_roi_results.parquet, outputs/stats/geometry_metrics.parquet, outputs/stats/geometry_brain_coupling.parquet, outputs/stats/roi_condition_stats.parquet, outputs/stats/roi_family_effect_panels.parquet, outputs/stats/roi_projected_maps/, data/interim/lppc_run_manifest.parquet, data/interim/sentence_spans_*.parquet, data/processed/alignment_triplets.parquet, data/processed/alignment_triplets_qc.parquet, data/interim/roi/harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz
- fig08_note=generated as ROI-projected descriptive cortex maps from ROI group means; not voxelwise encoding maps
## 2026-03-15T17:43:36+00:00 - Generated canonical paper figures

- generating_script=src/brain_subspace_paper/viz/figures.py
- fig01=outputs/figures/fig01_pipeline.png
- fig02=outputs/figures/fig02_dataset_alignment_overview.png
- fig03=outputs/figures/fig03_text_geometry_by_layer.png
- fig04=outputs/figures/fig04_main_confirmatory_roi_families.png
- fig05=outputs/figures/fig05_roi_condition_comparison.png
- fig06=outputs/figures/fig06_layer_curves_key_rois.png
- fig07=outputs/figures/fig07_geometry_to_brain_coupling.png
- fig08=outputs/figures/fig08_whole_brain_maps.png
- coupling_points=outputs/stats/geometry_brain_coupling_points.parquet
- source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/group_level_roi_results.parquet, outputs/stats/geometry_metrics.parquet, outputs/stats/geometry_brain_coupling.parquet, outputs/stats/roi_condition_stats.parquet, outputs/stats/roi_family_effect_panels.parquet, outputs/stats/roi_projected_maps/, data/interim/lppc_run_manifest.parquet, data/interim/sentence_spans_*.parquet, data/processed/alignment_triplets.parquet, data/processed/alignment_triplets_qc.parquet, data/interim/roi/harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz
- fig08_note=generated as ROI-projected descriptive cortex maps from ROI group means; not voxelwise encoding maps
## 2026-03-15T18:26:20+00:00 - Generated robustness figure

- generating_script=src/brain_subspace_paper/stats/robustness.py
- fig09=outputs/figures/fig09_robustness_summary.png
- robustness_cell_results=outputs/stats/robustness_cell_results.parquet
- source_stats=outputs/stats/robustness_cell_results.parquet
## 2026-03-15T22:38:38+00:00 - Generated robustness figure

- generating_script=src/brain_subspace_paper/stats/robustness.py
- fig09=outputs/figures/fig09_robustness_summary__smoke_nopitch.png
- robustness_cell_results=outputs/stats/robustness_cell_results__smoke_nopitch.parquet
- source_stats=outputs/stats/robustness_cell_results__smoke_nopitch.parquet
## 2026-03-15T22:40:17+00:00 - Generated robustness figure

- generating_script=src/brain_subspace_paper/stats/robustness.py
- fig09=outputs/figures/fig09_robustness_summary__smoke_lasttok.png
- robustness_cell_results=outputs/stats/robustness_cell_results__smoke_lasttok.parquet
- source_stats=outputs/stats/robustness_cell_results__smoke_lasttok.parquet
## 2026-03-15T22:40:26+00:00 - Generated robustness figure

- generating_script=src/brain_subspace_paper/stats/robustness.py
- fig09=outputs/figures/fig09_robustness_summary__smoke_merge.png
- robustness_cell_results=outputs/stats/robustness_cell_results__smoke_merge.parquet
- source_stats=outputs/stats/robustness_cell_results__smoke_merge.parquet
## 2026-03-15T22:43:11+00:00 - Generated robustness figure

- generating_script=src/brain_subspace_paper/stats/robustness.py
- fig09=outputs/figures/fig09_robustness_summary__smoke_nopitch.png
- robustness_cell_results=outputs/stats/robustness_cell_results__smoke_nopitch.parquet
- source_stats=outputs/stats/robustness_cell_results__smoke_nopitch.parquet
## 2026-03-15T22:57:35+00:00 - Generated robustness figure

- generating_script=src/brain_subspace_paper/stats/robustness.py
- fig09=outputs/figures/fig09_robustness_summary__smoke_lasttok_again.png
- robustness_cell_results=outputs/stats/robustness_cell_results__smoke_lasttok_again.parquet
- source_stats=outputs/stats/robustness_cell_results__smoke_lasttok_again.parquet
## 2026-03-15T23:08:28+00:00 - Generated robustness figure

- generating_script=src/brain_subspace_paper/stats/robustness.py
- fig09=outputs/figures/fig09_robustness_summary__smoke_t15_wrapper.png
- robustness_cell_results=outputs/stats/robustness_cell_results__smoke_t15_wrapper.parquet
- source_stats=outputs/stats/robustness_cell_results__smoke_t15_wrapper.parquet
## 2026-03-15T23:08:41+00:00 - Generated robustness figure

- generating_script=src/brain_subspace_paper/stats/robustness.py
- fig09=outputs/figures/fig09_robustness_summary__smoke_t15_wrapper.png
- robustness_cell_results=outputs/stats/robustness_cell_results__smoke_t15_wrapper.parquet
- source_stats=outputs/stats/robustness_cell_results__smoke_t15_wrapper.parquet
## 2026-03-16T17:05:09+00:00 - Generated robustness figure

- generating_script=src/brain_subspace_paper/stats/robustness.py
- fig09=outputs/figures/fig09_robustness_summary.png
- robustness_cell_results=outputs/stats/robustness_cell_results.parquet
- source_stats=outputs/stats/robustness_cell_results.parquet
