# Table Provenance

## Canonical Stats Bundle (`T13`)

- Generating command: `python -m brain_subspace_paper build-paper-stats`
- Generating script entrypoint: `src/brain_subspace_paper/cli.py`
- Core implementation: `src/brain_subspace_paper/stats/paper_level.py`
- Input stats:
  - `outputs/stats/xlmr_subject_level_roi_results__xlmr_fast_readout.parquet`
  - `outputs/stats/nllb_subject_level_roi_results__nllb_fast_readout.parquet`
- Produced canonical outputs:
  - `outputs/stats/subject_level_roi_results.parquet`
  - `outputs/stats/group_level_roi_results.parquet`
  - `outputs/stats/confirmatory_effects.parquet`
  - `outputs/stats/geometry_metrics.parquet`
  - `outputs/stats/geometry_brain_coupling.parquet`

## Table Generation Status

- Tables `1-5` now exist under `outputs/tables/`.
- `Table 5` is finalized from the completed `T15` robustness checkpoint.

## Table Sources

### Table 3 - Main confirmatory statistics

- Canonical source stats: `outputs/stats/confirmatory_effects.parquet`
- Generated CSV: `outputs/tables/table03_main_confirmatory_stats.csv`
- Generating script: `src/brain_subspace_paper/viz/tables.py`
- Expected content: one row per `model x language x hypothesis`

### Table 4 - Per-ROI main condition statistics

- Canonical source stats: `outputs/stats/group_level_roi_results.parquet`
- Supporting subject-level source: `outputs/stats/subject_level_roi_results.parquet`
- Generated CSV: `outputs/tables/table04_roi_condition_stats.csv`
- Derived helper stats: `outputs/stats/roi_condition_stats.parquet`
- Generating script: `src/brain_subspace_paper/viz/tables.py`
- Expected content: per `model x language x ROI` summaries for `RAW`, `SHARED`, `SPECIFIC`, and `MISMATCHED_SHARED`

### Tables 1, 2, and 5

- Table `1` generated: `outputs/tables/table01_dataset_summary.csv`
- Table `2` generated: `outputs/tables/table02_model_summary.csv`
- Table `5` generated: `outputs/tables/table05_robustness_summary.csv`

### Table 5 - Robustness summary

- Canonical source stats: `outputs/stats/robustness_summary.parquet`
- Supporting cell-level source: `outputs/stats/robustness_cell_results.parquet`
- Representative layer source: `outputs/stats/robustness_representative_layers.parquet`
- Generated CSV: `outputs/tables/table05_robustness_summary.csv`
- Generating script: `src/brain_subspace_paper/stats/robustness.py`
- Expected content: one row per robustness condition summarizing whether the representative-layer core effect kept the positive sign across the six `model x language` cells

## 2026-03-15 - T13 canonical stats sources

- `Table 03 main confirmatory stats`
  - source: `outputs/stats/confirmatory_effects.parquet`
  - fields: `model`, `language`, `hypothesis`, `mean_delta_mid`, `se`, `dz`, `p_perm`, `p_holm_primary`, `ci_low`, `ci_high`, `n_subjects`
  - note: current checkpoint contains `12` rows; all six `H1_shared_gt_specific_semantic` rows are Holm-significant and `H2_semantic_minus_auditory` remains mixed
- `Table 04 ROI condition stats`
  - source: `outputs/stats/group_level_roi_results.parquet`
  - fields to summarize: `model`, `language`, `layer_index`, `roi`, `roi_family`, `condition`, `mean_r`, `delta_vs_acoustic_only`, `delta_vs_mismatched_shared`
- `Table 02 model geometry summary`
  - source: `outputs/stats/geometry_metrics.parquet`
  - fields: `model`, `layer_index`, `layer_depth`, `align_mean`, `cas`, `retrieval_r1_mean`, `specificity_energy`
- `Geometry-brain coupling summary`
  - source: `outputs/stats/geometry_brain_coupling.parquet`
  - fields: `model`, `language`, `rho_spearman`, `p_perm`, `n_layers`

Current caveat:
- these canonical sources come from the fast-paper checkpoint built from `xlmr_fast_readout` and `nllb_fast_readout`
- regenerate table artifacts if the manuscript-final run definition changes

## 2026-03-15T16:57:42+00:00 - Generated canonical paper tables

- generating_script=src/brain_subspace_paper/viz/tables.py
- table01=outputs/tables/table01_dataset_summary.csv
- table02=outputs/tables/table02_model_summary.csv
- table03=outputs/tables/table03_main_confirmatory_stats.csv
- table04=outputs/tables/table04_roi_condition_stats.csv
- derived_roi_condition_stats=outputs/stats/roi_condition_stats.parquet
- derived_roi_family_panels=outputs/stats/roi_family_effect_panels.parquet
- source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/confirmatory_effects.parquet, outputs/stats/geometry_metrics.parquet, data/interim/lppc_run_manifest.parquet, data/interim/embeddings/embedding_manifest.parquet
## 2026-03-15T16:58:57+00:00 - Generated canonical paper tables

- generating_script=src/brain_subspace_paper/viz/tables.py
- table01=outputs/tables/table01_dataset_summary.csv
- table02=outputs/tables/table02_model_summary.csv
- table03=outputs/tables/table03_main_confirmatory_stats.csv
- table04=outputs/tables/table04_roi_condition_stats.csv
- derived_roi_condition_stats=outputs/stats/roi_condition_stats.parquet
- derived_roi_family_panels=outputs/stats/roi_family_effect_panels.parquet
- source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/confirmatory_effects.parquet, outputs/stats/geometry_metrics.parquet, data/interim/lppc_run_manifest.parquet, data/interim/embeddings/embedding_manifest.parquet
## 2026-03-15T16:59:59+00:00 - Generated canonical paper tables

- generating_script=src/brain_subspace_paper/viz/tables.py
- table01=outputs/tables/table01_dataset_summary.csv
- table02=outputs/tables/table02_model_summary.csv
- table03=outputs/tables/table03_main_confirmatory_stats.csv
- table04=outputs/tables/table04_roi_condition_stats.csv
- derived_roi_condition_stats=outputs/stats/roi_condition_stats.parquet
- derived_roi_family_panels=outputs/stats/roi_family_effect_panels.parquet
- source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/confirmatory_effects.parquet, outputs/stats/geometry_metrics.parquet, data/interim/lppc_run_manifest.parquet, data/interim/embeddings/embedding_manifest.parquet
## 2026-03-15T18:26:20+00:00 - Generated robustness table

- generating_script=src/brain_subspace_paper/stats/robustness.py
- table05=outputs/tables/table05_robustness_summary.csv
- robustness_cell_results=outputs/stats/robustness_cell_results.parquet
- robustness_summary=outputs/stats/robustness_summary.parquet
- representative_layers=outputs/stats/robustness_representative_layers.parquet
- source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/group_level_roi_results.parquet, data/processed/features/feature_manifest.parquet, data/interim/roi/*_roi_target_manifest.parquet, data/interim/roi/roi_metadata.parquet, data/interim/roi/harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz
## 2026-03-15T22:38:38+00:00 - Generated robustness table

- generating_script=src/brain_subspace_paper/stats/robustness.py
- table05=outputs/tables/table05_robustness_summary__smoke_nopitch.csv
- robustness_cell_results=outputs/stats/robustness_cell_results__smoke_nopitch.parquet
- robustness_summary=outputs/stats/robustness_summary__smoke_nopitch.parquet
- representative_layers=outputs/stats/robustness_representative_layers__smoke_nopitch.parquet
- source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/group_level_roi_results.parquet, data/processed/features/feature_manifest.parquet, data/interim/roi/*_roi_target_manifest.parquet, data/interim/roi/roi_metadata.parquet, data/interim/roi/harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz
## 2026-03-15T22:40:17+00:00 - Generated robustness table

- generating_script=src/brain_subspace_paper/stats/robustness.py
- table05=outputs/tables/table05_robustness_summary__smoke_lasttok.csv
- robustness_cell_results=outputs/stats/robustness_cell_results__smoke_lasttok.parquet
- robustness_summary=outputs/stats/robustness_summary__smoke_lasttok.parquet
- representative_layers=outputs/stats/robustness_representative_layers__smoke_lasttok.parquet
- source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/group_level_roi_results.parquet, data/processed/features/feature_manifest.parquet, data/interim/roi/*_roi_target_manifest.parquet, data/interim/roi/roi_metadata.parquet, data/interim/roi/harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz
## 2026-03-15T22:40:26+00:00 - Generated robustness table

- generating_script=src/brain_subspace_paper/stats/robustness.py
- table05=outputs/tables/table05_robustness_summary__smoke_merge.csv
- robustness_cell_results=outputs/stats/robustness_cell_results__smoke_merge.parquet
- robustness_summary=outputs/stats/robustness_summary__smoke_merge.parquet
- representative_layers=outputs/stats/robustness_representative_layers__smoke_merge.parquet
- source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/group_level_roi_results.parquet, data/processed/features/feature_manifest.parquet, data/interim/roi/*_roi_target_manifest.parquet, data/interim/roi/roi_metadata.parquet, data/interim/roi/harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz
## 2026-03-15T22:43:11+00:00 - Generated robustness table

- generating_script=src/brain_subspace_paper/stats/robustness.py
- table05=outputs/tables/table05_robustness_summary__smoke_nopitch.csv
- robustness_cell_results=outputs/stats/robustness_cell_results__smoke_nopitch.parquet
- robustness_summary=outputs/stats/robustness_summary__smoke_nopitch.parquet
- representative_layers=outputs/stats/robustness_representative_layers__smoke_nopitch.parquet
- source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/group_level_roi_results.parquet, data/processed/features/feature_manifest.parquet, data/interim/roi/*_roi_target_manifest.parquet, data/interim/roi/roi_metadata.parquet, data/interim/roi/harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz
## 2026-03-15T22:57:35+00:00 - Generated robustness table

- generating_script=src/brain_subspace_paper/stats/robustness.py
- table05=outputs/tables/table05_robustness_summary__smoke_lasttok_again.csv
- robustness_cell_results=outputs/stats/robustness_cell_results__smoke_lasttok_again.parquet
- robustness_summary=outputs/stats/robustness_summary__smoke_lasttok_again.parquet
- representative_layers=outputs/stats/robustness_representative_layers__smoke_lasttok_again.parquet
- source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/group_level_roi_results.parquet, data/processed/features/feature_manifest.parquet, data/interim/roi/*_roi_target_manifest.parquet, data/interim/roi/roi_metadata.parquet, data/interim/roi/harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz
## 2026-03-15T23:08:28+00:00 - Generated robustness table

- generating_script=src/brain_subspace_paper/stats/robustness.py
- table05=outputs/tables/table05_robustness_summary__smoke_t15_wrapper.csv
- robustness_cell_results=outputs/stats/robustness_cell_results__smoke_t15_wrapper.parquet
- robustness_summary=outputs/stats/robustness_summary__smoke_t15_wrapper.parquet
- representative_layers=outputs/stats/robustness_representative_layers__smoke_t15_wrapper.parquet
- source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/group_level_roi_results.parquet, data/processed/features/feature_manifest.parquet, data/interim/roi/*_roi_target_manifest.parquet, data/interim/roi/roi_metadata.parquet, data/interim/roi/harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz
## 2026-03-15T23:08:41+00:00 - Generated robustness table

- generating_script=src/brain_subspace_paper/stats/robustness.py
- table05=outputs/tables/table05_robustness_summary__smoke_t15_wrapper.csv
- robustness_cell_results=outputs/stats/robustness_cell_results__smoke_t15_wrapper.parquet
- robustness_summary=outputs/stats/robustness_summary__smoke_t15_wrapper.parquet
- representative_layers=outputs/stats/robustness_representative_layers__smoke_t15_wrapper.parquet
- source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/group_level_roi_results.parquet, data/processed/features/feature_manifest.parquet, data/interim/roi/*_roi_target_manifest.parquet, data/interim/roi/roi_metadata.parquet, data/interim/roi/harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz
## 2026-03-16T17:05:09+00:00 - Generated robustness table

- generating_script=src/brain_subspace_paper/stats/robustness.py
- table05=outputs/tables/table05_robustness_summary.csv
- robustness_cell_results=outputs/stats/robustness_cell_results.parquet
- robustness_summary=outputs/stats/robustness_summary.parquet
- representative_layers=outputs/stats/robustness_representative_layers.parquet
- source_stats=outputs/stats/subject_level_roi_results.parquet, outputs/stats/group_level_roi_results.parquet, data/processed/features/feature_manifest.parquet, data/interim/roi/*_roi_target_manifest.parquet, data/interim/roi/roi_metadata.parquet, data/interim/roi/harvard_oxford_cortical_resampled_to_lppc_bold.nii.gz
