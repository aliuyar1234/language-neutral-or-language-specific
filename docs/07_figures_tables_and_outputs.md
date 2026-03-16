# 07 - Figures, Tables, and Output Contracts

This document defines what outputs must exist for the paper.

## 1. Directory policy

All final outputs should land under:
```text
outputs/
  figures/
  tables/
  stats/
  manuscript/
  logs/
```

## 2. Figure list

## Figure 1 - Conceptual pipeline
A single schematic showing:

parallel sentence spans in EN/FR/ZH
-> multilingual model
-> `RAW / SHARED / SPECIFIC / FULL / MISMATCHED_SHARED`
-> target-language timeline placement
-> HRF convolution
-> ridge encoding
-> ROI and whole-brain outputs

Purpose:
- explain the project visually in one page

## Figure 2 - Dataset and alignment overview
Panels:
- participant counts by language
- canonical run structure
- number of sentence spans and final triplets
- merge-pattern frequencies
- alignment QC summary
- sentence duration histogram per language

## Figure 3 - Text-side multilingual geometry by layer
Per model:
- same-sentence cosine by layer
- contrastive alignment score by layer
- retrieval R@1 by layer
- specificity energy by layer

Use normalized layer depth on the x-axis.

## Figure 4 - Main confirmatory ROI-family effects
This is the main inferential figure.

Panels:
- semantic family `Delta^mid` by language and model
- auditory family `Delta^mid` by language and model
- semantic minus auditory difference by language and model

Use point + interval plots, not bar charts.

## Figure 5 - Per-ROI condition comparison
For selected ROIs:
- `RAW`
- `SHARED`
- `SPECIFIC`
- `MISMATCHED_SHARED`

Show held-out Fisher-z.

Suggested ROIs:
- left pMTG
- right pMTG
- left AG
- left Heschl
- left pSTG
- left IFGtri

## Figure 6 - Layer curves in key ROIs
For selected ROIs:
- x-axis = normalized layer depth
- y-axis = mean held-out Fisher-z
- one line each for `RAW`, `SHARED`, `SPECIFIC`

Do not clutter with too many ROIs in one panel; use small multiples.

## Figure 7 - Geometry-to-brain coupling
Scatter plot:
- x-axis = `CAS_l`
- y-axis = semantic-family layerwise shared advantage `B_l`
- one point per layer
- one regression or trend line per model
- separate panels by language if needed

## Figure 8 - Whole-brain maps (secondary)
For a representative layer per model and language:
- SHARED map
- SPECIFIC map
- SHARED minus SPECIFIC map

These are descriptive, not the main inference.

Representative layer rule:
- choose the layer with the maximum semantic-family group mean `SHARED - SPECIFIC`
- tie-break toward the shallower layer

Machine-readable artifacts are mandatory. For each `model x language x selected_layer`, save:
- `outputs/stats/whole_brain/<model>/<language>/layer_<selected_layer:02d>/shared_mean_z.nii.gz`
- `outputs/stats/whole_brain/<model>/<language>/layer_<selected_layer:02d>/specific_mean_z.nii.gz`
- `outputs/stats/whole_brain/<model>/<language>/layer_<selected_layer:02d>/shared_minus_specific_mean_z.nii.gz`
- `outputs/stats/whole_brain/<model>/<language>/layer_<selected_layer:02d>/brain_mask.nii.gz`
- `outputs/stats/whole_brain/<model>/<language>/layer_<selected_layer:02d>/manifest.json`

The `manifest.json` must record:
- source subject list
- selected layer index
- selected layer normalized depth
- exact generating script path
- input feature files
- whether the maps are mean-z, t, or another descriptive statistic

## Figure 9 - Robustness summary (optional)
A compact panel showing whether the core sign survives:
- FIR
- last-token pooling
- no pitch nuisance
- context window
- voxelwise-within-ROI target

Figure 9 is optional, but robustness itself is mandatory for the paper-complete target.

## Figure 10 - German appendix (optional)
Use only if the studyforrest appendix is completed.

## 3. Table list

## Table 1 - Dataset summary
Columns:
- dataset
- language
- subjects
- runs
- duration
- TR
- preprocessed derivatives available?
- annotations available?
- free download?

Include LPPC and, if used, studyforrest.

## Table 2 - Model summary
Columns:
- model
- Hugging Face id
- architecture
- layers
- hidden size
- main role in paper
- used in core paper? yes/no

Core paper rows:
- XLM-R-base
- NLLB-200-distilled-600M encoder

## Table 3 - Main confirmatory statistics
Rows:
- model x language x hypothesis

Columns:
- hypothesis
- roi_family
- mean_delta_mid
- se
- dz
- p_perm
- p_holm_primary
- ci_low
- ci_high

## Table 4 - Per-ROI main condition statistics
Rows:
- model x language x ROI

Columns:
- mean_z_raw
- mean_z_shared
- mean_z_specific
- mean_z_mismatched
- delta_shared_specific
- p_perm
- q_fdr

## Table 5 - Robustness summary
Rows:
- robustness condition

Columns:
- core effect sign preserved?
- note

Table 5 is mandatory for the paper-complete target.

## 4. Output file names

Recommended canonical paths:

```text
outputs/figures/fig01_pipeline.png
outputs/figures/fig02_dataset_alignment_overview.png
outputs/figures/fig03_text_geometry_by_layer.png
outputs/figures/fig04_main_confirmatory_roi_families.png
outputs/figures/fig05_roi_condition_comparison.png
outputs/figures/fig06_layer_curves_key_rois.png
outputs/figures/fig07_geometry_to_brain_coupling.png
outputs/figures/fig08_whole_brain_maps.png
outputs/figures/fig09_robustness_summary.png
outputs/figures/fig10_german_appendix.png
```

```text
outputs/tables/table01_dataset_summary.csv
outputs/tables/table02_model_summary.csv
outputs/tables/table03_main_confirmatory_stats.csv
outputs/tables/table04_roi_condition_stats.csv
outputs/tables/table05_robustness_summary.csv
```

## 5. Required machine-readable results

At minimum, the repo must save:

```text
outputs/stats/subject_level_roi_results.parquet
outputs/stats/group_level_roi_results.parquet
outputs/stats/confirmatory_effects.parquet
outputs/stats/geometry_metrics.parquet
outputs/stats/geometry_brain_coupling.parquet
outputs/stats/whole_brain/<model>/<language>/layer_<selected_layer:02d>/shared_mean_z.nii.gz
outputs/stats/whole_brain/<model>/<language>/layer_<selected_layer:02d>/specific_mean_z.nii.gz
outputs/stats/whole_brain/<model>/<language>/layer_<selected_layer:02d>/shared_minus_specific_mean_z.nii.gz
outputs/stats/whole_brain/<model>/<language>/layer_<selected_layer:02d>/brain_mask.nii.gz
outputs/stats/whole_brain/<model>/<language>/layer_<selected_layer:02d>/manifest.json
```

## 6. Plotting discipline

- Use readable axis labels
- Use normalized layer depth for cross-model layer plots
- Keep language families visually separate if necessary
- Prefer point/interval plots over bars
- Explicitly label SHARED, SPECIFIC, RAW, MISMATCHED_SHARED
- For whole-brain maps, use the same color scale across comparable conditions when possible

## 7. Figure-citation discipline

Every figure must have:
- a descriptive caption
- the exact generating script path
- the exact input data file(s)

Record this in:
`outputs/manuscript/figure_provenance.md`

## 8. Table-citation discipline

Every table must have:
- a generating script path
- its source stats file(s)

Record this in:
`outputs/manuscript/table_provenance.md`
