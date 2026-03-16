# 10 — Data Contracts

This document defines the canonical schemas for intermediate and final artifacts.

## 1. Principle

Every nontrivial derived artifact should have:
- a fixed schema
- a saved manifest
- a stable file path
- enough metadata to be reloaded without guessing

Prefer:
- Parquet / TSV for tables
- NPY / NPZ for large arrays
- JSON / YAML for manifests

## 2. Run manifest

Path:
- `data/interim/lppc_run_manifest.parquet`

Columns:
- `subject_id` (str)
- `language` (str: en/fr/zh)
- `task_name` (str)
- `original_run_label` (int)
- `canonical_run_index` (int in 1..9)
- `n_volumes` (int)
- `filepath` (str)
- `space` (str, expected `MNIColin27`)
- `is_preproc` (bool)

## 3. Sentence-span tables

Sentence spans are mandatory derived artifacts:
- `data/interim/sentence_spans_en.parquet`
- `data/interim/sentence_spans_fr.parquet`
- `data/interim/sentence_spans_zh.parquet`

Columns:
- `language`
- `section_index`
- `language_sentence_index`
- `text`
- `onset_sec`
- `offset_sec`
- `duration_sec`
- `n_words`
- `first_word_idx`
- `last_word_idx`

## 4. Alignment triplet table

Path:
- `data/processed/alignment_triplets.parquet`

Columns:
- `triplet_id` (int)
- `section_index` (int)
- `section_triplet_index` (int)
- `merge_pattern` (str, e.g. `1-1-1`, `1-2-1`)
- `en_text`
- `fr_text`
- `zh_text`
- `en_onset_sec`
- `en_offset_sec`
- `fr_onset_sec`
- `fr_offset_sec`
- `zh_onset_sec`
- `zh_offset_sec`
- `en_first_sentence_idx`
- `en_last_sentence_idx`
- `fr_first_sentence_idx`
- `fr_last_sentence_idx`
- `zh_first_sentence_idx`
- `zh_last_sentence_idx`

## 5. Alignment QC table

Path:
- `data/processed/alignment_triplets_qc.parquet`

Columns:
- `triplet_id`
- `section_index`
- `sim_en_fr`
- `sim_en_zh`
- `sim_fr_zh`
- `mean_pairwise_sim`
- `char_ratio_en_fr`
- `char_ratio_en_zh`
- `punct_mismatch_flag`
- `max_local_cost`
- `needs_manual_review`
- `manual_status` (approved/rejected/fixed)
- `notes`

## 6. Embedding manifest

Path:
- `data/interim/embeddings/embedding_manifest.parquet`

Columns:
- `model`
- `language`
- `layer_index`
- `layer_depth`
- `n_rows`
- `hidden_size`
- `filepath`
- `dtype`

## 7. Embedding arrays

Recommended path pattern:
```text
data/interim/embeddings/{model_slug}/{language}/layer_{layer_index:02d}.npy
```

Array shape:
- `(n_triplets, hidden_size)`

Triplet row order must match `alignment_triplets.parquet`.

## 8. Decomposition feature manifest

Path:
- `data/processed/features/feature_manifest.parquet`

Columns:
- `model`
- `language`
- `layer_index`
- `condition` (`raw`, `shared`, `specific`, `full`, `mismatched_shared`)
- `shuffle_index` (null for all conditions except `mismatched_shared`; `0..K-1` for mismatched controls)
- `filepath`
- `n_rows`
- `feature_dim`

## 9. Sentence-level feature arrays

Recommended path pattern:
```text
data/processed/features/{model_slug}/{language}/{condition}/layer_{layer_index:02d}.npy
```

For `mismatched_shared`, store one array per shuffle:
```text
data/processed/features/{model_slug}/{language}/mismatched_shared/shuffle_{shuffle_index:02d}/layer_{layer_index:02d}.npy
```

`RAW`, `SHARED`, `SPECIFIC` array shape:
- `(n_triplets, d)`

`FULL` shape:
- `(n_triplets, 2d)`

`MISMATCHED_SHARED` shape:
- `(n_triplets, d)` per shuffle

## 10. Design-matrix manifest

Path:
- `data/processed/designs/design_manifest.parquet`

Columns:
- `subject_id`
- `language`
- `model`
- `layer_index`
- `condition`
- `canonical_run_index`
- `filepath`
- `n_timepoints`
- `feature_dim`

## 11. ROI target manifest

Path:
- `data/processed/roi_targets/roi_target_manifest.parquet`

Columns:
- `subject_id`
- `language`
- `roi_name`
- `canonical_run_index`
- `filepath`
- `n_timepoints`

## 12. Subject-level ROI results

Path:
- `outputs/stats/subject_level_roi_results.parquet`

Columns:
- `subject_id`
- `language`
- `model`
- `roi_name`
- `roi_family`
- `layer_index`
- `layer_depth`
- `condition`
- `metric_name` (`r`, `z`, `r2`)
- `value`

## 13. Group-level ROI results

Path:
- `outputs/stats/group_level_roi_results.parquet`

Columns:
- `language`
- `model`
- `roi_name`
- `roi_family`
- `layer_index`
- `layer_depth`
- `condition`
- `mean_r`
- `mean_z`
- `mean_r2`
- `se_z`
- `n_subjects`

## 14. Confirmatory effects table

Path:
- `outputs/stats/confirmatory_effects.parquet`

Columns:
- `hypothesis`
- `language`
- `model`
- `roi_family`
- `mean_delta_mid`
- `se`
- `dz`
- `p_perm`
- `p_holm_primary`
- `ci_low`
- `ci_high`
- `n_subjects`

Use one shared table for both hypotheses:
- `hypothesis = H1_shared_gt_specific_semantic` with `roi_family = semantic`
- `hypothesis = H2_semantic_minus_auditory` with `roi_family = semantic_minus_auditory`

## 15. Geometry metrics

Path:
- `outputs/stats/geometry_metrics.parquet`

Columns:
- `model`
- `layer_index`
- `layer_depth`
- `align_mean`
- `cas`
- `retrieval_r1_mean`
- `specificity_energy`

## 16. Geometry-brain coupling

Path:
- `outputs/stats/geometry_brain_coupling.parquet`

Columns:
- `model`
- `language`
- `rho_spearman`
- `p_perm`
- `n_layers`

## 17. Whole-brain voxelwise artifacts

Root:
- `outputs/stats/whole_brain/`

Per `model x language x selected_layer`, save:
- `outputs/stats/whole_brain/{model}/{language}/layer_{selected_layer:02d}/shared_mean_z.nii.gz`
- `outputs/stats/whole_brain/{model}/{language}/layer_{selected_layer:02d}/specific_mean_z.nii.gz`
- `outputs/stats/whole_brain/{model}/{language}/layer_{selected_layer:02d}/shared_minus_specific_mean_z.nii.gz`
- `outputs/stats/whole_brain/{model}/{language}/layer_{selected_layer:02d}/brain_mask.nii.gz`
- `outputs/stats/whole_brain/{model}/{language}/layer_{selected_layer:02d}/manifest.json`

`manifest.json` must include:
- `model`
- `language`
- `selected_layer`
- `layer_depth`
- `source_subject_ids`
- `generating_script`
- `input_feature_files`
- `statistic_type`

## 18. Provenance files

Required:
- `outputs/manuscript/figure_provenance.md`
- `outputs/manuscript/table_provenance.md`
- `outputs/manuscript/claim_evidence_map.md`

## 19. Config and environment provenance

Required:
- `outputs/logs/config_snapshot.yaml`
- `outputs/logs/pip_freeze.txt`
- `outputs/logs/git_commit.txt`

These are part of the reproducibility contract.
