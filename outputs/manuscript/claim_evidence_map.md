# Claim Evidence Map

## Status As Of 2026-03-16

- Canonical tables and figures now exist for the paper-complete checkpoint, including `Table 5` and `Figure 9`.
- Every claim below is tied to an existing generated stats file and, where relevant, a generated figure or CSV table.
- `Figure 08` should be treated as descriptive visual context only because it is an ROI-projected cortex panel, not a voxelwise encoding analysis.
- The manuscript draft at `paper/manuscript.md` has been refreshed against the completed robustness outputs and the current fast-paper-derived canonical stats bundle.

## Claim 1

- Claim: In the semantic ROI family, `SHARED` outperforms `SPECIFIC` across both model families and all three languages.
- Status: supported in the current canonical checkpoint.
- Evidence:
  - `outputs/stats/confirmatory_effects.parquet`
  - rows with `hypothesis = H1_shared_gt_specific_semantic`
  - all six `model x language` rows have positive `mean_delta_mid`
  - all six rows survive Holm correction (`p_holm_primary = 0.0012`)
- Current table anchor:
  - `outputs/tables/table03_main_confirmatory_stats.csv`
- Current figure anchor:
  - `outputs/figures/fig04_main_confirmatory_roi_families.png`

## Claim 2

- Claim: The semantic-family shared advantage exceeds the auditory-family shared advantage.
- Status: not cleanly supported in the current canonical checkpoint.
- Evidence:
  - `outputs/stats/confirmatory_effects.parquet`
  - rows with `hypothesis = H2_semantic_minus_auditory`
  - XLM-R is negative or near zero in `en`, `fr`, and `zh`
  - NLLB is negative in `en` and `fr`, with a small positive trend in `zh`
- Current table anchor:
  - `outputs/tables/table03_main_confirmatory_stats.csv`
- Current figure anchor:
  - `outputs/figures/fig04_main_confirmatory_roi_families.png`
- Writing constraint:
  - treat this as a mixed result and the main caveat, not as a confirmed primary claim

## Claim 3

- Claim: The paper now has canonical text-side geometry metrics for both model families.
- Status: supported as a descriptive results block.
- Evidence:
  - `outputs/stats/geometry_metrics.parquet`
  - metrics available: `align_mean`, `cas`, `retrieval_r1_mean`, `specificity_energy`
- Current figure anchor:
  - `outputs/figures/fig03_text_geometry_by_layer.png`

## Claim 4

- Claim: Geometry-to-brain coupling has been computed as a secondary analysis, but it is not yet strong enough for a bold interpretive claim.
- Status: computed; interpretation should remain cautious.
- Evidence:
  - `outputs/stats/geometry_brain_coupling.parquet`
  - XLM-R `rho_spearman` values are small in magnitude across `en`, `fr`, and `zh`
  - NLLB reaches `rho_spearman = 1.0` in each language, but only across `3` sampled layers with exact `p_perm = 0.428571`
- Current figure anchor:
  - `outputs/figures/fig07_geometry_to_brain_coupling.png`

## Claim 5

- Claim: Control comparisons remain directionally sane in the current checkpoint.
- Status: descriptively supported; not yet elevated to a dedicated confirmatory table.
- Evidence:
  - `outputs/stats/group_level_roi_results.parquet`
  - at mid layers, semantic-family `mean_z(shared) > mean_z(mismatched_shared)` for both models in `en`, `fr`, and `zh`
- Current table anchor:
  - `outputs/tables/table04_roi_condition_stats.csv`
- Current figure anchor:
  - `outputs/figures/fig05_roi_condition_comparison.png`

## Claim 6

- Claim: The paper includes a descriptive cortex-wide localization panel that shows where the main ROI-level effects sit in atlas space.
- Status: supported as descriptive visualization only.
- Evidence:
  - `outputs/figures/fig08_whole_brain_maps.png`
  - `outputs/stats/roi_projected_maps/`
  - per-panel manifests under `outputs/stats/roi_projected_maps/<model>/<language>/layer_XX/manifest.json`
- Writing constraint:
  - describe this as an ROI-projected atlas visualization, not as a voxelwise confirmatory map

## Claim 7

- Claim: robustness is now computed and supportive, but it adds an explicit caveat rather than a new headline claim.
- Status: supported with a nontrivial caveat.
- Evidence:
  - `outputs/stats/robustness_cell_results.parquet`
  - `outputs/stats/robustness_summary.parquet`
  - `outputs/tables/table05_robustness_summary.csv`
  - `outputs/figures/fig09_robustness_summary.png`
- Current support:
  - the representative-layer reference base is positive in all `6/6` model-language cells
  - `fir_4lag`, `no_acoustic_nuisance`, `no_pitch_nuisance`, and `previous_2_sentence_context` preserve the sign in `6/6` cells
  - `last_token_pooling` preserves the sign in `5/6` cells, with only a near-zero XLM-R / `zh` miss (`-0.001`)
  - `voxelwise_within_roi_mean_z` preserves the sign in `4/6` cells and turns negative in French for both model families (`XLM-R = -0.029`, `NLLB encoder = -0.025`)
- Writing constraint:
  - describe robustness as supportive but bounded; do not write it as a uniformly clean success because the voxelwise French rows are genuinely negative

## Abstract and Conclusion Coverage

- Abstract sentence on the replicated semantic-family `SHARED > SPECIFIC` result:
  - supported by `outputs/stats/confirmatory_effects.parquet`
  - anchored by `outputs/tables/table03_main_confirmatory_stats.csv` and `outputs/figures/fig04_main_confirmatory_roi_families.png`
- Abstract sentence on the mixed semantic-versus-auditory result:
  - supported by the `H2_semantic_minus_auditory` rows in `outputs/stats/confirmatory_effects.parquet`
  - anchored by `outputs/tables/table03_main_confirmatory_stats.csv` and `outputs/figures/fig04_main_confirmatory_roi_families.png`
- Abstract / discussion sentence on correct-content controls:
  - supported by `outputs/stats/group_level_roi_results.parquet`
  - anchored by `outputs/tables/table04_roi_condition_stats.csv` and `outputs/figures/fig05_roi_condition_comparison.png`
- Abstract / discussion sentence on weak geometry-to-brain coupling:
  - supported by `outputs/stats/geometry_brain_coupling.parquet`
  - anchored by `outputs/figures/fig07_geometry_to_brain_coupling.png`
- Conclusion sentence on descriptive cortex-wide localization:
  - supported by `outputs/stats/roi_projected_maps/` and `outputs/figures/fig08_whole_brain_maps.png`
  - writing constraint: descriptive only, not voxelwise inferential

## 2026-03-16 - Current supported claims from the paper-complete checkpoint

- Claim: multilingual semantic-family `SHARED > SPECIFIC` replicates across both model families and all three languages.
  - Evidence: `outputs/stats/confirmatory_effects.parquet`
  - Rows: all `H1_shared_gt_specific_semantic` rows for `xlmr` and `nllb_encoder` in `en`, `fr`, and `zh`
  - Current support: yes; all six H1 rows are positive and Holm-significant in the current checkpoint
- Claim: the stronger semantic-vs-auditory dissociation is not cleanly supported and must be caveated.
  - Evidence: `outputs/stats/confirmatory_effects.parquet`
  - Rows: all `H2_semantic_minus_auditory` rows
  - Current support: mixed; no H2 row survives Holm correction
- Claim: control behavior remains healthy in the model-specific readouts.
  - Evidence: `outputs/logs/xlmr_encoding_qc_report__xlmr_fast_readout.md`, `outputs/logs/nllb_encoding_qc_report__nllb_fast_readout.md`
  - Current support: yes for the fast-paper checkpoint; `RAW > acoustic_only` and `SHARED > MISMATCHED_SHARED` are reported
- Claim: geometry-to-brain coupling is not yet a supported headline result.
  - Evidence: `outputs/stats/geometry_brain_coupling.parquet`
  - Current support: no strong evidence in the current checkpoint; observed correlations are weak or non-significant

Current caveat:
- the current evidence map is still based on the fast-paper subject-level checkpoint built from `xlmr_fast_readout` and `nllb_fast_readout`
- the completed robustness suite strengthens manuscript completeness, but it does not erase the already logged fast-paper implementation deviations

