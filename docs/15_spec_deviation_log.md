# Spec Deviation Log

This is the public, paper-facing copy of the implementation deviation record for the current checkpoint.

## 2026-03-16 - Current status

- The current paper-complete checkpoint does include formally logged implementation deviations.
- The frozen canonical stats bundle still inherits the runtime-optimized fast-paper subject-level path documented below.
- The scientific aim is unchanged; these deviations affect implementation scope and interpretive caution, not the thesis of the paper.

## 2026-03-15 - Current canonical T13 implementation

- The current canonical `T13` outputs were generated from the merged fast-paper subject tables:
  - `outputs/stats/xlmr_subject_level_roi_results__xlmr_fast_readout.parquet`
  - `outputs/stats/nllb_subject_level_roi_results__nllb_fast_readout.parquet`
- This means the present checkpoint inherits the fast-path run definition:
  - `--mismatched-shuffles 1` for the model-specific ROI runs
  - NLLB restricted to layers `4,6,8`
- Canonical aggregation was run directly with `python -m brain_subspace_paper build-paper-stats`, which applies `10000` permutations and `10000` bootstraps at the paper-level stats stage instead of requiring separate `xlmr_final` and `nllb_final` merge tags first.
- Scientific aim unchanged. This is an implementation shortcut for the current checkpoint and should be revisited during manuscript finalization if fuller reruns are required.

## 2026-03-15 - Canonical fast-paper T13 checkpoint

- `outputs/stats/subject_level_roi_results.parquet`,
  `outputs/stats/group_level_roi_results.parquet`,
  `outputs/stats/confirmatory_effects.parquet`,
  `outputs/stats/geometry_metrics.parquet`, and
  `outputs/stats/geometry_brain_coupling.parquet` were built from the fast-paper merged subject-level readouts rather than a separately frozen final run definition.
- Current inputs:
  - `xlmr_fast_readout`: all `13` layers, `mismatched_shuffles=1`
  - `nllb_fast_readout`: layers `4,6,8`, `mismatched_shuffles=1`
- Rationale:
  - establish the combined confirmatory checkpoint and unblock T14 implementation without rerunning expensive subject-level encoding
- Consequence:
  - regenerate the canonical paper stats if the manuscript-final run definition changes, especially if NLLB layer coverage or mismatch-shuffle counts are expanded

## 2026-03-15 - Figure 08 fallback for the current paper checkpoint

- `Figure 08` is currently implemented as an ROI-projected descriptive cortex panel rather than a voxelwise whole-brain encoding analysis.
- Current generated assets:
  - `outputs/figures/fig08_whole_brain_maps.png`
  - `outputs/stats/roi_projected_maps/`
- Rationale:
  - the exact representative-layer voxelwise path is substantially more expensive and was deferred in favor of a fast, honest descriptive visualization for the present manuscript-quality checkpoint
- Scientific constraint:
  - this figure must be described as an atlas projection of ROI group means, not as a voxelwise statistical map
- Scientific aim unchanged:
  - the confirmatory claims remain ROI-first and are anchored by `outputs/stats/confirmatory_effects.parquet`, `outputs/tables/table03_main_confirmatory_stats.csv`, and `outputs/figures/fig04_main_confirmatory_roi_families.png`
