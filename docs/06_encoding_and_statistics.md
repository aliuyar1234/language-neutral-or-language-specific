# 06 — Encoding and Statistics

This document defines the exact encoding pipeline and statistics.

## 1. Data split policy

Use **leave-one-run-out** cross-validation.

For LPPC:
- 9 outer folds
- each fold leaves one canonical run out

Never use:
- random timepoint splits
- random sentence splits
- time-shuffled CV

## 2. Target variables

### Subject inclusion rule
For the main confirmatory LPPC analysis, include only subjects with:
- all 9 canonical runs present
- exact preprocessed scan counts after canonical remapping by default
- at most isolated documented plus_minus_one volume exceptions
- usable ROI targets for all required runs

Exclude incomplete subjects from confirmatory analyses rather than creating a complex variable-fold setup.
Document every exclusion.


### Primary target
For each ROI and subject:
- extract voxel time series from the resampled ROI mask
- z-score each voxel time series within run
- average voxels within ROI
- concatenate runs after keeping run ids

This yields one target time series per ROI per subject.

### Secondary target
Voxelwise-within-ROI or whole-brain voxelwise targets are allowed only after the ROI pipeline is stable.

## 3. Nuisance matrix \(Z\)

Required nuisance terms:
- run intercepts
- run-wise linear trends
- high-pass cosine basis (128 s)
- RMS envelope
- f0 / pitch
- word rate
- sentence onset impulses
- EN/CN picture-event and picture-block regressors for run 1
- motion regressors if they are available in shipped derivatives, confounds, or metadata

If motion regressors are unavailable:
- omit them for the main paper
- log the omission explicitly
- do not recompute motion from raw MRI for the main paper


### 3.1 Acoustic-only baseline definition

The acoustic-only baseline is a **separate primary design**, not just the nuisance matrix.

Use as baseline predictors:
- RMS envelope
- f0 / pitch
- word rate
- sentence onset impulses
- picture-event / picture-block regressors for EN/CN run 1 when relevant

Still treat the following as nuisance for this baseline:
- run intercepts
- run-wise linear trends
- high-pass basis
- motion regressors if available

This baseline answers whether language-model-derived representations add value beyond low-level acoustics and timing.

## 4. Residualization

Use fold-local Frisch–Waugh–Lovell residualization.

### Training fold
\[
Y'_{\mathrm{train}} = Y_{\mathrm{train}} - P_{Z_{\mathrm{train}}}Y_{\mathrm{train}}
\]
\[
X'_{\mathrm{train}} = X_{\mathrm{train}} - P_{Z_{\mathrm{train}}}X_{\mathrm{train}}
\]

where
\[
P_{Z} = Z(Z^\top Z)^{-1}Z^\top
\]

### Test fold
Project test \(X\) and \(Y\) using nuisance weights learned from the training fold only.

This is non-negotiable.  
Never residualize with the full dataset.

## 5. Standardization and PCA

Within each fold:

1. standardize feature columns using training statistics only
2. fit PCA on training features only
3. transform training and test with the train-fit PCA

Default PCA retention:
- keep at most 128 PCs
- or enough PCs to explain 95% variance
- use the smaller choice

Store explained variance curves.

## 6. Ridge regression

For each condition / layer / ROI / subject:
\[
\hat B_\lambda = \arg\min_B \|Y - XB\|_2^2 + \lambda \|B\|_2^2
\]

Use a log-spaced alpha grid:
\[
\lambda \in 10^{[-2,6]}
\]
with 15 values.

Choose alpha on training data only.

Exact alpha-selection rule:
- use inner leave-one-run-out CV on the 8 outer-training runs
- refit the full pipeline inside each inner split:
  - nuisance residualization on inner-train only
  - standardization on inner-train only
  - PCA on inner-train only
  - ridge on inner-train only
- score candidate alphas by mean Fisher-z transformed Pearson `r` across inner held-out runs
- break ties by choosing the larger alpha

## 7. Metrics

### Primary metric
Pearson correlation on held-out data:
\[
r = \mathrm{corr}(\hat y, y)
\]

### Fisher transform
Before averaging across folds or subjects:
\[
z = \frac{1}{2}\ln\frac{1+r}{1-r}
\]

Exact outer-fold aggregation rule:
- compute fold-level Pearson `r`
- Fisher transform each fold-level `r`
- average the 9 fold `z` values to obtain the primary subject-level summary
- invert with `tanh` only when an `r` summary is explicitly needed

### Secondary metric
\[
R^2 = 1 - \frac{\sum_t (y_t - \hat y_t)^2}{\sum_t (y_t - \bar y)^2}
\]

For `R^2` only:
- concatenate the 9 held-out predictions in chronological order
- compute one subject-level `R^2` from the concatenated held-out predictions and targets
- do not average `R^2` foldwise

### Incremental contribution
\[
\Delta R^2_{\mathrm{specific}|shared} = R^2_{FULL} - R^2_{SHARED}
\]
\[
\Delta R^2_{\mathrm{shared}|specific} = R^2_{FULL} - R^2_{SPECIFIC}
\]

## 8. Confirmatory tests

### H1
For each model and language, test:
\[
\Delta^{mid}_{semantic} > 0
\]

### H2
For each model and language, test:
\[
\Delta^{mid}_{semantic} - \Delta^{mid}_{auditory} > 0
\]

Use sign-flip permutation over subject-level summary effects.

Default:
- 10,000 permutations

Report:
- mean
- SE
- Cohen’s \(d_z\)
- one-sided `p_perm`
- `p_holm_primary` across all 12 primary tests
- bootstrap 95% CI

Primary multiplicity rule:
- treat each `model x language x hypothesis` slice as its own confirmatory test
- that yields 12 primary tests total
- cross-language pooled summaries remain descriptive only

## 9. Secondary tests

These are descriptive / exploratory:
- per-ROI per-layer SHARED vs SPECIFIC
- RAW vs acoustic-only
- SHARED vs MISMATCHED_SHARED
- FULL vs SHARED
- FULL vs SPECIFIC
- hemisphere contrasts
- geometry-to-brain coupling

Apply FDR within sensible families:
- per model, per language, across ROI × layer comparisons

## 10. Meta-analytic summary across languages

Because LPPC languages come from different sites/scanners, pooled language summaries should be descriptive and cautious.

Recommended summary:
- compute language-specific means and SEs
- show them separately
- optionally report inverse-variance weighted pooled mean effect

Do not interpret pooled magnitude as a scanner-free truth.

## 11. Bootstrap confidence intervals

For subject-level group summaries, compute bootstrap CIs:
- resample subjects with replacement within language
- 10,000 bootstrap samples
- BCa or percentile intervals are acceptable

## 12. Geometry-to-brain coupling statistics

For each model and language:
- compute \(CAS_l\) and \(B_l\) across layers
- use Spearman correlation
- permute layer order 10,000 times for a p-value

This is a secondary analysis and does not need to be oversold.

## 13. Robustness analyses

Run after the core result exists.
Robustness is mandatory for the paper-complete target and optional only for a narrower core-analysis-complete milestone.

### Required robustness set
1. mean pooling vs last-token pooling
2. canonical HRF vs 4-lag FIR
3. with vs without acoustic nuisance
4. with vs without pitch nuisance
5. sentence-only input vs previous-2-sentence context
6. ROI-mean target vs voxelwise-within-ROI mean-z target

### Optional robustness set
7. alternative PCA caps
8. alternative atlas thresholds
9. exclude top-motion or lowest-quality subjects if motion or QC measures exist

## 14. Representative statistical output tables

### Table A — subject-level summary effects
Columns:
- model
- language
- ROI_family
- mean_delta_mid
- se
- dz
- p_perm
- ci_low
- ci_high

### Table B — per-ROI per-layer results
Columns:
- model
- language
- roi
- layer_index
- layer_depth
- z_shared_mean
- z_specific_mean
- z_raw_mean
- z_mismatched_mean
- delta_shared_specific
- p_perm
- q_fdr

### Table C — geometry-to-brain coupling
Columns:
- model
- language
- rho_spearman
- p_perm

## 15. QC failure conditions

Do not write the paper if any of these are true:
- same-sentence cross-language similarity does not beat mismatched similarity
- `SHARED` does not beat `MISMATCHED_SHARED` anywhere
- `RAW` fails to beat acoustic-only in language ROIs
- results hinge on one single subject
- fold leakage is detected

## 16. Claim grounding rule

Before manuscript writing, create:
`outputs/manuscript/claim_evidence_map.md`

Every main claim must point to:
- a table row or figure panel
- the exact result file that generated it

If a claim cannot be mapped, it does not belong in the paper.
