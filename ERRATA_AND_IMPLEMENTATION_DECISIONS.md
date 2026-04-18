# ERRATA AND IMPLEMENTATION DECISIONS

This file resolves ambiguities discovered during implementation review. It is part of the SSOT handoff.

## Scope Of This File

This file does not change the thesis, dataset, core models, or main hypotheses. It removes ambiguity from implementation defaults, statistics, storage contracts, and completion criteria.

## 1. Final Precedence And Read Order

### Final precedence
Use this precedence order:

1. `SSOT.md` for thesis, scientific scope, and core formulas
2. `ERRATA_AND_IMPLEMENTATION_DECISIONS.md` for resolved ambiguities and exact defaults
3. `docs/*.md`
4. `configs/*.yaml`

If a detailed doc conflicts with this file on a resolved ambiguity, this file wins. If this file appears to conflict with thesis-level content in `SSOT.md`, preserve `SSOT.md`.

### Final read order
Read files in this order before implementation:

1. `README.md`
2. `SSOT.md`
3. `ERRATA_AND_IMPLEMENTATION_DECISIONS.md`
4. `docs/02_datasets_and_downloads.md`
5. `docs/03_models_and_extraction.md`
6. `docs/04_alignment_algorithm.md`
7. `docs/05_novel_method_and_formulas.md`
8. `docs/06_encoding_and_statistics.md`
9. `docs/07_figures_tables_and_outputs.md`
10. `docs/10_data_contracts.md`
11. `configs/*.yaml`

## 2. Final Decisions

Use:
- `1a+`
- `2a`
- `3a`
- `4a+`
- `5a*`

Where:
- `1a+` = option 1a, strengthened with a local tri-lingual reconciliation pass
- `4a+` = option 4a, plus mandatory Holm-adjusted transparency p-values across all 12 primary tests
- `5a*` = option 5a, except robustness is mandatory for paper-complete, not optional

## 3. Alignment Conflict Resolution Rule

### Decision
Use `1a+`, not plain `1a`.

Do not let EN-FR win.
Do not let EN-ZH win.
Do not keep incompatible English blockings.
Do not duplicate target-language spans just to force a merge.

### Exact rule
1. Run pairwise monotonic dynamic programs for EN-FR and EN-ZH.
2. If the implied English block boundaries match, merge directly into triplets.
3. If they conflict, isolate the minimal conflicting English window whose left and right edges are common boundaries in both pairwise paths.
4. Extract the corresponding local EN, FR, and ZH sentence windows.
5. Re-run alignment on that local window with a tri-lingual monotonic dynamic program using candidate span lengths `{1, 2}` in each language.

### Tri-lingual local cost

```text
C_tri(i,p,j,q,k,r)
= 1.0 * C_EN_FR(i,p,j,q)
+ 1.0 * C_EN_ZH(i,p,k,r)
+ 0.5 * C_FR_ZH(j,q,k,r)
```

### Tri-lingual recurrence

```text
D(i,j,k)
= min over p,q,r in {1,2} of
  D(i-p, j-q, k-r)
  + C_tri(i-p+1, p, j-q+1, q, k-r+1, r)
```

Base case:

```text
D(0,0,0) = 0
```

Treat invalid states as `+inf`.

### Conflict retry rule
If no feasible local tri-lingual path exists under the current minimal window:
- expand the window by one sentence on each side where possible
- retry once

If it still fails:
- mark the affected local region as `manual_status = needs_fix`
- exclude only those unresolved rows from the confirmatory analysis
- log the event in `outputs/logs/alignment_qc_report.md`
- if unresolved rows exceed 1% of all triplets, stop and inspect before continuing

### Required outcome
The final `alignment_triplets.parquet` must be a single globally consistent triplet table.

## 4. Hidden-State Layer Indexing And Normalized Depth

### Decision
Use `2a`.

### Exact rule
- `hidden_states[0]` is the embedding layer output and counts as `layer_index = 0`
- `hidden_states[1]` is the output of transformer block 1
- continue upward through the final returned hidden state

If a model returns `N` hidden states total, valid layer indices are `0..N-1`.

### Normalized depth

```text
delta_l = l / (N - 1)
```

where `N` is the number of returned hidden states for that model and `l` is the zero-based `layer_index`.

This normalized depth definition is used everywhere:
- middle-late layer sets
- cross-model layer plots
- figure x-axes
- group summaries by depth

## 5. Alpha Selection And Fold-Score Aggregation

### Decision
Use `3a`, with explicit `R^2` handling.

### Outer CV
- outer CV is leave-one-canonical-run-out across the 9 canonical runs
- there are exactly 9 outer folds for each included subject

### Inner CV for alpha selection
Within each outer fold:
1. restrict to the 8 training runs
2. run inner leave-one-run-out CV on those training runs
3. for each alpha candidate, refit the full inner pipeline:
   - nuisance residualization fit on inner-train only
   - feature standardization fit on inner-train only
   - PCA fit on inner-train only
   - ridge fit on inner-train only
4. score each alpha by mean Fisher-z transformed Pearson `r` across inner held-out runs
5. choose the alpha with the highest mean inner Fisher-z
6. break ties by choosing the larger alpha

### Outer held-out scoring
For each outer fold, after choosing alpha:
1. refit on the full outer-training data
2. predict the held-out run
3. compute fold-level Pearson `r`
4. transform to Fisher `z`

### Subject-level aggregation for primary analyses

```text
z_subject = (1 / 9) * sum_{f=1..9} z_f
r_subject = tanh(z_subject)
```

The confirmatory H1/H2 analyses use the `z`-based subject summaries. Only invert with `tanh` when an `r` summary is needed.

### Secondary metric aggregation
For `R^2` only:
- concatenate the 9 held-out predictions in chronological order across runs
- compute one subject-level `R^2` from the concatenated held-out predictions and targets
- do not average `R^2` foldwise

## 6. Primary Multiplicity Policy

### Decision
Use `4a+`.

### Primary inferential unit
Treat each `model x language x hypothesis` slice as its own confirmatory test:
- 2 models x 3 languages x 2 hypotheses = 12 primary tests
- H1 and H2 are distinct hypotheses
- cross-language pooled summaries remain descriptive only

### Required reporting
For each of the 12 primary tests, report:
- unadjusted one-sided sign-flip permutation `p_perm`
- effect size and CI
- Holm-adjusted `p_holm_primary` across all 12 primary tests

### Interpretation rule
- main narrative should emphasize effect size, confidence intervals, sign consistency, and replication across model families
- the paper may discuss both unadjusted slice-wise confirmatory results and Holm-adjusted transparency results
- do not base the paper on a pooled cross-language significance claim

## 7. Inclusion, Scan-Count, Motion, Acoustic, And Picture-Event Rules

### 7.1 Main-analysis inclusion rule
Main confirmatory inclusion requires:
- all 9 canonical runs present
- all required annotation sections present
- usable ROI targets for all 9 canonical runs
- run counts matching the language template exactly, except for a maximum deviation of `+/-1` volume only when the deviation is isolated, explicitly documented, and logged

### 7.2 Replace vague scan-count wording
Whenever earlier docs say `plausible` or `approximately`, interpret them as:
- exact match to the canonical language-specific scan-count template by default
- `+/-1` volume allowed only if the subject still has all 9 runs, the mismatch is isolated, and the reason is documented in `outputs/logs/data_integrity_report.md`
- any larger or unexplained mismatch excludes the subject from the main confirmatory analysis

### 7.3 Motion rule
Main-paper motion regressors may come only from:
- shipped derivative confounds or confound TSVs
- derivative metadata that provide per-volume motion quantities

Do not estimate motion regressors from raw MRI or by re-running preprocessing for the main paper.

If per-volume motion is unavailable for a subject:
- omit motion regressors for that subject
- set a manifest flag such as `motion_available = false`
- log the omission in `outputs/logs/data_integrity_report.md`

### 7.4 Acoustic feature fallback
Use this priority order for acoustic predictors:
1. shipped `*_prosody.csv` or equivalent provided acoustic summaries
2. if missing or corrupt for a required run, recompute RMS and `f0` from the publicly available audio stimulus for that run
3. if neither provided summaries nor public audio are available, stop and inspect rather than silently omitting acoustic predictors

### 7.5 Word rate and sentence onsets
- derive word rate from `TextGrid` or word timing annotations if not already tabulated
- derive sentence onset impulses from the final sentence-span tables
- both are required for the core baseline

### 7.6 Picture-event fallback
For EN and ZH canonical run 1:
1. use shipped event files if present
2. otherwise reconstruct picture regressors from the fixed timings below

Published picture timings remain:
- onsets: `10 s`, `35 s`, `60 s`
- durations: `15 s`, `20 s`, `15 s`

## 8. Output-Contract Normalization

### 8.1 Sentence spans are mandatory
The per-language sentence-span tables are mandatory core artifacts, not optional intermediates:
- `data/interim/sentence_spans_en.parquet`
- `data/interim/sentence_spans_fr.parquet`
- `data/interim/sentence_spans_zh.parquet`

### 8.2 Confirmatory effects storage shape
Store H1 and H2 in the same canonical file:
- `outputs/stats/confirmatory_effects.parquet`

Required columns:
- `hypothesis` with values:
  - `H1_shared_gt_specific_semantic`
  - `H2_semantic_minus_auditory`
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

For H1:
- `roi_family = semantic`

For H2:
- `roi_family = semantic_minus_auditory`

### 8.3 Figure 8 whole-brain artifact contract
Figure 8 is descriptive but still requires saved machine-readable artifacts.

For each `model x language x selected_layer`, save:
- `outputs/stats/whole_brain/{model}/{language}/layer_{selected_layer:02d}/shared_mean_z.nii.gz`
- `outputs/stats/whole_brain/{model}/{language}/layer_{selected_layer:02d}/specific_mean_z.nii.gz`
- `outputs/stats/whole_brain/{model}/{language}/layer_{selected_layer:02d}/shared_minus_specific_mean_z.nii.gz`
- `outputs/stats/whole_brain/{model}/{language}/layer_{selected_layer:02d}/brain_mask.nii.gz`
- `outputs/stats/whole_brain/{model}/{language}/layer_{selected_layer:02d}/manifest.json`

The `manifest.json` must include:
- source subject list
- selected layer index
- selected layer normalized depth
- exact generating script path
- input feature files
- whether the maps are mean-z, t, or another descriptive statistic

### 8.4 Selected layer rule for Figure 8
For each model and language, choose the representative layer as:
- the layer with the maximum semantic-family group mean `SHARED - SPECIFIC`
- if there is a tie, choose the shallower layer

### 8.5 Provenance files are mandatory
The following are mandatory for a paper-complete deliverable:
- `outputs/manuscript/figure_provenance.md`
- `outputs/manuscript/table_provenance.md`
- `outputs/manuscript/claim_evidence_map.md`

## 9. Robustness And Completion Policy

### Decision
For a complete research paper, robustness is mandatory.

### Milestones
Use two milestones:

#### Core-analysis complete
This means:
- all core data, alignment, feature extraction, ROI stats, Figures 1-8, and Tables 1-4 exist
- no manuscript-final claim yet depends on robustness

#### Paper-complete
This is the actual completion criterion for this project.
It additionally requires the robustness suite and manuscript grounding.

### Required robustness set for paper-complete
Run the required robustness checks already named in `docs/06_encoding_and_statistics.md`:
1. mean pooling vs last-token pooling
2. canonical HRF vs 4-lag FIR
3. with vs without acoustic nuisance baseline comparison context
4. with vs without pitch nuisance
5. sentence-only vs previous-2-sentence context
6. ROI-mean target vs voxelwise-within-ROI mean-z target

These feed:
- `outputs/tables/table05_robustness_summary.csv`
- optional `outputs/figures/fig09_robustness_summary.png`

Figure 9 remains optional. Table 5 is mandatory for paper-complete.

## 10. Scaffold Authority Rule

The scaffold is intentionally non-runnable and non-authoritative.
Treat `repo_scaffold/` as a layout suggestion only.
If any scaffold file conflicts with:
- data contracts
- output paths
- formulas
- statistical rules
- done definition

then the scaffold loses immediately. Implementation must satisfy the docs and data contracts even if that means replacing most of the scaffold.
