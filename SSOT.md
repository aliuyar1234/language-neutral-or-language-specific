# SSOT.md

This file is the canonical research specification for the project.

## 1. Canonical thesis

**Working title**  
**Language-Neutral or Language-Specific? Disentangling Multilingual LLM Subspaces with Naturalistic fMRI**

**Core claim**  
During naturalistic story listening, higher-level language regions are better predicted by a **leave-target-out shared cross-lingual representation** than by a **target-language-specific residual**, while earlier auditory and form-heavier regions retain explanatory power for the specific residual.

This project is a **mechanistic multilingual brain-encoding paper**, not a benchmark paper.

---

## 2. Core scope

### 2.1 Core paper
Use only:

- **LPPC-fMRI / OpenNeuro ds003643**
- languages: **English, French, Chinese**
- models:
  - `FacebookAI/xlm-roberta-base`
  - `facebook/nllb-200-distilled-600M` (**encoder only**)
- alignment QC model:
  - `sentence-transformers/LaBSE`
- analysis granularity:
  - **sentence-level**
  - **ROI-first**
- required feature sets:
  - `RAW`
  - `SHARED`
  - `SPECIFIC`
  - `FULL`
  - `MISMATCHED_SHARED`
- required controls:
  - acoustic/nuisance-only
  - `RAW`
  - `MISMATCHED_SHARED`
  - `FULL`

### 2.2 Optional appendix only after core success
- German studyforrest extension (`ds000113`)

The German extension is **not** part of the core paper and must not drive the main claim.

---

## 3. The exact scientific question

For a target language listener, if we build a sentence representation from the **other languages only**, does that shared representation explain the target-language listener’s fMRI responses better than the target-language-specific residual?

Concrete example:

- target: French listener
- sentence content: aligned triplet across EN/FR/ZH
- `SHARED` for French = average of English and Chinese sentence embeddings for the same aligned sentence content
- `SPECIFIC` for French = French embedding minus the shared component, orthogonalized to shared
- regress both onto the French listener’s BOLD response using the French sentence timings

This is the essential experiment.

---

## 4. Why this topic won

This topic was chosen because it satisfies all of the following:

1. **Novel enough**
   - multilingual model geometry and multilingual naturalistic fMRI each exist
   - their direct mechanistic connection via shared-vs-specific factorization is still underexplored

2. **Interesting enough**
   - it asks a real neuroscience question
   - it is not a benchmark-only project

3. **Feasible enough**
   - no large-scale finetuning
   - only feature extraction + linear encoding
   - suitable for one strong local GPU

4. **Freely reproducible**
   - the core dataset is openly downloadable
   - the models are publicly downloadable

---

## 5. Non-negotiables

1. Do not turn the paper into “which model gets the highest brain score?”
2. Do not add many extra models to look comprehensive.
3. Do not skip `MISMATCHED_SHARED`.
4. Do not compare absolute encoding magnitudes across languages as if site/scanner differences were irrelevant.
5. Do not use German as the main experiment.
6. Do not claim a syntax/semantics dissociation unless you directly test syntax separately.
7. Do not use the NLLB decoder.
8. Do not rely on hidden chat context; this file is the authority.

---

## 6. Dataset choice and free access

### 6.1 Core dataset
**Le Petit Prince multilingual naturalistic fMRI corpus (LPPC-fMRI)**  
OpenNeuro accession: `ds003643`

Reasons:
- same story in multiple languages
- released publicly
- includes preprocessed derivatives
- includes annotations in `annotation/`
- includes multilingual participants:
  - English: 49
  - Chinese: 35
  - French: 28

### 6.2 Optional appendix dataset
**studyforrest / Forrest Gump**  
OpenNeuro accession: `ds000113`

Use only after the LPPC paper works.

### 6.3 Free-access rule
All required core data and models must be freely downloadable. No paid APIs. No private datasets.

---

## 7. LPPC implementation facts that are easy to miss

These facts are critical and must be honored.

### 7.1 LPPC includes derivatives
The LPPC GitHub / OpenNeuro dataset contains a `derivatives/` directory with preprocessed BOLD volumes.

### 7.2 Derivative files can use non-consecutive raw run labels
Example: a subject can have derivative files named with runs `15` to `23` rather than `1` to `9`.

**Rule:** never assume the filename run numbers are the canonical section indices.  
Instead:
- discover all derivative run files per subject
- sort by original run number
- map them to canonical run indices `1..9` in sorted order
- save this mapping in a run manifest

### 7.3 Preprocessed scans start at audio onset
For the preprocessed LPPC derivatives, use **audio onset as time 0** for sentence timings.

The dataset README indicates:
- Chinese preprocessed scans match the audio length
- English preprocessed scans keep only the extra scans **after** audio offset
- French preprocessed scans include extra scans **after** audio offset

Therefore, for **preprocessed** BOLD:
- treat sentence timings as starting at time 0 = audio onset
- handle any post-offset extra scans by leaving the design matrix as zeros after the story ends

### 7.4 Expected LPPC preprocessed scan counts by language and canonical run
These numbers are from the dataset README and should be used as a sanity check after canonical run remapping.

**Chinese preprocessed scans**
- run1: 283
- run2: 322
- run3: 322
- run4: 307
- run5: 293
- run6: 392
- run7: 364
- run8: 293
- run9: 401

**English preprocessed scans**
- run1: 282
- run2: 298
- run3: 340
- run4: 303
- run5: 265
- run6: 343
- run7: 325
- run8: 292
- run9: 368

**French preprocessed scans**
- run1: 309
- run2: 326
- run3: 354
- run4: 315
- run5: 293
- run6: 378
- run7: 332
- run8: 294
- run9: 336

If discovered run manifests do not match these counts exactly by default, stop and inspect.
An isolated difference of 1 volume is allowed only when a documented dataset quirk explains it.


### 7.4b Canonical inclusion rule
Main analysis subject inclusion requires:
- all 9 canonical runs present in derivatives
- all required annotation sections present
- run volume counts matching the language template exactly or within a tolerance of 1 volume when a documented dataset quirk explains the difference

If a subject is missing runs or has unexplained scan-count mismatches:
- exclude that subject from the **main confirmatory analysis**
- document the exclusion in `outputs/logs/data_integrity_report.md`
- optionally include them in sensitivity analyses later

This keeps the core paper simple and avoids variable-fold cross-validation edge cases.

### 7.5 Visual picture events in section 1 for EN/CN
For English and Chinese, the dataset paper reports visual picture cues in the first section and recommends picture-event / picture-block regressors.

**Rule:** include picture nuisance regressors for EN/CN canonical run 1 if the event files are present.
If they are not present, reconstruct them from the published timings below.

Published timings:
- picture onsets: 10 s, 35 s, 60 s
- picture blocks with durations: 15 s, 20 s, 15 s

These belong in the nuisance matrix for run 1 of EN/CN only.


### 7.5b Random seed
Use a fixed project seed for all deterministic procedures involving randomization:
- default seed: `20260314`

Use this for:
- mismatched-shared shuffles
- bootstrap resampling
- any other randomized QC or plotting subsampling

---

## 8. Sentence unit and timing rules

### 8.1 Primary unit
The primary unit is the **sentence**, not the word.

Why:
- cross-language alignment is much cleaner at the sentence level
- fMRI is slow enough for sentence-level regressors
- the shared/specific factorization is easier to define and interpret

### 8.2 Timing rule
For each target language sentence span:
- onset = onset of the first word in that sentence span
- offset = offset of the last word in that sentence span

Use the target language’s own timings when building regressors, even for `SHARED` features.

This is crucial:

> `SHARED` features are content derived from the other languages, but they must be placed on the **target language listener’s timeline**.

### 8.3 Merge rule
If alignment requires merging 2 adjacent sentences into 1 span in a language:
- concatenate the raw sentence texts with a single separator space
- onset = onset of the first sentence
- offset = offset of the last sentence

Do not allow spans larger than 2 sentences in the core pipeline.

---

## 9. Canonical representation formulas

Let:

- \(m\) = model
- \(l\) = layer
- \(s\) = aligned triplet id
- \(\ell \in \{en, fr, zh\}\) = target language
- \(L = 3\) = number of languages
- \(h_{s,\ell}^{(m,l)} \in \mathbb{R}^d\) = pooled sentence embedding for target language \(\ell\)

### 9.1 L2 normalization
\[
\tilde h_{s,\ell}^{(m,l)} = \frac{h_{s,\ell}^{(m,l)}}{\|h_{s,\ell}^{(m,l)}\|_2 + \epsilon}
\]

Use \(\epsilon = 10^{-8}\).

### 9.2 Leave-target-out shared component
\[
u_{s,\ell}^{(m,l)} = \frac{1}{L - 1}\sum_{j \neq \ell} \tilde h_{s,j}^{(m,l)}
\]

Interpretation:
- a cross-lingual shared representation built **without using the target language**

### 9.3 Raw residual
\[
v_{s,\ell}^{(m,l)} = \tilde h_{s,\ell}^{(m,l)} - u_{s,\ell}^{(m,l)}
\]

### 9.4 Orthogonalized specific residual
\[
v_{s,\ell}^{\perp (m,l)} =
v_{s,\ell}^{(m,l)}
-
\frac{\left(u_{s,\ell}^{(m,l)}\right)^\top v_{s,\ell}^{(m,l)}}{\|u_{s,\ell}^{(m,l)}\|_2^2 + \epsilon}
u_{s,\ell}^{(m,l)}
\]

This is the canonical `SPECIFIC` feature.

### 9.5 Feature sets
- `RAW` = \( \tilde h_{s,\ell}^{(m,l)} \)
- `SHARED` = \( u_{s,\ell}^{(m,l)} \)
- `SPECIFIC` = \( v_{s,\ell}^{\perp (m,l)} \)
- `FULL` = concatenation \( [u_{s,\ell}^{(m,l)} ; v_{s,\ell}^{\perp (m,l)}] \)
- `MISMATCHED_SHARED` = same construction as `SHARED`, but aligned triplet ids are shuffled within run before feature placement

### 9.6 Mismatched-shared rule
`MISMATCHED_SHARED` must be generated by shuffling triplet ids **within each canonical run** of the target language.

Default:
- use `K = 5` independent run-local shuffles
- compute encoding scores for each shuffle
- average the final control score across the 5 shuffles

This reduces Monte Carlo noise.

---

## 10. Non-obvious text-space metrics

These metrics are required because they connect representation geometry to brain effects.

### 10.1 Same-sentence cross-language cosine
\[
\mathrm{Align}_{l}^{(m)}
=
\mathbb{E}_{s,\ell \neq \ell'}
\left[
\cos\left(\tilde h_{s,\ell}^{(m,l)}, \tilde h_{s,\ell'}^{(m,l)}\right)
\right]
\]

### 10.2 Contrastive alignment score
\[
\mathrm{CAS}_{l}^{(m)}
=
\mathbb{E}\left[\cos(\tilde h_{s,\ell}^{(m,l)}, \tilde h_{s,\ell'}^{(m,l)})\right]
-
\mathbb{E}\left[\cos(\tilde h_{s,\ell}^{(m,l)}, \tilde h_{s',\ell'}^{(m,l)})\right], \quad s' \neq s
\]

### 10.3 Retrieval accuracy
For each pair of languages, retrieve the correct aligned sentence in the other language using cosine similarity.

\[
R@1_{l}^{(m)} =
\frac{1}{S}
\sum_{s=1}^{S}
\mathbf{1}
\left[
s = \arg\max_{s'}
\cos\left(\tilde h_{s,\ell}^{(m,l)}, \tilde h_{s',\ell'}^{(m,l)}\right)
\right]
\]

Use pairwise retrieval for EN-FR, EN-ZH, and FR-ZH, then average.

### 10.4 Specificity energy
\[
E_{l}^{(m)} =
\frac{\mathbb{E}\|v^\perp\|_2^2}
{\mathbb{E}\|u\|_2^2 + \mathbb{E}\|v^\perp\|_2^2}
\]

---

## 11. Design-matrix construction

### 11.1 Fine-grid sentence boxcar
Let sentence \(s\) have target-language onset \(a_{s,\ell}\) and offset \(b_{s,\ell}\). Let \(f_{s,j}\) be feature dimension \(j\).

On a fine temporal grid:
\[
x_j(t) = \sum_s f_{s,j} \mathbf{1}[a_{s,\ell} \le t < b_{s,\ell}]
\]

### 11.2 HRF convolution
Let \(g(t)\) be the canonical HRF.

\[
\tilde x_j(t) = (x_j * g)(t)
\]

### 11.3 Downsample to TR
At scan times \(t_n\):
\[
X_{n,j} = \tilde x_j(t_n)
\]

Canonical defaults:
- fine grid = 10 Hz
- TR = 2 s
- HRF = Glover or SPM-style canonical HRF implemented by Nilearn

### 11.4 Important placement rule
When building `SHARED`, place the vector \(u_{s,\ell}\) on the **target language** sentence timeline \((a_{s,\ell}, b_{s,\ell})\).

Do **not** attempt to use other-language audio timings for the target subject.

---

## 12. Nuisance matrix

The canonical nuisance matrix \(Z\) should contain:

1. run intercepts
2. run-wise linear trends
3. high-pass cosine basis, cutoff 128 s
4. RMS acoustic envelope
5. pitch / f0
6. word rate
7. sentence onset impulses
8. picture-event / picture-block regressors for EN/CN run 1 when available
9. motion regressors **if available in the shipped derivatives, confounds, or metadata**

### Motion fallback rule
If motion regressors are not available from the dataset derivatives or metadata, do **not** re-preprocess the raw MRI for the main paper.  
Proceed with the nuisance matrix without motion regressors and state this explicitly in limitations and robustness notes.

This is a deliberate scope decision to keep the core paper tractable and reproducible.

### Acoustic fallback rule
For RMS and pitch:
- use shipped prosody summaries first
- if they are missing or corrupt, recompute them from public audio
- if neither source exists, stop and inspect rather than silently dropping the acoustic baseline

---

## 13. Residualization and leakage control

Use Frisch–Waugh–Lovell residualization **inside each training fold**.

For a training fold:
\[
Y'_{\text{train}} = Y_{\text{train}} - Z_{\text{train}}(Z_{\text{train}}^\top Z_{\text{train}})^{-1}Z_{\text{train}}^\top Y_{\text{train}}
\]

\[
X'_{\text{train}} = X_{\text{train}} - Z_{\text{train}}(Z_{\text{train}}^\top Z_{\text{train}})^{-1}Z_{\text{train}}^\top X_{\text{train}}
\]

For test data, apply nuisance weights estimated on training data only.

Never residualize on the full dataset before cross-validation.

---

## 14. Dimensionality reduction and encoding

### 14.1 Standardization
Standardize feature columns using train-fold mean and std only.

### 14.2 PCA
Fit PCA on train features only.

Default:
- keep at most 128 PCs
- or enough PCs to explain 95% variance
- choose the smaller of the two

### 14.3 Ridge regression
For target \(Y\) and design \(X\):
\[
\hat B_\lambda = \arg\min_B \|Y - X B\|_F^2 + \lambda \|B\|_F^2
\]

Closed form:
\[
\hat B_\lambda = (X^\top X + \lambda I)^{-1}X^\top Y
\]

Default alpha grid:
\[
\lambda \in 10^{[-2,6]}
\]
with 15 log-spaced values.

Alpha selection must use training data only.

Exact alpha-selection rule:
- use inner leave-one-run-out CV on the 8 outer-training runs
- inside each inner split, refit the full pipeline:
  - nuisance residualization on inner-train only
  - standardization on inner-train only
  - PCA on inner-train only
  - ridge on inner-train only
- score alpha candidates by mean Fisher-z transformed Pearson `r` across inner held-out runs
- break ties by choosing the larger alpha

### 14.4 Cross-validation
Use **leave-one-run-out** outer CV.

For LPPC:
- 9 outer folds

Never use random timepoint splits.

Outer-fold aggregation rule:
- compute fold-level Pearson `r`
- Fisher transform each fold
- average the 9 fold `z` values to define the primary subject-level summary
- invert with `tanh` only if an `r` summary is explicitly needed

For `R^2` only:
- concatenate the 9 held-out predictions in chronological order
- compute one subject-level `R^2` from the concatenated held-out predictions and targets
- do not average `R^2` foldwise

---

## 15. ROI strategy

### 15.1 Primary atlas
Use Harvard-Oxford cortical atlas from Nilearn, resampled to the LPPC derivative space.

### 15.2 Template mismatch caveat
LPPC derivatives are in **MNIColin27** space. Harvard-Oxford is typically distributed in MNI152 space.

Operational rule:
- resample the atlas to the BOLD image space using nearest-neighbor interpolation
- keep this limitation explicit in the manuscript

### 15.3 Primary ROI families

**Semantic / higher-level family**
- left and right posterior middle temporal gyrus
- left and right angular gyrus
- left and right temporal pole
- left and right inferior frontal gyrus pars triangularis

**Auditory / form-heavier family**
- left and right Heschl’s gyrus
- left and right posterior superior temporal gyrus
- left and right anterior superior temporal gyrus

**Control family**
- left and right precentral gyrus
- left and right occipital pole

If atlas label names differ slightly, map to the nearest anatomically appropriate labels and record the mapping.

### 15.4 Primary ROI target
Use the mean BOLD timeseries of each ROI after voxelwise z-scoring within run.

Voxelwise-within-ROI analyses are secondary.

---

## 16. Primary and secondary hypotheses

### H1 (primary)
In the semantic ROI family, the shared component has positive advantage over the specific component:
\[
\Delta^{mid}_{semantic} > 0
\]

### H2 (primary)
The shared advantage is larger in the semantic ROI family than in the auditory family:
\[
\Delta^{mid}_{semantic} > \Delta^{mid}_{auditory}
\]

### H3 (secondary)
The shared advantage peaks in middle-to-late layers.

### H4 (secondary)
Layers with stronger cross-language convergence in text space show stronger shared-brain advantage.

### H5 (secondary / descriptive)
Specific residuals retain non-zero explanatory power in auditory/form-sensitive regions.

---

## 17. Canonical primary summary effect

For subject \(i\), model \(m\), language \(\ell\), ROI family \(F\), and layer set \(\mathcal L_{mid}\):

Define normalized layer depth:
\[
\delta_l = \frac{l}{N_m - 1}
\]

where \(N_m\) is the number of returned hidden states for model \(m\), including the embedding-layer output.
Use `hidden_states[0]` as the embedding-layer output with `layer_index = 0`, then continue upward through the transformer blocks.

Define the middle-late layer set:
\[
\mathcal L_{mid} = \{ l : 0.33 \le \delta_l \le 0.83 \}
\]

Let \(z^{cond}_{i,m,\ell,r,l}\) be the Fisher-z transformed held-out correlation for condition `cond`, ROI \(r\), layer \(l\).
For each subject and condition, this value is obtained by computing fold-level held-out Pearson correlations, Fisher-transforming them, and averaging those 9 fold `z` values.

Define per-layer ROI advantage:
\[
\Delta z_{i,m,\ell,r,l} =
z^{SHARED}_{i,m,\ell,r,l} - z^{SPECIFIC}_{i,m,\ell,r,l}
\]

Define family-level middle-late effect:
\[
\Delta^{mid}_{i,m,\ell,F}
=
\frac{1}{|F||\mathcal L_{mid}|}
\sum_{r \in F}
\sum_{l \in \mathcal L_{mid}}
\Delta z_{i,m,\ell,r,l}
\]

This is the **confirmatory subject-level summary statistic**.

Primary tests:
- H1: test \(\Delta^{mid}_{i,m,\ell,F_{semantic}} > 0\)
- H2: test \(\Delta^{mid}_{i,m,\ell,F_{semantic}} - \Delta^{mid}_{i,m,\ell,F_{auditory}} > 0\)

Use sign-flip permutation within each language and model; summarize across languages by inverse-variance meta-analysis for description only.
Treat each `model x language x hypothesis` slice as its own confirmatory test, for 12 primary tests total.
Also compute Holm-adjusted transparency p-values across those 12 tests and store them as `p_holm_primary`.

Layer-by-layer results remain secondary / exploratory.

---

## 18. Geometry-to-brain coupling

For model \(m\) and language \(\ell\), define semantic family layerwise brain effect:
\[
B_l^{(m,\ell)}
=
\frac{1}{|F_{semantic}|}
\sum_{r \in F_{semantic}}
\left(
\frac{1}{N_\ell}
\sum_{i=1}^{N_\ell}
\Delta z_{i,m,\ell,r,l}
\right)
\]

Let
\[
G_l^{(m)} = CAS_l^{(m)}
\]

Compute:
\[
\rho_{m,\ell} = \mathrm{SpearmanCorr}\left(\{G_l^{(m)}\}_l, \{B_l^{(m,\ell)}\}_l\right)
\]

Test significance by permuting layer order.

This is the canonical text-to-brain coupling analysis.

---

## 19. Required statistical tests

### 19.1 Subject-level primary contrasts
Use paired sign-flip permutation tests across subjects within language.
Use one-sided `p_perm` for each primary contrast.
Compute Holm-adjusted `p_holm_primary` across all 12 primary tests.

### 19.2 Multiple comparisons
For secondary ROI × layer analyses, apply FDR correction within reasonable families:
- per model
- per language
- across ROIs × layers

### 19.3 Report at minimum
- mean effect
- standard error
- Cohen’s \(d_z\)
- permutation p-value
- Holm-adjusted primary p-value where relevant
- FDR q-value where relevant
- bootstrap 95% CI

### 19.4 Do not over-pool languages
Scanner/site differences mean cross-language pooling should be descriptive and cautious.

---

## 20. Required controls

These are mandatory.

1. **Acoustic/nuisance-only**
Interpret `acoustic/nuisance-only` as follows:

- primary design = acoustic predictors:
  - RMS
  - f0 / pitch
  - word rate
  - sentence onset impulses
  - picture regressors for EN/CN run 1 when relevant
- nuisance terms for this baseline still include:
  - run intercepts
  - run-wise linear trends
  - high-pass cosine basis
  - motion regressors if available in shipped derivatives, confounds, or metadata

Do **not** define the acoustic baseline as “everything in \(Z\)” and then residualize it away. It must remain a meaningful predictive baseline.

2. **RAW**
3. **SHARED**
4. **SPECIFIC**
5. **FULL**
6. **MISMATCHED_SHARED**

Interpretation:
- `SHARED > MISMATCHED_SHARED` proves correct content alignment matters
- `RAW > acoustic-only` proves the model adds value beyond acoustics
- `FULL > SHARED` or `FULL > SPECIFIC` tests incremental contributions

---

## 21. Success conditions

The paper is successful if the following are true:

1. `SHARED > SPECIFIC` in the semantic family
2. `SHARED > MISMATCHED_SHARED` in at least some core language ROIs
3. `RAW > acoustic-only` in language ROIs
4. the sign of the main semantic-family effect is consistent across both model families
5. the paper is written around these grounded outputs, not imagined ones

For the paper-complete target, robustness outputs are also required:
- Table 5 is mandatory
- Figure 9 remains optional

If only 1–3 hold, the paper is still viable.
If 1–5 hold, the paper is strong.

---

## 22. What would ruin the paper

1. Treating it as a benchmark leaderboard.
2. Hardcoding run indices and misaligning LPPC runs.
3. Ignoring the preprocessed scan conventions.
4. Using other-language timings for target subjects.
5. Skipping `MISMATCHED_SHARED`.
6. Overclaiming universal semantics or syntax/semantics separation.
7. Making studyforrest the main analysis.
8. Writing a paper not grounded in actual figures and tables.

---

## 23. Final operational recommendation

Implement in this order:

1. download LPPC and inspect file structure
2. build canonical run manifests
3. derive sentence timings and sentence spans
4. align EN/FR/ZH sentence triplets
5. run alignment QC
6. extract XLM-R hidden states
7. build `RAW / SHARED / SPECIFIC / FULL / MISMATCHED_SHARED`
8. run ROI encoding for English only
9. expand to all languages
10. run NLLB encoder
11. run primary statistics
12. generate figures and tables
13. write manuscript
14. only then consider voxelwise maps or German appendix

That order is not optional. It is the intended path.
