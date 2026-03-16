# METHOD_RATIONALE.md

This file explains why the project made its non-obvious methodological choices.

Use this file as an explanatory companion to:
- `SSOT.md`
- `ERRATA_AND_IMPLEMENTATION_DECISIONS.md`
- `docs/04_alignment_algorithm.md`
- `docs/06_encoding_and_statistics.md`

This file is not a second SSOT and does not override the canonical spec. Its job is to make the fixed choices legible to a future implementer, reviewer, or writer.

## 1. Why this paper looks the way it does

### Core scientific bet
Decision:
- test whether a leave-target-out shared multilingual representation explains language-network activity better than a target-language-specific residual

Why:
- this asks a mechanistic question rather than a benchmark question
- it turns multilingual representation geometry into a falsifiable brain-encoding hypothesis
- it lets the paper separate content shared across languages from residual language-specific variance

Prevents:
- collapsing the project into "which model scores highest"
- overclaiming universal semantic coding without a control against language-specific structure

### Why LPPC is the core dataset
Decision:
- use LPPC only for the core paper

Why:
- the same story exists in multiple languages
- the data and annotations are public
- the dataset is large enough for a real paper but still feasible on one local machine

Prevents:
- project sprawl
- dependence on private data or paid APIs
- turning the studyforrest appendix into the real project

### Why English, French, and Chinese
Decision:
- keep EN, FR, and ZH as the core language set

Why:
- those are the LPPC languages with the relevant public derivatives and annotations
- they provide meaningful cross-language variation without adding a second dataset

Prevents:
- introducing new dataset confounds just to add more languages

### Why only two main models
Decision:
- use XLM-R-base and the NLLB encoder as the core paper models

LaBSE is a required alignment-QC model, but not a main brain-encoding model.

Why:
- they give two distinct multilingual model families without exploding scope
- the paper needs replication across model families more than a large model zoo

Prevents:
- benchmark-style sprawl
- spending compute budget on breadth instead of controls and robustness

### Why sentence-level and ROI-first
Decision:
- use sentence-level regressors and ROI-first encoding

Why:
- sentence alignment across languages is much cleaner than word-level alignment
- fMRI temporal resolution is compatible with sentence-scale predictors
- ROI-first gives interpretable, tractable confirmatory tests before secondary voxelwise maps

Prevents:
- fragile word-level alignment logic
- premature whole-brain exploration before the main analysis is stable

## 2. Why the representation formulas are defined this way

### Why L2 normalization comes first
Decision:
- normalize sentence embeddings before shared/specific decomposition

Why:
- raw embedding norm can vary by model, layer, and language for reasons unrelated to the shared-vs-specific question
- the paper wants directional structure, not magnitude accidents

Prevents:
- norm differences masquerading as semantic sharing
- one model or layer dominating simply because its vectors are larger

### Why SHARED is leave-target-out
Decision:
- define `SHARED` for language `l` as the average of the other languages only

Why:
- it forces the shared representation to be independent of the target-language sentence embedding
- it turns the test into a real cross-lingual generalization question

Prevents:
- target-language leakage into the shared predictor
- circular claims that the target language predicts itself

### Why SPECIFIC is an orthogonalized residual
Decision:
- define `SPECIFIC` as the target-language residual after removing the projection onto `SHARED`

Why:
- the paper wants `SHARED` and `SPECIFIC` to capture separable explanatory directions
- orthogonalization makes the comparison cleaner and more interpretable

Prevents:
- shared variance being counted twice
- ambiguous comparisons where `SPECIFIC` still partly contains the shared component

### Why FULL and MISMATCHED_SHARED are required
Decision:
- keep `FULL` and `MISMATCHED_SHARED` as required controls

Why:
- `FULL` tests whether the combined decomposition retains useful information
- `MISMATCHED_SHARED` tests whether the effect depends on correct content alignment rather than generic cross-lingual smoothness

Prevents:
- claiming the decomposition works without proving that true alignment matters
- mistaking temporal autocorrelation or coarse discourse similarity for sentence-level content matching

## 3. Why the alignment algorithm uses pairwise DP plus local tri-lingual repair

### Why start with EN-FR and EN-ZH pairwise DPs
Decision:
- run pairwise monotonic alignment first, then reconcile only where they disagree

Why:
- pairwise DP is simpler and more stable than forcing a full tri-lingual DP over the entire corpus
- most local windows do not need tri-lingual repair

Prevents:
- unnecessary global complexity
- making the hardest part of the algorithm the default path

### Why resolve conflicts locally
Decision:
- isolate the minimal conflicting English window and repair it locally

Why:
- only the incompatible region needs extra work
- local repair preserves the rest of the already-stable alignment path

Prevents:
- global resegmentation when only a small region is problematic
- drift in unaffected parts of the alignment table

### Why span lengths are restricted to `{1, 2}`
Decision:
- allow only 1-sentence or 2-sentence merges in each language

Why:
- sentence-level alignment is the core design
- small merges capture ordinary translation granularity differences without making units too coarse

Prevents:
- giant spans that are hard to interpret in time and meaning
- the confirmatory analysis drifting away from sentence-scale predictors

### Why the pairwise cost is sentence-normalized and lightly merge-regularized
Decision:
- scale the base pairwise mismatch term by average span length and add a small merge penalty

Why:
- otherwise the DP receives a structural reward for using fewer, longer blocks even when the underlying match quality is similar
- sentence normalization keeps the objective closer to the intended sentence-level unit of analysis
- a light merge penalty still allows necessary `2`-sentence merges without letting them dominate the table

Prevents:
- pathological overuse of `2-2-2` triplets
- a large manual-review burden created by objective-function bias rather than real ambiguity

### Why the tri-lingual cost uses weights `1.0 / 1.0 / 0.5`
Decision:
- weight EN-FR and EN-ZH equally at `1.0`, with FR-ZH at `0.5`

Why:
- English is the common bridge language in the pairwise setup
- FR-ZH still matters as a consistency term, but should not overpower the two bridge constraints

Prevents:
- letting the auxiliary FR-ZH comparison dominate the local repair
- collapsing the repair into a hidden third pairwise objective unrelated to the main bridge structure

### Why the final triplet table must be globally consistent
Decision:
- the final `alignment_triplets.parquet` must be a single coherent table

Why:
- every downstream stage assumes one consistent sentence inventory across languages
- SHARED and SPECIFIC only make sense if a triplet means one aligned content unit

Prevents:
- duplicate spans
- language-specific blocking artifacts that make the brain analysis logically incoherent

## 4. Why hidden-state indexing and layer depth are fixed this way

### Why `hidden_states[0]` counts as layer 0
Decision:
- treat the embedding output as `layer_index = 0`

Why:
- that matches what the models actually return
- it makes cross-model comparisons honest about how many representational stages are available

Prevents:
- off-by-one ambiguity
- hidden disagreement between code and figures

### Why normalized depth is `l / (N - 1)`
Decision:
- define layer depth over all returned hidden states, including the embedding output

Why:
- cross-model plots need a comparable depth axis even when the raw number of returned states differs
- the project compares representational stage, not raw layer count

Prevents:
- misleading cross-model depth comparisons
- a fake "middle layer" created by indexing conventions rather than model structure

## 5. Why the encoding pipeline uses nested leave-one-run-out CV

### Why the outer split is by run
Decision:
- use leave-one-canonical-run-out outer CV with 9 folds

Why:
- runs are the natural independent blocks in naturalistic fMRI
- random timepoint splits would leak temporal structure

Prevents:
- optimistic scores caused by train-test dependence within a run

### Why alpha is chosen with inner leave-one-run-out CV
Decision:
- tune ridge alpha only inside the outer-training runs

Why:
- hyperparameter search is part of the model and must stay inside training data
- runwise inner CV matches the temporal dependence structure of the outer problem

Prevents:
- test-set leakage
- alpha choices that accidentally exploit held-out runs

### Why the full pipeline is refit inside each inner split
Decision:
- refit residualization, standardization, PCA, and ridge inside each inner split

Why:
- all of those steps estimate parameters from data
- leakage can happen through preprocessing, not just the regression fit

Prevents:
- hidden leakage through PCA bases or nuisance regression
- inflated cross-validated performance

### Why fold `r` values are Fisher-z averaged
Decision:
- average held-out fold scores in Fisher-z space, then invert with `tanh` only if needed

Why:
- Pearson `r` is not additive on its raw scale
- Fisher-z gives a better-behaved summary across folds

Prevents:
- biased subject-level summaries from averaging correlations directly

### Why `R^2` is computed from concatenated held-out predictions
Decision:
- compute one subject-level `R^2` from the concatenated held-out predictions and targets

Why:
- `R^2` is a variance-accounted-for summary, so one combined held-out series is more interpretable than an average of foldwise `R^2` values

Prevents:
- unstable or misleading foldwise `R^2` averages

### Why alpha ties go to the larger value
Decision:
- if alphas tie, choose the larger alpha

Why:
- when performance is indistinguishable, the more regularized solution is usually more stable

Prevents:
- selecting a needlessly flexible model without evidence it is better

## 6. Why the confirmatory hypotheses and multiplicity policy are set this way

### Why H1 is in the semantic family
Decision:
- H1 asks whether `SHARED > SPECIFIC` in semantic ROIs

Why:
- the core theory is about higher-level content abstraction
- semantic regions are the most direct place where a shared cross-language content code should emerge

Prevents:
- testing the main hypothesis in a region family not conceptually tied to the claim

### Why H2 compares semantic vs auditory families
Decision:
- H2 asks whether the shared advantage is larger in semantic than auditory/form-heavier ROIs

Why:
- it turns the paper from a pure "is there an effect anywhere?" design into a regional dissociation test
- auditory regions are the natural comparison because they plausibly retain more form-specific variance

Prevents:
- overinterpreting a globally positive effect as specifically semantic

### Why the 12 primary tests stay separate
Decision:
- each `model x language x hypothesis` slice is its own confirmatory test

Each `model x language x hypothesis` slice is confirmatory on its own; additionally, Holm-adjusted p-values across all 12 primary tests must still be computed and reported for transparency.

Why:
- languages are not exchangeable because LPPC mixes sites and scanners
- the paper wants replication across slices, not one pooled p-value to carry the whole claim

Prevents:
- hiding inconsistency behind pooled significance
- treating scanner/site differences as irrelevant

### Why Holm-adjusted `p_holm_primary` is reported
Decision:
- compute Holm-adjusted transparency p-values across the 12 primary tests

Why:
- the repo should show both slice-wise confirmatory results and family-wise error control across the whole primary set
- Holm is simple, interpretable, and less blunt than Bonferroni

Prevents:
- a false impression that multiplicity was ignored
- overstating the certainty of the full primary result family

## 7. Why the ROI, figure, and output decisions look the way they do

### Why ROI-first remains the main inference
Decision:
- treat ROI analyses as confirmatory and whole-brain maps as secondary

Why:
- the main question is about interpretable region families, not a voxelwise fishing expedition
- ROI-first is much more stable for a first complete paper

Prevents:
- letting noisy descriptive maps drive the headline claim

### Why Figure 8 uses one representative layer per model and language
Decision:
- choose the representative layer by the maximum semantic-family group mean `SHARED - SPECIFIC`, tie-breaking shallower

Why:
- Figure 8 is descriptive and should show the clearest layer for that `model x language` slice
- using the semantic-family ROI effect ties the descriptive map back to the main inferential target

Prevents:
- picking arbitrary whole-brain layers
- choosing a late flashy layer disconnected from the main result

### Why Figure 8 artifacts are mandatory even though the figure is secondary
Decision:
- save NIfTI maps, mask, and manifest for each Figure 8 slice

Why:
- descriptive figures still need machine-readable provenance
- later writing and inspection depend on knowing exactly what was plotted

Prevents:
- irreproducible voxelwise figures
- hand-made figures with no auditable source

### Why Table 5 is mandatory but Figure 9 is optional
Decision:
- robustness must exist in a table for paper-complete; the figure is optional

Robustness is mandatory for the project's target milestone, which is paper-complete; only a narrower core-analysis milestone may omit it.

Why:
- the paper needs robustness evidence in machine-readable form
- a figure is presentation sugar, while the table is the real reporting contract

Prevents:
- a "complete" paper with no explicit robustness record
- blocking completion on a nonessential visualization

### Why provenance files are mandatory
Decision:
- require figure provenance, table provenance, and a claim-evidence map

Why:
- this repo is designed to write the paper from actual outputs, not memory
- provenance is the bridge from computation to manuscript claims

Prevents:
- paper text drifting away from generated results
- future confusion about which script produced which figure or claim

## 8. Why the data-integrity and fallback rules are strict

### Why scan counts default to exact match
Decision:
- exact run-volume match by default, with only isolated documented `+/-1` exceptions

Why:
- LPPC run mapping is easy to get subtly wrong
- exact counts are a strong sanity check that the canonical run mapping is correct

Prevents:
- silent run misassignment
- building design matrices against the wrong BOLD series

### Why motion is only used if shipped derivatives provide it
Decision:
- use shipped motion/confound information when available; do not recompute motion from raw MRI for the main paper

Why:
- the paper is an encoding study, not a re-preprocessing project
- recomputing motion would add a major, unnecessary branch of pipeline complexity

Prevents:
- scope creep
- hidden dependence on a custom preprocessing workflow

### Why acoustics fall back to public audio rather than being silently dropped
Decision:
- if shipped prosody is missing, recompute RMS and `f0` from public audio; otherwise stop and inspect

Why:
- the acoustic baseline is a required control
- silently losing it would weaken the interpretability of `RAW` and `SHARED`

Prevents:
- a paper that cannot show model value above acoustics

### Why picture events are reconstructed from fixed timings if needed
Decision:
- for EN and ZH run 1, reconstruct picture events from published timings when event files are missing

Why:
- the nuisance event is conceptually known even if a file is absent
- omitting it without replacement would make the nuisance model depend on file accidents

Prevents:
- avoidable inconsistency across subjects or languages

## 9. Why the scaffold is explicitly non-authoritative

Decision:
- treat `repo_scaffold/` as a skeleton only

Why:
- the scaffold exists to suggest structure, not to prove completeness
- the scientific contract lives in the SSOT, errata, docs, and data contracts

Prevents:
- assuming the repo is runnable because a stub file exists
- letting placeholder code override the written spec

## 10. How to use this file

Use `METHOD_RATIONALE.md` when you need to answer:
- why this exact formula exists
- why an algorithm was chosen over a tempting alternative
- why a control is mandatory
- why a result is confirmatory, descriptive, or optional
- how to explain the project choices in Methods, Discussion, Limitations, or peer review

Use `SSOT.md` and `ERRATA_AND_IMPLEMENTATION_DECISIONS.md` when you need to answer:
- what exactly to implement
- which path or schema is canonical
- which statistical rule is binding

If this file and the canonical spec ever seem to disagree, the canonical spec wins and this file should be updated to match it.
