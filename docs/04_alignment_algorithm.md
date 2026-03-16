# 04 — Alignment Algorithm

This document defines the sentence alignment algorithm. This is one of the most important non-obvious parts of the project.

## 1. Goal

Construct a multilingual triplet table where each row corresponds to the **same story content** across:

- English
- French
- Chinese


Alignment is always performed **within matching section / canonical run only**.  
Never align sentences across different sections.

Each row may contain either:
- a 1-sentence span in each language, or
- a 2-sentence merged span in one or more languages

The core pipeline allows only:
- 1:1
- 1:2
- 2:1
- 2:2

Do not allow spans larger than 2 sentences in the core pipeline.

## 2. Why alignment is hard

The story is the same across languages, but:
- sentence counts differ by language
- punctuation and segmentation differ
- speech rates differ
- direct sentence order is mostly monotonic, but not perfectly 1:1

A robust monotonic alignment is therefore required.

## 3. Source-of-truth principle

The alignment must be constructed from **dataset-provided text and timing information**.

Use this source priority:

1. sentence structure from `*_tree.csv` or `*_dependency.csv` if sentence ids and sentence texts are explicit
2. else sentence reconstruction from `*_word_information.csv`
3. word timings from `*_section*.TextGrid`

Do not use machine translation to create the core LPPC triplets.

## 4. Step 1 — Build per-language sentence spans

For each language and section:

1. load the ordered words from `word_information.csv` or equivalent
2. determine sentence boundaries using the dataset’s sentence-level annotations if available
3. otherwise reconstruct boundaries from punctuation and parser outputs
4. parse the matching section TextGrid
5. align the ordered word list to the TextGrid word intervals
6. compute sentence-span onset/offset from first and last matched word

Output:
- one table per language with ordered sentence spans

Required columns:
- `language`
- `section_index`
- `language_sentence_index`
- `text`
- `onset_sec`
- `offset_sec`
- `n_words`
- `first_word_idx`
- `last_word_idx`

## 5. Step 2 — Create pivot-language sequences

Use **English as the pivot language**.

Rationale:
- English annotations are widely used in prior LPPC work
- EN-to-FR and EN-to-ZH pairwise monotonic alignment is easy to reason about
- triplets can then be merged on English block ids

This does not mean English is theoretically privileged. It is only the implementation pivot.

## 6. Step 3 — Compute QC embeddings for candidate spans

Use LaBSE embeddings only for alignment quality control.

For each language and section:
- embed each 1-sentence unit
- create on-the-fly merged 2-sentence span embeddings by averaging or re-embedding concatenated span text

Preferred rule:
- concatenate raw sentence texts with a single space
- embed the merged text directly with LaBSE

## 7. Step 4 — Pairwise monotonic dynamic programming

For each section, align:
- English ↔ French
- English ↔ Chinese

Let English spans be \(A_1, \dots, A_I\) and the other-language spans be \(B_1, \dots, B_J\).

Allow candidate span lengths:
\[
p, q \in \{1, 2\}
\]

Let \(A_{i,p}\) denote the concatenated English span from \(i\) to \(i+p-1\).  
Let \(B_{j,q}\) denote the concatenated non-English span from \(j\) to \(j+q-1\).

### 7.1 Alignment cost

Define:
- \(e(A_{i,p})\): LaBSE embedding of the candidate English span
- \(e(B_{j,q})\): LaBSE embedding of the candidate other-language span
- `chars(X)`: number of Unicode characters in span \(X\)
- `punct_type(X)`: coarse punctuation ending class, e.g. declarative / question / exclamation / none

The canonical local cost is:
\[
C(i,p,j,q) =
\left(
\alpha \left(1 - \cos(e(A_{i,p}), e(B_{j,q}))\right)
+
\beta \left|\log\frac{\mathrm{chars}(A_{i,p}) + 1}{\mathrm{chars}(B_{j,q}) + 1}\right|
+
\gamma \cdot \mathbf{1}\left[\mathrm{punct\_type}(A_{i,p}) \neq \mathrm{punct\_type}(B_{j,q})\right]
\right)
\cdot \frac{p + q}{2}
+
\lambda \left((p - 1) + (q - 1)\right)
\]

Default weights:
- \(\alpha = 1.0\)
- \(\beta = 0.25\)
- \(\gamma = 0.10\)
- \(\lambda = 0.10\)

The \(\frac{p + q}{2}\) factor makes the objective sentence-normalized instead of block-count-normalized.
Without it, the DP gets an artificial reward for collapsing several neighboring sentences into one long block.

The \(\lambda\) term is a small merge regularizer. It does not ban `2`-sentence spans, but it makes them pay their way.

These defaults are not magical; they are stable starting points that kept the triplet table interpretable and the manual-review burden reasonable.

### 7.2 Dynamic program

Let \(D(i,j)\) be the minimum alignment cost for the first \(i\) English sentences and first \(j\) other-language sentences.

\[
D(i,j) = \min_{p,q \in \{1,2\}} D(i-p, j-q) + C(i-p+1,p,j-q+1,q)
\]

subject to valid indices.

Base case:
\[
D(0,0) = 0
\]

All invalid states are \(+\infty\).

Backtrack to recover the optimal monotonic alignment path.

## 8. Step 5 — Triplet merge via English blocks

After EN↔FR and EN↔ZH are aligned, create triplets by English block id.

Each final triplet row should contain:

- English span indices
- French span indices
- Chinese span indices
- English text span
- French text span
- Chinese text span
- English onset/offset
- French onset/offset
- Chinese onset/offset

Triplet ids should be unique and monotonic within section.

Recommended convention:
- `triplet_id` = global integer id across the dataset
- also store `section_triplet_index` = monotonic index within section

This makes section-local debugging easier while preserving a single global row key.

### 8.1 Conflict resolution via local tri-lingual reconciliation

Do not let EN-FR win by default.
Do not let EN-ZH win by default.
Do not keep incompatible English blockings.
Do not duplicate target-language spans just to force a merge.

If the implied English block boundaries from EN-FR and EN-ZH match, merge directly.

If they conflict:
1. isolate the minimal conflicting English window whose edges are common boundaries in both pairwise paths
2. extract the corresponding local EN, FR, and ZH sentence windows
3. re-run alignment on that local window with a tri-lingual monotonic DP using span lengths `{1, 2}` in each language

Use local cost:
\[
C_{tri}(i,p,j,q,k,r)
= 1.0 \cdot C_{EN,FR}(i,p,j,q)
+ 1.0 \cdot C_{EN,ZH}(i,p,k,r)
+ 0.5 \cdot C_{FR,ZH}(j,q,k,r)
\]

Use recurrence:
\[
D(i,j,k)
=
\min_{p,q,r \in \{1,2\}}
D(i-p, j-q, k-r)
+ C_{tri}(i-p+1, p, j-q+1, q, k-r+1, r)
\]

Base case:
\[
D(0,0,0) = 0
\]

If no feasible path exists:
1. expand the local window by one sentence on each side and retry once
2. if it still fails, mark that region `needs_fix`, exclude only those rows from the confirmatory analysis, and log it
3. if unresolved rows exceed 1% of all triplets, stop and inspect before continuing

The final `alignment_triplets.parquet` must be a single globally consistent triplet table.

## 9. Step 6 — Quality-control flags

Every triplet gets QC features:

- `sim_en_fr`
- `sim_en_zh`
- `sim_fr_zh`
- `mean_pairwise_sim`
- `max_local_cost`
- `char_ratio_en_fr`
- `char_ratio_en_zh`
- `punct_mismatch_flag`
- `merge_pattern` (e.g. 1-1-1, 1-2-1, 2-1-1, etc.)

Flag rows for manual inspection when one or more of the following hold:

- pairwise cosine < 0.35
- mean pairwise cosine < 0.40
- any local cost > 1.25
- suspicious punctuation mismatch
- 2-sentence merges in more than one language for the same triplet
- non-monotonicity (should never happen)
- duration or character ratios far outside the median range

These thresholds are defaults. They are for flagging, not automatic deletion.

## 10. Manual inspection protocol

Randomly inspect at least 100 triplets, stratified by:
- section
- merge pattern
- low QC scores
- high QC scores

Inspect:
- content equivalence
- obvious merge/split errors
- onset/offset plausibility
- whether triplets preserve story order

Write a short report:
- `outputs/logs/alignment_qc_report.md`

## 11. Freeze rule

Once the alignment passes QC:
- freeze it
- version it
- do not silently mutate it later

Canonical outputs:
- `data/processed/alignment_triplets.parquet`
- `data/processed/alignment_triplets.tsv`
- `data/processed/alignment_triplets_qc.parquet`

## 12. Sentence-span timing from TextGrid

The TextGrid is the source for word timing.

Procedure:
1. parse the word tier from the section TextGrid
2. remove silence and empty intervals
3. normalize tokens lightly for matching (Unicode normalization, strip trivial whitespace)
4. align ordered annotation tokens to TextGrid tokens using monotonic token matching
5. derive sentence onset/offset from the matched word intervals

If tokenization mismatches are minor:
- allow monotonic fuzzy matching
- log unmatched tokens
- do not silently reorder

## 13. Failure modes to avoid

Do not:
- align by raw absolute timing across languages
- align by naive sentence index alone
- rely on translation APIs
- allow arbitrary many-to-many spans
- use the main paper models as the QC aligner
- keep changing the triplet table after model extraction begins

## 14. Minimal acceptance criteria

Before moving to hidden-state extraction, the alignment must satisfy:

1. > 95% of triplets are unflagged or manually approved
2. same-triplet LaBSE similarities exceed mismatched similarities by a healthy margin
3. section-wise order is monotonic and complete
4. every triplet has target-language onset/offset values for EN/FR/ZH
5. triplet ids are stable and versioned
