# 03 — Models and Hidden-State Extraction

This document defines the exact model choices and extraction rules.

## 1. Core models

### Model A — XLM-R-base
- Hugging Face id: `FacebookAI/xlm-roberta-base`
- type: multilingual encoder
- role: strong multilingual encoder baseline

### Model B — NLLB-200-distilled-600M
- Hugging Face id: `facebook/nllb-200-distilled-600M`
- type: multilingual seq2seq translation model
- role: explicitly multilingual shared-space model
- **rule:** use encoder hidden states only

## 2. Alignment QC model

### LaBSE
- Hugging Face id: `sentence-transformers/LaBSE`
- role: alignment quality control only
- not part of main brain encoding analyses
- do not report LaBSE as a main paper model

The QC model should remain independent from the two paper models.

## 3. Optional model policy

Do **not** add Qwen or other large models to the core pipeline.

If, after the full paper is complete, there is extra time:
- a modern multilingual base model may be added as an appendix
- but the core paper must be complete first

## 4. Model licenses and practical note

- XLM-R-base: public model card available on Hugging Face
- NLLB-200-distilled-600M: public model card available on Hugging Face; note the non-commercial license
- LaBSE: public model card available on Hugging Face

For the academic paper, this is acceptable.  
If any downstream use becomes commercial, re-check the NLLB license and replace it if needed.

## 5. Hidden-state extraction rules

### 5.1 Output type
Always request `output_hidden_states=True`.

### 5.2 Sentence text input
Use the aligned sentence-span text exactly as defined by the alignment table.

Do:
- preserve punctuation
- preserve original script
- keep capitalization
- keep language-specific tokens intact

Do not:
- lowercase blindly
- strip punctuation
- apply external translation for core LPPC inputs

### 5.3 Pooling rule
Pool over **current sentence-span tokens only**.

Primary pooling:
- masked mean pooling across non-special tokens belonging to the current sentence span

Do not use CLS/BOS as the primary representation.

### 5.4 Context rule
Core v1 input is sentence-span text only.

Optional robustness:
- prepend previous two aligned sentence spans as context
- but still pool only tokens belonging to the current sentence span

Do not let context complicate v1.

## 6. NLLB extraction rule

This is important:

> For NLLB, use the **encoder** hidden states only.

Do not:
- generate text
- decode translations
- use decoder hidden states
- mix encoder and decoder representations

Operationally:
- tokenize the sentence span
- run the encoder forward
- pool encoder token states over the current sentence span tokens

## 7. Token-span tracking

When context is added or special tokens exist, the code must know which tokens belong to the current sentence span.

Required implementation:
- store token offsets or token indices for the current sentence span
- pool only that slice
- exclude special tokens

Save this token-span metadata if helpful for debugging.

## 8. Numeric precision and caching

Recommended:
- forward pass in `bfloat16` or `float16` where safe
- convert pooled sentence embeddings to `float32` before normalization and saving

Cache per model/language/layer outputs.

Suggested cache layout:
```text
data/interim/embeddings/
  xlmr/
    en/
      layer_00.npy
      ...
    fr/
    zh/
  nllb_encoder/
    en/
    fr/
    zh/
```

Add a companion manifest:
```text
data/interim/embeddings/embedding_manifest.parquet
```

## 9. Canonical embedding object

Each sentence-span embedding record should contain:
- `triplet_id`
- `language`
- `model`
- `layer_index`
- `n_tokens_pooled`
- `embedding_vector`

Keep high-dimensional vectors in NPY / NPZ or Parquet with array support.

## 10. Batching strategy

Recommended:
- batch by total token count, not fixed sentence count
- sort candidate inputs by length within language/model extraction jobs
- save checkpoints often
- allow restart from partial caches

## 11. Layer indexing convention

Use zero-based layer indexing in code and results.

Exact convention:
- `hidden_states[0]` is the embedding layer output and is `layer_index = 0`
- `hidden_states[1]` is transformer block 1 output
- continue upward through the final returned hidden state

If a model returns `N` hidden states total, store normalized layer depth as:
\[
\delta_l = \frac{l}{N - 1}
\]

This is required for cross-model layer comparisons and layer-curve plots.

## 12. Quality checks after extraction

For each model/language:
1. no NaNs in embeddings
2. expected number of sentence-span rows exists
3. same-sentence cross-language cosine exceeds mismatched cosine for at least some middle layers
4. layerwise norms are sane
5. LaBSE-based alignment QC agrees that the triplet table passes the alignment QC thresholds or approved-review path

Save a summary:
- `outputs/logs/model_extraction_report.md`

## 13. Minimal extraction order

Do not extract everything at once.

Order:
1. XLM-R on 100 aligned triplets only
2. verify same-sentence > mismatched similarity
3. scale XLM-R to full dataset
4. build the full pipeline on English ROI analysis
5. only then extract NLLB encoder states
