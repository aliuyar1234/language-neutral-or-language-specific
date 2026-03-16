# Language-Neutral or Language-Specific?

This repository is the paper-complete research release for:

**Language-Neutral or Language-Specific? Disentangling Multilingual LLM Subspaces with Naturalistic fMRI**

The project tests whether naturalistic fMRI responses during multilingual story listening are better predicted by a leave-target-out shared cross-lingual representation or by a target-language-specific residual. The core dataset is the multilingual *Le Petit Prince* corpus (LPPC; OpenNeuro `ds003643`), and the core model families are XLM-R-base and the encoder of NLLB-200-distilled-600M.

## Final Paper

The public paper PDF is:

- `paper/final/language_neutral_or_language_specific_disentangling_multilingual_llm_subspaces_with_naturalistic_fmri_ali_uyar.pdf`

## What This Repository Contains

- reproducible analysis code under `src/brain_subspace_paper/`
- exact configuration files under `configs/`
- canonical alignment and feature manifests under `data/`
- canonical paper figures under `outputs/figures/`
- canonical paper tables under `outputs/tables/`
- canonical statistical outputs under `outputs/stats/`
- manuscript-facing provenance under `outputs/manuscript/`

## Main Scientific Scope

- dataset: LPPC / OpenNeuro `ds003643`
- languages: English, French, Chinese
- models: `FacebookAI/xlm-roberta-base` and `facebook/nllb-200-distilled-600M` encoder
- analysis style: sentence-level, ROI-first naturalistic encoding
- key conditions: `SHARED`, `SPECIFIC`, `RAW`, `FULL`, `MISMATCHED_SHARED`

## Public Reading Order

If you want the shortest path through the release, read:

1. `paper/final/...pdf`
2. `outputs/manuscript/claim_evidence_map.md`
3. `outputs/manuscript/figure_provenance.md`
4. `outputs/manuscript/table_provenance.md`
5. `docs/06_encoding_and_statistics.md`
6. `docs/15_spec_deviation_log.md`

## Reproducibility Surface

The repository keeps the paper-facing artifacts needed to inspect or rerun the released checkpoint:

- code and configuration
- data/manifests needed to understand derived artifacts
- canonical figures, tables, and stats used in the paper
- claim-evidence and provenance files for manuscript audit

Raw LPPC data, large intermediate caches, local scratch outputs, and private workflow handoff files are intentionally excluded from the public repository.

## Key Files

- `SSOT.md`: canonical research specification
- `ERRATA_AND_IMPLEMENTATION_DECISIONS.md`: resolved implementation defaults and boundary decisions
- `METHOD_RATIONALE.md`: method-design rationale
- `docs/02_datasets_and_downloads.md`: data acquisition instructions
- `docs/04_alignment_algorithm.md`: multilingual alignment procedure
- `docs/06_encoding_and_statistics.md`: encoding and inference details
- `outputs/tables/table03_main_confirmatory_stats.csv`: primary confirmatory statistics
- `outputs/tables/table05_robustness_summary.csv`: robustness summary
- `outputs/stats/robustness_summary.parquet`: canonical robustness output

## Interpretation Notes

The main semantic-family `SHARED > SPECIFIC` result is positive in all six model-by-language confirmatory tests. The stronger semantic-versus-auditory dissociation remains mixed, and the robustness suite is supportive rather than uniformly clean. The release therefore supports a narrower and more defensible claim than a simple universal-language-code story.
