# Language-Neutral or Language-Specific? Disentangling Multilingual LLM Subspaces with Naturalistic fMRI

[![DOI](https://zenodo.org/badge/1183637689.svg)](https://doi.org/10.5281/zenodo.19185476)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-B31B1B?style=flat-square&logo=adobeacrobatreader&logoColor=white)](paper/final/language_neutral_or_language_specific_disentangling_multilingual_llm_subspaces_with_naturalistic_fmri_ali_uyar.pdf)
[![Manuscript Source](https://img.shields.io/badge/Manuscript-Source-1D4ED8?style=flat-square&logo=markdown&logoColor=white)](paper/manuscript.md)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/aliuyar1234/language-neutral-or-language-specific)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Scope](https://img.shields.io/badge/Scope-ROI--First%20Confirmatory%20Study-5B4B8A?style=flat-square)](#scope)

Ali Uyar
Independent Researcher

**Paper title:** *Language-Neutral or Language-Specific? Disentangling Multilingual LLM Subspaces with Naturalistic fMRI*

This repository accompanies a mechanistic study at the interface of multilingual representation learning and cognitive neuroscience. During naturalistic story listening in English, French, and Chinese, it asks whether higher-level language regions are better predicted by a leave-target-out cross-lingual representation shared across languages or by a residual component that remains specific to the heard language. The analysis is built on the Le Petit Prince multilingual fMRI corpus (LPPC; OpenNeuro `ds003643`) with two multilingual encoders: `FacebookAI/xlm-roberta-base` and the encoder of `facebook/nllb-200-distilled-600M`.

## Abstract

Model-brain alignment studies increasingly use large language model features to explain naturalistic fMRI responses, but most such work remains effectively monolingual. Multilingual encoders, however, contain both cross-lingually shared structure and language-specific residual structure, raising a mechanistic question about which component better tracks human brain activity during comprehension. We addressed this question in the Le Petit Prince multilingual naturalistic fMRI corpus by aligning English, French, and Chinese sentence spans, extracting multilingual sentence embeddings from XLM-R-base and the encoder of NLLB-200-distilled-600M, decomposing each target-language representation into a leave-target-out shared component and an orthogonalized target-language-specific residual, and testing those components with cross-validated ROI encoding models. Across both model families and all three languages, the shared component showed a positive advantage over the specific residual in the semantic ROI family, with all six confirmatory `model x language` H1 tests surviving Holm correction. The stronger prediction that the semantic-family shared advantage would exceed the auditory-family advantage was mixed, because auditory and superior temporal ROIs also showed robust shared-over-specific effects. Correct-content controls remained directionally sane, including `SHARED > MISMATCHED_SHARED`, while geometry-to-brain coupling was weak and non-significant in the present analysis. These results support the use of multilingual LLMs as factorized explanatory probes for naturalistic language neuroscience while arguing for a narrower claim than a clean semantic-versus-auditory dissociation.

## Main Finding

The central confirmatory result replicates cleanly across model families and languages: in higher-level semantic ROIs, a leave-target-out `SHARED` representation outperforms the orthogonalized target-language-specific residual.

| Model | Language | Semantic-family `delta_shared_specific` | H1 Holm-significant |
| ----- | -------- | --------------------------------------- | ------------------- |
| xlmr  | EN       | 0.0609                                  | Yes                 |
| xlmr  | FR       | 0.0589                                  | Yes                 |
| xlmr  | ZH       | 0.0324                                  | Yes                 |
| nllb  | EN       | 0.0415                                  | Yes                 |
| nllb  | FR       | 0.0429                                  | Yes                 |
| nllb  | ZH       | 0.0194                                  | Yes                 |

All six confirmatory H1 tests survived Holm correction (`p_holm_primary = 0.0012`). However, the stronger prediction that the semantic-family advantage would exceed the auditory-family advantage (H2) did not come out cleanly: all 36 of 36 auditory ROI rows also showed positive `SHARED - SPECIFIC` effects, with XLM-R French `R_pSTG` reaching `0.1096`. The paper therefore supports a replicated semantic-family `SHARED > SPECIFIC` effect, but not a clean confirmatory semantic-over-auditory dissociation or a broad "universal language code" claim. Geometry-to-brain coupling is computed but remains weak and non-significant.

## Contributions

1. A reproducible multilingual sentence-span pipeline with canonical run remapping and tri-lingual alignment QC over 1,323 aligned triplets across English, French, and Chinese.
2. A `SHARED` / `SPECIFIC` / `FULL` / `MISMATCHED_SHARED` feature decomposition in which all features, including `SHARED`, are placed on the target listener's own sentence timeline.
3. ROI-first naturalistic encoding with required controls: `RAW`, `FULL`, `MISMATCHED_SHARED`, and an acoustic-only baseline under leave-one-run-out cross-validation.
4. A text-side multilingual geometry analysis that is kept descriptive and compared against the brain-side shared advantage as a secondary rather than headline result.
5. A full paper-facing public release surface: canonical figures, tables, statistical outputs, claim-evidence tracing, and public-facing methods and deviation documentation.

## Scope

This release is intentionally narrow.

- One public naturalistic fMRI corpus: LPPC (`ds003643`), three language cohorts (English, French, Chinese)
- Two multilingual encoder families: XLM-R-base and the encoder of NLLB-200-distilled-600M
- ROI-first analysis on a Harvard-Oxford atlas resampled into LPPC derivative space
- Sentence-span analysis rather than long-context comprehension modeling
- Anatomical ROIs, not subject-specific functional ROIs
- Cross-language magnitudes are descriptive because the language cohorts come from different acquisition sites
- The cortex-wide panel is an ROI-projected visualization, not a voxelwise inferential whole-brain map

The manuscript is deliberate about not overclaiming a clean semantic-versus-auditory dissociation or a universal language-invariant code. The contribution is a replicated semantic-family `SHARED > SPECIFIC` mechanism under strict controls, not a broad universality result.

## Paper

- Compiled PDF: [`paper/final/language_neutral_or_language_specific_disentangling_multilingual_llm_subspaces_with_naturalistic_fmri_ali_uyar.pdf`](paper/final/language_neutral_or_language_specific_disentangling_multilingual_llm_subspaces_with_naturalistic_fmri_ali_uyar.pdf)
- Manuscript source: [`paper/manuscript.md`](paper/manuscript.md)
- Primary confirmatory statistics: [`outputs/tables/table03_main_confirmatory_stats.csv`](outputs/tables/table03_main_confirmatory_stats.csv)
- Robustness summary: [`outputs/tables/table05_robustness_summary.csv`](outputs/tables/table05_robustness_summary.csv)
- Claim-evidence map: [`outputs/manuscript/claim_evidence_map.md`](outputs/manuscript/claim_evidence_map.md)

## Repository Layout

- [`src/brain_subspace_paper/`](src/brain_subspace_paper/) — implementation of data handling, alignment, model extraction, feature decomposition, encoding, statistics, and figure/table generation
- [`configs/`](configs/) — canonical YAML configuration for project scope, models, ROIs, outputs, and pipeline defaults
- [`docs/`](docs/) — public methods, data contracts, implementation workplan, and deviation log
- [`paper/`](paper/) — manuscript source, captions, bibliography seed, and final PDF
- [`outputs/`](outputs/) — canonical figures, tables, statistics, manuscript provenance, and execution logs
- [`scripts/`](scripts/) — pipeline entrypoints and paper-facing refresh scripts

## Reproducibility

- [`SSOT.md`](SSOT.md) — canonical research specification
- [`ERRATA_AND_IMPLEMENTATION_DECISIONS.md`](ERRATA_AND_IMPLEMENTATION_DECISIONS.md) — resolved defaults and implementation decisions
- [`METHOD_RATIONALE.md`](METHOD_RATIONALE.md) — explanatory rationale for key design choices
- [`docs/02_datasets_and_downloads.md`](docs/02_datasets_and_downloads.md) — data acquisition instructions
- [`docs/04_alignment_algorithm.md`](docs/04_alignment_algorithm.md) — multilingual alignment procedure
- [`docs/06_encoding_and_statistics.md`](docs/06_encoding_and_statistics.md) — encoding and inference details
- [`docs/15_spec_deviation_log.md`](docs/15_spec_deviation_log.md) — implementation deviations from the specification

Raw LPPC data, large intermediate caches, and local scratch products are not committed in full. The repository exposes the contracts, manifests, and generated paper-facing outputs needed to audit the release.

## Citation

```bibtex
@unpublished{uyar2026languageneutral,
  author = {Uyar, Ali},
  title  = {Language-Neutral or Language-Specific? Disentangling Multilingual {LLM} Subspaces with Naturalistic {fMRI}},
  year   = {2026},
  doi    = {10.5281/zenodo.19185476},
  note   = {Independent research}
}
```
