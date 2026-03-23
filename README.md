# Language-Neutral or Language-Specific?

[![DOI](https://zenodo.org/badge/1183637689.svg)](https://doi.org/10.5281/zenodo.19185476)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-B31B1B?style=flat&logo=adobeacrobatreader&logoColor=white)](https://github.com/aliuyar1234/language-neutral-or-language-specific/raw/main/paper/final/language_neutral_or_language_specific_disentangling_multilingual_llm_subspaces_with_naturalistic_fmri_ali_uyar.pdf)

This repository is the paper-complete public research release for:

**Language-Neutral or Language-Specific? Disentangling Multilingual LLM Subspaces with Naturalistic fMRI**

The project asks a mechanistic question at the interface of multilingual representation learning and cognitive neuroscience: during naturalistic story listening, are brain responses in higher-level language regions better predicted by a leave-target-out cross-lingual representation that is shared across languages, or by a residual component that remains specific to the heard language?

The released analysis is centered on the multilingual *Le Petit Prince* fMRI corpus (LPPC; OpenNeuro `ds003643`) and two multilingual encoder families: `FacebookAI/xlm-roberta-base` and the encoder of `facebook/nllb-200-distilled-600M`.

## At a Glance

- **Scientific scope:** multilingual naturalistic fMRI, sentence-level encoding, ROI-first analysis
- **Languages:** English, French, Chinese
- **Core feature families:** `RAW`, `SHARED`, `SPECIFIC`, `FULL`, `MISMATCHED_SHARED`
- **Primary outcome:** semantic-family `SHARED > SPECIFIC` is positive across both model families and all three languages
- **Interpretive stance:** this release supports a narrow mechanistic claim, not a broad "universal language code" claim

## Main Paper

- [Download the final paper PDF](https://github.com/aliuyar1234/language-neutral-or-language-specific/raw/main/paper/final/language_neutral_or_language_specific_disentangling_multilingual_llm_subspaces_with_naturalistic_fmri_ali_uyar.pdf)
- [Read the manuscript source](paper/manuscript.md)

## What This Repository Contains

- reproducible analysis code in `src/brain_subspace_paper/`
- exact project configuration in `configs/`
- canonical paper figures in `outputs/figures/`
- canonical paper tables in `outputs/tables/`
- canonical statistical outputs in `outputs/stats/`
- manuscript-facing provenance and evidence mapping in `outputs/manuscript/`
- public-facing methods and implementation documentation in `docs/`

## Core Scientific Question

For a target language listener, the key test is whether BOLD responses are better predicted by:

- `SHARED`: a leave-target-out representation built from the other two languages for the same aligned sentence content
- `SPECIFIC`: the target-language residual after orthogonalization against that shared component

All features, including `SHARED`, are placed on the **target language's own timeline**. This is a multilingual brain-encoding paper designed to distinguish cross-lingually shared content structure from target-language-specific residual structure under naturalistic listening.

## Main Supported Findings

- In the semantic ROI family, `SHARED > SPECIFIC` is positive in all six confirmatory `model x language` tests.
- All six semantic-family H1 tests survive Holm correction in the released canonical checkpoint.
- The stronger semantic-versus-auditory dissociation remains mixed and is not presented as a clean confirmatory success.
- Correct-content controls remain directionally sane, including `SHARED > MISMATCHED_SHARED`.
- Geometry-to-brain coupling is computed, but the current evidence is weak and should be interpreted cautiously.
- Robustness is supportive overall, but not uniformly clean across every perturbation.

## Reading Order

If you want the fastest path through the release, read:

1. [Final paper PDF](https://github.com/aliuyar1234/language-neutral-or-language-specific/raw/main/paper/final/language_neutral_or_language_specific_disentangling_multilingual_llm_subspaces_with_naturalistic_fmri_ali_uyar.pdf)
2. [Claim-evidence map](outputs/manuscript/claim_evidence_map.md)
3. [Figure provenance](outputs/manuscript/figure_provenance.md)
4. [Table provenance](outputs/manuscript/table_provenance.md)
5. [Encoding and statistics](docs/06_encoding_and_statistics.md)
6. [Spec deviation log](docs/15_spec_deviation_log.md)

## Repository Structure

- `src/brain_subspace_paper/`: end-to-end implementation of data handling, alignment, model extraction, feature decomposition, encoding, statistics, and figure/table generation
- `configs/`: canonical YAML configuration for project scope, models, ROIs, outputs, and pipeline defaults
- `docs/`: detailed methods, data contracts, implementation workplan, and public-facing deviation log
- `paper/`: manuscript source, captions, bibliography seed, and final PDF
- `outputs/`: canonical release artifacts used to support the manuscript claims

## Reproducibility Surface

This repository is intended to be inspectable and rerunnable without hidden context. The public release includes:

- code and configuration needed to understand the analysis
- canonical figures, tables, and statistical outputs used in the manuscript
- claim-to-evidence tracing for the main scientific statements
- provenance files for figure and table generation
- execution and deviation logs under `outputs/logs/`

Raw LPPC data, large intermediate caches, and local scratch products are intentionally not committed in full. The repository instead exposes the contracts, manifests, and generated paper-facing outputs needed to audit the release.

## Important Caveats

- The release is ROI-first. Whole-brain outputs are secondary.
- Figure 8 is a descriptive ROI-projected cortex visualization, not a voxelwise inferential whole-brain analysis.
- Cross-language pooled magnitudes should be interpreted cautiously because the language cohorts come from different sites and scanners.
- The current canonical stats bundle inherits documented fast-paper implementation shortcuts; see [docs/15_spec_deviation_log.md](docs/15_spec_deviation_log.md).
- The manuscript is careful not to overclaim a clean semantic-versus-auditory dissociation or a universal language-invariant code.

## Key Files

- [SSOT.md](SSOT.md): canonical research specification
- [ERRATA_AND_IMPLEMENTATION_DECISIONS.md](ERRATA_AND_IMPLEMENTATION_DECISIONS.md): resolved defaults and implementation decisions
- [METHOD_RATIONALE.md](METHOD_RATIONALE.md): explanatory rationale for key design choices
- [docs/02_datasets_and_downloads.md](docs/02_datasets_and_downloads.md): data acquisition instructions
- [docs/04_alignment_algorithm.md](docs/04_alignment_algorithm.md): multilingual alignment procedure
- [docs/06_encoding_and_statistics.md](docs/06_encoding_and_statistics.md): encoding and inference details
- [outputs/tables/table03_main_confirmatory_stats.csv](outputs/tables/table03_main_confirmatory_stats.csv): primary confirmatory statistics
- [outputs/tables/table05_robustness_summary.csv](outputs/tables/table05_robustness_summary.csv): robustness summary
- [outputs/manuscript/claim_evidence_map.md](outputs/manuscript/claim_evidence_map.md): manuscript claim grounding

## Citation

If you use this repository or build on the released analysis, please cite the paper and the Zenodo archive linked above.
