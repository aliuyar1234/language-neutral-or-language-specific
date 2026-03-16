# Language-Neutral or Language-Specific? Codex SSOT Handoff

This package is the **single source of truth (SSOT)** for implementing, executing, analyzing, and writing the paper:

**Language-Neutral or Language-Specific? Disentangling Multilingual LLM Subspaces with Naturalistic fMRI**

It is designed as a handoff to Codex CLI. Codex should be able to implement the project without access to any prior chat context.

## What this package contains

- the scientific thesis and why it was chosen
- the exact core scope and non-negotiables
- all non-obvious formulas and algorithms
- the rationale behind the major method and design decisions
- exact dataset and model choices
- download instructions and file-structure expectations
- alignment, encoding, statistics, and figure specifications
- paper-writing rules and a manuscript template
- anti-drift guardrails
- a machine-readable work breakdown
- a starter repo scaffold with stub modules and scripts
- an errata file with resolved ambiguities and exact defaults

## Read order for Codex

1. `AGENTS.md`
2. `README.md`
3. `SSOT.md`
4. `ERRATA_AND_IMPLEMENTATION_DECISIONS.md`
5. `docs/02_datasets_and_downloads.md`
6. `docs/03_models_and_extraction.md`
7. `docs/04_alignment_algorithm.md`
8. `docs/05_novel_method_and_formulas.md`
9. `docs/06_encoding_and_statistics.md`
10. `docs/07_figures_tables_and_outputs.md`
11. `docs/10_data_contracts.md`
12. `docs/11_implementation_workplan.md`
13. `tasks/work_breakdown.yaml`
14. `configs/*.yaml`
15. `paper/manuscript_template.md`

## Precedence rules

- `AGENTS.md` controls **operational behavior**.
- `SSOT.md` controls **research content and thesis direction**.
- `ERRATA_AND_IMPLEMENTATION_DECISIONS.md` resolves known ambiguities and exact defaults.
- `docs/*.md` provide detailed specifications.
- `configs/*.yaml` provide machine-readable defaults.
- `repo_scaffold/` is a non-authoritative skeleton; it may be replaced.

If any file conflicts with `SSOT.md` on research content, **`SSOT.md` wins**.
If any lower-priority file conflicts with `ERRATA_AND_IMPLEMENTATION_DECISIONS.md` on a resolved ambiguity, **the errata file wins**.

## Core project scope

Core paper:
- dataset: **LPPC-fMRI / OpenNeuro ds003643**
- models: **FacebookAI/xlm-roberta-base** and **facebook/nllb-200-distilled-600M** (encoder only)
- analysis: **sentence-level**, **ROI-first**, **SHARED vs SPECIFIC vs MISMATCHED_SHARED**
- goal: **complete reproducible paper**, not a benchmark exercise

Optional appendix after core success:
- German studyforrest extension (`ds000113`)

## Why this project exists

The paper is not about benchmark-chasing. It asks a mechanistic question:

> When the same story is heard in different languages, does the brain track a language-neutral shared component of multilingual LLM representations more strongly than a target-language-specific residual?

The novelty is the explicit factorization of multilingual sentence representations and the test of that factorization against naturalistic fMRI.

## Contents

- `AGENTS.md` - strict instructions for Codex
- `SSOT.md` - canonical thesis, formulas, hypotheses, scope
- `ERRATA_AND_IMPLEMENTATION_DECISIONS.md` - resolved ambiguities and exact implementation defaults
- `METHOD_RATIONALE.md` - why the major method and design choices were fixed this way
- `docs/` - detailed specifications
- `configs/` - exact defaults and manifests
- `tasks/` - stepwise work breakdown and acceptance criteria
- `paper/` - manuscript and figure-caption templates
- `prompt_pack/` - prompts to paste into Codex CLI
- `repo_scaffold/` - starter repository layout with stub modules

## Important non-negotiables

- Do **not** change the thesis.
- Do **not** turn this into a benchmark paper.
- Do **not** skip the `MISMATCHED_SHARED` control.
- Do **not** compare absolute encoding scores across languages as if scanner/site differences do not matter.
- Do **not** overclaim "syntax vs semantics."
- Do **not** make the German extension the main paper.

## Human quick-start

If a human operator uses this package with Codex CLI, the safest sequence is:

1. unpack the zip
2. start Codex in the unpacked directory
3. paste `prompt_pack/master_prompt.md`
4. ask Codex to execute the work plan stage by stage
5. require Codex to keep all claims tied to generated tables and figures

## Current status and session handoff

For a fresh session that needs to recover context quickly, read these after the canonical SSOT files:

- `docs/11_implementation_workplan.md` for the stable stage structure
- `docs/14_current_status_and_session_runbook.md` for the mutable "what is done / what is next" status
- `outputs/logs/progress_log.md` for the dated execution trail

The runbook is the main operational handoff file and should be updated at the end of every work session.

## Expected end state

A completed repo should produce:

- downloaded and indexed LPPC data
- aligned sentence triplets across EN/FR/ZH
- cached hidden states for XLM-R and NLLB encoder layers
- SHARED / SPECIFIC / RAW / FULL / MISMATCHED_SHARED feature sets
- ROI encoding results with cross-validation
- statistics tables, robustness outputs, and all required paper figures
- a completed manuscript grounded in actual outputs
- a claim-evidence map linking each conclusion to specific figures/tables
