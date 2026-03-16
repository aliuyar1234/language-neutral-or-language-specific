# 02 — Datasets and Downloads

This document defines the exact data sources and how to obtain them.

## 1. Core dataset

### LPPC-fMRI
- name: Le Petit Prince multilingual naturalistic fMRI corpus
- OpenNeuro accession: `ds003643`
- public dataset page: `https://openneuro.org/datasets/ds003643`
- public GitHub mirror: `https://github.com/OpenNeuroDatasets/ds003643`

Core languages:
- English
- Chinese
- French

Public annotations and files of interest:
- `annotation/EN`, `annotation/CN`, `annotation/FR`
- `derivatives/`
- `stimuli/`
- subject directories such as `sub-EN057`, `sub-CN001`, `sub-FR001`

## 2. Optional appendix dataset

### studyforrest / Forrest Gump
- OpenNeuro accession: `ds000113`
- public dataset page: `https://openneuro.org/datasets/ds000113`
- public GitHub mirror: `https://github.com/OpenNeuroDatasets/ds000113`

Use this dataset only after the core LPPC paper is complete.

## 3. Download methods

Use one of the two methods below.

### Method A — Official OpenNeuro CLI (preferred)
This is the official route.

Install:
```bash
# install Deno first if needed
deno install -A --global jsr:@openneuro/cli -n openneuro
```

Login:
```bash
export OPENNEURO_API_KEY=<your_api_key>
openneuro login --error-reporting true
```

Download LPPC snapshot:
```bash
mkdir -p data/raw
openneuro download ds003643 data/raw/ds003643
```

Then retrieve annexed files:
```bash
cd data/raw/ds003643
datalad get annotation stimuli derivatives
```

### Method B — Public GitHub mirror + DataLad / git-annex fallback
If CLI login becomes annoying, use the public GitHub mirror.

```bash
mkdir -p data/raw
cd data/raw
git clone https://github.com/OpenNeuroDatasets/ds003643.git
cd ds003643
datalad get annotation stimuli derivatives
```

If `datalad get` fails, inspect the repository’s git-annex configuration and try `git annex get .` after ensuring the OpenNeuro special remote is configured.

## 4. Required LPPC subtrees

The core paper only requires these subtrees:

- `annotation/`
- `derivatives/`
- enough metadata to inspect subject/run structure
- optional `stimuli/` if acoustic features need to be recomputed or verified

Raw functional MRI is **not** required for the main paper if derivatives are intact.

## 5. Download integrity checks

After download, create `outputs/logs/data_integrity_report.md` and confirm:

1. `annotation/EN`, `annotation/CN`, `annotation/FR` exist
2. `derivatives/` exists
3. all three language groups have expected subject counts
4. derivative NIfTI files exist for each subject and roughly 9 runs per subject
5. run manifests can be built without assuming consecutive run labels
6. annotation files include:
   - `*_section1.TextGrid` ... `*_section9.TextGrid`
   - `*_prosody.csv`
   - `*_word_information.csv`
   - `*_tree.csv` or equivalent
   - `*_dependency.csv`

## 6. Canonical LPPC file expectations

### 6.1 Annotation
Expected structure:
```text
annotation/
  EN/
    lppEN_section1.TextGrid
    ...
    lppEN_section9.TextGrid
    lppEN_prosody.csv
    lppEN_word_information.csv
    lppEN_tree.csv
    lppEN_dependency.csv
    ...
  CN/
    ...
  FR/
    ...
```

### 6.2 Derivatives
Expected structure:
```text
derivatives/
  sub-EN057/
    func/
      sub-EN057_task-lppEN_run-15_space-MNIColin27_desc-preproc_bold.nii.gz
      ...
  sub-CN001/
    func/
      ...
  sub-FR001/
    func/
      ...
```

Important:
- the original run labels in derivative filenames may not be 1..9
- canonical run indices must be discovered by sorting run labels within subject

## 7. LPPC run-manifest algorithm

### Main-analysis subject inclusion policy
For the confirmatory core analysis, keep only subjects with:
- all 9 canonical runs present
- exact scan counts after canonical remapping by default, with only isolated documented plus_minus_one exceptions
- all required annotation sections present

Operational default from the errata:
- exact match by default
- +/-1 volume allowed only when isolated and explicitly documented
- otherwise exclude from the main confirmatory analysis

This is deliberate. It avoids variable-fold edge cases in the core paper.


For each subject:

1. collect all derivative preprocessed BOLD files for that subject’s language task
2. parse the original run label from filename
3. sort by original run label ascending
4. assign canonical run index `1..R`
5. record:
   - `subject_id`
   - `language`
   - `original_run_label`
   - `canonical_run_index`
   - `n_volumes`
   - `filepath`

Save as:
- `data/interim/lppc_run_manifest.parquet`
- `outputs/logs/data_integrity_report.md`

## 8. LPPC sanity-check counts

Use the expected preprocessed scan counts in `SSOT.md`.

Expected behavior:
- exact match is preferred
- a difference of 1 volume may be tolerated only if a documented dataset quirk explains it
- otherwise the subject should be excluded from the main confirmatory analysis and the reason logged

If a subject’s discovered run volume counts differ severely from the language template:
- inspect for corrupted download
- inspect for missing runs
- inspect wrong file selection
- do not proceed blindly

## 9. Visual-picture nuisance events

For English and Chinese section 1:
- check whether event or metadata files already encode the picture timings
- if not, use the published timings in `SSOT.md`
- build picture-event and picture-block nuisance regressors only for run 1 of EN/CN

## 10. Acoustic features

Preferred route:
- use the provided `*_prosody.csv` files for RMS/f0
- use `TextGrid` and/or `word_information.csv` to derive word onsets

If shipped prosody summaries are missing or corrupt, recompute RMS and f0 from public audio.
If neither source exists, stop and inspect rather than silently dropping the acoustic baseline.

## 11. Free-access note

The core data route must remain:
- public
- scriptable
- reproducible
- free of charge

If any chosen workflow requires payment or private access, it violates scope and must be changed.

## 12. studyforrest appendix note

If the optional German appendix is attempted later:
- use only public files from `ds000113`
- use the public speech annotation resource associated with studyforrest
- keep it clearly separate from the core LPPC claims
