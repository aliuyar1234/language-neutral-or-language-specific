from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import io
import gzip
import re
import struct

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import pandas as pd

from brain_subspace_paper.config import project_config, project_root
from brain_subspace_paper.logging_utils import append_markdown_log, bootstrap_logs, write_text


LANGUAGE_CODE_MAP = {"EN": "en", "FR": "fr", "CN": "zh"}
TASK_MAP = {"en": "lppEN", "fr": "lppFR", "zh": "lppCN"}
OPENNEURO_BUCKET = "openneuro.org"

BOLD_PATTERN = re.compile(
    r"sub-(?P<subject_lang>EN|FR|CN)\d+_task-(?P<task>lppEN|lppFR|lppCN)_run-(?P<run>\d+)_space-(?P<space>[^_]+)_desc-preproc_bold\.nii(?:\.gz)?$"
)


@dataclass(slots=True)
class InspectionSummary:
    manifest_path: Path
    report_path: Path
    rows: int
    included_subjects: int
    excluded_subjects: int


def _dataset_root() -> Path:
    root = project_root()
    raw_root = root / project_config()["paths"]["raw_data_root"]
    accession = project_config()["data"]["core_dataset"]["accession"]
    return raw_root / accession


def _unsigned_s3_client():
    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 8, "mode": "standard"}),
    )


def _parse_nifti_header(header: bytes, source: str) -> int:
    if len(header) < 56:
        raise ValueError(f"File is too small to be a NIfTI image: {source}")

    little = struct.unpack("<i", header[0:4])[0]
    big = struct.unpack(">i", header[0:4])[0]
    if little == 348:
        endian = "<"
    elif big == 348:
        endian = ">"
    else:
        raise ValueError(f"Unrecognized NIfTI header size in {source}")

    dims = struct.unpack(f"{endian}8h", header[40:56])
    ndim = dims[0]
    if ndim >= 4 and dims[4] > 0:
        return int(dims[4])
    return 1


def _read_nifti_n_volumes(path: Path, dataset_root: Path) -> int:
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as handle:
            header = handle.read(348)
        return _parse_nifti_header(header, str(path))
    except Exception:
        accession = project_config()["data"]["core_dataset"]["accession"]
        key = f"{accession}/{path.relative_to(dataset_root).as_posix()}"
        client = _unsigned_s3_client()
        for stop in (4095, 16383, 65535, 262143):
            response = client.get_object(
                Bucket=OPENNEURO_BUCKET,
                Key=key,
                Range=f"bytes=0-{stop}",
            )
            chunk = response["Body"].read()
            if path.suffix == ".gz":
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(chunk), mode="rb") as handle:
                        header = handle.read(348)
                except Exception:
                    continue
            else:
                header = chunk[:348]
            try:
                return _parse_nifti_header(header, key)
            except ValueError:
                continue
        raise ValueError(f"Could not read a valid NIfTI header from {key}")


def _manifest_row_for_item(
    *,
    root: Path,
    dataset_root: Path,
    subject_id: str,
    language: str,
    canonical_index: int,
    item: dict[str, object],
) -> dict[str, object]:
    path = Path(item["path"])
    volume_count = _read_nifti_n_volumes(path, dataset_root=dataset_root)
    return {
        "subject_id": subject_id,
        "language": language,
        "task_name": item["task_name"],
        "original_run_label": int(item["original_run_label"]),
        "canonical_run_index": canonical_index,
        "n_volumes": volume_count,
        "filepath": path.relative_to(root).as_posix(),
        "space": item["space"],
        "is_preproc": True,
    }


def _annotation_summary(annotation_root: Path) -> tuple[dict[str, dict[str, object]], list[str]]:
    issues: list[str] = []
    summary: dict[str, dict[str, object]] = {}
    for lang, folder in {"en": "EN", "zh": "CN", "fr": "FR"}.items():
        lang_dir = annotation_root / folder
        section_presence = [(lang_dir / f"lpp{folder}_section{section}.TextGrid").exists() for section in range(1, 10)]
        fallback_section_presence = [bool(list(lang_dir.glob(f"*section{section}.TextGrid"))) for section in range(1, 10)]
        sections_ok = all(a or b for a, b in zip(section_presence, fallback_section_presence))
        required_patterns = {
            "prosody": "*prosody.csv",
            "word_information": "*word_information.csv",
            "tree": "*tree.csv",
            "dependency": "*dependency.csv",
        }
        files_present = {name: bool(list(lang_dir.glob(pattern))) for name, pattern in required_patterns.items()}
        summary[lang] = {
            "exists": lang_dir.exists(),
            "sections_ok": sections_ok,
            "files_present": files_present,
        }
        if not lang_dir.exists():
            issues.append(f"Missing annotation directory for {lang}: {lang_dir}")
        if not sections_ok:
            issues.append(f"Missing one or more section TextGrids for {lang}.")
        for name, present in files_present.items():
            if not present:
                issues.append(f"Missing annotation file pattern `{name}` for {lang}.")
    return summary, issues


def build_run_manifest() -> InspectionSummary:
    bootstrap_logs()
    root = project_root()
    dataset_root = _dataset_root()
    derivatives_root = dataset_root / "derivatives"
    annotation_root = dataset_root / "annotation"

    rows: list[dict[str, object]] = []
    subject_issues: list[str] = []
    summary_rows: list[dict[str, object]] = []
    expected_counts = project_config()["lppc_expected_preproc_scan_counts"]
    tolerance = int(project_config()["subject_inclusion"]["scan_count_tolerance_volumes"])

    annotation_summary, annotation_issues = _annotation_summary(annotation_root)

    subject_dirs = sorted(derivatives_root.glob("sub-*"))
    for subject_number, subject_dir in enumerate(subject_dirs, start=1):
        print(f"Inspecting {subject_dir.name} ({subject_number}/{len(subject_dirs)})")
        func_dir = subject_dir / "func"
        if not func_dir.exists():
            subject_issues.append(f"{subject_dir.name}: missing func directory.")
            continue

        bold_files = sorted(func_dir.glob("*_desc-preproc_bold.nii.gz"))
        if not bold_files:
            bold_files = sorted(func_dir.glob("*_desc-preproc_bold.nii"))
        parsed: list[dict[str, object]] = []
        for path in bold_files:
            match = BOLD_PATTERN.match(path.name)
            if not match:
                continue
            language = LANGUAGE_CODE_MAP[match.group("subject_lang")]
            parsed.append(
                {
                    "path": path,
                    "language": language,
                    "task_name": match.group("task"),
                    "original_run_label": int(match.group("run")),
                    "space": match.group("space"),
                }
            )

        if not parsed:
            subject_issues.append(f"{subject_dir.name}: no matching preprocessed BOLD files found.")
            continue

        parsed.sort(key=lambda item: int(item["original_run_label"]))
        language = str(parsed[0]["language"])
        expected_task = TASK_MAP[language]
        if any(str(item["task_name"]) != expected_task for item in parsed):
            subject_issues.append(f"{subject_dir.name}: mixed or unexpected task labels.")

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_rows = [
                executor.submit(
                    _manifest_row_for_item,
                    root=root,
                    dataset_root=dataset_root,
                    subject_id=subject_dir.name,
                    language=language,
                    canonical_index=canonical_index,
                    item=item,
                )
                for canonical_index, item in enumerate(parsed, start=1)
            ]
            subject_rows = [future.result() for future in future_rows]

        subject_rows.sort(key=lambda row: int(row["canonical_run_index"]))
        n_volumes = [int(row["n_volumes"]) for row in subject_rows]
        rows.extend(subject_rows)

        expected = list(expected_counts[language])
        mismatch_diffs = []
        if len(n_volumes) == len(expected):
            mismatch_diffs = [actual - exp for actual, exp in zip(n_volumes, expected)]
        include_main = False
        if len(n_volumes) == 9:
            non_zero = [diff for diff in mismatch_diffs if diff != 0]
            if not non_zero:
                include_main = True
            elif len(non_zero) == 1 and abs(non_zero[0]) <= tolerance:
                include_main = True

        summary_rows.append(
            {
                "subject_id": subject_dir.name,
                "language": language,
                "n_runs": len(n_volumes),
                "scan_count_diffs": mismatch_diffs,
                "main_analysis_included": include_main,
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise FileNotFoundError(
            f"No LPPC preprocessed BOLD files were discovered under {derivatives_root.as_posix()}."
        )

    manifest.sort_values(
        by=["language", "subject_id", "canonical_run_index"],
        inplace=True,
    )
    manifest_path = root / "data" / "interim" / "lppc_run_manifest.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)

    subject_summary = pd.DataFrame(summary_rows)
    included_subjects = int(subject_summary["main_analysis_included"].sum()) if not subject_summary.empty else 0
    excluded_subjects = int(len(subject_summary) - included_subjects)

    subject_counts = (
        manifest.groupby("language")["subject_id"].nunique().to_dict()
        if not manifest.empty
        else {}
    )
    expected_subject_counts = project_config()["lppc_expected_subject_counts"]

    report_lines = [
        "# Data Integrity Report",
        "",
        f"- dataset root: `{dataset_root.as_posix()}`",
        f"- annotation exists: `{annotation_root.exists()}`",
        f"- derivatives exists: `{derivatives_root.exists()}`",
        "",
        "## Annotation Summary",
        "",
    ]
    for language in ("en", "zh", "fr"):
        entry = annotation_summary.get(language, {})
        files_present = entry.get("files_present", {})
        report_lines.extend(
            [
                f"### {language}",
                f"- directory exists: `{entry.get('exists', False)}`",
                f"- sections 1-9 present: `{entry.get('sections_ok', False)}`",
                f"- prosody csv present: `{files_present.get('prosody', False)}`",
                f"- word_information csv present: `{files_present.get('word_information', False)}`",
                f"- tree csv present: `{files_present.get('tree', False)}`",
                f"- dependency csv present: `{files_present.get('dependency', False)}`",
                "",
            ]
        )

    report_lines.extend(
        [
            "## Derivative Summary",
            "",
            f"- total manifest rows: `{len(manifest)}`",
            f"- subjects included for main confirmatory analysis: `{included_subjects}`",
            f"- subjects excluded from main confirmatory analysis: `{excluded_subjects}`",
            "",
        ]
    )
    for language in ("en", "zh", "fr"):
        report_lines.extend(
            [
                f"### {language}",
                f"- discovered subjects: `{subject_counts.get(language, 0)}`",
                f"- expected subjects: `{expected_subject_counts[language]}`",
                "",
            ]
        )

    issues = annotation_issues + subject_issues
    mismatched_subjects = subject_summary.loc[~subject_summary["main_analysis_included"]] if not subject_summary.empty else pd.DataFrame()
    if not mismatched_subjects.empty:
        report_lines.extend(["## Excluded Or Flagged Subjects", ""])
        for row in mismatched_subjects.itertuples(index=False):
            report_lines.append(
                f"- `{row.subject_id}` ({row.language}) runs={row.n_runs} diffs={row.scan_count_diffs}"
            )
        report_lines.append("")

    if issues:
        report_lines.extend(["## Issues", ""])
        for issue in issues:
            report_lines.append(f"- {issue}")
        report_lines.append("")

    report_path = root / "outputs" / "logs" / "data_integrity_report.md"
    write_text(report_path, "\n".join(report_lines))

    append_markdown_log(
        root / "outputs" / "logs" / "progress_log.md",
        "LPPC inspection",
        [
            f"Built canonical run manifest at {manifest_path.relative_to(root).as_posix()}.",
            f"Wrote data integrity report to {report_path.relative_to(root).as_posix()}.",
            f"Included {included_subjects} subjects and excluded {excluded_subjects} subjects for the main confirmatory policy.",
        ],
    )

    return InspectionSummary(
        manifest_path=manifest_path,
        report_path=report_path,
        rows=len(manifest),
        included_subjects=included_subjects,
        excluded_subjects=excluded_subjects,
    )
