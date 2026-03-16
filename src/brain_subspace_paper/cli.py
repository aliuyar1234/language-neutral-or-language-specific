from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import typer

from brain_subspace_paper.config import project_root
from brain_subspace_paper.data.download import download_lppc
from brain_subspace_paper.data.inspect_lppc import build_run_manifest
from brain_subspace_paper.data.sentence_spans import build_all_sentence_spans, build_sentence_spans
from brain_subspace_paper.encoding.english_prototype import run_english_roi_prototype
from brain_subspace_paper.encoding.xlmr_roi_pipeline import (
    merge_subject_result_chunks,
    run_nllb_roi_pipeline,
    run_xlmr_roi_pipeline,
)
from brain_subspace_paper.features.decomposition import build_decomposition_features
from brain_subspace_paper.logging_utils import append_markdown_log, bootstrap_logs
from brain_subspace_paper.models.download import download_model
from brain_subspace_paper.roi.targets import extract_roi_targets
from brain_subspace_paper.stats.paper_level import build_paper_level_stats
from brain_subspace_paper.stats.robustness import build_paper_robustness, merge_paper_robustness
from brain_subspace_paper.stats.whole_brain import build_paper_whole_brain
from brain_subspace_paper.viz import build_paper_figures, build_paper_tables


app = typer.Typer(
    add_completion=False,
    help="CLI for the multilingual shared-vs-specific fMRI paper.",
    no_args_is_help=True,
)


def _run_subprocess_python(code: str, *, env: dict[str, str] | None = None) -> None:
    try:
        subprocess.run([sys.executable, "-c", code], check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise typer.Exit(code=exc.returncode) from exc


def _parse_layer_indices(spec: str) -> tuple[int, ...]:
    values = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise typer.BadParameter("At least one layer index is required.")
    return tuple(values)


def _parse_optional_layer_indices(spec: str) -> tuple[int, ...] | None:
    if spec.strip().lower() == "all":
        return None
    return _parse_layer_indices(spec)


def _parse_csv_values(spec: str) -> tuple[str, ...]:
    values = tuple(chunk.strip() for chunk in spec.split(",") if chunk.strip())
    if not values:
        raise typer.BadParameter("At least one value is required.")
    return values


def _parse_csv_paths(spec: str) -> tuple[Path, ...]:
    paths = tuple(Path(chunk.strip()) for chunk in spec.split(",") if chunk.strip())
    if not paths:
        raise typer.BadParameter("At least one path is required.")
    return paths


def _model_stem(model_name: str) -> str:
    return "nllb" if model_name == "nllb_encoder" else model_name


def _fast_chunk_tag(model_name: str, language: str, tag_suffix: str) -> str:
    return f"{_model_stem(model_name)}_{language}_{tag_suffix}"


def _subject_result_path_for_tag(model_name: str, output_tag: str) -> Path:
    stem = _model_stem(model_name)
    return project_root() / "outputs" / "stats" / f"{stem}_subject_level_roi_results__{output_tag}.parquet"


def _subject_metadata_path_for_tag(model_name: str, output_tag: str) -> Path:
    subject_path = _subject_result_path_for_tag(model_name, output_tag)
    return subject_path.with_name(f"{subject_path.stem}__metadata.json")


def _chunk_launcher_log_path(model_name: str, language: str, output_tag: str) -> Path:
    stem = _model_stem(model_name)
    return project_root() / "outputs" / "logs" / "chunk_launcher" / f"{stem}_{language}__{output_tag}.log"


def _merged_output_paths(model_name: str, output_tag: str) -> dict[str, Path]:
    stem = _model_stem(model_name)
    root = project_root()
    return {
        "subject": root / "outputs" / "stats" / f"{stem}_subject_level_roi_results__{output_tag}.parquet",
        "group": root / "outputs" / "stats" / f"{stem}_group_level_roi_results__{output_tag}.parquet",
        "confirmatory": root / "outputs" / "stats" / f"{stem}_confirmatory_effects__{output_tag}.parquet",
        "report": root / "outputs" / "logs" / f"{stem}_encoding_qc_report__{output_tag}.md",
    }


def _thread_capped_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def _run_parallel_fast_chunks(
    *,
    model_name: str,
    languages: tuple[str, ...],
    layers: str,
    max_subjects: int | None,
    mismatched_shuffles: int,
    tag_suffix: str,
    resume: bool,
) -> tuple[Path, ...]:
    bootstrap_logs()
    command_name = "run-xlmr-roi-pipeline" if model_name == "xlmr" else "run-nllb-roi-pipeline"
    env = _thread_capped_env()
    chunk_paths = tuple(_subject_result_path_for_tag(model_name, _fast_chunk_tag(model_name, language, tag_suffix)) for language in languages)

    typer.echo(f"Launching {model_name} fast-paper chunks for: {', '.join(languages)}")
    typer.echo("BLAS thread caps set to 1 for each child process.")
    with ExitStack() as stack:
        processes: list[tuple[str, str, Path, subprocess.Popen[str]]] = []
        for language in languages:
            output_tag = _fast_chunk_tag(model_name, language, tag_suffix)
            log_path = _chunk_launcher_log_path(model_name, language, output_tag)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(log_path.open("w", encoding="utf-8"))
            command = [
                sys.executable,
                "-m",
                "brain_subspace_paper",
                command_name,
                "--languages",
                language,
                "--layers",
                layers,
                "--mismatched-shuffles",
                str(mismatched_shuffles),
                "--subject-only",
                "--output-tag",
                output_tag,
            ]
            command.append("--resume" if resume else "--no-resume")
            if max_subjects is not None:
                command.extend(["--max-subjects", str(max_subjects)])
            process = subprocess.Popen(
                command,
                cwd=project_root(),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((language, output_tag, log_path, process))
            typer.echo(f"- {language}: output_tag={output_tag}")
            typer.echo(f"  log={log_path}")

        failed: list[tuple[str, Path, int]] = []
        for language, output_tag, log_path, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failed.append((language, log_path, return_code))
            else:
                typer.echo(f"Completed {language}: {output_tag}")

    if failed:
        for language, log_path, return_code in failed:
            typer.echo(f"FAILED {language}: exit={return_code} log={log_path}", err=True)
        raise typer.Exit(code=1)

    append_markdown_log(
        project_root() / "outputs" / "logs" / "progress_log.md",
        f"{model_name} fast-paper chunk launcher",
        [
            f"languages={','.join(languages)}",
            f"layers={layers}",
            f"max_subjects={max_subjects if max_subjects is not None else 'all'}",
            f"mismatched_shuffles={mismatched_shuffles}",
            f"tag_suffix={tag_suffix}",
            f"resume={resume}",
            "Completed all chunk jobs successfully.",
        ],
    )
    return chunk_paths


def _merge_fast_chunks(
    *,
    model_name: str,
    languages: tuple[str, ...],
    tag_suffix: str,
    permutations: int,
    bootstraps: int,
    output_tag: str,
) -> None:
    bootstrap_logs()
    input_paths = tuple(_subject_result_path_for_tag(model_name, _fast_chunk_tag(model_name, language, tag_suffix)) for language in languages)
    missing = [path for path in input_paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise typer.BadParameter(
            f"Missing subject-level chunk outputs: {missing_text}. "
            f"Run the corresponding fast chunk command first."
        )

    summary = merge_subject_result_chunks(
        model_name=model_name,
        input_paths=input_paths,
        n_permutations=permutations,
        n_bootstraps=bootstraps,
        output_tag=output_tag,
    )
    append_markdown_log(
        project_root() / "outputs" / "logs" / "progress_log.md",
        f"{model_name} fast-paper merge",
        [
            f"languages={','.join(languages)}",
            f"tag_suffix={tag_suffix}",
            f"permutations={permutations}",
            f"bootstraps={bootstraps}",
            f"output_tag={output_tag}",
            f"subject_results={summary.subject_results_path}",
            f"group_results={summary.group_results_path}",
            f"confirmatory={summary.confirmatory_path}",
        ],
    )
    typer.echo(f"Subject results: {summary.subject_results_path}")
    if summary.group_results_path is not None:
        typer.echo(f"Group results: {summary.group_results_path}")
    if summary.confirmatory_path is not None:
        typer.echo(f"Confirmatory effects: {summary.confirmatory_path}")
    if summary.report_path is not None:
        typer.echo(f"Report: {summary.report_path}")
    typer.echo(f"Languages: {', '.join(summary.languages)}")
    typer.echo(f"Subjects: {summary.n_subjects}")
    typer.echo(f"Rows: {summary.n_rows}")


def _roi_manifest_subject_count(language: str) -> int:
    path = project_root() / "data" / "interim" / "roi" / f"{language}_roi_target_manifest.parquet"
    if not path.exists():
        return 0
    return int(pd.read_parquet(path, columns=["subject_id"])["subject_id"].nunique())


def _chunk_status_rows(
    *,
    model_name: str,
    languages: tuple[str, ...],
    tag_suffix: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for language in languages:
        output_tag = _fast_chunk_tag(model_name, language, tag_suffix)
        subject_path = _subject_result_path_for_tag(model_name, output_tag)
        metadata_path = _subject_metadata_path_for_tag(model_name, output_tag)
        log_path = _chunk_launcher_log_path(model_name, language, output_tag)
        metadata: dict[str, object] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        available_subjects = _roi_manifest_subject_count(language)
        max_subjects = metadata.get("max_subjects")
        target_subjects = available_subjects if max_subjects in (None, "None") else min(int(max_subjects), available_subjects)
        completed_subjects = 0
        total_rows = 0
        if subject_path.exists():
            df = pd.read_parquet(subject_path, columns=["subject_id", "language"])
            completed_subjects = int(df.loc[df["language"] == language, "subject_id"].nunique())
            total_rows = int(len(df))
        if completed_subjects == 0 and not subject_path.exists():
            status = "not_started"
        elif target_subjects > 0 and completed_subjects >= target_subjects:
            status = "complete"
        else:
            status = "partial"
        rows.append(
            {
                "language": language,
                "status": status,
                "completed_subjects": completed_subjects,
                "target_subjects": target_subjects,
                "subject_path": subject_path,
                "metadata_path": metadata_path,
                "log_path": log_path,
                "total_rows": total_rows,
            }
        )
    return rows


def _merge_status(
    *,
    model_name: str,
    output_tag: str,
) -> dict[str, object]:
    paths = _merged_output_paths(model_name, output_tag)
    exists = {name: path.exists() for name, path in paths.items()}
    if all(exists.values()):
        status = "complete"
    elif any(exists.values()):
        status = "partial"
    else:
        status = "not_started"
    return {
        "status": status,
        "paths": paths,
        "exists": exists,
    }


def _next_fast_step(
    *,
    xlmr_chunk_rows: list[dict[str, object]],
    xlmr_merge: dict[str, object],
    nllb_chunk_rows: list[dict[str, object]],
    nllb_merge: dict[str, object],
) -> str:
    if any(str(row["status"]) != "complete" for row in xlmr_chunk_rows):
        return "python -m brain_subspace_paper run-paper-fast-xlmr-chunks"
    if str(xlmr_merge["status"]) != "complete":
        return "python -m brain_subspace_paper merge-paper-fast-xlmr"
    if any(str(row["status"]) != "complete" for row in nllb_chunk_rows):
        return "python -m brain_subspace_paper run-paper-fast-nllb-chunks"
    if str(nllb_merge["status"]) != "complete":
        return "python -m brain_subspace_paper merge-paper-fast-nllb"
    return "Fast-paper Plan B readout complete. Next step: final 10000/10000 merges or T13 implementation."


@app.command("bootstrap")
def bootstrap() -> None:
    """Create required log files and config snapshots."""
    artifacts = bootstrap_logs()
    typer.echo("Bootstrapped logs and snapshots:")
    for name, path in artifacts.items():
        typer.echo(f"- {name}: {path}")


@app.command("download-lppc")
def download_lppc_command(
    method: str = typer.Option(
        "auto",
        "--method",
        help="Download backend: auto, git_annex, or s3_public.",
    ),
    include_stimuli: bool = typer.Option(
        False,
        "--include-stimuli",
        help="Also download stimuli/ for future acoustic recomputation.",
    ),
    max_files: int | None = typer.Option(
        None,
        "--max-files",
        min=1,
        help="Limit download for smoke testing.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Re-download files even if a same-sized local copy exists.",
    ),
) -> None:
    """Download the required LPPC dataset files from the public OpenNeuro S3 bucket."""
    summary = download_lppc(
        method=method,
        include_stimuli=include_stimuli,
        max_files=max_files,
        overwrite=overwrite,
    )
    typer.echo(f"Dataset root: {summary.dataset_root}")
    typer.echo(f"Method: {summary.method}")
    typer.echo(f"Selected prefixes: {', '.join(summary.selected_prefixes)}")
    typer.echo(f"Objects considered: {summary.object_count}")
    typer.echo(f"Files downloaded: {summary.downloaded_files}")
    typer.echo(f"Files skipped: {summary.skipped_files}")
    typer.echo(f"Bytes transferred: {summary.downloaded_bytes}")
    typer.echo(f"Manifest: {summary.manifest_path}")


@app.command("build-run-manifest")
@app.command("inspect-lppc")
def inspect_lppc_command() -> None:
    """Build the canonical LPPC run manifest and data integrity report."""
    summary = build_run_manifest()
    typer.echo(f"Manifest: {summary.manifest_path}")
    typer.echo(f"Report: {summary.report_path}")
    typer.echo(f"Rows: {summary.rows}")
    typer.echo(f"Included subjects: {summary.included_subjects}")
    typer.echo(f"Excluded subjects: {summary.excluded_subjects}")


@app.command("build-sentence-spans")
def build_sentence_spans_command(
    language: str = typer.Option(
        "all",
        "--language",
        help="Language to build: en, fr, zh, or all.",
    ),
) -> None:
    """Build sentence-span tables from LPPC annotations."""
    if language == "all":
        summaries = build_all_sentence_spans()
    else:
        summaries = [build_sentence_spans(language)]
    for summary in summaries:
        typer.echo(f"{summary.language}: {summary.n_sentences} sentences -> {summary.output_path}")
        typer.echo(f"report: {summary.report_path}")


@app.command("build-alignment-triplets")
def build_alignment_triplets_command() -> None:
    """Build the multilingual EN/FR/ZH triplet table and alignment QC artifacts."""
    _run_subprocess_python(
        "\n".join(
            [
                "import torch",
                "from brain_subspace_paper.data.alignment import build_alignment_triplets",
                "summary = build_alignment_triplets()",
                "print(f'Triplets: {summary.n_triplets} -> {summary.triplets_path}')",
                "print(f'QC: {summary.qc_path}')",
                "print(f'Report: {summary.report_path}')",
                "print(f'TSV: {summary.tsv_path}')",
                "print(f'Flagged: {summary.n_flagged}')",
                "print(f'Conflict windows: {summary.n_conflict_windows}')",
                "print(f'Unresolved: {summary.n_unresolved}')",
            ]
        )
    )


@app.command("download-model")
def download_model_command(
    model: str = typer.Option(
        ...,
        "--model",
        help="Model slug: xlmr, nllb_encoder, or labse.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download even if the local directory already looks complete.",
    ),
) -> None:
    """Download one of the required Hugging Face models into the local models/ directory."""
    summary = download_model(model_name=model, force=force)
    typer.echo(f"Model: {summary.model_name}")
    typer.echo(f"HF id: {summary.hf_id}")
    typer.echo(f"Local dir: {summary.local_dir}")
    typer.echo(f"Downloaded: {summary.downloaded}")


@app.command("extract-xlmr-pilot")
def extract_xlmr_pilot_command(
    n_triplets: int = typer.Option(
        100,
        "--n-triplets",
        min=1,
        help="Number of aligned triplets to extract for the XLM-R pilot stage.",
    ),
    max_tokens_per_batch: int = typer.Option(
        2048,
        "--max-tokens-per-batch",
        min=64,
        help="Approximate token budget per XLM-R forward batch.",
    ),
) -> None:
    """Extract XLM-R hidden states for the pilot triplet subset and write geometry QC artifacts."""
    _run_subprocess_python(
        "\n".join(
            [
                "import torch",
                "from brain_subspace_paper.models.extraction import extract_xlmr_pilot",
                f"summary = extract_xlmr_pilot(n_triplets={n_triplets}, max_tokens_per_batch={max_tokens_per_batch})",
                "print(f'Model: {summary.model_name}')",
                "print(f'Triplets: {summary.n_triplets}')",
                "print(f'Layers: {summary.n_layers}')",
                "print(f'Hidden size: {summary.hidden_size}')",
                "print(f'Triplet ids: {summary.triplet_ids_path}')",
                "print(f'Manifest: {summary.manifest_path}')",
                "print(f'Token metadata: {summary.token_metadata_path}')",
                "print(f'Geometry: {summary.geometry_path}')",
                "print(f'Plot: {summary.plot_path}')",
                "print(f'Report: {summary.report_path}')",
            ]
        )
    )


@app.command("extract-xlmr-full")
def extract_xlmr_full_command(
    max_tokens_per_batch: int = typer.Option(
        2048,
        "--max-tokens-per-batch",
        min=64,
        help="Approximate token budget per XLM-R forward batch.",
    ),
) -> None:
    """Extract the full XLM-R EN/FR/ZH cache into the contract embedding paths."""
    _run_subprocess_python(
        "\n".join(
            [
                "import torch",
                "from brain_subspace_paper.models.extraction import extract_xlmr_full",
                f"summary = extract_xlmr_full(max_tokens_per_batch={max_tokens_per_batch})",
                "print(f'Model: {summary.model_name}')",
                "print(f'Triplets: {summary.n_triplets}')",
                "print(f'Layers: {summary.n_layers}')",
                "print(f'Hidden size: {summary.hidden_size}')",
                "print(f'Triplet ids: {summary.triplet_ids_path}')",
                "print(f'Manifest: {summary.manifest_path}')",
                "print(f'Token metadata: {summary.token_metadata_path}')",
            ]
        )
    )


@app.command("extract-nllb-full")
def extract_nllb_full_command(
    max_tokens_per_batch: int = typer.Option(
        2048,
        "--max-tokens-per-batch",
        min=64,
        help="Approximate token budget per NLLB encoder forward batch.",
    ),
) -> None:
    """Extract the full NLLB encoder EN/FR/ZH cache into the contract embedding paths."""
    _run_subprocess_python(
        "\n".join(
            [
                "import torch",
                "from brain_subspace_paper.models.extraction import extract_nllb_full",
                f"summary = extract_nllb_full(max_tokens_per_batch={max_tokens_per_batch})",
                "print(f'Model: {summary.model_name}')",
                "print(f'Triplets: {summary.n_triplets}')",
                "print(f'Layers: {summary.n_layers}')",
                "print(f'Hidden size: {summary.hidden_size}')",
                "print(f'Triplet ids: {summary.triplet_ids_path}')",
                "print(f'Manifest: {summary.manifest_path}')",
                "print(f'Token metadata: {summary.token_metadata_path}')",
            ]
        )
    )


@app.command("build-features")
def build_features_command(
    model: str = typer.Option(
        "xlmr",
        "--model",
        help="Model slug to decompose into RAW/SHARED/SPECIFIC/FULL/MISMATCHED_SHARED.",
    ),
) -> None:
    """Build decomposition feature arrays and manifests from full embedding caches."""
    summary = build_decomposition_features(model_name=model)
    typer.echo(f"Model: {summary.model_name}")
    typer.echo(f"Triplets: {summary.n_triplets}")
    typer.echo(f"Layers: {summary.n_layers}")
    typer.echo(f"Hidden size: {summary.hidden_size}")
    typer.echo(f"Feature arrays: {summary.n_feature_arrays}")
    typer.echo(f"Feature manifest: {summary.feature_manifest_path}")
    typer.echo(f"Permutation manifest: {summary.permutation_manifest_path}")
    typer.echo(f"Report: {summary.report_path}")


@app.command("extract-roi-targets")
def extract_roi_targets_command(
    language: str = typer.Option(
        "en",
        "--language",
        help="Target language: en, fr, zh, or all.",
    ),
    max_subjects: int | None = typer.Option(
        None,
        "--max-subjects",
        min=1,
        help="Optional cap for a faster smoke run.",
    ),
) -> None:
    """Extract run-level ROI timeseries targets for one or more LPPC languages."""
    languages = ("en", "fr", "zh") if language == "all" else _parse_csv_values(language)
    for lang in languages:
        summary = extract_roi_targets(language=lang, max_subjects=max_subjects)
        typer.echo(f"Language: {summary.language}")
        typer.echo(f"Subjects: {summary.n_subjects}")
        typer.echo(f"Runs: {summary.n_runs}")
        typer.echo(f"ROIs: {summary.n_rois}")
        typer.echo(f"Atlas: {summary.atlas_path}")
        typer.echo(f"ROI metadata: {summary.roi_metadata_path}")
        typer.echo(f"Manifest: {summary.manifest_path}")
        typer.echo(f"Report: {summary.report_path}")


@app.command("run-english-roi-prototype")
def run_english_roi_prototype_command(
    max_subjects: int | None = typer.Option(
        None,
        "--max-subjects",
        min=1,
        help="Optional cap for a faster prototype smoke run.",
    ),
    layers: str = typer.Option(
        "0,6,12",
        "--layers",
        help="Comma-separated layer indices to evaluate.",
    ),
) -> None:
    """Run the English XLM-R ROI prototype encoding stage end-to-end."""
    summary = run_english_roi_prototype(
        max_subjects=max_subjects,
        layer_indices=_parse_layer_indices(layers),
    )
    typer.echo(f"Results: {summary.results_path}")
    typer.echo(f"Plot: {summary.plot_path}")
    typer.echo(f"Report: {summary.report_path}")
    typer.echo(f"Subjects: {summary.n_subjects}")
    typer.echo(f"ROIs: {summary.n_rois}")
    typer.echo(f"Layers: {summary.n_layers}")


@app.command("run-xlmr-roi-pipeline")
def run_xlmr_roi_pipeline_command(
    languages: str = typer.Option(
        "en,fr,zh",
        "--languages",
        help="Comma-separated target languages to run.",
    ),
    max_subjects: int | None = typer.Option(
        None,
        "--max-subjects",
        min=1,
        help="Optional cap for a faster staged verification run.",
    ),
    layers: str = typer.Option(
        "all",
        "--layers",
        help="Comma-separated layer indices, or 'all' for every XLM-R layer.",
    ),
    permutations: int | None = typer.Option(
        None,
        "--permutations",
        min=1,
        help="Override the confirmatory permutation count.",
    ),
    bootstraps: int | None = typer.Option(
        None,
        "--bootstraps",
        min=1,
        help="Override the confirmatory bootstrap count.",
    ),
    mismatched_shuffles: int | None = typer.Option(
        None,
        "--mismatched-shuffles",
        min=1,
        help="Override the number of mismatched_shared shuffles used during encoding.",
    ),
    output_tag: str | None = typer.Option(
        None,
        "--output-tag",
        help="Optional suffix for output files so chunked runs do not collide.",
    ),
    subject_only: bool = typer.Option(
        False,
        "--subject-only",
        help="Write only subject-level rows and skip group / confirmatory aggregation.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume/--no-resume",
        help="For subject-only runs, reuse an existing tagged subject parquet and skip completed subjects.",
    ),
) -> None:
    """Run the full XLM-R ROI pipeline and write stage-specific stats tables."""
    summary = run_xlmr_roi_pipeline(
        languages=_parse_csv_values(languages),
        max_subjects=max_subjects,
        layer_indices=_parse_optional_layer_indices(layers),
        n_permutations=permutations,
        n_bootstraps=bootstraps,
        mismatch_shuffles=mismatched_shuffles,
        output_tag=output_tag,
        subject_only=subject_only,
        resume=resume,
    )
    typer.echo(f"Subject results: {summary.subject_results_path}")
    if summary.group_results_path is not None:
        typer.echo(f"Group results: {summary.group_results_path}")
    if summary.confirmatory_path is not None:
        typer.echo(f"Confirmatory effects: {summary.confirmatory_path}")
    if summary.report_path is not None:
        typer.echo(f"Report: {summary.report_path}")
    typer.echo(f"Languages: {', '.join(summary.languages)}")
    typer.echo(f"Subjects: {summary.n_subjects}")
    typer.echo(f"Rows: {summary.n_rows}")


@app.command("run-nllb-roi-pipeline")
def run_nllb_roi_pipeline_command(
    languages: str = typer.Option(
        "en,fr,zh",
        "--languages",
        help="Comma-separated target languages to run.",
    ),
    max_subjects: int | None = typer.Option(
        None,
        "--max-subjects",
        min=1,
        help="Optional cap for a faster staged verification run.",
    ),
    layers: str = typer.Option(
        "all",
        "--layers",
        help="Comma-separated layer indices, or 'all' for every extracted NLLB encoder layer.",
    ),
    permutations: int | None = typer.Option(
        None,
        "--permutations",
        min=1,
        help="Override the confirmatory permutation count.",
    ),
    bootstraps: int | None = typer.Option(
        None,
        "--bootstraps",
        min=1,
        help="Override the confirmatory bootstrap count.",
    ),
    mismatched_shuffles: int | None = typer.Option(
        None,
        "--mismatched-shuffles",
        min=1,
        help="Override the number of mismatched_shared shuffles used during encoding.",
    ),
    output_tag: str | None = typer.Option(
        None,
        "--output-tag",
        help="Optional suffix for output files so chunked runs do not collide.",
    ),
    subject_only: bool = typer.Option(
        False,
        "--subject-only",
        help="Write only subject-level rows and skip group / confirmatory aggregation.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume/--no-resume",
        help="For subject-only runs, reuse an existing tagged subject parquet and skip completed subjects.",
    ),
) -> None:
    """Run the full NLLB encoder ROI pipeline and write stage-specific stats tables."""
    summary = run_nllb_roi_pipeline(
        languages=_parse_csv_values(languages),
        max_subjects=max_subjects,
        layer_indices=_parse_optional_layer_indices(layers),
        n_permutations=permutations,
        n_bootstraps=bootstraps,
        mismatch_shuffles=mismatched_shuffles,
        output_tag=output_tag,
        subject_only=subject_only,
        resume=resume,
    )
    typer.echo(f"Subject results: {summary.subject_results_path}")
    if summary.group_results_path is not None:
        typer.echo(f"Group results: {summary.group_results_path}")
    if summary.confirmatory_path is not None:
        typer.echo(f"Confirmatory effects: {summary.confirmatory_path}")
    if summary.report_path is not None:
        typer.echo(f"Report: {summary.report_path}")
    typer.echo(f"Languages: {', '.join(summary.languages)}")
    typer.echo(f"Subjects: {summary.n_subjects}")
    typer.echo(f"Rows: {summary.n_rows}")


@app.command("merge-roi-results")
def merge_roi_results_command(
    model: str = typer.Option(
        ...,
        "--model",
        help="Model slug: xlmr or nllb_encoder.",
    ),
    inputs: str = typer.Option(
        ...,
        "--inputs",
        help="Comma-separated subject-level parquet paths to merge.",
    ),
    permutations: int | None = typer.Option(
        None,
        "--permutations",
        min=1,
        help="Override the confirmatory permutation count.",
    ),
    bootstraps: int | None = typer.Option(
        None,
        "--bootstraps",
        min=1,
        help="Override the confirmatory bootstrap count.",
    ),
    output_tag: str | None = typer.Option(
        None,
        "--output-tag",
        help="Optional suffix for merged output files.",
    ),
) -> None:
    """Merge chunked subject-level ROI result files and write aggregated outputs."""
    summary = merge_subject_result_chunks(
        model_name=model,
        input_paths=_parse_csv_paths(inputs),
        n_permutations=permutations,
        n_bootstraps=bootstraps,
        output_tag=output_tag,
    )
    typer.echo(f"Subject results: {summary.subject_results_path}")
    if summary.group_results_path is not None:
        typer.echo(f"Group results: {summary.group_results_path}")
    if summary.confirmatory_path is not None:
        typer.echo(f"Confirmatory effects: {summary.confirmatory_path}")
    if summary.report_path is not None:
        typer.echo(f"Report: {summary.report_path}")
    typer.echo(f"Languages: {', '.join(summary.languages)}")
    typer.echo(f"Subjects: {summary.n_subjects}")
    typer.echo(f"Rows: {summary.n_rows}")


@app.command("run-paper-fast-xlmr-chunks")
def run_paper_fast_xlmr_chunks_command(
    languages: str = typer.Option(
        "en,fr,zh",
        "--languages",
        help="Comma-separated target languages to run in parallel.",
    ),
    max_subjects: int | None = typer.Option(
        None,
        "--max-subjects",
        min=1,
        help="Optional subject cap for a smaller checkpoint run.",
    ),
    layers: str = typer.Option(
        "all",
        "--layers",
        help="Comma-separated XLM-R layer indices, or 'all'.",
    ),
    mismatched_shuffles: int = typer.Option(
        1,
        "--mismatched-shuffles",
        min=1,
        help="Mismatched_shared shuffles for the fast main pass.",
    ),
    tag_suffix: str = typer.Option(
        "fast",
        "--tag-suffix",
        help="Suffix appended to per-language chunk output tags.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Resume the same tagged chunk set by skipping completed subjects.",
    ),
) -> None:
    """Launch the fast-paper XLM-R subject-only language chunks in parallel."""
    chunk_paths = _run_parallel_fast_chunks(
        model_name="xlmr",
        languages=_parse_csv_values(languages),
        layers=layers,
        max_subjects=max_subjects,
        mismatched_shuffles=mismatched_shuffles,
        tag_suffix=tag_suffix,
        resume=resume,
    )
    typer.echo("Chunk outputs:")
    for path in chunk_paths:
        typer.echo(f"- {path}")
    typer.echo("Next step: `python -m brain_subspace_paper merge-paper-fast-xlmr`")


@app.command("merge-paper-fast-xlmr")
def merge_paper_fast_xlmr_command(
    languages: str = typer.Option(
        "en,fr,zh",
        "--languages",
        help="Comma-separated chunk languages to merge.",
    ),
    permutations: int = typer.Option(
        256,
        "--permutations",
        min=1,
        help="Confirmatory permutation count for the merge step.",
    ),
    bootstraps: int = typer.Option(
        256,
        "--bootstraps",
        min=1,
        help="Confirmatory bootstrap count for the merge step.",
    ),
    tag_suffix: str = typer.Option(
        "fast",
        "--tag-suffix",
        help="Suffix used by the per-language chunk output tags.",
    ),
    output_tag: str = typer.Option(
        "xlmr_fast_readout",
        "--output-tag",
        help="Tag for the merged XLM-R outputs.",
    ),
) -> None:
    """Merge the fast-paper XLM-R chunk outputs into one aggregated readout."""
    _merge_fast_chunks(
        model_name="xlmr",
        languages=_parse_csv_values(languages),
        tag_suffix=tag_suffix,
        permutations=permutations,
        bootstraps=bootstraps,
        output_tag=output_tag,
    )


@app.command("run-paper-fast-nllb-chunks")
def run_paper_fast_nllb_chunks_command(
    languages: str = typer.Option(
        "en,fr,zh",
        "--languages",
        help="Comma-separated target languages to run in parallel.",
    ),
    max_subjects: int | None = typer.Option(
        None,
        "--max-subjects",
        min=1,
        help="Optional subject cap for a smaller checkpoint run.",
    ),
    layers: str = typer.Option(
        "4,6,8",
        "--layers",
        help="Comma-separated NLLB layer indices for confirmatory replication.",
    ),
    mismatched_shuffles: int = typer.Option(
        1,
        "--mismatched-shuffles",
        min=1,
        help="Mismatched_shared shuffles for the fast main pass.",
    ),
    tag_suffix: str = typer.Option(
        "fast",
        "--tag-suffix",
        help="Suffix appended to per-language chunk output tags.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Resume the same tagged chunk set by skipping completed subjects.",
    ),
) -> None:
    """Launch the fast-paper NLLB subject-only language chunks in parallel."""
    chunk_paths = _run_parallel_fast_chunks(
        model_name="nllb_encoder",
        languages=_parse_csv_values(languages),
        layers=layers,
        max_subjects=max_subjects,
        mismatched_shuffles=mismatched_shuffles,
        tag_suffix=tag_suffix,
        resume=resume,
    )
    typer.echo("Chunk outputs:")
    for path in chunk_paths:
        typer.echo(f"- {path}")
    typer.echo("Next step: `python -m brain_subspace_paper merge-paper-fast-nllb`")


@app.command("merge-paper-fast-nllb")
def merge_paper_fast_nllb_command(
    languages: str = typer.Option(
        "en,fr,zh",
        "--languages",
        help="Comma-separated chunk languages to merge.",
    ),
    permutations: int = typer.Option(
        256,
        "--permutations",
        min=1,
        help="Confirmatory permutation count for the merge step.",
    ),
    bootstraps: int = typer.Option(
        256,
        "--bootstraps",
        min=1,
        help="Confirmatory bootstrap count for the merge step.",
    ),
    tag_suffix: str = typer.Option(
        "fast",
        "--tag-suffix",
        help="Suffix used by the per-language chunk output tags.",
    ),
    output_tag: str = typer.Option(
        "nllb_fast_readout",
        "--output-tag",
        help="Tag for the merged NLLB outputs.",
    ),
) -> None:
    """Merge the fast-paper NLLB chunk outputs into one aggregated readout."""
    _merge_fast_chunks(
        model_name="nllb_encoder",
        languages=_parse_csv_values(languages),
        tag_suffix=tag_suffix,
        permutations=permutations,
        bootstraps=bootstraps,
        output_tag=output_tag,
    )


@app.command("build-paper-stats")
def build_paper_stats_command(
    xlmr_tag: str = typer.Option(
        "xlmr_fast_readout",
        "--xlmr-tag",
        help="Merged XLM-R output tag to use as the subject-level paper input.",
    ),
    nllb_tag: str = typer.Option(
        "nllb_fast_readout",
        "--nllb-tag",
        help="Merged NLLB output tag to use as the subject-level paper input.",
    ),
    permutations: int | None = typer.Option(
        None,
        "--permutations",
        min=1,
        help="Override the confirmatory and coupling permutation count.",
    ),
    bootstraps: int | None = typer.Option(
        None,
        "--bootstraps",
        min=1,
        help="Override the confirmatory bootstrap count.",
    ),
) -> None:
    """Build canonical paper-level stats files from merged XLM-R and NLLB readouts."""
    xlmr_subject_path = _subject_result_path_for_tag("xlmr", xlmr_tag)
    nllb_subject_path = _subject_result_path_for_tag("nllb_encoder", nllb_tag)
    missing = [path for path in (xlmr_subject_path, nllb_subject_path) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise typer.BadParameter(
            f"Missing merged subject-level readouts: {missing_text}. "
            "Run the appropriate merge-paper-fast-* command first."
        )

    summary = build_paper_level_stats(
        xlmr_subject_results_path=xlmr_subject_path,
        nllb_subject_results_path=nllb_subject_path,
        n_permutations=permutations,
        n_bootstraps=bootstraps,
    )
    typer.echo(f"Subject results: {summary.subject_results_path}")
    typer.echo(f"Group results: {summary.group_results_path}")
    typer.echo(f"Confirmatory effects: {summary.confirmatory_path}")
    typer.echo(f"Geometry metrics: {summary.geometry_metrics_path}")
    typer.echo(f"Geometry-brain coupling: {summary.geometry_brain_coupling_path}")
    typer.echo(f"Subject rows: {summary.n_subject_rows}")
    typer.echo(f"Group rows: {summary.n_group_rows}")
    typer.echo(f"Confirmatory rows: {summary.n_confirmatory_rows}")
    typer.echo(f"Geometry rows: {summary.n_geometry_rows}")
    typer.echo(f"Coupling rows: {summary.n_coupling_rows}")


@app.command("build-paper-tables")
def build_paper_tables_command(
    permutations: int | None = typer.Option(
        None,
        "--permutations",
        min=1,
        help="Override the Table 4 secondary-test permutation count.",
    ),
    bootstraps: int | None = typer.Option(
        None,
        "--bootstraps",
        min=1,
        help="Override the Figure 4/Table-derived interval bootstrap count.",
    ),
) -> None:
    """Build canonical paper tables and reusable derived stats from T13 outputs."""
    summary = build_paper_tables(
        n_permutations=permutations,
        n_bootstraps=bootstraps,
    )
    typer.echo(f"Table 01: {summary.table01_path}")
    typer.echo(f"Table 02: {summary.table02_path}")
    typer.echo(f"Table 03: {summary.table03_path}")
    typer.echo(f"Table 04: {summary.table04_path}")
    typer.echo(f"ROI condition stats: {summary.roi_condition_stats_path}")
    typer.echo(f"ROI family panels: {summary.roi_family_panels_path}")
    typer.echo(f"Table 01 rows: {summary.n_table01_rows}")
    typer.echo(f"Table 02 rows: {summary.n_table02_rows}")
    typer.echo(f"Table 03 rows: {summary.n_table03_rows}")
    typer.echo(f"Table 04 rows: {summary.n_table04_rows}")


@app.command("build-paper-figures")
def build_paper_figures_command(
    dpi: int = typer.Option(
        200,
        "--dpi",
        min=72,
        help="Raster DPI for the canonical PNG outputs.",
    ),
) -> None:
    """Build the canonical paper figures that are currently supported by ROI-side outputs."""
    summary = build_paper_figures(dpi=dpi)
    for name, path in sorted(summary.figure_paths.items()):
        typer.echo(f"{name}: {path}")
    typer.echo(f"Coupling points: {summary.coupling_points_path}")
    typer.echo(f"Figure 08 note: {summary.fig08_note}")


@app.command("build-paper-robustness")
def build_paper_robustness_command(
    permutations: int | None = typer.Option(
        None,
        "--permutations",
        min=1,
        help="Override the robustness permutation count.",
    ),
    bootstraps: int | None = typer.Option(
        None,
        "--bootstraps",
        min=1,
        help="Override the robustness bootstrap count.",
    ),
    conditions: str = typer.Option(
        "all",
        "--conditions",
        help="Comma-separated subset of robustness conditions, or 'all'.",
    ),
    max_subjects: int | None = typer.Option(
        None,
        "--max-subjects",
        min=1,
        help="Optional subject cap for a smoke or checkpoint run.",
    ),
    max_tokens_per_batch: int = typer.Option(
        2048,
        "--max-tokens-per-batch",
        min=64,
        help="Approximate token budget for robustness re-extraction batches.",
    ),
    render_figure: bool = typer.Option(
        True,
        "--render-figure/--no-render-figure",
        help="Also render the optional Figure 09 robustness panel.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Reuse already-computed robustness conditions from the saved cell-results file.",
    ),
    output_tag: str | None = typer.Option(
        None,
        "--output-tag",
        help="Optional suffix for robustness outputs so independent condition runs do not collide.",
    ),
    include_base: bool = typer.Option(
        True,
        "--include-base/--no-include-base",
        help="Whether to include the canonical representative-layer base rows in this run.",
    ),
) -> None:
    """Build Table 05 and optional Figure 09 from the required robustness suite."""
    condition_expr = "None" if conditions.strip().lower() == "all" else repr(_parse_csv_values(conditions))
    _run_subprocess_python(
        "\n".join(
            [
                "import torch",
                "from brain_subspace_paper.stats.robustness import build_paper_robustness",
                (
                    "summary = build_paper_robustness("
                    f"n_permutations={repr(permutations)}, "
                    f"n_bootstraps={repr(bootstraps)}, "
                    f"conditions={condition_expr}, "
                    f"max_subjects={repr(max_subjects)}, "
                    f"max_tokens_per_batch={max_tokens_per_batch}, "
                    f"render_figure={render_figure}, "
                    f"resume={resume}, "
                    f"output_tag={repr(output_tag)}, "
                    f"include_base={include_base})"
                ),
                "print(f'Robustness cell results: {summary.cell_results_path}')",
                "print(f'Robustness summary: {summary.summary_path}')",
                "print(f'Representative layers: {summary.representative_layers_path}')",
                "print(f'Table 05: {summary.table05_path}')",
                "print(f'Figure 09: {summary.figure09_path}')",
                "print(f'Conditions: {summary.n_conditions}')",
                "print(f'Cells: {summary.n_cells}')",
            ]
        ),
        env=_thread_capped_env(),
    )


@app.command("run-paper-t15")
def run_paper_t15_command(
    permutations: int | None = typer.Option(
        None,
        "--permutations",
        min=1,
        help="Override the robustness permutation count.",
    ),
    bootstraps: int | None = typer.Option(
        None,
        "--bootstraps",
        min=1,
        help="Override the robustness bootstrap count.",
    ),
    max_subjects: int | None = typer.Option(
        None,
        "--max-subjects",
        min=1,
        help="Optional subject cap for a smoke run.",
    ),
    max_tokens_per_batch: int = typer.Option(
        2048,
        "--max-tokens-per-batch",
        min=64,
        help="Approximate token budget for robustness re-extraction batches.",
    ),
    output_tag: str | None = typer.Option(
        None,
        "--output-tag",
        help="Optional suffix for robustness outputs so this T15 wrapper can use isolated outputs.",
    ),
    include_voxelwise: bool = typer.Option(
        True,
        "--include-voxelwise/--no-include-voxelwise",
        help="Include the voxelwise-within-ROI robustness row in the one-command T15 run.",
    ),
    render_figure: bool = typer.Option(
        True,
        "--render-figure/--no-render-figure",
        help="Render the final Figure 09 after the suite finishes.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Reuse already-computed robustness conditions from the saved cell-results file.",
    ),
) -> None:
    """Run the full T15 robustness suite in one resumable command."""
    conditions = "all" if include_voxelwise else "last_token_pooling,fir_4lag,no_acoustic_nuisance,no_pitch_nuisance,previous_2_sentence_context"
    condition_expr = "None" if conditions == "all" else repr(_parse_csv_values(conditions))
    render_code = "\n".join(
        [
            "import torch",
            "from brain_subspace_paper.stats.robustness import build_paper_robustness",
            (
                "summary = build_paper_robustness("
                f"n_permutations={repr(permutations)}, "
                f"n_bootstraps={repr(bootstraps)}, "
                f"conditions={condition_expr}, "
                f"max_subjects={repr(max_subjects)}, "
                f"max_tokens_per_batch={max_tokens_per_batch}, "
                "render_figure=False, "
                f"resume={resume}, "
                f"output_tag={repr(output_tag)}, "
                "include_base=True)"
            ),
            "print(f'Robustness cell results: {summary.cell_results_path}')",
            "print(f'Robustness summary: {summary.summary_path}')",
            "print(f'Representative layers: {summary.representative_layers_path}')",
            "print(f'Table 05: {summary.table05_path}')",
            "print(f'Figure 09: {summary.figure09_path}')",
            "print(f'Conditions: {summary.n_conditions}')",
            "print(f'Cells: {summary.n_cells}')",
        ]
    )
    _run_subprocess_python(render_code, env=_thread_capped_env())

    if render_figure:
        final_render_code = "\n".join(
            [
                "import torch",
                "from brain_subspace_paper.stats.robustness import build_paper_robustness",
                (
                    "summary = build_paper_robustness("
                    f"n_permutations={repr(permutations)}, "
                    f"n_bootstraps={repr(bootstraps)}, "
                    f"conditions={condition_expr}, "
                    f"max_subjects={repr(max_subjects)}, "
                    f"max_tokens_per_batch={max_tokens_per_batch}, "
                    "render_figure=True, "
                    "resume=True, "
                    f"output_tag={repr(output_tag)}, "
                    "include_base=True)"
                ),
                "print(f'Final Figure 09: {summary.figure09_path}')",
            ]
        )
        _run_subprocess_python(final_render_code, env=_thread_capped_env())


@app.command("merge-paper-robustness")
def merge_paper_robustness_command(
    inputs: str = typer.Option(
        ...,
        "--inputs",
        help="Comma-separated robustness output tags to merge.",
    ),
    output_tag: str | None = typer.Option(
        None,
        "--output-tag",
        help="Optional suffix for merged robustness outputs.",
    ),
    render_figure: bool = typer.Option(
        True,
        "--render-figure/--no-render-figure",
        help="Render Figure 09 for the merged robustness outputs.",
    ),
) -> None:
    """Merge tagged robustness chunk outputs into one robustness summary/table set."""
    tags = _parse_csv_values(inputs)
    _run_subprocess_python(
        "\n".join(
            [
                "from brain_subspace_paper.stats.robustness import merge_paper_robustness",
                (
                    "summary = merge_paper_robustness("
                    f"input_tags={repr(tags)}, "
                    f"output_tag={repr(output_tag)}, "
                    f"render_figure={render_figure})"
                ),
                "print(f'Robustness cell results: {summary.cell_results_path}')",
                "print(f'Robustness summary: {summary.summary_path}')",
                "print(f'Representative layers: {summary.representative_layers_path}')",
                "print(f'Table 05: {summary.table05_path}')",
                "print(f'Figure 09: {summary.figure09_path}')",
                "print(f'Conditions: {summary.n_conditions}')",
                "print(f'Cells: {summary.n_cells}')",
            ]
        ),
        env=_thread_capped_env(),
    )


@app.command("build-paper-whole-brain")
def build_paper_whole_brain_command(
    models: str = typer.Option(
        "xlmr,nllb_encoder",
        "--models",
        help="Comma-separated models to process.",
    ),
    languages: str = typer.Option(
        "en,fr,zh",
        "--languages",
        help="Comma-separated languages to process.",
    ),
    chunk_size: int = typer.Option(
        8192,
        "--chunk-size",
        min=256,
        help="Voxel chunk size for the whole-brain ridge loop.",
    ),
    max_subjects: int | None = typer.Option(
        None,
        "--max-subjects",
        min=1,
        help="Optional subject cap for a smoke or preview run.",
    ),
    min_subject_coverage_fraction: float = typer.Option(
        0.9,
        "--min-subject-coverage-fraction",
        min=0.1,
        max=1.0,
        help="Coverage fraction required for the group brain mask.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Reuse cached subject maps when they already exist.",
    ),
    render_figure: bool = typer.Option(
        True,
        "--render-figure/--no-render-figure",
        help="Render the canonical Figure 08 after artifact generation.",
    ),
) -> None:
    """Build representative-layer whole-brain artifacts and Figure 08."""
    summary = build_paper_whole_brain(
        models=_parse_csv_values(models),
        languages=_parse_csv_values(languages),
        chunk_size=chunk_size,
        max_subjects=max_subjects,
        min_subject_coverage_fraction=min_subject_coverage_fraction,
        resume=resume,
        render_figure=render_figure,
    )
    typer.echo(f"Representative layers: {summary.representative_layers_path}")
    typer.echo(f"Figure 08: {summary.figure_path}")
    typer.echo(f"Artifact roots: {len(summary.artifact_roots)}")
    typer.echo(f"Computed subject maps: {summary.n_subject_maps}")
    typer.echo(f"Skipped subject maps: {summary.skipped_subject_maps}")


@app.command("paper-fast-status")
def paper_fast_status_command(
    languages: str = typer.Option(
        "en,fr,zh",
        "--languages",
        help="Comma-separated languages to inspect for the fast-paper path.",
    ),
    tag_suffix: str = typer.Option(
        "fast",
        "--tag-suffix",
        help="Suffix used by the fast chunk wrapper commands.",
    ),
    xlmr_output_tag: str = typer.Option(
        "xlmr_fast_readout",
        "--xlmr-output-tag",
        help="Merged XLM-R output tag to inspect.",
    ),
    nllb_output_tag: str = typer.Option(
        "nllb_fast_readout",
        "--nllb-output-tag",
        help="Merged NLLB output tag to inspect.",
    ),
    show_paths: bool = typer.Option(
        False,
        "--show-paths",
        help="Also print the chunk and merge file paths.",
    ),
) -> None:
    """Show the current fast-paper chunk and merge status, including resumable progress."""
    languages_tuple = _parse_csv_values(languages)
    xlmr_chunk_rows = _chunk_status_rows(model_name="xlmr", languages=languages_tuple, tag_suffix=tag_suffix)
    nllb_chunk_rows = _chunk_status_rows(model_name="nllb_encoder", languages=languages_tuple, tag_suffix=tag_suffix)
    xlmr_merge = _merge_status(model_name="xlmr", output_tag=xlmr_output_tag)
    nllb_merge = _merge_status(model_name="nllb_encoder", output_tag=nllb_output_tag)

    typer.echo("Fast-paper status")
    typer.echo("")

    for label, rows in (("XLM-R chunks", xlmr_chunk_rows), ("NLLB chunks", nllb_chunk_rows)):
        typer.echo(label)
        for row in rows:
            typer.echo(
                f"- {row['language']}: {row['status']} "
                f"({row['completed_subjects']}/{row['target_subjects']} subjects)"
            )
            if show_paths:
                typer.echo(f"  subject={row['subject_path']}")
                typer.echo(f"  metadata={row['metadata_path']}")
                typer.echo(f"  launcher_log={row['log_path']}")
        typer.echo("")

    typer.echo(f"XLM-R merge: {xlmr_merge['status']}")
    if show_paths:
        for name, path in xlmr_merge["paths"].items():
            exists_flag = "present" if xlmr_merge["exists"][name] else "missing"
            typer.echo(f"- {name}: {exists_flag} {path}")
    typer.echo("")

    typer.echo(f"NLLB merge: {nllb_merge['status']}")
    if show_paths:
        for name, path in nllb_merge["paths"].items():
            exists_flag = "present" if nllb_merge["exists"][name] else "missing"
            typer.echo(f"- {name}: {exists_flag} {path}")
    typer.echo("")

    typer.echo("Next recommended step:")
    typer.echo(_next_fast_step(
        xlmr_chunk_rows=xlmr_chunk_rows,
        xlmr_merge=xlmr_merge,
        nllb_chunk_rows=nllb_chunk_rows,
        nllb_merge=nllb_merge,
    ))


def main() -> None:
    app()
