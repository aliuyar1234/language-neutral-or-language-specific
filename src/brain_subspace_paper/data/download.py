from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Iterable

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from boto3.s3.transfer import TransferConfig

from brain_subspace_paper.config import project_config, project_root
from brain_subspace_paper.logging_utils import append_markdown_log, bootstrap_logs, utc_now_iso, write_text


OPENNEURO_BUCKET = "openneuro.org"


@dataclass(slots=True)
class DownloadSummary:
    dataset_root: Path
    method: str
    object_count: int
    downloaded_files: int
    skipped_files: int
    downloaded_bytes: int
    selected_prefixes: tuple[str, ...]
    manifest_path: Path


def _unsigned_s3_client():
    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 8, "mode": "standard"}),
    )


def _dataset_root() -> Path:
    root = project_root()
    raw_root = root / project_config()["paths"]["raw_data_root"]
    raw_root.mkdir(parents=True, exist_ok=True)
    accession = project_config()["data"]["core_dataset"]["accession"]
    return raw_root / accession


def _git_annex_available() -> bool:
    return shutil.which("git") is not None and (
        shutil.which("git-annex") is not None or shutil.which("git-annex.exe") is not None
    )


def _dataset_repo_url() -> str:
    return str(project_config()["data"]["core_dataset"]["github_url"])


def _ensure_git_annex_clone(dataset_root: Path) -> None:
    if (dataset_root / ".git").exists():
        return
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "clone",
            _dataset_repo_url(),
            str(dataset_root),
        ],
        check=True,
    )


def _download_via_git_annex(dataset_root: Path, include_stimuli: bool) -> DownloadSummary:
    bootstrap_logs()
    root = project_root()
    _ensure_git_annex_clone(dataset_root)
    selected_paths = ["annotation", "derivatives"]
    if include_stimuli:
        selected_paths.append("stimuli")

    print(f"Fetching LPPC content with git-annex for: {', '.join(selected_paths)}")
    subprocess.run(
        ["git", "-C", str(dataset_root), "annex", "get", "-J4", *selected_paths],
        check=True,
    )

    downloaded_files = sum(1 for path in dataset_root.rglob("*") if path.is_file())
    downloaded_bytes = sum(path.stat().st_size for path in dataset_root.rglob("*") if path.is_file())
    manifest_path = dataset_root / ".download_manifest.json"

    import json

    manifest = {
        "timestamp_utc": utc_now_iso(),
        "method": "git_annex",
        "dataset_accession": project_config()["data"]["core_dataset"]["accession"],
        "repo_url": _dataset_repo_url(),
        "selected_prefixes": selected_paths,
        "downloaded_files_present": downloaded_files,
        "downloaded_bytes_present": downloaded_bytes,
    }
    write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")

    append_markdown_log(
        root / "outputs" / "logs" / "progress_log.md",
        "LPPC download",
        [
            "Fetched LPPC content through git-annex.",
            f"Selected paths: {', '.join(selected_paths)}.",
            f"Files present after fetch: {downloaded_files}.",
            f"Bytes present after fetch: {downloaded_bytes}.",
        ],
    )

    return DownloadSummary(
        dataset_root=dataset_root,
        method="git_annex",
        object_count=downloaded_files,
        downloaded_files=downloaded_files,
        skipped_files=0,
        downloaded_bytes=downloaded_bytes,
        selected_prefixes=tuple(selected_paths),
        manifest_path=manifest_path,
    )


def _should_include_key(relative_key: str, include_stimuli: bool) -> bool:
    if not relative_key or relative_key.endswith("/"):
        return False
    parts = relative_key.split("/", 1)
    first = parts[0]
    if len(parts) == 1:
        return True
    if first in {"annotation", "derivatives"}:
        return True
    if include_stimuli and first == "stimuli":
        return True
    return False


def _iter_lppc_objects(include_stimuli: bool) -> Iterable[dict]:
    client = _unsigned_s3_client()
    prefix = f"{project_config()['data']['core_dataset']['accession']}/"
    token: str | None = None
    while True:
        kwargs = {"Bucket": OPENNEURO_BUCKET, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        for item in response.get("Contents", []):
            relative = item["Key"][len(prefix) :]
            if _should_include_key(relative, include_stimuli=include_stimuli):
                yield item
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")


def _priority(relative_key: str) -> tuple[int, str]:
    if "/" not in relative_key:
        return (0, relative_key)
    if relative_key.startswith("annotation/"):
        return (1, relative_key)
    if relative_key.startswith("stimuli/"):
        return (2, relative_key)
    if relative_key.startswith("derivatives/"):
        return (3, relative_key)
    return (4, relative_key)


def download_lppc(
    include_stimuli: bool = False,
    max_files: int | None = None,
    overwrite: bool = False,
    method: str = "auto",
) -> DownloadSummary:
    bootstrap_logs()
    root = project_root()
    dataset_root = _dataset_root()
    dataset_root.mkdir(parents=True, exist_ok=True)

    if method not in {"auto", "git_annex", "s3_public"}:
        raise ValueError(f"Unsupported download method: {method}")
    if method == "auto":
        method = "git_annex" if _git_annex_available() else "s3_public"
    if method == "git_annex":
        if max_files is not None:
            raise ValueError("`max_files` is only supported for the S3 smoke-test backend.")
        if overwrite:
            raise ValueError("`overwrite` is not supported for the git-annex backend.")
        return _download_via_git_annex(dataset_root, include_stimuli=include_stimuli)

    objects = list(_iter_lppc_objects(include_stimuli=include_stimuli))
    objects.sort(key=lambda item: _priority(item["Key"].split("/", 1)[1]))
    if max_files is not None:
        objects = objects[:max_files]
    total_bytes = sum(int(item["Size"]) for item in objects)

    client = _unsigned_s3_client()
    transfer_config = TransferConfig(max_concurrency=8, multipart_threshold=32 * 1024 * 1024)

    downloaded_files = 0
    skipped_files = 0
    downloaded_bytes = 0
    accession = project_config()["data"]["core_dataset"]["accession"]

    print(
        f"Starting LPPC download from public S3: {len(objects)} objects, "
        f"{total_bytes} bytes total."
    )
    for index, item in enumerate(objects, start=1):
        relative = item["Key"][len(accession) + 1 :]
        destination = dataset_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)

        expected_size = int(item["Size"])
        if destination.exists() and not overwrite and destination.stat().st_size == expected_size:
            skipped_files += 1
            if index == len(objects) or index % 50 == 0:
                print(
                    f"[{index}/{len(objects)}] skipped existing file: {relative}"
                )
            continue

        client.download_file(
            OPENNEURO_BUCKET,
            item["Key"],
            str(destination),
            Config=transfer_config,
        )
        downloaded_files += 1
        downloaded_bytes += expected_size
        if index == len(objects) or index % 25 == 0:
            print(
                f"[{index}/{len(objects)}] downloaded {relative} "
                f"({downloaded_bytes} bytes transferred this run)"
            )

    selected_prefixes = ["annotation", "derivatives"]
    if include_stimuli:
        selected_prefixes.append("stimuli")

    manifest_path = dataset_root / ".download_manifest.json"
    manifest = {
        "timestamp_utc": utc_now_iso(),
        "method": "openneuro_public_s3_unsigned",
        "bucket": OPENNEURO_BUCKET,
        "dataset_accession": accession,
        "selected_prefixes": selected_prefixes,
        "object_count": len(objects),
        "downloaded_files": downloaded_files,
        "skipped_files": skipped_files,
        "downloaded_bytes": downloaded_bytes,
        "include_stimuli": include_stimuli,
        "max_files": max_files,
    }
    import json

    write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")

    append_markdown_log(
        root / "outputs" / "logs" / "progress_log.md",
        "LPPC download",
        [
            "Downloaded required LPPC files from the public OpenNeuro S3 bucket.",
            f"Selected prefixes: {', '.join(selected_prefixes)}.",
            f"Files downloaded: {downloaded_files}.",
            f"Files skipped because they already matched size: {skipped_files}.",
            f"Bytes transferred in this run: {downloaded_bytes}.",
        ],
    )

    return DownloadSummary(
        dataset_root=dataset_root,
        method="openneuro_public_s3_unsigned",
        object_count=len(objects),
        downloaded_files=downloaded_files,
        skipped_files=skipped_files,
        downloaded_bytes=downloaded_bytes,
        selected_prefixes=tuple(selected_prefixes),
        manifest_path=manifest_path,
    )
