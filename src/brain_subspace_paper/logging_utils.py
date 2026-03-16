from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess

import yaml

from brain_subspace_paper.config import project_root


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> Path:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")
    return path


def append_markdown_log(path: Path, title: str, lines: list[str]) -> Path:
    ensure_parent(path)
    timestamp = utc_now_iso()
    entry = [f"## {timestamp} - {title}", ""] + [f"- {line}" for line in lines] + [""]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(entry))
    return path


def bootstrap_logs() -> dict[str, Path]:
    root = project_root()
    logs_root = root / "outputs" / "logs"
    manuscript_root = root / "outputs" / "manuscript"
    logs_root.mkdir(parents=True, exist_ok=True)
    manuscript_root.mkdir(parents=True, exist_ok=True)

    progress_log = logs_root / "progress_log.md"
    if not progress_log.exists():
        write_text(progress_log, "# Progress Log\n\n")

    spec_deviation_log = logs_root / "spec_deviation_log.md"
    if not spec_deviation_log.exists():
        write_text(spec_deviation_log, "# Spec Deviation Log\n\n")

    config_snapshot = logs_root / "config_snapshot.yaml"
    config_dir = root / "configs"
    snapshot: dict[str, object] = {}
    for config_file in sorted(config_dir.glob("*.yaml")):
        with config_file.open("r", encoding="utf-8") as handle:
            snapshot[config_file.name] = yaml.safe_load(handle)
    write_text(config_snapshot, yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=False))

    pip_freeze = logs_root / "pip_freeze.txt"
    freeze = subprocess.run(
        ["python", "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    write_text(pip_freeze, freeze.stdout or freeze.stderr)

    git_commit = logs_root / "git_commit.txt"
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if git_result.returncode == 0:
        git_text = git_result.stdout.strip() + "\n"
    else:
        git_text = "not a git repository\n"
    write_text(git_commit, git_text)

    for manuscript_file in (
        manuscript_root / "figure_provenance.md",
        manuscript_root / "table_provenance.md",
        manuscript_root / "claim_evidence_map.md",
    ):
        if not manuscript_file.exists():
            write_text(manuscript_file, f"# {manuscript_file.stem.replace('_', ' ').title()}\n\n")

    append_markdown_log(
        progress_log,
        "Bootstrap",
        [
            "Initialized required log files and manuscript provenance files.",
            "Wrote config snapshot and pip freeze.",
            f"Recorded git state in {git_commit.relative_to(root).as_posix()}.",
        ],
    )

    return {
        "progress_log": progress_log,
        "config_snapshot": config_snapshot,
        "pip_freeze": pip_freeze,
        "git_commit": git_commit,
    }
