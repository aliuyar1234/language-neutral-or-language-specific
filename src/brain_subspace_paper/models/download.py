from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from brain_subspace_paper.config import model_config, project_root
from brain_subspace_paper.logging_utils import append_markdown_log, bootstrap_logs


MODEL_DIR_NAMES = {
    "xlmr": "xlmr",
    "nllb_encoder": "nllb_encoder",
    "labse": "labse",
}


@dataclass(slots=True)
class ModelDownloadSummary:
    model_name: str
    hf_id: str
    local_dir: Path
    downloaded: bool


def _iter_model_entries() -> list[dict[str, Any]]:
    config = model_config()
    entries = list(config.get("models", {}).get("core", []))
    alignment_qc = config.get("alignment_qc_model")
    if alignment_qc:
        entries.append(alignment_qc)
    return entries


def get_model_entry(model_name: str) -> dict[str, Any]:
    for entry in _iter_model_entries():
        if entry.get("name") == model_name:
            return entry
    raise KeyError(f"Unknown model name: {model_name}")


def model_local_dir(model_name: str) -> Path:
    dirname = MODEL_DIR_NAMES.get(model_name)
    if dirname is None:
        raise KeyError(f"No local directory rule for model: {model_name}")
    return project_root() / "models" / dirname


def _allow_patterns(model_name: str) -> list[str]:
    if model_name == "labse":
        return ["*.json", "*.txt", "*.model", "*.safetensors", "*.bin", "1_Pooling/**", "modules.json"]
    return [
        "*.json",
        "*.txt",
        "*.model",
        "*.bpe",
        "*.safetensors",
        "*.bin",
        "sentencepiece.bpe.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]


def _looks_downloaded(model_name: str, local_dir: Path) -> bool:
    required = [local_dir / "config.json"]
    if model_name == "labse":
        required.extend([local_dir / "modules.json", local_dir / "1_Pooling" / "config.json"])
    tokenizer_candidates = [
        local_dir / "tokenizer.json",
        local_dir / "sentencepiece.bpe.model",
        local_dir / "spiece.model",
    ]
    weight_candidates = [
        local_dir / "model.safetensors",
        local_dir / "pytorch_model.bin",
    ]
    return all(path.exists() for path in required) and any(
        path.exists() for path in tokenizer_candidates
    ) and any(path.exists() for path in weight_candidates)


def download_model(model_name: str, *, force: bool = False) -> ModelDownloadSummary:
    bootstrap_logs()
    entry = get_model_entry(model_name)
    hf_id = str(entry["hf_id"])
    local_dir = model_local_dir(model_name)

    if local_dir.exists() and _looks_downloaded(model_name, local_dir) and not force:
        downloaded = False
    else:
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=hf_id,
            local_dir=str(local_dir),
            allow_patterns=_allow_patterns(model_name),
            force_download=force,
        )
        downloaded = True

    append_markdown_log(
        project_root() / "outputs" / "logs" / "progress_log.md",
        "Model Download",
        [
            f"model={model_name}",
            f"hf_id={hf_id}",
            f"local_dir={local_dir.as_posix()}",
            f"downloaded={downloaded}",
        ],
    )
    return ModelDownloadSummary(
        model_name=model_name,
        hf_id=hf_id,
        local_dir=local_dir,
        downloaded=downloaded,
    )
