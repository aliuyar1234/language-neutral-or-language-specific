from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brain_subspace_paper.config import project_root
from brain_subspace_paper.logging_utils import append_markdown_log, bootstrap_logs, write_text
from brain_subspace_paper.models.download import model_local_dir


LANGUAGES = ("en", "fr", "zh")
LANGUAGE_TEXT_COLUMNS = {
    "en": "en_text",
    "fr": "fr_text",
    "zh": "zh_text",
}
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "zh": "zho_Hans",
}


@dataclass(slots=True)
class PilotExtractionSummary:
    model_name: str
    model_dir: Path
    triplet_ids_path: Path
    manifest_path: Path
    token_metadata_path: Path
    geometry_path: Path
    plot_path: Path
    report_path: Path
    n_triplets: int
    n_layers: int
    hidden_size: int


@dataclass(slots=True)
class FullExtractionSummary:
    model_name: str
    model_dir: Path
    triplet_ids_path: Path
    manifest_path: Path
    token_metadata_path: Path
    n_triplets: int
    n_layers: int
    hidden_size: int


def _pilot_root(model_name: str) -> Path:
    return project_root() / "data" / "interim" / "embeddings" / f"{model_name}_pilot"


def _triplets_path() -> Path:
    return project_root() / "data" / "processed" / "alignment_triplets.parquet"


def _triplets_qc_path() -> Path:
    return project_root() / "data" / "processed" / "alignment_triplets_qc.parquet"


def _stats_path(model_name: str) -> Path:
    return project_root() / "outputs" / "stats" / f"{model_name}_pilot_geometry_by_layer.parquet"


def _plot_path(model_name: str) -> Path:
    return project_root() / "outputs" / "figures" / f"{model_name}_pilot_same_vs_mismatched.png"


def _report_path() -> Path:
    return project_root() / "outputs" / "logs" / "model_extraction_report.md"


@lru_cache(maxsize=1)
def _load_xlmr() -> tuple[Any, Any, torch.device]:
    model_dir = model_local_dir("xlmr")
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Local XLM-R directory missing at {model_dir}. Download the model before extraction."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModel.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


@lru_cache(maxsize=1)
def _load_nllb_encoder() -> tuple[Any, Any, torch.device]:
    model_dir = model_local_dir("nllb_encoder")
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Local NLLB encoder directory missing at {model_dir}. Download the model before extraction."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModel.from_pretrained(str(model_dir))
    if not hasattr(model, "get_encoder"):
        raise RuntimeError("Expected the NLLB model to expose get_encoder() for encoder-only extraction.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _load_model_bundle(model_name: str) -> tuple[Any, Any, torch.device]:
    if model_name == "xlmr":
        return _load_xlmr()
    if model_name == "nllb_encoder":
        return _load_nllb_encoder()
    raise KeyError(f"Unsupported extraction model: {model_name}")


def _load_pilot_triplets(n_triplets: int) -> pd.DataFrame:
    triplets = pd.read_parquet(_triplets_path()).copy()
    qc = pd.read_parquet(_triplets_qc_path())[["triplet_id", "manual_status"]].copy()
    merged = triplets.merge(qc, on="triplet_id", how="left")
    merged = merged.loc[merged["manual_status"].fillna("") != "needs_fix"].copy()
    merged = merged.sort_values("triplet_id").head(n_triplets).reset_index(drop=True)
    if len(merged) < n_triplets:
        raise RuntimeError(f"Requested {n_triplets} pilot triplets but only found {len(merged)} usable rows.")
    return merged


def _load_all_triplets() -> pd.DataFrame:
    return pd.read_parquet(_triplets_path()).sort_values("triplet_id").reset_index(drop=True)


def _batch_index_groups(token_lengths: list[int], max_tokens_per_batch: int) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for idx, length in enumerate(token_lengths):
        token_cost = max(1, int(length))
        if current and current_tokens + token_cost > max_tokens_per_batch:
            groups.append(current)
            current = [idx]
            current_tokens = token_cost
            continue
        current.append(idx)
        current_tokens += token_cost
    if current:
        groups.append(current)
    return groups


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return array / norms


def _extract_language_embeddings(
    *,
    model_name: str,
    language: str,
    triplets: pd.DataFrame,
    max_tokens_per_batch: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    tokenizer, model, device = _load_model_bundle(model_name)
    texts = triplets[LANGUAGE_TEXT_COLUMNS[language]].astype(str).tolist()
    tokenizer_kwargs: dict[str, Any] = {"add_special_tokens": True, "truncation": True}
    if model_name == "nllb_encoder":
        tokenizer.src_lang = NLLB_LANGUAGE_CODES[language]
    token_lengths = [len(ids) for ids in tokenizer(texts, **tokenizer_kwargs)["input_ids"]]
    batches = _batch_index_groups(token_lengths, max_tokens_per_batch=max_tokens_per_batch)
    row_index_by_triplet_id = {
        int(triplet_id): row_index
        for row_index, triplet_id in enumerate(triplets["triplet_id"].astype(int).tolist())
    }

    pooled_by_layer: list[np.ndarray] | None = None
    token_metadata: list[dict[str, Any]] = []
    hidden_size: int | None = None

    for batch_indices in batches:
        batch_triplets = triplets.iloc[batch_indices].copy()
        encoded = tokenizer(
            batch_triplets[LANGUAGE_TEXT_COLUMNS[language]].astype(str).tolist(),
            padding=True,
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        pooling_mask = encoded["attention_mask"].bool() & ~encoded["special_tokens_mask"].bool()

        with torch.no_grad():
            if model_name == "nllb_encoder":
                outputs = model.get_encoder()(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                outputs = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )

        hidden_states = outputs.hidden_states
        if pooled_by_layer is None:
            hidden_size = int(hidden_states[0].shape[-1])
            pooled_by_layer = [
                np.zeros((len(triplets), hidden_size), dtype=np.float32)
                for _ in range(len(hidden_states))
            ]

        token_counts = pooling_mask.sum(dim=1).clamp(min=1)

        for row_offset, row in enumerate(batch_triplets.itertuples(index=False)):
            active_positions = torch.nonzero(pooling_mask[row_offset], as_tuple=False).flatten()
            first_idx = int(active_positions[0].item()) if len(active_positions) else -1
            last_idx = int(active_positions[-1].item()) if len(active_positions) else -1
            token_metadata.append(
                {
                    "triplet_id": int(row.triplet_id),
                    "language": language,
                    "sequence_length": int(encoded["attention_mask"][row_offset].sum().item()),
                    "n_tokens_pooled": int(token_counts[row_offset].item()),
                    "first_pooled_token_index": first_idx,
                    "last_pooled_token_index": last_idx,
                }
            )

        for layer_index, hidden in enumerate(hidden_states):
            masked = hidden * pooling_mask.unsqueeze(-1)
            pooled = masked.sum(dim=1) / token_counts.unsqueeze(-1)
            pooled_np = pooled.detach().cpu().to(torch.float32).numpy().astype(np.float32, copy=False)
            pooled_np = _normalize_rows(pooled_np)
            for row_offset, triplet_id in enumerate(batch_triplets["triplet_id"].astype(int).tolist()):
                dest_index = row_index_by_triplet_id[triplet_id]
                pooled_by_layer[layer_index][dest_index] = pooled_np[row_offset]

    if pooled_by_layer is None or hidden_size is None:
        raise RuntimeError(f"No embeddings were extracted for language={language}.")

    return pooled_by_layer, token_metadata


def _save_language_arrays(
    *,
    model_name: str,
    language: str,
    pooled_by_layer: list[np.ndarray],
) -> list[dict[str, Any]]:
    pilot_root = _pilot_root(model_name)
    language_root = pilot_root / language
    language_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    n_layers = len(pooled_by_layer)
    for layer_index, array in enumerate(pooled_by_layer):
        path = language_root / f"layer_{layer_index:02d}.npy"
        np.save(path, array.astype(np.float32, copy=False))
        manifest_rows.append(
            {
                "subset": "pilot",
                "model": model_name,
                "language": language,
                "layer_index": layer_index,
                "layer_depth": layer_index / (n_layers - 1),
                "n_rows": int(array.shape[0]),
                "hidden_size": int(array.shape[1]),
                "filepath": path.as_posix(),
                "dtype": "float32",
            }
        )
    return manifest_rows


def _save_full_language_arrays(
    *,
    model_name: str,
    language: str,
    pooled_by_layer: list[np.ndarray],
) -> list[dict[str, Any]]:
    root = project_root() / "data" / "interim" / "embeddings" / model_name
    language_root = root / language
    language_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    n_layers = len(pooled_by_layer)
    for layer_index, array in enumerate(pooled_by_layer):
        path = language_root / f"layer_{layer_index:02d}.npy"
        np.save(path, array.astype(np.float32, copy=False))
        manifest_rows.append(
            {
                "model": model_name,
                "language": language,
                "layer_index": layer_index,
                "layer_depth": layer_index / (n_layers - 1),
                "n_rows": int(array.shape[0]),
                "hidden_size": int(array.shape[1]),
                "filepath": path.as_posix(),
                "dtype": "float32",
            }
        )
    return manifest_rows


def _geometry_rows(
    arrays_by_language: dict[str, list[np.ndarray]],
    *,
    model_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    language_pairs = [("en", "fr"), ("en", "zh"), ("fr", "zh")]
    n_layers = len(next(iter(arrays_by_language.values())))

    for layer_index in range(n_layers):
        for left_language, right_language in language_pairs:
            left = arrays_by_language[left_language][layer_index]
            right = arrays_by_language[right_language][layer_index]
            same = float(np.mean(np.sum(left * right, axis=1)))
            mismatched = float(np.mean(np.sum(left * np.roll(right, shift=1, axis=0), axis=1)))
            rows.append(
                {
                    "model": model_name,
                    "layer_index": layer_index,
                    "layer_depth": layer_index / (n_layers - 1),
                    "language_pair": f"{left_language}-{right_language}",
                    "same_mean_cosine": same,
                    "mismatched_mean_cosine": mismatched,
                    "delta_same_minus_mismatched": same - mismatched,
                }
            )
    return pd.DataFrame(rows)


def _save_geometry_plot(geometry_df: pd.DataFrame, *, model_name: str) -> Path:
    plot_path = _plot_path(model_name)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    language_pairs = ["en-fr", "en-zh", "fr-zh"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, language_pair in zip(axes, language_pairs, strict=False):
        subset = geometry_df.loc[geometry_df["language_pair"] == language_pair].copy()
        axis.plot(
            subset["layer_depth"],
            subset["same_mean_cosine"],
            label="same",
            linewidth=2.0,
        )
        axis.plot(
            subset["layer_depth"],
            subset["mismatched_mean_cosine"],
            label="mismatched",
            linewidth=2.0,
        )
        axis.set_title(language_pair)
        axis.set_xlabel("Normalized layer depth")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean cosine")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def _write_report(
    *,
    model_name: str,
    manifest_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    token_metadata_df: pd.DataFrame,
    n_triplets: int,
) -> Path:
    report_path = _report_path()
    geometry_summary = (
        geometry_df.groupby("language_pair")["delta_same_minus_mismatched"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    lines = [
        "# Model Extraction Report",
        "",
        f"- model: `{model_name}`",
        f"- subset: `pilot`",
        f"- triplets: `{n_triplets}`",
        f"- layers: `{manifest_df['layer_index'].nunique()}`",
        f"- hidden size: `{int(manifest_df['hidden_size'].iloc[0])}`",
        f"- token metadata rows: `{len(token_metadata_df)}`",
        f"- manifest: `{(_pilot_root(model_name) / 'pilot_embedding_manifest.parquet').as_posix()}`",
        f"- geometry stats: `{_stats_path(model_name).as_posix()}`",
        f"- geometry plot: `{_plot_path(model_name).as_posix()}`",
        "",
        "## Geometry Check",
        "",
    ]
    for row in geometry_summary.itertuples(index=False):
        lines.append(
            f"- `{row.language_pair}`: mean delta `{row.mean:.4f}`, "
            f"min `{row.min:.4f}`, max `{row.max:.4f}`"
        )
    lines.extend(
        [
            "",
            "## Extraction Checks",
            "",
            f"- no NaNs in saved arrays: `{manifest_df['has_nans'].eq(False).all()}`",
            f"- pooled token counts all positive: `{(token_metadata_df['n_tokens_pooled'] > 0).all()}`",
            f"- same > mismatched for at least one layer in every pair: "
            f"`{geometry_df.groupby('language_pair')['delta_same_minus_mismatched'].max().gt(0).all()}`",
            "",
        ]
    )
    return write_text(report_path, "\n".join(lines) + "\n")


def _extract_model_pilot(
    *,
    model_name: str,
    n_triplets: int = 100,
    max_tokens_per_batch: int = 2048,
) -> PilotExtractionSummary:
    bootstrap_logs()
    triplets = _load_pilot_triplets(n_triplets)
    pilot_root = _pilot_root(model_name)
    pilot_root.mkdir(parents=True, exist_ok=True)

    arrays_by_language: dict[str, list[np.ndarray]] = {}
    manifest_rows: list[dict[str, Any]] = []
    token_metadata_rows: list[dict[str, Any]] = []

    for language in LANGUAGES:
        pooled_by_layer, token_metadata = _extract_language_embeddings(
            model_name=model_name,
            language=language,
            triplets=triplets,
            max_tokens_per_batch=max_tokens_per_batch,
        )
        arrays_by_language[language] = pooled_by_layer
        manifest_rows.extend(
            _save_language_arrays(
                model_name=model_name,
                language=language,
                pooled_by_layer=pooled_by_layer,
            )
        )
        token_metadata_rows.extend(token_metadata)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df["has_nans"] = manifest_df["filepath"].map(lambda path: bool(np.isnan(np.load(path)).any()))
    manifest_path = pilot_root / "pilot_embedding_manifest.parquet"
    manifest_df.to_parquet(manifest_path, index=False)

    triplet_ids_path = pilot_root / "pilot_triplet_ids.parquet"
    triplets[["triplet_id"]].to_parquet(triplet_ids_path, index=False)

    token_metadata_df = pd.DataFrame(token_metadata_rows).sort_values(["language", "triplet_id"]).reset_index(drop=True)
    token_metadata_path = pilot_root / "token_pooling_metadata.parquet"
    token_metadata_df.to_parquet(token_metadata_path, index=False)

    geometry_df = _geometry_rows(arrays_by_language, model_name=model_name)
    geometry_path = _stats_path(model_name)
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_df.to_parquet(geometry_path, index=False)
    plot_path = _save_geometry_plot(geometry_df, model_name=model_name)
    report_path = _write_report(
        model_name=model_name,
        manifest_df=manifest_df,
        geometry_df=geometry_df,
        token_metadata_df=token_metadata_df,
        n_triplets=n_triplets,
    )

    append_markdown_log(
        project_root() / "outputs" / "logs" / "progress_log.md",
        f"{model_name} Pilot Extraction",
        [
            f"triplets={n_triplets}",
            f"layers={manifest_df['layer_index'].nunique()}",
            f"manifest={manifest_path.as_posix()}",
            f"geometry={geometry_path.as_posix()}",
            f"plot={plot_path.as_posix()}",
        ],
    )

    return PilotExtractionSummary(
        model_name=model_name,
        model_dir=model_local_dir(model_name),
        triplet_ids_path=triplet_ids_path,
        manifest_path=manifest_path,
        token_metadata_path=token_metadata_path,
        geometry_path=geometry_path,
        plot_path=plot_path,
        report_path=report_path,
        n_triplets=n_triplets,
        n_layers=int(manifest_df["layer_index"].nunique()),
        hidden_size=int(manifest_df["hidden_size"].iloc[0]),
    )


def extract_xlmr_pilot(
    *,
    n_triplets: int = 100,
    max_tokens_per_batch: int = 2048,
) -> PilotExtractionSummary:
    return _extract_model_pilot(
        model_name="xlmr",
        n_triplets=n_triplets,
        max_tokens_per_batch=max_tokens_per_batch,
    )


def _extract_model_full(
    *,
    model_name: str,
    max_tokens_per_batch: int = 2048,
) -> FullExtractionSummary:
    bootstrap_logs()
    triplets = _load_all_triplets()
    root = project_root() / "data" / "interim" / "embeddings" / model_name
    root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    token_metadata_rows: list[dict[str, Any]] = []

    for language in LANGUAGES:
        pooled_by_layer, token_metadata = _extract_language_embeddings(
            model_name=model_name,
            language=language,
            triplets=triplets,
            max_tokens_per_batch=max_tokens_per_batch,
        )
        manifest_rows.extend(
            _save_full_language_arrays(
                model_name=model_name,
                language=language,
                pooled_by_layer=pooled_by_layer,
            )
        )
        token_metadata_rows.extend(token_metadata)

    manifest_df = pd.DataFrame(manifest_rows)[
        [
            "model",
            "language",
            "layer_index",
            "layer_depth",
            "n_rows",
            "hidden_size",
            "filepath",
            "dtype",
        ]
    ]
    manifest_path = project_root() / "data" / "interim" / "embeddings" / "embedding_manifest.parquet"
    if manifest_path.exists():
        existing = pd.read_parquet(manifest_path).copy()
        manifest_df = pd.concat([existing.loc[existing["model"] != model_name].copy(), manifest_df], ignore_index=True)
    manifest_df = manifest_df.sort_values(["model", "language", "layer_index"]).reset_index(drop=True)
    manifest_df.to_parquet(manifest_path, index=False)

    triplet_ids_path = root / "triplet_ids.parquet"
    triplets[["triplet_id"]].to_parquet(triplet_ids_path, index=False)

    token_metadata_df = pd.DataFrame(token_metadata_rows).sort_values(["language", "triplet_id"]).reset_index(drop=True)
    token_metadata_path = root / "token_pooling_metadata.parquet"
    token_metadata_df.to_parquet(token_metadata_path, index=False)

    append_markdown_log(
        project_root() / "outputs" / "logs" / "progress_log.md",
        f"{model_name} Full Extraction",
        [
            f"triplets={len(triplets)}",
            f"layers={manifest_df['layer_index'].nunique()}",
            f"manifest={manifest_path.as_posix()}",
            f"token_metadata={token_metadata_path.as_posix()}",
        ],
    )

    return FullExtractionSummary(
        model_name=model_name,
        model_dir=model_local_dir(model_name),
        triplet_ids_path=triplet_ids_path,
        manifest_path=manifest_path,
        token_metadata_path=token_metadata_path,
        n_triplets=len(triplets),
        n_layers=int(manifest_df["layer_index"].nunique()),
        hidden_size=int(manifest_df["hidden_size"].iloc[0]),
    )


def extract_xlmr_full(
    *,
    max_tokens_per_batch: int = 2048,
) -> FullExtractionSummary:
    return _extract_model_full(model_name="xlmr", max_tokens_per_batch=max_tokens_per_batch)


def extract_nllb_full(
    *,
    max_tokens_per_batch: int = 2048,
) -> FullExtractionSummary:
    return _extract_model_full(model_name="nllb_encoder", max_tokens_per_batch=max_tokens_per_batch)
