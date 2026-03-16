from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import inf, log
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

from brain_subspace_paper.config import project_config, project_root
from brain_subspace_paper.logging_utils import append_markdown_log, bootstrap_logs, write_text


ALPHA = 1.0
BETA = 0.25
GAMMA = 0.10
MERGE_PENALTY = 0.10
PAIRWISE_BATCH_SIZE = 64


@dataclass(slots=True)
class AlignmentBuildSummary:
    triplets_path: Path
    qc_path: Path
    report_path: Path
    tsv_path: Path
    n_triplets: int
    n_flagged: int
    n_conflict_windows: int
    n_unresolved: int


@dataclass(slots=True)
class SpanCandidate:
    language: str
    section_index: int
    start0: int
    length: int
    end0: int
    first_sentence_idx: int
    last_sentence_idx: int
    text: str
    onset_sec: float
    offset_sec: float
    char_count: int
    punct_type: str
    embedding: np.ndarray


@dataclass(slots=True)
class PairwiseBlock:
    en_start0: int
    en_len: int
    other_start0: int
    other_len: int
    local_cost: float
    cosine: float


@dataclass(slots=True)
class TripletBlock:
    en_start0: int
    en_len: int
    fr_start0: int
    fr_len: int
    zh_start0: int
    zh_len: int
    en_fr_cost: float
    en_zh_cost: float
    fr_zh_cost: float
    sim_en_fr: float
    sim_en_zh: float
    sim_fr_zh: float
    source: str


def _random_seed() -> int:
    return int(project_config().get("random_seed", 20260314))


def _labse_dir() -> Path:
    return project_root() / "models" / "labse"


@lru_cache(maxsize=1)
def _load_labse_model() -> Any:
    model_dir = _labse_dir()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"LaBSE directory missing at {model_dir}. Download sentence-transformers/LaBSE first."
        )
    try:
        return SentenceTransformer(str(model_dir))
    except OSError as exc:
        raise RuntimeError(
            "LaBSE files are incomplete. Re-run the model download so the alignment-QC "
            "stage can load the local weights."
        ) from exc


def _sentence_span_path(language: str) -> Path:
    return project_root() / "data" / "interim" / f"sentence_spans_{language}.parquet"


def _load_sentence_spans(language: str) -> pd.DataFrame:
    path = _sentence_span_path(language)
    if not path.exists():
        raise FileNotFoundError(
            f"Sentence spans missing for {language}: {path}. Build sentence spans first."
        )
    df = pd.read_parquet(path).copy()
    df = df.sort_values(["section_index", "language_sentence_index"]).reset_index(drop=True)
    return df


def _merge_texts(language: str, texts: list[str]) -> str:
    parts = [text.strip() for text in texts if str(text).strip()]
    if not parts:
        return ""
    if language == "zh":
        return " ".join(parts)
    return " ".join(parts)


def _punct_type(text: str) -> str:
    stripped = str(text).strip()
    if not stripped:
        return "none"
    last = stripped[-1]
    if last in {"?", "?"}:
        return "question"
    if last in {"!", "!"}:
        return "exclamation"
    if last in {".", "。"}:
        return "declarative"
    if last in {";", ":", "；", "：", "…"}:
        return "pause"
    return "none"


def _build_candidates_for_section(language: str, section_df: pd.DataFrame) -> dict[tuple[int, int], SpanCandidate]:
    section_df = section_df.sort_values("language_sentence_index").reset_index(drop=True)
    texts: list[str] = []
    metas: list[tuple[int, int, int, pd.DataFrame]] = []
    for start0 in range(len(section_df)):
        for length in (1, 2):
            end0 = start0 + length
            if end0 > len(section_df):
                continue
            rows = section_df.iloc[start0:end0]
            merged_text = _merge_texts(language, rows["text"].astype(str).tolist())
            texts.append(merged_text)
            metas.append((start0, length, end0 - 1, rows))

    embeddings = _load_labse_model().encode(
        texts,
        batch_size=PAIRWISE_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    candidates: dict[tuple[int, int], SpanCandidate] = {}
    for text, embedding, (start0, length, end0, rows) in zip(texts, embeddings, metas, strict=False):
        candidates[(start0, length)] = SpanCandidate(
            language=language,
            section_index=int(rows["section_index"].iloc[0]),
            start0=start0,
            length=length,
            end0=end0,
            first_sentence_idx=int(rows["language_sentence_index"].iloc[0]),
            last_sentence_idx=int(rows["language_sentence_index"].iloc[-1]),
            text=text,
            onset_sec=float(rows["onset_sec"].min()),
            offset_sec=float(rows["offset_sec"].max()),
            char_count=len(text),
            punct_type=_punct_type(text),
            embedding=np.asarray(embedding, dtype=np.float32),
        )
    return candidates


def _get_or_create_candidate(
    candidates: dict[tuple[int, int], SpanCandidate],
    language: str,
    section_df: pd.DataFrame,
    start0: int,
    length: int,
) -> SpanCandidate:
    key = (start0, length)
    if key in candidates:
        return candidates[key]

    rows = section_df.sort_values("language_sentence_index").reset_index(drop=True).iloc[start0 : start0 + length]
    text = _merge_texts(language, rows["text"].astype(str).tolist())
    embedding = _load_labse_model().encode(
        [text],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    candidate = SpanCandidate(
        language=language,
        section_index=int(rows["section_index"].iloc[0]),
        start0=start0,
        length=length,
        end0=start0 + length - 1,
        first_sentence_idx=int(rows["language_sentence_index"].iloc[0]),
        last_sentence_idx=int(rows["language_sentence_index"].iloc[-1]),
        text=text,
        onset_sec=float(rows["onset_sec"].min()),
        offset_sec=float(rows["offset_sec"].max()),
        char_count=len(text),
        punct_type=_punct_type(text),
        embedding=np.asarray(embedding, dtype=np.float32),
    )
    candidates[key] = candidate
    return candidate


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _pair_cost(
    left: SpanCandidate,
    right: SpanCandidate,
) -> tuple[float, float]:
    cosine = _cosine(left.embedding, right.embedding)
    char_term = abs(log((left.char_count + 1) / (right.char_count + 1)))
    punct_term = 1.0 if left.punct_type != right.punct_type else 0.0
    base_cost = ALPHA * (1.0 - cosine) + BETA * char_term + GAMMA * punct_term
    # Normalize by span size so the DP does not get a free reward for collapsing
    # many neighboring sentences into a single long block.
    length_scale = (left.length + right.length) / 2.0
    merge_term = MERGE_PENALTY * ((left.length - 1) + (right.length - 1))
    cost = base_cost * length_scale + merge_term
    return cost, cosine


def _pairwise_dp(
    en_candidates: dict[tuple[int, int], SpanCandidate],
    other_candidates: dict[tuple[int, int], SpanCandidate],
    n_en: int,
    n_other: int,
) -> list[PairwiseBlock]:
    dp = np.full((n_en + 1, n_other + 1), np.inf, dtype=np.float64)
    back: dict[tuple[int, int], tuple[int, int, int, int, float, float]] = {}
    dp[0, 0] = 0.0

    for i in range(n_en + 1):
        for j in range(n_other + 1):
            current = dp[i, j]
            if not np.isfinite(current):
                continue
            for en_len in (1, 2):
                if i + en_len > n_en:
                    continue
                en_span = en_candidates[(i, en_len)]
                for other_len in (1, 2):
                    if j + other_len > n_other:
                        continue
                    other_span = other_candidates[(j, other_len)]
                    local_cost, cosine = _pair_cost(en_span, other_span)
                    new_cost = current + local_cost
                    if new_cost < dp[i + en_len, j + other_len]:
                        dp[i + en_len, j + other_len] = new_cost
                        back[(i + en_len, j + other_len)] = (
                            i,
                            j,
                            en_len,
                            other_len,
                            local_cost,
                            cosine,
                        )

    if not np.isfinite(dp[n_en, n_other]):
        raise RuntimeError("Pairwise alignment DP did not reach the terminal state.")

    blocks: list[PairwiseBlock] = []
    i = n_en
    j = n_other
    while i > 0 or j > 0:
        prev_i, prev_j, en_len, other_len, local_cost, cosine = back[(i, j)]
        blocks.append(
            PairwiseBlock(
                en_start0=prev_i,
                en_len=en_len,
                other_start0=prev_j,
                other_len=other_len,
                local_cost=local_cost,
                cosine=cosine,
            )
        )
        i, j = prev_i, prev_j
    blocks.reverse()
    return blocks


def _collect_boundary_map(blocks: list[PairwiseBlock], start_idx: int) -> dict[int, int]:
    return {
        block.en_start0 + block.en_len: idx + 1
        for idx, block in enumerate(blocks[start_idx:], start=start_idx)
    }


def _find_conflict_window(
    pair_fr: list[PairwiseBlock],
    pair_zh: list[PairwiseBlock],
    fr_idx: int,
    zh_idx: int,
) -> tuple[int, int]:
    fr_boundaries = _collect_boundary_map(pair_fr, fr_idx)
    zh_boundaries = _collect_boundary_map(pair_zh, zh_idx)
    current_boundary = pair_fr[fr_idx].en_start0
    common = sorted(boundary for boundary in fr_boundaries if boundary in zh_boundaries and boundary > current_boundary)
    if not common:
        raise RuntimeError("Could not find a common English boundary for conflict repair.")
    window_end = common[0]
    return fr_boundaries[window_end], zh_boundaries[window_end]


def _window_block_indices(
    blocks: list[PairwiseBlock],
    start0: int,
    end0: int,
) -> tuple[int, int]:
    start_idx = next(
        idx
        for idx, block in enumerate(blocks)
        if block.en_start0 + block.en_len > start0
    )
    end_idx = next(
        (idx for idx, block in enumerate(blocks[start_idx:], start=start_idx) if block.en_start0 >= end0),
        len(blocks),
    )
    return start_idx, end_idx


def _index_blocks_by_en_start(blocks: list[PairwiseBlock]) -> dict[int, int]:
    start_to_index: dict[int, int] = {}
    for idx, block in enumerate(blocks):
        if block.en_start0 in start_to_index:
            raise RuntimeError(f"Duplicate English start boundary in pairwise path: {block.en_start0}")
        start_to_index[block.en_start0] = idx
    return start_to_index


def _path_boundaries(blocks: list[PairwiseBlock]) -> list[int]:
    boundaries = {0}
    for block in blocks:
        boundaries.add(block.en_start0)
        boundaries.add(block.en_start0 + block.en_len)
    return sorted(boundaries)


def _expand_to_common_boundaries(
    common_boundaries: list[int],
    *,
    desired_start0: int,
    desired_end0: int,
) -> tuple[int, int]:
    start0 = max(boundary for boundary in common_boundaries if boundary <= desired_start0)
    end0 = min(boundary for boundary in common_boundaries if boundary >= desired_end0)
    return start0, end0


def _tri_dp_window(
    en_candidates: dict[tuple[int, int], SpanCandidate],
    fr_candidates: dict[tuple[int, int], SpanCandidate],
    zh_candidates: dict[tuple[int, int], SpanCandidate],
    *,
    en_start0: int,
    en_count: int,
    fr_start0: int,
    fr_count: int,
    zh_start0: int,
    zh_count: int,
) -> list[TripletBlock]:
    states: dict[tuple[int, int, int], float] = {(0, 0, 0): 0.0}
    back: dict[tuple[int, int, int], tuple[int, int, int, int, int, int, float, float, float, float, float, float]] = {}

    for i in range(en_count + 1):
        for j in range(fr_count + 1):
            for k in range(zh_count + 1):
                state = (i, j, k)
                current = states.get(state, inf)
                if not np.isfinite(current):
                    continue
                for en_len in (1, 2):
                    if i + en_len > en_count:
                        continue
                    en_span = en_candidates[(en_start0 + i, en_len)]
                    for fr_len in (1, 2):
                        if j + fr_len > fr_count:
                            continue
                        fr_span = fr_candidates[(fr_start0 + j, fr_len)]
                        en_fr_cost, sim_en_fr = _pair_cost(en_span, fr_span)
                        for zh_len in (1, 2):
                            if k + zh_len > zh_count:
                                continue
                            zh_span = zh_candidates[(zh_start0 + k, zh_len)]
                            en_zh_cost, sim_en_zh = _pair_cost(en_span, zh_span)
                            fr_zh_cost, sim_fr_zh = _pair_cost(fr_span, zh_span)
                            local_cost = en_fr_cost + en_zh_cost + 0.5 * fr_zh_cost
                            next_state = (i + en_len, j + fr_len, k + zh_len)
                            new_cost = current + local_cost
                            if new_cost < states.get(next_state, inf):
                                states[next_state] = new_cost
                                back[next_state] = (
                                    i,
                                    j,
                                    k,
                                    en_len,
                                    fr_len,
                                    zh_len,
                                    en_fr_cost,
                                    en_zh_cost,
                                    fr_zh_cost,
                                    sim_en_fr,
                                    sim_en_zh,
                                    sim_fr_zh,
                                )

    final_state = (en_count, fr_count, zh_count)
    if final_state not in back and final_state != (0, 0, 0):
        raise RuntimeError(
            "Tri-lingual repair DP did not produce a feasible path: "
            f"en_start0={en_start0}, en_count={en_count}, "
            f"fr_start0={fr_start0}, fr_count={fr_count}, "
            f"zh_start0={zh_start0}, zh_count={zh_count}."
        )

    blocks: list[TripletBlock] = []
    i, j, k = final_state
    while i > 0 or j > 0 or k > 0:
        (
            prev_i,
            prev_j,
            prev_k,
            en_len,
            fr_len,
            zh_len,
            en_fr_cost,
            en_zh_cost,
            fr_zh_cost,
            sim_en_fr,
            sim_en_zh,
            sim_fr_zh,
        ) = back[(i, j, k)]
        blocks.append(
            TripletBlock(
                en_start0=en_start0 + prev_i,
                en_len=en_len,
                fr_start0=fr_start0 + prev_j,
                fr_len=fr_len,
                zh_start0=zh_start0 + prev_k,
                zh_len=zh_len,
                en_fr_cost=en_fr_cost,
                en_zh_cost=en_zh_cost,
                fr_zh_cost=fr_zh_cost,
                sim_en_fr=sim_en_fr,
                sim_en_zh=sim_en_zh,
                sim_fr_zh=sim_fr_zh,
                source="tri_repair",
            )
        )
        i, j, k = prev_i, prev_j, prev_k
    blocks.reverse()
    return blocks


def _direct_triplet_block(
    fr_block: PairwiseBlock,
    zh_block: PairwiseBlock,
    en_candidates: dict[tuple[int, int], SpanCandidate],
    fr_candidates: dict[tuple[int, int], SpanCandidate],
    zh_candidates: dict[tuple[int, int], SpanCandidate],
) -> TripletBlock:
    en_span = en_candidates[(fr_block.en_start0, fr_block.en_len)]
    fr_span = fr_candidates[(fr_block.other_start0, fr_block.other_len)]
    zh_span = zh_candidates[(zh_block.other_start0, zh_block.other_len)]
    fr_zh_cost, sim_fr_zh = _pair_cost(fr_span, zh_span)
    return TripletBlock(
        en_start0=fr_block.en_start0,
        en_len=fr_block.en_len,
        fr_start0=fr_block.other_start0,
        fr_len=fr_block.other_len,
        zh_start0=zh_block.other_start0,
        zh_len=zh_block.other_len,
        en_fr_cost=fr_block.local_cost,
        en_zh_cost=zh_block.local_cost,
        fr_zh_cost=fr_zh_cost,
        sim_en_fr=fr_block.cosine,
        sim_en_zh=zh_block.cosine,
        sim_fr_zh=sim_fr_zh,
        source="pairwise_agree",
    )


def _merge_pairwise_paths(
    pair_fr: list[PairwiseBlock],
    pair_zh: list[PairwiseBlock],
    en_candidates: dict[tuple[int, int], SpanCandidate],
    fr_candidates: dict[tuple[int, int], SpanCandidate],
    zh_candidates: dict[tuple[int, int], SpanCandidate],
) -> tuple[list[TripletBlock], int, int]:
    triplets: list[TripletBlock] = []
    conflict_windows = 0
    unresolved_rows = 0
    total_en = max(candidate.end0 for candidate in en_candidates.values()) + 1
    fr_start_to_index = _index_blocks_by_en_start(pair_fr)
    zh_start_to_index = _index_blocks_by_en_start(pair_zh)
    common_boundaries = sorted(set(_path_boundaries(pair_fr)).intersection(_path_boundaries(pair_zh)))
    cursor = 0

    while cursor < total_en:
        if cursor not in fr_start_to_index or cursor not in zh_start_to_index:
            raise RuntimeError(f"Missing pairwise block at English boundary {cursor}.")
        fr_idx = fr_start_to_index[cursor]
        zh_idx = zh_start_to_index[cursor]
        fr_block = pair_fr[fr_idx]
        zh_block = pair_zh[zh_idx]

        if fr_block.en_len == zh_block.en_len:
            triplets.append(
                _direct_triplet_block(
                    fr_block,
                    zh_block,
                    en_candidates=en_candidates,
                    fr_candidates=fr_candidates,
                    zh_candidates=zh_candidates,
                )
            )
            cursor += fr_block.en_len
            continue

        conflict_windows += 1
        fr_end_idx, zh_end_idx = _find_conflict_window(pair_fr, pair_zh, fr_idx, zh_idx)
        en_start0 = cursor
        en_end0 = pair_fr[fr_end_idx - 1].en_start0 + pair_fr[fr_end_idx - 1].en_len
        fr_start0 = pair_fr[fr_idx].other_start0
        fr_end0 = pair_fr[fr_end_idx - 1].other_start0 + pair_fr[fr_end_idx - 1].other_len
        zh_start0 = pair_zh[zh_idx].other_start0
        zh_end0 = pair_zh[zh_end_idx - 1].other_start0 + pair_zh[zh_end_idx - 1].other_len
        try:
            resolved = _tri_dp_window(
                en_candidates=en_candidates,
                fr_candidates=fr_candidates,
                zh_candidates=zh_candidates,
                en_start0=en_start0,
                en_count=en_end0 - en_start0,
                fr_start0=fr_start0,
                fr_count=fr_end0 - fr_start0,
                zh_start0=zh_start0,
                zh_count=zh_end0 - zh_start0,
            )
            triplets.extend(resolved)
            cursor = en_end0
            continue
        except RuntimeError:
            desired_start0 = max(0, en_start0 - 1)
            desired_end0 = min(total_en, en_end0 + 1)
            expanded_start0, expanded_end0 = _expand_to_common_boundaries(
                common_boundaries,
                desired_start0=desired_start0,
                desired_end0=desired_end0,
            )
            if expanded_start0 == en_start0 and expanded_end0 == en_end0:
                raise

            while triplets and triplets[-1].en_start0 + triplets[-1].en_len > expanded_start0:
                triplets.pop()

            expanded_fr_start_idx, expanded_fr_end_idx = _window_block_indices(
                pair_fr,
                expanded_start0,
                expanded_end0,
            )
            expanded_zh_start_idx, expanded_zh_end_idx = _window_block_indices(
                pair_zh,
                expanded_start0,
                expanded_end0,
            )
            expanded_fr_start0 = pair_fr[expanded_fr_start_idx].other_start0
            expanded_fr_end0 = (
                pair_fr[expanded_fr_end_idx - 1].other_start0 + pair_fr[expanded_fr_end_idx - 1].other_len
            )
            expanded_zh_start0 = pair_zh[expanded_zh_start_idx].other_start0
            expanded_zh_end0 = (
                pair_zh[expanded_zh_end_idx - 1].other_start0 + pair_zh[expanded_zh_end_idx - 1].other_len
            )
            try:
                resolved = _tri_dp_window(
                    en_candidates=en_candidates,
                    fr_candidates=fr_candidates,
                    zh_candidates=zh_candidates,
                    en_start0=expanded_start0,
                    en_count=expanded_end0 - expanded_start0,
                    fr_start0=expanded_fr_start0,
                    fr_count=expanded_fr_end0 - expanded_fr_start0,
                    zh_start0=expanded_zh_start0,
                    zh_count=expanded_zh_end0 - expanded_zh_start0,
                )
                triplets.extend(resolved)
                cursor = expanded_end0
            except RuntimeError:
                unresolved_rows += 1
                triplets.append(
                    TripletBlock(
                        en_start0=expanded_start0,
                        en_len=expanded_end0 - expanded_start0,
                        fr_start0=expanded_fr_start0,
                        fr_len=expanded_fr_end0 - expanded_fr_start0,
                        zh_start0=expanded_zh_start0,
                        zh_len=expanded_zh_end0 - expanded_zh_start0,
                        en_fr_cost=float("nan"),
                        en_zh_cost=float("nan"),
                        fr_zh_cost=float("nan"),
                        sim_en_fr=float("nan"),
                        sim_en_zh=float("nan"),
                        sim_fr_zh=float("nan"),
                        source="needs_fix",
                    )
                )
                cursor = expanded_end0

    if cursor != total_en:
        raise RuntimeError("Pairwise merge ended with leftover blocks.")
    return triplets, conflict_windows, unresolved_rows


def _build_triplet_and_qc_rows(
    section_index: int,
    triplet_blocks: list[TripletBlock],
    en_candidates: dict[tuple[int, int], SpanCandidate],
    fr_candidates: dict[tuple[int, int], SpanCandidate],
    zh_candidates: dict[tuple[int, int], SpanCandidate],
    en_section: pd.DataFrame,
    fr_section: pd.DataFrame,
    zh_section: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    triplet_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    for section_triplet_index, block in enumerate(triplet_blocks, start=1):
        en_span = _get_or_create_candidate(en_candidates, "en", en_section, block.en_start0, block.en_len)
        fr_span = _get_or_create_candidate(fr_candidates, "fr", fr_section, block.fr_start0, block.fr_len)
        zh_span = _get_or_create_candidate(zh_candidates, "zh", zh_section, block.zh_start0, block.zh_len)
        merge_pattern = f"{block.en_len}-{block.fr_len}-{block.zh_len}"
        if np.isnan(block.sim_en_fr) or np.isnan(block.sim_en_zh) or np.isnan(block.sim_fr_zh):
            en_fr_cost, sim_en_fr = _pair_cost(en_span, fr_span)
            en_zh_cost, sim_en_zh = _pair_cost(en_span, zh_span)
            fr_zh_cost, sim_fr_zh = _pair_cost(fr_span, zh_span)
        else:
            en_fr_cost, sim_en_fr = block.en_fr_cost, block.sim_en_fr
            en_zh_cost, sim_en_zh = block.en_zh_cost, block.sim_en_zh
            fr_zh_cost, sim_fr_zh = block.fr_zh_cost, block.sim_fr_zh
        mean_pairwise_sim = float(np.mean([sim_en_fr, sim_en_zh, sim_fr_zh]))
        char_ratio_en_fr = (en_span.char_count + 1) / (fr_span.char_count + 1)
        char_ratio_en_zh = (en_span.char_count + 1) / (zh_span.char_count + 1)
        punct_mismatch_flag = len({en_span.punct_type, fr_span.punct_type, zh_span.punct_type}) > 1
        max_local_cost = max(en_fr_cost, en_zh_cost, fr_zh_cost)
        needs_manual_review = any(
            [
                sim_en_fr < 0.35,
                sim_en_zh < 0.35,
                sim_fr_zh < 0.35,
                mean_pairwise_sim < 0.40,
                max_local_cost > 1.25,
                punct_mismatch_flag,
                sum(length == 2 for length in (block.en_len, block.fr_len, block.zh_len)) > 1,
                block.source == "needs_fix",
            ]
        )
        triplet_rows.append(
            {
                "section_index": section_index,
                "section_triplet_index": section_triplet_index,
                "merge_pattern": merge_pattern,
                "en_text": en_span.text,
                "fr_text": fr_span.text,
                "zh_text": zh_span.text,
                "en_onset_sec": en_span.onset_sec,
                "en_offset_sec": en_span.offset_sec,
                "fr_onset_sec": fr_span.onset_sec,
                "fr_offset_sec": fr_span.offset_sec,
                "zh_onset_sec": zh_span.onset_sec,
                "zh_offset_sec": zh_span.offset_sec,
                "en_first_sentence_idx": en_span.first_sentence_idx,
                "en_last_sentence_idx": en_span.last_sentence_idx,
                "fr_first_sentence_idx": fr_span.first_sentence_idx,
                "fr_last_sentence_idx": fr_span.last_sentence_idx,
                "zh_first_sentence_idx": zh_span.first_sentence_idx,
                "zh_last_sentence_idx": zh_span.last_sentence_idx,
            }
        )
        qc_rows.append(
            {
                "section_index": section_index,
                "sim_en_fr": sim_en_fr,
                "sim_en_zh": sim_en_zh,
                "sim_fr_zh": sim_fr_zh,
                "mean_pairwise_sim": mean_pairwise_sim,
                "char_ratio_en_fr": char_ratio_en_fr,
                "char_ratio_en_zh": char_ratio_en_zh,
                "punct_mismatch_flag": punct_mismatch_flag,
                "max_local_cost": max_local_cost,
                "needs_manual_review": needs_manual_review,
                "manual_status": "needs_fix" if block.source == "needs_fix" else ("" if needs_manual_review else "approved"),
                "notes": "" if not needs_manual_review else f"auto_flagged:{merge_pattern}:{block.source}",
            }
        )

    return triplet_rows, qc_rows


def _flag_ratio_outliers(qc_df: pd.DataFrame) -> pd.DataFrame:
    result = qc_df.copy()
    for column in ("char_ratio_en_fr", "char_ratio_en_zh"):
        values = result[column].astype(float)
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flag = (values < lower) | (values > upper)
        result.loc[flag, "needs_manual_review"] = True
        result.loc[flag & (result["notes"] == ""), "notes"] = f"auto_flagged:{column}_outlier"
    return result


def _manual_review_sample(
    triplets_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    n_samples: int = 100,
) -> pd.DataFrame:
    merged = triplets_df.merge(
        qc_df[["triplet_id", "needs_manual_review", "mean_pairwise_sim"]],
        on="triplet_id",
        how="left",
    )
    rng = np.random.default_rng(_random_seed())
    selected_ids: list[int] = []

    def take_rows(frame: pd.DataFrame, count: int) -> None:
        if count <= 0 or frame.empty:
            return
        available = frame.loc[~frame["triplet_id"].isin(selected_ids)]
        if available.empty:
            return
        take = min(len(available), count)
        sampled = available.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
        selected_ids.extend(sampled["triplet_id"].tolist())

    flagged = merged.loc[merged["needs_manual_review"]].copy()
    take_rows(flagged, min(n_samples // 2, len(flagged)))

    low_qc = merged.nsmallest(min(20, len(merged)), columns=["mean_pairwise_sim"])
    high_qc = merged.nlargest(min(20, len(merged)), columns=["mean_pairwise_sim"])
    take_rows(low_qc, 20)
    take_rows(high_qc, 20)

    for section_index in sorted(merged["section_index"].unique().tolist()):
        take_rows(merged.loc[merged["section_index"] == section_index], 2)

    for merge_pattern in sorted(merged["merge_pattern"].unique().tolist()):
        take_rows(merged.loc[merged["merge_pattern"] == merge_pattern], 2)

    if len(selected_ids) < n_samples:
        remaining = merged.loc[~merged["triplet_id"].isin(selected_ids)]
        take_rows(remaining, n_samples - len(selected_ids))

    review_df = merged.loc[merged["triplet_id"].isin(selected_ids)].copy()
    review_df = review_df.drop_duplicates(subset=["triplet_id"]).sort_values("triplet_id")
    return review_df.head(n_samples)


def build_alignment_triplets() -> AlignmentBuildSummary:
    bootstrap_logs()
    en_df = _load_sentence_spans("en")
    fr_df = _load_sentence_spans("fr")
    zh_df = _load_sentence_spans("zh")

    triplet_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    conflict_windows = 0
    unresolved_rows = 0

    for section_index in sorted(en_df["section_index"].astype(int).unique()):
        en_section = en_df.loc[en_df["section_index"].astype(int) == section_index].copy()
        fr_section = fr_df.loc[fr_df["section_index"].astype(int) == section_index].copy()
        zh_section = zh_df.loc[zh_df["section_index"].astype(int) == section_index].copy()
        en_candidates = _build_candidates_for_section("en", en_section)
        fr_candidates = _build_candidates_for_section("fr", fr_section)
        zh_candidates = _build_candidates_for_section("zh", zh_section)
        pair_fr = _pairwise_dp(
            en_candidates=en_candidates,
            other_candidates=fr_candidates,
            n_en=len(en_section),
            n_other=len(fr_section),
        )
        pair_zh = _pairwise_dp(
            en_candidates=en_candidates,
            other_candidates=zh_candidates,
            n_en=len(en_section),
            n_other=len(zh_section),
        )
        triplet_blocks, section_conflicts, section_unresolved = _merge_pairwise_paths(
            pair_fr=pair_fr,
            pair_zh=pair_zh,
            en_candidates=en_candidates,
            fr_candidates=fr_candidates,
            zh_candidates=zh_candidates,
        )
        section_triplets, section_qc = _build_triplet_and_qc_rows(
            section_index=section_index,
            triplet_blocks=triplet_blocks,
            en_candidates=en_candidates,
            fr_candidates=fr_candidates,
            zh_candidates=zh_candidates,
            en_section=en_section,
            fr_section=fr_section,
            zh_section=zh_section,
        )
        conflict_windows += section_conflicts
        unresolved_rows += section_unresolved
        triplet_rows.extend(section_triplets)
        qc_rows.extend(section_qc)

    triplets_df = pd.DataFrame(triplet_rows)
    triplets_df.insert(0, "triplet_id", np.arange(1, len(triplets_df) + 1, dtype=np.int64))
    if len(triplets_df) and unresolved_rows / len(triplets_df) > 0.01:
        raise RuntimeError(
            f"Unresolved alignment rows exceed 1% of the triplet table: "
            f"{unresolved_rows}/{len(triplets_df)}."
        )
    qc_df = pd.DataFrame(qc_rows)
    qc_df.insert(0, "triplet_id", triplets_df["triplet_id"].to_numpy())
    qc_df = _flag_ratio_outliers(qc_df)
    qc_df.loc[qc_df["manual_status"] == "", "manual_status"] = ""
    qc_df = qc_df[
        [
            "triplet_id",
            "section_index",
            "sim_en_fr",
            "sim_en_zh",
            "sim_fr_zh",
            "mean_pairwise_sim",
            "char_ratio_en_fr",
            "char_ratio_en_zh",
            "punct_mismatch_flag",
            "max_local_cost",
            "needs_manual_review",
            "manual_status",
            "notes",
        ]
    ]

    processed_root = project_root() / "data" / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)
    triplets_path = processed_root / "alignment_triplets.parquet"
    tsv_path = processed_root / "alignment_triplets.tsv"
    qc_path = processed_root / "alignment_triplets_qc.parquet"
    triplets_df.to_parquet(triplets_path, index=False)
    triplets_df.to_csv(tsv_path, index=False, sep="\t")
    qc_df.to_parquet(qc_path, index=False)

    review_df = _manual_review_sample(triplets_df, qc_df, n_samples=100)
    report_lines = [
        "# Alignment QC Report",
        "",
        f"- triplets: `{len(triplets_df)}`",
        f"- flagged for manual review: `{int(qc_df['needs_manual_review'].sum())}`",
        f"- conflict windows repaired with tri-lingual DP: `{conflict_windows}`",
        f"- unresolved rows: `{unresolved_rows}`",
        "",
        "## QC Threshold Summary",
        "",
        "- pairwise cosine < `0.35`",
        "- mean pairwise cosine < `0.40`",
        "- max local cost > `1.25`",
        "- punctuation mismatch",
        "- 2-sentence merges in more than one language",
        "- character-ratio outlier",
        "",
        "## Merge Pattern Counts",
        "",
    ]
    for merge_pattern, count in triplets_df["merge_pattern"].value_counts().sort_index().items():
        report_lines.append(f"- `{merge_pattern}`: `{int(count)}`")

    if unresolved_rows:
        report_lines.extend(
            [
                "",
                "## Needs-Fix Regions",
                "",
            ]
        )
        unresolved_df = triplets_df.merge(
            qc_df.loc[qc_df["manual_status"] == "needs_fix", ["triplet_id", "notes"]],
            on="triplet_id",
            how="inner",
        )
        for row in unresolved_df[["triplet_id", "section_index", "merge_pattern", "notes"]].itertuples(index=False):
            report_lines.append(
                f"- triplet `{int(row.triplet_id)}` section `{int(row.section_index)}` "
                f"merge `{row.merge_pattern}` note `{row.notes}`"
            )

    report_lines.extend(
        [
            "",
            "## Manual Review Sample",
            "",
            "The following triplets are the auto-selected sample for the required manual alignment review.",
            "",
        ]
    )
    for row in review_df[["triplet_id", "section_index", "merge_pattern", "en_text", "fr_text", "zh_text"]].itertuples(index=False):
        report_lines.extend(
            [
                f"### triplet {int(row.triplet_id)}",
                f"- section: `{int(row.section_index)}`",
                f"- merge pattern: `{row.merge_pattern}`",
                f"- en: `{row.en_text}`",
                f"- fr: `{row.fr_text}`",
                f"- zh: `{row.zh_text}`",
                "",
            ]
        )

    report_path = project_root() / "outputs" / "logs" / "alignment_qc_report.md"
    write_text(report_path, "\n".join(report_lines))
    append_markdown_log(
        project_root() / "outputs" / "logs" / "progress_log.md",
        "Alignment triplets",
        [
            f"Built {len(triplets_df)} alignment triplets.",
            f"Wrote triplets to {triplets_path.relative_to(project_root()).as_posix()} and {tsv_path.relative_to(project_root()).as_posix()}.",
            f"Wrote QC table to {qc_path.relative_to(project_root()).as_posix()}.",
            f"Wrote alignment QC report to {report_path.relative_to(project_root()).as_posix()}.",
        ],
    )

    return AlignmentBuildSummary(
        triplets_path=triplets_path,
        qc_path=qc_path,
        report_path=report_path,
        tsv_path=tsv_path,
        n_triplets=len(triplets_df),
        n_flagged=int(qc_df["needs_manual_review"].sum()),
        n_conflict_windows=conflict_windows,
        n_unresolved=unresolved_rows,
    )
