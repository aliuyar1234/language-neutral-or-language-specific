from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import json
import re
import subprocess
import unicodedata

import pandas as pd

from brain_subspace_paper.config import project_root
from brain_subspace_paper.logging_utils import append_markdown_log, bootstrap_logs, write_text


LANGUAGE_SPECS = {
    "en": {"folder": "EN", "stem": "lppEN"},
    "fr": {"folder": "FR", "stem": "lppFR"},
    "zh": {"folder": "CN", "stem": "lppCN"},
}


@dataclass(slots=True)
class SentenceSpanBuildSummary:
    language: str
    output_path: Path
    report_path: Path
    n_sentences: int
    n_sections: int


@dataclass(slots=True)
class TextGridAlignmentSummary:
    section_index: int
    row_count: int
    interval_count: int
    matched_rows: int
    matched_intervals: int
    unmatched_rows: int
    unmatched_intervals: int
    row_coverage: float
    interval_coverage: float
    status: str


def _dataset_root() -> Path:
    return project_root() / "data" / "raw" / "ds003643"


def _annotation_dir(language: str) -> Path:
    spec = LANGUAGE_SPECS[language]
    return _dataset_root() / "annotation" / spec["folder"]


def _normalize_token(token: str) -> str:
    text = unicodedata.normalize("NFKC", str(token)).strip().lower()
    text = text.replace("`", "'").replace("\ufeff", "")
    text = text.replace("â€™", "'").replace("\x92", "'").replace("’", "'")
    text = text.replace("…", "...").replace("“", '"').replace("”", '"')
    return text


def _strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _canonical_forms(tokens: list[str], language: str) -> set[str]:
    normalized = [_normalize_token(token) for token in tokens if _normalize_token(token)]
    if not normalized:
        return set()

    collapsed = [token.replace("'", "").replace("-", "") for token in normalized]
    folded = [_strip_diacritics(token) for token in normalized]
    folded_collapsed = [token.replace("'", "").replace("-", "") for token in folded]
    forms = {
        " ".join(normalized),
        "".join(normalized),
        " ".join(collapsed),
        "".join(collapsed),
        " ".join(folded),
        "".join(folded),
        " ".join(folded_collapsed),
        "".join(folded_collapsed),
    }

    if language == "fr":
        values = tuple(normalized)
        if values == ("de", "le"):
            forms.add("du")
        if values == ("de", "les"):
            forms.add("des")
        if values in {("\u00e0", "le"), ("a", "le")}:
            forms.add("au")
        if values in {("\u00e0", "les"), ("a", "les")}:
            forms.add("aux")

    return {form for form in forms if form}


def _groups_equivalent(lhs: list[str], rhs: list[str], language: str) -> bool:
    return bool(_canonical_forms(lhs, language) & _canonical_forms(rhs, language))


def _resolve_annex_content_path(path: Path) -> Path:
    dataset_root = _dataset_root()
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        if path.stat().st_size > 1024:
            return path
        content = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return path

    if ".git/annex/objects/" not in content:
        return path

    relative_path = path.relative_to(dataset_root).as_posix()
    info_result = subprocess.run(
        [
            "git",
            "-C",
            str(dataset_root),
            "annex",
            "info",
            relative_path,
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(info_result.stdout.strip() or "{}")
    if not info.get("present", False):
        subprocess.run(
            [
                "git",
                "-C",
                str(dataset_root),
                "annex",
                "get",
                relative_path,
            ],
            check=True,
        )
        info_result = subprocess.run(
            [
                "git",
                "-C",
                str(dataset_root),
                "annex",
                "info",
                relative_path,
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        info = json.loads(info_result.stdout.strip() or "{}")

    key = str(info.get("key", "")).strip()
    if key:
        location_result = subprocess.run(
            [
                "git",
                "-C",
                str(dataset_root),
                "annex",
                "contentlocation",
                key,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        location = location_result.stdout.strip()
        if location:
            target = (dataset_root / location).resolve()
            if target.exists():
                return target

    rel = content.splitlines()[0].strip()
    target = (path.parent / rel).resolve()
    if target.exists():
        return target
    return path


def _read_text(path: Path) -> str:
    resolved = _resolve_annex_content_path(path)
    data = resolved.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(_read_text(path)))


def _parse_tree_leaves(tree_line: str) -> list[str]:
    text = tree_line.strip()
    if not text:
        return []

    n_chars = len(text)
    pointer = 0
    leaves: list[str] = []

    def skip_ws(index: int) -> int:
        while index < n_chars and text[index].isspace():
            index += 1
        return index

    def parse_node(index: int) -> tuple[list[str], int]:
        assert text[index] == "("
        index += 1
        index = skip_ws(index)
        while index < n_chars and not text[index].isspace() and text[index] not in "()":
            index += 1
        index = skip_ws(index)

        node_leaves: list[str] = []
        if index < n_chars and text[index] == "(":
            while index < n_chars and text[index] == "(":
                child_leaves, index = parse_node(index)
                node_leaves.extend(child_leaves)
                index = skip_ws(index)
        else:
            start = index
            while index < n_chars and text[index] != ")":
                index += 1
            token = text[start:index].strip()
            if token:
                node_leaves.extend(part for part in token.split() if part)

        if index < n_chars and text[index] == ")":
            index += 1
        return node_leaves, skip_ws(index)

    while pointer < n_chars:
        pointer = skip_ws(pointer)
        if pointer < n_chars and text[pointer] == "(":
            node_leaves, pointer = parse_node(pointer)
            leaves.extend(node_leaves)
        else:
            pointer += 1

    return leaves


def _parse_tree_sentences(path: Path) -> list[list[str]]:
    return [_parse_tree_leaves(line) for line in _read_text(path).splitlines() if line.strip()]


def _parse_textgrid_intervals(path: Path, language: str) -> list[dict[str, object]]:
    content = _read_text(path).replace("\x00", "")
    pattern = re.compile(
        r"intervals \[\d+\]:\s*xmin = ([0-9eE+\-.]+)\s*xmax = ([0-9eE+\-.]+)\s*text = \"(.*?)\"",
        re.S,
    )
    intervals: list[dict[str, object]] = []
    for onset, offset, token in pattern.findall(content):
        text = token.strip()
        normalized = _normalize_token(text)
        if not normalized:
            continue
        if language == "en" and normalized == "#":
            continue
        if language == "zh" and normalized == "sil":
            continue
        intervals.append(
            {
                "onset_sec": float(onset),
                "offset_sec": float(offset),
                "text": text,
                "normalized": normalized,
            }
        )
    return intervals


def _is_soft_skip_token(token: str) -> bool:
    normalized = _normalize_token(token)
    if not normalized:
        return True
    return normalized == "..." or re.fullmatch(r"[-.?!,:;\"']+", normalized) is not None


def _prepare_word_rows(language: str) -> pd.DataFrame:
    spec = LANGUAGE_SPECS[language]
    path = _annotation_dir(language) / f"{spec['stem']}_word_information.csv"
    df = _read_csv(path).copy()
    if "section" not in df.columns:
        raise ValueError(f"`section` column missing from {path}")

    df["word"] = df["word"].fillna("").astype(str)
    df["lemma"] = df["lemma"].fillna("").astype(str)
    df["surface_token"] = df["word"].fillna("").astype(str)
    df["alignment_token"] = [
        lemma if str(lemma).strip() else word
        for word, lemma in zip(df["surface_token"], df["lemma"], strict=False)
    ]
    df["alignment_token"] = df["alignment_token"].astype(str)
    df["normalized_alignment_token"] = df["alignment_token"].map(_normalize_token)
    df["normalized_surface_token"] = df["surface_token"].map(_normalize_token)
    df["word_row_index"] = list(range(len(df)))
    return df


def _align_section_rows_to_textgrid(
    *,
    language: str,
    section_index: int,
    section_df: pd.DataFrame,
) -> tuple[pd.DataFrame, TextGridAlignmentSummary]:
    spec = LANGUAGE_SPECS[language]
    tg_path = _annotation_dir(language) / f"{spec['stem']}_section{section_index}.TextGrid"
    intervals = _parse_textgrid_intervals(tg_path, language=language)

    aligned = section_df.copy()
    aligned["textgrid_onset_sec"] = float("nan")
    aligned["textgrid_offset_sec"] = float("nan")

    spoken_mask = aligned["normalized_surface_token"].astype(str).str.len() > 0
    spoken_rows = aligned.loc[spoken_mask].copy()
    row_tokens = spoken_rows["normalized_surface_token"].tolist()
    interval_tokens = [str(interval["normalized"]) for interval in intervals]
    assignments: list[tuple[int, int] | None] = [None] * len(spoken_rows)

    matched_rows = 0
    matched_intervals = 0
    unmatched_rows = 0
    unmatched_intervals = 0
    row_pointer = 0
    interval_pointer = 0
    preferred_spans = [
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
    ]

    while row_pointer < len(row_tokens) and interval_pointer < len(interval_tokens):
        matched = False
        for row_span, interval_span in preferred_spans:
            if row_pointer + row_span > len(row_tokens):
                continue
            if interval_pointer + interval_span > len(interval_tokens):
                continue
            if not _groups_equivalent(
                row_tokens[row_pointer : row_pointer + row_span],
                interval_tokens[interval_pointer : interval_pointer + interval_span],
                language=language,
            ):
                continue

            for local_row_idx in range(row_pointer, row_pointer + row_span):
                assignments[local_row_idx] = (interval_pointer, interval_pointer + interval_span - 1)
            matched_rows += row_span
            matched_intervals += interval_span
            row_pointer += row_span
            interval_pointer += interval_span
            matched = True
            break

        if matched:
            continue

        current_row = row_tokens[row_pointer]
        current_interval = interval_tokens[interval_pointer]
        if _is_soft_skip_token(current_interval):
            unmatched_intervals += 1
            interval_pointer += 1
            continue
        if _is_soft_skip_token(current_row):
            unmatched_rows += 1
            row_pointer += 1
            continue

        grid_lookahead = next(
            (
                delta
                for delta in range(1, 4)
                if interval_pointer + delta < len(interval_tokens)
                and _groups_equivalent(
                    [current_row],
                    [interval_tokens[interval_pointer + delta]],
                    language=language,
                )
            ),
            None,
        )
        row_lookahead = next(
            (
                delta
                for delta in range(1, 4)
                if row_pointer + delta < len(row_tokens)
                and _groups_equivalent(
                    [row_tokens[row_pointer + delta]],
                    [current_interval],
                    language=language,
                )
            ),
            None,
        )

        if grid_lookahead is not None and (row_lookahead is None or grid_lookahead <= row_lookahead):
            unmatched_intervals += grid_lookahead
            interval_pointer += grid_lookahead
            continue
        if row_lookahead is not None:
            unmatched_rows += row_lookahead
            row_pointer += row_lookahead
            continue

        unmatched_rows += 1
        unmatched_intervals += 1
        row_pointer += 1
        interval_pointer += 1

    unmatched_rows += len(row_tokens) - row_pointer
    unmatched_intervals += len(interval_tokens) - interval_pointer

    for local_row_idx, assignment in enumerate(assignments):
        if assignment is None:
            continue
        original_index = spoken_rows.index[local_row_idx]
        first_interval, last_interval = assignment
        aligned.at[original_index, "textgrid_onset_sec"] = float(intervals[first_interval]["onset_sec"])
        aligned.at[original_index, "textgrid_offset_sec"] = float(intervals[last_interval]["offset_sec"])

    row_count = len(row_tokens)
    interval_count = len(interval_tokens)
    row_coverage = matched_rows / row_count if row_count else 0.0
    interval_coverage = matched_intervals / interval_count if interval_count else 0.0
    status = "exact" if unmatched_rows == 0 and unmatched_intervals == 0 else "approx"

    return aligned, TextGridAlignmentSummary(
        section_index=section_index,
        row_count=row_count,
        interval_count=interval_count,
        matched_rows=matched_rows,
        matched_intervals=matched_intervals,
        unmatched_rows=unmatched_rows,
        unmatched_intervals=unmatched_intervals,
        row_coverage=row_coverage,
        interval_coverage=interval_coverage,
        status=status,
    )


def _align_sentence_to_rows(
    tree_tokens: list[str],
    row_tokens: list[str],
    language: str,
) -> tuple[int, float]:
    sentence_tokens = [_normalize_token(token) for token in tree_tokens if _normalize_token(token)]
    if not sentence_tokens:
        return 0, 0.0

    n_rows = len(row_tokens)
    if n_rows == 0:
        raise ValueError("No word rows left for sentence alignment.")

    if language == "zh":
        max_rows = min(n_rows, max(len(sentence_tokens) + 12, len(sentence_tokens) * 3))
        max_group = 5
        cost_limit = max(4.0, 0.40 * len(sentence_tokens))
    else:
        max_rows = min(n_rows, max(len(sentence_tokens) + 8, len(sentence_tokens) * 2))
        max_group = 4
        cost_limit = max(3.0, 0.25 * len(sentence_tokens))
    rows = row_tokens[:max_rows]
    inf = 10**9

    dp_cost = [[inf] * (max_rows + 1) for _ in range(len(sentence_tokens) + 1)]
    dp_match = [[-1] * (max_rows + 1) for _ in range(len(sentence_tokens) + 1)]
    dp_cost[0][0] = 0
    dp_match[0][0] = 0

    def relax(i: int, j: int, new_cost: float, new_match: int) -> None:
        if new_cost < dp_cost[i][j] or (new_cost == dp_cost[i][j] and new_match > dp_match[i][j]):
            dp_cost[i][j] = new_cost
            dp_match[i][j] = new_match

    for i in range(len(sentence_tokens) + 1):
        for j in range(max_rows + 1):
            current_cost = dp_cost[i][j]
            if current_cost >= inf:
                continue
            current_match = dp_match[i][j]
            if i < len(sentence_tokens):
                relax(i + 1, j, current_cost + 1.0, current_match)
            if j < max_rows:
                relax(i, j + 1, current_cost + 1.0, current_match)
            for tree_span in range(1, max_group + 1):
                for row_span in range(1, max_group + 1):
                    if i + tree_span > len(sentence_tokens) or j + row_span > max_rows:
                        continue
                    if _groups_equivalent(
                        sentence_tokens[i : i + tree_span],
                        rows[j : j + row_span],
                        language=language,
                    ):
                        relax(
                            i + tree_span,
                            j + row_span,
                            current_cost,
                            current_match + max(tree_span, row_span),
                        )

    target_rows = min(max_rows, max(1, len(sentence_tokens)))
    best_j = None
    best_key = None
    for j in range(1, max_rows + 1):
        cost = dp_cost[len(sentence_tokens)][j]
        if cost >= inf:
            continue
        key = (cost, abs(j - target_rows), -dp_match[len(sentence_tokens)][j], j)
        if best_key is None or key < best_key:
            best_key = key
            best_j = j

    if best_j is None:
        raise ValueError("No monotonic alignment path found for sentence tokens.")

    best_cost = float(dp_cost[len(sentence_tokens)][best_j])
    if best_cost > cost_limit:
        raise ValueError(f"Sentence alignment cost too high ({best_cost}) for language {language}.")
    return best_j, best_cost


def _render_sentence_text(rows: pd.DataFrame, language: str) -> str:
    tokens = [token for token in rows["surface_token"].astype(str).tolist() if token.strip()]
    if language == "zh":
        return "".join(tokens)
    return " ".join(tokens)


def build_sentence_spans(language: str) -> SentenceSpanBuildSummary:
    bootstrap_logs()
    if language not in LANGUAGE_SPECS:
        raise ValueError(f"Unsupported language: {language}")

    spec = LANGUAGE_SPECS[language]
    word_df = _prepare_word_rows(language)
    tree_path = _annotation_dir(language) / f"{spec['stem']}_tree.csv"
    tree_sentences = _parse_tree_sentences(tree_path)

    section_textgrid_summaries: dict[int, TextGridAlignmentSummary] = {}
    aligned_sections: list[pd.DataFrame] = []
    for section_index in sorted(word_df["section"].astype(int).unique()):
        section_df = word_df.loc[word_df["section"].astype(int) == section_index]
        aligned_section_df, textgrid_summary = _align_section_rows_to_textgrid(
            language=language,
            section_index=section_index,
            section_df=section_df,
        )
        aligned_sections.append(aligned_section_df)
        section_textgrid_summaries[section_index] = textgrid_summary
    word_df = pd.concat(aligned_sections).sort_values("word_row_index").reset_index(drop=True)

    section_sentence_counts: dict[int, int] = {}
    rows: list[dict[str, object]] = []
    pointer = 0
    global_sentence_index = 0
    for tree_sentence in tree_sentences:
        while pointer < len(word_df) and not str(word_df.iloc[pointer]["alignment_token"]).strip():
            pointer += 1
        if pointer >= len(word_df):
            break

        current_section = int(word_df.iloc[pointer]["section"])
        section_end = pointer
        while section_end < len(word_df) and int(word_df.iloc[section_end]["section"]) == current_section:
            section_end += 1

        row_tokens = word_df.iloc[pointer:section_end]["normalized_alignment_token"].tolist()
        consumed_rows, alignment_cost = _align_sentence_to_rows(
            tree_sentence,
            row_tokens=row_tokens,
            language=language,
        )
        sentence_rows = word_df.iloc[pointer : pointer + consumed_rows].copy()
        if sentence_rows.empty:
            raise ValueError(f"Empty sentence slice for {language} section {current_section}.")

        timed_rows = sentence_rows.loc[
            sentence_rows["textgrid_onset_sec"].notna() & sentence_rows["textgrid_offset_sec"].notna()
        ]
        if timed_rows.empty:
            raise ValueError(
                f"No TextGrid-aligned timings for {language} section {current_section} "
                f"sentence {global_sentence_index + 1}."
            )

        section_sentence_counts[current_section] = section_sentence_counts.get(current_section, 0) + 1
        global_sentence_index += 1
        onset_sec = float(timed_rows["textgrid_onset_sec"].astype(float).min())
        offset_sec = float(timed_rows["textgrid_offset_sec"].astype(float).max())
        rows.append(
            {
                "language": language,
                "section_index": current_section,
                "language_sentence_index": global_sentence_index,
                "section_sentence_index": section_sentence_counts[current_section],
                "text": _render_sentence_text(sentence_rows, language=language),
                "onset_sec": onset_sec,
                "offset_sec": offset_sec,
                "duration_sec": offset_sec - onset_sec,
                "n_words": int((sentence_rows["surface_token"].astype(str).str.strip() != "").sum()),
                "first_word_idx": int(sentence_rows["word_row_index"].iloc[0]),
                "last_word_idx": int(sentence_rows["word_row_index"].iloc[-1]),
                "alignment_cost": alignment_cost,
            }
        )
        pointer += consumed_rows

    span_df = pd.DataFrame(rows)
    output_path = project_root() / "data" / "interim" / f"sentence_spans_{language}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    span_df.to_parquet(output_path, index=False)

    report_lines = [
        f"# Sentence Span Report ({language})",
        "",
        f"- output: `{output_path.relative_to(project_root()).as_posix()}`",
        f"- sentences: `{len(span_df)}`",
        f"- sections: `{span_df['section_index'].nunique()}`",
        f"- max alignment cost: `{span_df['alignment_cost'].max() if not span_df.empty else 0}`",
        "",
        "## Section Summary",
        "",
    ]
    for section_index in sorted(section_sentence_counts):
        textgrid_summary = section_textgrid_summaries[section_index]
        report_lines.extend(
            [
                f"### section {section_index}",
                f"- sentence count: `{section_sentence_counts[section_index]}`",
                f"- textgrid status: `{textgrid_summary.status}`",
                f"- word coverage: `{textgrid_summary.row_coverage:.3f}`",
                f"- interval coverage: `{textgrid_summary.interval_coverage:.3f}`",
                f"- unmatched word tokens: `{textgrid_summary.unmatched_rows}`",
                f"- unmatched TextGrid intervals: `{textgrid_summary.unmatched_intervals}`",
                "",
            ]
        )

    report_path = project_root() / "outputs" / "logs" / f"sentence_span_report_{language}.md"
    write_text(report_path, "\n".join(report_lines))

    append_markdown_log(
        project_root() / "outputs" / "logs" / "progress_log.md",
        f"Sentence spans ({language})",
        [
            f"Built sentence spans for {language}.",
            f"Wrote {len(span_df)} rows to {output_path.relative_to(project_root()).as_posix()}.",
            f"Wrote section report to {report_path.relative_to(project_root()).as_posix()}.",
        ],
    )

    return SentenceSpanBuildSummary(
        language=language,
        output_path=output_path,
        report_path=report_path,
        n_sentences=len(span_df),
        n_sections=int(span_df["section_index"].nunique()) if not span_df.empty else 0,
    )


def build_all_sentence_spans() -> list[SentenceSpanBuildSummary]:
    return [build_sentence_spans(language) for language in ("en", "fr", "zh")]
