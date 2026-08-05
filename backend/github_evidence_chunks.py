"""Deterministic backend-only chunks derived from raw GitHub sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, TypedDict

from backend import evidence_memory
from backend.github_raw_storage import ALLOWED_SOURCE_TYPES, safe_metadata_summary


DEFAULT_GITHUB_EVIDENCE_CHUNKS_PATH = (
    Path(__file__).resolve().parents[1] / "information" / "github_evidence_chunks.jsonl"
)
MAX_CHUNK_CHARS = 3600
MAX_CHUNKS_PER_SOURCE = 64
MAX_KEYWORDS = 32
MAX_TECHNICAL_TAGS = 16
ALLOWED_CHUNK_TYPES = frozenset(
    {"diff_hunk", "file_section", "readme_section", "log_entry", "text_window", "unknown"}
)
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,63}\b")
_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9_+.#-]{1,64}$")
_SYMBOL_PATTERNS = (
    re.compile(r"\b(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"\bfunction\s+([A-Za-z_$][A-Za-z0-9_$]*)"),
    re.compile(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s*)?\([^\n]*\)\s*=>"),
)
_TECHNICAL_TERMS = (
    ("quality gate", "quality_gate"),
    ("chroma", "chroma"),
    ("evidence", "evidence"),
    ("fallback", "fallback"),
    ("latex", "latex"),
    ("merge", "merge"),
    ("rerank", "rerank"),
    ("retrieval", "retrieval"),
    ("sqlite", "sqlite"),
    ("validation", "validation"),
    ("cache", "cache"),
    ("diff", "diff"),
)


class GithubEvidenceChunkRecord(TypedDict):
    chunk_id: str
    source_id: str
    project_id: str
    repo: str
    source_type: str
    chunk_type: str
    path: str
    commit_sha: str
    symbol: str
    text: str
    summary: str
    keywords: list[str]
    technical_tags: list[str]
    start_line: int | None
    end_line: int | None
    raw_hash: str
    text_hash: str
    text_chars: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _Segment:
    text: str
    chunk_type: str
    path: str
    start_line: int | None
    end_line: int | None


def _string(value: Any, field_name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    return value


def _optional_line(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer or None")
    return value


def _bounded_strings(values: Any, field_name: str, limit: int) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{field_name} must be a list or tuple")
    normalized = {
        item.casefold()
        for value in values
        if (item := _string(value, field_name).strip()) and _SAFE_LABEL_RE.fullmatch(item)
    }
    return sorted(normalized)[:limit]


def extract_github_evidence_keywords(text: str, path: str = "") -> list[str]:
    text_value = _string(text, "text")
    path_value = _string(path, "path")
    values = {token.casefold() for token in _IDENTIFIER_RE.findall(f"{path_value}\n{text_value}")}
    return sorted(values)[:MAX_KEYWORDS]


def extract_github_evidence_technical_tags(text: str) -> list[str]:
    haystack = _string(text, "text").casefold()
    return sorted(tag for term, tag in _TECHNICAL_TERMS if term in haystack)[:MAX_TECHNICAL_TAGS]


def extract_github_evidence_symbol(text: str) -> str:
    text_value = _string(text, "text")
    for pattern in _SYMBOL_PATTERNS:
        match = pattern.search(text_value)
        if match:
            return match.group(1)[:128]
    hunk = re.search(r"^@@[^@]*@@\s*(.+)$", text_value, flags=re.MULTILINE)
    if hunk:
        tail = hunk.group(1).strip()
        for pattern in _SYMBOL_PATTERNS:
            match = pattern.search(tail)
            if match:
                return match.group(1)[:128]
        token = _IDENTIFIER_RE.search(tail)
        if token:
            return token.group(0)[:128]
    return ""


def build_github_evidence_chunk_record(
    *,
    source_id: str = "",
    project_id: str = "",
    repo: str = "",
    source_type: str = "unknown",
    chunk_type: str = "unknown",
    path: str = "",
    commit_sha: str = "",
    symbol: str = "",
    text: str = "",
    summary: str = "",
    keywords: list[str] | tuple[str, ...] | None = None,
    technical_tags: list[str] | tuple[str, ...] | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    raw_hash: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> GithubEvidenceChunkRecord:
    text_value = _string(text, "text")
    if len(text_value) > MAX_CHUNK_CHARS:
        raise ValueError("text exceeds MAX_CHUNK_CHARS")
    source_type_value = _string(source_type, "source_type").strip().casefold()
    if source_type_value not in ALLOWED_SOURCE_TYPES:
        source_type_value = "unknown"
    chunk_type_value = _string(chunk_type, "chunk_type").strip().casefold()
    if chunk_type_value not in ALLOWED_CHUNK_TYPES:
        chunk_type_value = "unknown"
    path_value = _string(path, "path").strip()
    keyword_values = (
        extract_github_evidence_keywords(text_value, path_value)
        if keywords is None
        else _bounded_strings(keywords, "keywords", MAX_KEYWORDS)
    )
    tag_values = (
        extract_github_evidence_technical_tags(text_value)
        if technical_tags is None
        else _bounded_strings(technical_tags, "technical_tags", MAX_TECHNICAL_TAGS)
    )
    requested_symbol = _string(symbol, "symbol").strip()
    symbol_value = (
        requested_symbol
        if _SAFE_LABEL_RE.fullmatch(requested_symbol)
        else extract_github_evidence_symbol(text_value)
    )
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")
    text_hash = evidence_memory.stable_hash(text_value)
    stable_fields = [
        _string(source_id, "source_id").strip(),
        _string(project_id, "project_id").strip(),
        source_type_value,
        chunk_type_value,
        path_value,
        _string(commit_sha, "commit_sha").strip(),
        symbol_value,
        _optional_line(start_line, "start_line"),
        _optional_line(end_line, "end_line"),
        _string(raw_hash, "raw_hash").strip(),
        text_hash,
    ]
    return {
        "chunk_id": evidence_memory.stable_record_id("chk", stable_fields),
        "source_id": stable_fields[0],
        "project_id": stable_fields[1],
        "repo": _string(repo, "repo").strip(),
        "source_type": source_type_value,
        "chunk_type": chunk_type_value,
        "path": path_value,
        "commit_sha": stable_fields[5],
        "symbol": symbol_value[:128],
        "text": text_value,
        "summary": _string(summary, "summary")[:512],
        "keywords": keyword_values,
        "technical_tags": tag_values,
        "start_line": stable_fields[7],
        "end_line": stable_fields[8],
        "raw_hash": stable_fields[9],
        "text_hash": text_hash,
        "text_chars": len(text_value),
        "metadata": dict(metadata or {}),
    }


def _bounded_windows(text: str, chunk_type: str, path: str, start_line: int = 1) -> list[_Segment]:
    if not text:
        return []
    segments: list[_Segment] = []
    current: list[str] = []
    current_chars = 0
    current_start = start_line
    line_number = start_line
    for line in text.splitlines(keepends=True):
        pieces = [line[index:index + MAX_CHUNK_CHARS] for index in range(0, len(line), MAX_CHUNK_CHARS)] or [""]
        for piece in pieces:
            if current and current_chars + len(piece) > MAX_CHUNK_CHARS:
                segments.append(_Segment("".join(current), chunk_type, path, current_start, line_number - 1))
                current, current_chars, current_start = [], 0, line_number
            current.append(piece)
            current_chars += len(piece)
            if current_chars == MAX_CHUNK_CHARS:
                segments.append(_Segment("".join(current), chunk_type, path, current_start, line_number))
                current, current_chars, current_start = [], 0, line_number
        line_number += 1
    if current:
        segments.append(_Segment("".join(current), chunk_type, path, current_start, line_number - 1))
    return segments


def _commit_patch_segments(text: str, fallback_path: str) -> list[_Segment]:
    lines = text.splitlines(keepends=True)
    file_starts = [index for index, line in enumerate(lines) if line.startswith("diff --git ")]
    if not file_starts:
        return _bounded_windows(text, "diff_hunk", fallback_path)
    segments: list[_Segment] = []
    for position, file_start in enumerate(file_starts):
        file_end = file_starts[position + 1] if position + 1 < len(file_starts) else len(lines)
        header = lines[file_start]
        match = re.match(r"diff --git a/(.*?) b/(.*?)(?:\r?\n)?$", header)
        current_path = match.group(2) if match else fallback_path
        hunk_starts = [
            index for index in range(file_start + 1, file_end) if lines[index].startswith("@@")
        ]
        if not hunk_starts:
            body_start = file_start + 1
            body = "".join(lines[body_start:file_end])
            segments.extend(_bounded_windows(body, "diff_hunk", current_path, body_start + 1))
            continue
        for hunk_position, hunk_start in enumerate(hunk_starts):
            hunk_end = hunk_starts[hunk_position + 1] if hunk_position + 1 < len(hunk_starts) else file_end
            body = "".join(lines[hunk_start:hunk_end])
            segments.extend(_bounded_windows(body, "diff_hunk", current_path, hunk_start + 1))
    return segments


def _section_segments(text: str, pattern: re.Pattern[str], chunk_type: str, path: str) -> list[_Segment]:
    lines = text.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if pattern.match(line)]
    if not starts:
        return _bounded_windows(text, "text_window", path)
    if starts[0] > 0 and "".join(lines[:starts[0]]).strip():
        starts.insert(0, 0)
    segments: list[_Segment] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        segments.extend(_bounded_windows("".join(lines[start:end]), chunk_type, path, start + 1))
    return segments


def build_github_evidence_chunks_from_raw_source(
    raw_record: Mapping[str, Any],
) -> list[GithubEvidenceChunkRecord]:
    if not isinstance(raw_record, Mapping):
        raise TypeError("raw_record must be a mapping")
    text = _string(raw_record.get("raw_text"), "raw_text")
    if not text:
        return []
    source_type = _string(raw_record.get("source_type", "unknown"), "source_type").strip().casefold()
    path = _string(raw_record.get("path"), "path").strip()
    if source_type == "commit_patch":
        segments = _commit_patch_segments(text, path)
    elif source_type == "readme":
        segments = _section_segments(text, re.compile(r"^#{1,6}\s+"), "readme_section", path)
    elif source_type == "file_snapshot":
        structure = re.compile(
            r"^\s*(?:(?:async\s+)?def\s+|class\s+|(?:export\s+)?function\s+|[A-Za-z_$][\w$]*\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)"
        )
        segments = _section_segments(text, structure, "file_section", path)
    elif source_type == "log":
        segments = _section_segments(text, re.compile(r"^\S"), "log_entry", path)
    else:
        segments = _bounded_windows(text, "text_window", path)
    common = {
        "source_id": raw_record.get("source_id"),
        "project_id": raw_record.get("project_id"),
        "repo": raw_record.get("repo"),
        "source_type": source_type,
        "commit_sha": raw_record.get("commit_sha"),
        "raw_hash": raw_record.get("raw_hash"),
        "metadata": raw_record.get("metadata"),
    }
    return [
        build_github_evidence_chunk_record(
            **common,
            chunk_type=segment.chunk_type,
            path=segment.path,
            text=segment.text,
            start_line=segment.start_line,
            end_line=segment.end_line,
        )
        for segment in segments[:MAX_CHUNKS_PER_SOURCE]
        if segment.text
    ]


def _normalize_chunk(record: Mapping[str, Any]) -> GithubEvidenceChunkRecord:
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    return build_github_evidence_chunk_record(
        source_id=record.get("source_id"),
        project_id=record.get("project_id"),
        repo=record.get("repo"),
        source_type=record.get("source_type", "unknown"),
        chunk_type=record.get("chunk_type", "unknown"),
        path=record.get("path"),
        commit_sha=record.get("commit_sha"),
        symbol=record.get("symbol"),
        text=record.get("text"),
        summary=record.get("summary"),
        keywords=record.get("keywords"),
        technical_tags=record.get("technical_tags"),
        start_line=record.get("start_line"),
        end_line=record.get("end_line"),
        raw_hash=record.get("raw_hash"),
        metadata=record.get("metadata"),
    )


def append_github_evidence_chunk_record(
    record: Mapping[str, Any],
    artifact_path: str | Path = DEFAULT_GITHUB_EVIDENCE_CHUNKS_PATH,
) -> GithubEvidenceChunkRecord:
    normalized = _normalize_chunk(record)
    evidence_memory.upsert_jsonl_record(artifact_path, normalized, "chunk_id")
    return normalized


def load_github_evidence_chunk_records(
    artifact_path: str | Path = DEFAULT_GITHUB_EVIDENCE_CHUNKS_PATH,
) -> list[GithubEvidenceChunkRecord]:
    return [_normalize_chunk(record) for record in evidence_memory.read_jsonl(artifact_path)]


def redact_github_evidence_chunk_record(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_chunk(record)
    return {
        "chunk_id": normalized["chunk_id"],
        "source_id": normalized["source_id"],
        "project_id": normalized["project_id"],
        "repo": normalized["repo"],
        "source_type": normalized["source_type"],
        "chunk_type": normalized["chunk_type"],
        "path": normalized["path"],
        "commit_sha": normalized["commit_sha"],
        "symbol": normalized["symbol"],
        "raw_hash": normalized["raw_hash"],
        "text_hash": normalized["text_hash"],
        "text_chars": normalized["text_chars"],
        "text_available": bool(normalized["text"]),
        "keywords": normalized["keywords"],
        "technical_tags": normalized["technical_tags"],
        "metadata": safe_metadata_summary(normalized["metadata"]),
    }
