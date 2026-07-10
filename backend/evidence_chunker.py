"""Deterministic Phase 2 evidence chunking helpers."""

from __future__ import annotations

import os
import re
from typing import Any

import evidence_memory


PHASE2_FLAG_ENV = "USE_GITHUB_CONTEXT_PHASE2"
ENABLED_VALUES = {"1", "true", "yes", "on"}
PHASE2_CHUNK_TARGET_CHARS = 3000
PHASE2_CHUNK_MAX_CHARS = 6000
PHASE2_CHUNK_MIN_CHARS = 80
PHASE2_CHUNK_SUMMARY_CHARS = 240

KEYWORD_PATTERNS = [
    ("bullet_depth_profile", r"\bbullet_depth_profile\b"),
    ("final_bullets", r"\bfinal_bullets\b"),
    ("quality_gate", r"\bquality[_ -]?gate\b"),
    ("project_memory", r"\bproject[_ -]?memory\b"),
    ("template pollution", r"\btemplate\s+pollution\b"),
    ("retrieval", r"\bretrieval\b"),
    ("rerank", r"\brerank(?:ing)?\b"),
    ("Chroma", r"\bchroma\b"),
    ("SQLite", r"\bsqlite\b"),
    ("LaTeX", r"(^|[^A-Za-z0-9])latex([^A-Za-z0-9]|$)"),
    ("validation", r"\bvalidat(?:e|es|ed|ion|ing)\b"),
    ("merge", r"\bmerge\b"),
    ("GitHub", r"\bgithub\b"),
    ("commit", r"\bcommit\b"),
    ("diff", r"\bdiff\b"),
    ("cache", r"\bcach(?:e|ed|ing)\b"),
    ("fallback", r"\bfallback\b"),
    ("coverage", r"\bcoverage\b"),
    ("evidence", r"\bevidence\b"),
    ("FastAPI", r"\bfastapi\b"),
    ("React", r"\breact\b"),
    ("API", r"\bapi\b"),
    ("endpoint", r"\bendpoint\b"),
    ("configuration", r"\bconfiguration\b|\bconfig\b"),
    ("testing", r"\btest(?:s|ing)?\b"),
]

TECHNICAL_TAG_RULES = {
    "retrieval": {"retrieval", "rerank"},
    "storage": {"Chroma", "SQLite", "cache"},
    "validation": {"validation", "quality_gate"},
    "latex": {"LaTeX"},
    "resume_generation": {"bullet_depth_profile", "final_bullets"},
    "quality_gate": {"quality_gate"},
    "diff_analysis": {"diff", "commit"},
    "project_memory": {"project_memory"},
    "frontend": {"React"},
    "backend": {"FastAPI", "API", "endpoint"},
    "testing": {"testing"},
    "configuration": {"configuration"},
}


def phase2_enabled() -> bool:
    return str(os.getenv(PHASE2_FLAG_ENV, "1")).strip().lower() in ENABLED_VALUES


def build_evidence_chunks_from_raw_sources(raw_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for raw_source in raw_sources:
        chunks.extend(build_evidence_chunks_from_raw_source(raw_source))
    return chunks


def build_evidence_chunks_from_raw_source(raw_source: dict[str, Any]) -> list[dict[str, Any]]:
    text = normalize_text(raw_source.get("raw_text"))
    if not text:
        return []

    source_type = str(raw_source.get("source_type") or "unknown")
    if source_type == "commit_patch" or looks_like_diff(text):
        candidates = diff_chunks(raw_source, text)
    elif source_type == "readme" or looks_like_markdown(text):
        candidates = boundary_chunks(raw_source, text, "readme_section", markdown_sections(text))
    elif source_type == "log":
        candidates = boundary_chunks(raw_source, text, "log_entry", log_entries(text))
    else:
        candidates = unknown_chunks(raw_source, text)

    chunks = []
    for index, candidate in enumerate(candidates):
        chunk = make_chunk(raw_source, candidate, index)
        if chunk is not None:
            chunks.append(chunk)
    if chunks:
        return chunks

    fallback_text = text[:PHASE2_CHUNK_TARGET_CHARS].strip()
    if fallback_text:
        fallback = make_chunk(
            raw_source,
            {
                "text": fallback_text,
                "chunk_type": chunk_type_for_source_type(source_type),
                "path": raw_source.get("path") or "",
                "start_line": 1,
                "end_line": count_lines(fallback_text),
            },
            0,
            force_keep=True,
        )
        return [fallback] if fallback is not None else []
    return []


def chunk_phase2_raw_sources(
    project_id: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    if not phase2_enabled():
        return {
            "enabled": False,
            "phase": "phase2",
            "project_id": project_id or None,
            "processed_raw_sources": 0,
            "created_chunks": 0,
            "updated_chunks": 0,
            "created_or_updated_chunks": 0,
            "chunks_count": 0,
            "message": "GitHub context evidence memory is disabled.",
            "errors": [],
        }

    raw_sources = evidence_memory.read_github_raw_sources(project_id=project_id)
    if limit is not None:
        safe_limit = max(0, int(limit))
        raw_sources = raw_sources[:safe_limit]

    existing_chunk_ids = {
        str(record.get("chunk_id") or "")
        for record in evidence_memory.read_records(evidence_memory.EVIDENCE_CHUNKS)
    }
    created_chunks = 0
    updated_chunks = 0
    created_or_updated_chunks = 0
    source_summaries: list[dict[str, Any]] = []

    for raw_source in raw_sources:
        chunks = build_evidence_chunks_from_raw_source(raw_source)
        source_created_or_updated = 0
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            evidence_memory.upsert_evidence_chunk(chunk)
            if chunk_id in existing_chunk_ids:
                updated_chunks += 1
            else:
                created_chunks += 1
                existing_chunk_ids.add(chunk_id)
            created_or_updated_chunks += 1
            source_created_or_updated += 1
        source_summaries.append(
            {
                "source_id": str(raw_source.get("source_id") or ""),
                "project_id": str(raw_source.get("project_id") or ""),
                "repo": str(raw_source.get("repo") or ""),
                "created_or_updated_chunks": source_created_or_updated,
            }
        )

    counts = evidence_memory.get_phase2_memory_counts(project_id=project_id)
    return {
        "enabled": True,
        "phase": "phase2",
        "project_id": project_id or None,
        "processed_raw_sources": len(raw_sources),
        "created_chunks": created_chunks,
        "updated_chunks": updated_chunks,
        "created_or_updated_chunks": created_or_updated_chunks,
        "chunks_count": counts["chunks_count"],
        "message": "Phase 2 evidence chunks built successfully.",
        "sources": source_summaries,
    }


def diff_chunks(raw_source: dict[str, Any], text: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    file_blocks = split_file_diff_blocks(text)
    if file_blocks:
        for block in file_blocks:
            path = parse_diff_path(block) or str(raw_source.get("path") or "")
            chunks.extend(hunk_chunks_from_block(block, path))
    else:
        chunks.extend(hunk_chunks_from_block(text, str(raw_source.get("path") or "")))
    if chunks:
        return chunks
    return [
        {
            "text": part,
            "chunk_type": "diff_hunk",
            "path": raw_source.get("path") or "",
            "start_line": None,
            "end_line": None,
        }
        for part in bounded_windows(text)
    ]


def hunk_chunks_from_block(block: str, path: str) -> list[dict[str, Any]]:
    hunk_positions = [match.start() for match in re.finditer(r"(?m)^@@\s", block)]
    if not hunk_positions:
        return [
            {
                "text": part,
                "chunk_type": "diff_hunk",
                "path": path,
                "start_line": None,
                "end_line": None,
            }
            for part in bounded_windows(block)
        ]

    preamble = block[: hunk_positions[0]]
    boundaries = hunk_positions + [len(block)]
    chunks: list[dict[str, Any]] = []
    for index, start in enumerate(hunk_positions):
        end = boundaries[index + 1]
        hunk = block[start:end]
        text = (preamble + hunk).strip()
        header = first_line(hunk)
        start_line, end_line = parse_hunk_lines(header)
        for window in bounded_windows(text):
            chunks.append(
                {
                    "text": window,
                    "chunk_type": "diff_hunk",
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "hunk_header": header,
                }
            )
    return chunks


def unknown_chunks(raw_source: dict[str, Any], text: str) -> list[dict[str, Any]]:
    if looks_like_diff(text):
        return diff_chunks(raw_source, text)
    if looks_like_markdown(text):
        return boundary_chunks(raw_source, text, "readme_section", markdown_sections(text))
    segments = visible_boundary_segments(text)
    return boundary_chunks(raw_source, text, "unknown", segments)


def boundary_chunks(
    raw_source: dict[str, Any],
    text: str,
    chunk_type: str,
    segments: list[str],
) -> list[dict[str, Any]]:
    if not segments:
        segments = bounded_windows(text)
    chunks: list[dict[str, Any]] = []
    path = str(raw_source.get("path") or "")
    for segment in pack_segments(segments):
        chunks.append(
            {
                "text": segment,
                "chunk_type": chunk_type,
                "path": path,
                "start_line": None,
                "end_line": None,
            }
        )
    return chunks


def make_chunk(
    raw_source: dict[str, Any],
    candidate: dict[str, Any],
    chunk_index: int,
    force_keep: bool = False,
) -> dict[str, Any] | None:
    text = str(candidate.get("text") or "").strip()
    if not text:
        return None
    keywords = extract_keywords(text)
    if not force_keep and not should_keep_chunk(text, keywords):
        return None

    project_id = str(raw_source.get("project_id") or "")
    repo = str(raw_source.get("repo") or "")
    source_id = str(raw_source.get("source_id") or "")
    chunk_type = str(candidate.get("chunk_type") or chunk_type_for_source_type(str(raw_source.get("source_type") or "")))
    path = str(candidate.get("path") or raw_source.get("path") or "")
    symbol = extract_symbol(text)
    text_hash = evidence_memory.stable_hash(text)
    chunk_id = evidence_memory.stable_record_id(
        "evidence_chunk",
        [source_id, project_id, repo, path, chunk_type, chunk_index, text_hash],
    )
    return evidence_memory.make_evidence_chunk(
        chunk_id=chunk_id,
        source_id=source_id,
        project_id=project_id,
        repo=repo,
        path=path,
        symbol=symbol,
        chunk_type=chunk_type,
        text=text,
        summary=chunk_summary(text, chunk_type, path, symbol, str(candidate.get("hunk_header") or "")),
        keywords=keywords,
        technical_tags=technical_tags_for_keywords(keywords),
        start_line=candidate.get("start_line"),
        end_line=candidate.get("end_line"),
        metadata={
            "source_type": str(raw_source.get("source_type") or ""),
            "raw_hash": str(raw_source.get("raw_hash") or ""),
            "commit_sha": str(raw_source.get("commit_sha") or ""),
            "chunk_index": chunk_index,
            "hunk_header": str(candidate.get("hunk_header") or ""),
        },
    )


def normalize_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def looks_like_diff(text: str) -> bool:
    return bool(re.search(r"(?m)^(diff --git |@@\s)", text))


def looks_like_markdown(text: str) -> bool:
    return bool(re.search(r"(?m)^#{1,3}\s+\S", text))


def split_file_diff_blocks(text: str) -> list[str]:
    starts = [match.start() for match in re.finditer(r"(?m)^diff --git\s", text)]
    if not starts:
        return []
    boundaries = starts + [len(text)]
    return [text[boundaries[index] : boundaries[index + 1]].strip() for index in range(len(starts))]


def parse_diff_path(block: str) -> str:
    match = re.search(r"(?m)^diff --git\s+a/(.*?)\s+b/(.*?)\s*$", block)
    if match:
        return match.group(2).strip()
    match = re.search(r"(?m)^\+\+\+\s+b/(.*?)\s*$", block)
    return match.group(1).strip() if match else ""


def parse_hunk_lines(header: str) -> tuple[int | None, int | None]:
    match = re.search(r"\+(\d+)(?:,(\d+))?", header)
    if not match:
        return None, None
    start = int(match.group(1))
    length = int(match.group(2) or "1")
    return start, start + max(0, length - 1)


def markdown_sections(text: str) -> list[str]:
    starts = [match.start() for match in re.finditer(r"(?m)^#{1,3}\s+\S", text)]
    if not starts:
        return []
    if starts[0] != 0 and text[: starts[0]].strip():
        starts = [0] + starts
    boundaries = starts + [len(text)]
    return [text[boundaries[index] : boundaries[index + 1]].strip() for index in range(len(starts)) if text[boundaries[index] : boundaries[index + 1]].strip()]


def log_entries(text: str) -> list[str]:
    return [entry.strip() for entry in re.split(r"\n\s*\n", text) if entry.strip()]


def visible_boundary_segments(text: str) -> list[str]:
    markers = r"(?=\n#{1,3}\s|\ndiff --git\s|\n@@\s|\nFile:|\nPath:|\n\s*\n)"
    parts = [part.strip() for part in re.split(markers, text) if part.strip()]
    return parts if len(parts) > 1 else bounded_windows(text)


def pack_segments(segments: list[str]) -> list[str]:
    packed: list[str] = []
    current = ""
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        if len(segment) > PHASE2_CHUNK_MAX_CHARS:
            if current:
                packed.append(current.strip())
                current = ""
            packed.extend(bounded_windows(segment))
            continue
        candidate = f"{current}\n\n{segment}".strip() if current else segment
        if len(candidate) > PHASE2_CHUNK_TARGET_CHARS and current:
            packed.append(current.strip())
            current = segment
        else:
            current = candidate
    if current:
        packed.append(current.strip())
    return [
        window
        for chunk in packed
        for window in (bounded_windows(chunk) if len(chunk) > PHASE2_CHUNK_MAX_CHARS else [chunk])
    ]


def bounded_windows(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= PHASE2_CHUNK_MAX_CHARS:
        return [text]
    windows = []
    start = 0
    while start < len(text):
        end = min(len(text), start + PHASE2_CHUNK_TARGET_CHARS)
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end))
            if boundary > start + PHASE2_CHUNK_MIN_CHARS:
                end = boundary
        window = text[start:end].strip()
        if window:
            windows.append(window)
        start = end
    return windows


def should_keep_chunk(text: str, keywords: list[str]) -> bool:
    return len(text) >= PHASE2_CHUNK_MIN_CHARS or bool(keywords)


def chunk_type_for_source_type(source_type: str) -> str:
    if source_type == "commit_patch":
        return "diff_hunk"
    if source_type == "readme":
        return "readme_section"
    if source_type == "log":
        return "log_entry"
    return "unknown"


def extract_keywords(text: str) -> list[str]:
    found: list[str] = []
    for keyword, pattern in KEYWORD_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(keyword)
    for identifier in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b", text):
        if "_" not in identifier:
            continue
        if identifier not in found:
            found.append(identifier)
        if len(found) >= 16:
            break
    return found[:16]


def technical_tags_for_keywords(keywords: list[str]) -> list[str]:
    keyword_set = set(keywords)
    tags = [
        tag
        for tag, triggers in TECHNICAL_TAG_RULES.items()
        if keyword_set.intersection(triggers)
    ]
    return tags[:12]


def extract_symbol(text: str) -> str:
    patterns = [
        r"\basync\s+def\s+([A-Za-z_][\w]*)\s*\(",
        r"\bdef\s+([A-Za-z_][\w]*)\s*\(",
        r"\bclass\s+([A-Za-z_][\w]*)\b",
        r"\bexport\s+function\s+([A-Za-z_][\w]*)\s*\(",
        r"\bfunction\s+([A-Za-z_][\w]*)\s*\(",
        r"\bconst\s+([A-Za-z_][\w]*)\s*=",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def chunk_summary(text: str, chunk_type: str, path: str, symbol: str, hunk_header: str = "") -> str:
    if chunk_type == "diff_hunk":
        target = symbol or first_meaningful_line(text)
        parts = [f"Diff hunk from {path or 'unknown path'}"]
        if hunk_header:
            parts.append(hunk_header)
        if target:
            parts.append(f"touching {target}")
        return truncate_summary(" ".join(parts))
    if chunk_type == "readme_section":
        heading = first_markdown_heading(text)
        return truncate_summary(heading or first_meaningful_line(text))
    if chunk_type == "log_entry":
        return truncate_summary(first_meaningful_line(text))
    return truncate_summary(first_meaningful_line(text))


def first_markdown_heading(text: str) -> str:
    match = re.search(r"(?m)^#{1,3}\s+(.+)$", text)
    return match.group(1).strip() if match else ""


def first_meaningful_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def first_line(text: str) -> str:
    return text.splitlines()[0].strip() if text.splitlines() else ""


def truncate_summary(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= PHASE2_CHUNK_SUMMARY_CHARS:
        return text
    return text[: PHASE2_CHUNK_SUMMARY_CHARS - 3].rstrip() + "..."


def count_lines(text: str) -> int:
    return len(text.splitlines()) or 1
