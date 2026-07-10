"""Phase 2 evidence memory schema and JSONL storage helpers."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, TypedDict


PHASE2_EVIDENCE_MEMORY_DIR_ENV = "PHASE2_EVIDENCE_MEMORY_DIR"
DEFAULT_STORAGE_ROOT = Path(__file__).resolve().parents[1] / "information" / "phase2_evidence_memory"

GITHUB_RAW_SOURCES = "github_raw_sources"
EVIDENCE_CHUNKS = "evidence_chunks"
RAW_CHANGE_SUMMARIES = "raw_change_summaries"
EVIDENCE_CARDS = "evidence_cards"
CAPABILITY_FACTS = "capability_facts"

RECORD_FILES = {
    GITHUB_RAW_SOURCES: "github_raw_sources.jsonl",
    EVIDENCE_CHUNKS: "evidence_chunks.jsonl",
    RAW_CHANGE_SUMMARIES: "raw_change_summaries.jsonl",
    EVIDENCE_CARDS: "evidence_cards.jsonl",
    CAPABILITY_FACTS: "capability_facts.jsonl",
}

RECORD_TYPE_ALIASES = {
    "raw_sources": GITHUB_RAW_SOURCES,
    "chunks": EVIDENCE_CHUNKS,
    "change_summaries": RAW_CHANGE_SUMMARIES,
    "cards": EVIDENCE_CARDS,
    "capabilities": CAPABILITY_FACTS,
}

COUNT_KEYS = {
    GITHUB_RAW_SOURCES: "raw_sources_count",
    EVIDENCE_CHUNKS: "chunks_count",
    RAW_CHANGE_SUMMARIES: "raw_change_summaries_count",
    EVIDENCE_CARDS: "evidence_cards_count",
    CAPABILITY_FACTS: "capability_facts_count",
}

SourceType = Literal["commit_patch", "file_snapshot", "readme", "issue", "pr", "log", "unknown"]
ChunkType = Literal[
    "function",
    "class",
    "endpoint",
    "diff_hunk",
    "config",
    "log_entry",
    "readme_section",
    "unknown",
]
Confidence = Literal["low", "medium", "high"]
MetricSupport = Literal["none", "approximate", "explicit"]

ALLOWED_SOURCE_TYPES = {"commit_patch", "file_snapshot", "readme", "issue", "pr", "log", "unknown"}
ALLOWED_CHUNK_TYPES = {
    "function",
    "class",
    "endpoint",
    "diff_hunk",
    "config",
    "log_entry",
    "readme_section",
    "unknown",
}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}
ALLOWED_METRIC_SUPPORT = {"none", "approximate", "explicit"}


class GithubRawSource(TypedDict):
    source_id: str
    project_id: str
    repo: str
    source_type: SourceType
    path: str
    commit_sha: str
    raw_text: str
    raw_hash: str
    created_at: str
    metadata: dict[str, Any]


class EvidenceChunk(TypedDict):
    chunk_id: str
    source_id: str
    project_id: str
    repo: str
    path: str
    symbol: str
    chunk_type: ChunkType
    text: str
    summary: str
    keywords: list[Any]
    technical_tags: list[Any]
    start_line: int | None
    end_line: int | None
    hash: str
    metadata: dict[str, Any]


class RawChangeSummary(TypedDict):
    change_id: str
    project_id: str
    source_chunk_ids: list[Any]
    files_changed: list[Any]
    symbols_changed: list[Any]
    raw_change_type: list[Any]
    what_changed: str
    direct_code_evidence: list[Any]
    uncertain_intent: list[Any]
    metadata: dict[str, Any]


class EvidenceCard(TypedDict):
    evidence_id: str
    project_id: str
    source_chunk_ids: list[Any]
    problem: str
    mechanism: str
    implementation_details: list[Any]
    safe_impact: str
    resume_angle: str
    confidence: Confidence
    metric_support: MetricSupport
    allowed_claims: list[Any]
    forbidden_claims: list[Any]
    metadata: dict[str, Any]


class CapabilityFact(TypedDict):
    capability_id: str
    project_id: str
    capability_type: str
    present: bool
    confidence: Confidence
    mechanisms: list[Any]
    source_evidence_ids: list[Any]
    allowed_resume_claims: list[Any]
    forbidden_claims: list[Any]
    metric_support: MetricSupport
    metadata: dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def stable_record_id(prefix: str, parts: Any) -> str:
    return f"{prefix}_{stable_hash(_canonical_json(parts))[:32]}"


def get_phase2_memory_dir(storage_dir: str | Path | None = None) -> Path:
    if storage_dir is not None:
        return Path(storage_dir)
    configured = os.getenv(PHASE2_EVIDENCE_MEMORY_DIR_ENV, "").strip()
    if configured:
        return Path(configured)
    return DEFAULT_STORAGE_ROOT


def get_record_path(record_type: str, storage_dir: str | Path | None = None) -> Path:
    normalized_type = normalize_record_type(record_type)
    return get_phase2_memory_dir(storage_dir) / RECORD_FILES[normalized_type]


def normalize_record_type(record_type: str) -> str:
    normalized_type = RECORD_TYPE_ALIASES.get(record_type, record_type)
    if normalized_type not in RECORD_FILES:
        raise ValueError(f"Unknown Phase 2 evidence memory record type: {record_type}")
    return normalized_type


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
            records.append(payload)
    return records


def write_jsonl(path: str | Path, records: list[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def upsert_jsonl_record(path: str | Path, record: Mapping[str, Any], id_field: str) -> dict[str, Any]:
    record_dict = dict(record)
    record_id = str(record_dict.get(id_field) or "").strip()
    if not record_id:
        raise ValueError(f"Cannot upsert Phase 2 record without {id_field}")

    records = read_jsonl(path)
    updated = False
    next_records: list[dict[str, Any]] = []
    for existing in records:
        if str(existing.get(id_field) or "") == record_id:
            next_records.append(record_dict)
            updated = True
        else:
            next_records.append(existing)
    if not updated:
        next_records.append(record_dict)
    write_jsonl(path, next_records)
    return record_dict


def read_records(record_type: str, storage_dir: str | Path | None = None) -> list[dict[str, Any]]:
    return read_jsonl(get_record_path(record_type, storage_dir))


def write_records(
    record_type: str,
    records: list[Mapping[str, Any]],
    storage_dir: str | Path | None = None,
) -> None:
    write_jsonl(get_record_path(record_type, storage_dir), records)


def read_records_by_project(
    record_type: str,
    project_id: str,
    storage_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    requested_project_id = str(project_id)
    return [
        record
        for record in read_records(record_type, storage_dir)
        if str(record.get("project_id") or "") == requested_project_id
    ]


def count_records(
    record_type: str,
    project_id: str | None = None,
    storage_dir: str | Path | None = None,
) -> int:
    if project_id is None:
        return len(read_records(record_type, storage_dir))
    return len(read_records_by_project(record_type, project_id, storage_dir))


def get_phase2_memory_counts(
    project_id: str | None = None,
    storage_dir: str | Path | None = None,
) -> dict[str, int]:
    return {
        count_key: count_records(record_type, project_id=project_id, storage_dir=storage_dir)
        for record_type, count_key in COUNT_KEYS.items()
    }


def read_github_raw_sources(
    project_id: str | None = None,
    storage_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    if project_id:
        return read_records_by_project(GITHUB_RAW_SOURCES, project_id, storage_dir)
    return read_records(GITHUB_RAW_SOURCES, storage_dir)


def github_raw_source_stats(
    project_id: str | None = None,
    storage_dir: str | Path | None = None,
) -> dict[str, Any]:
    records = read_github_raw_sources(project_id=project_id, storage_dir=storage_dir)
    chunk_records = read_records_by_project(EVIDENCE_CHUNKS, project_id, storage_dir) if project_id else read_records(EVIDENCE_CHUNKS, storage_dir)
    summary_records = (
        read_records_by_project(RAW_CHANGE_SUMMARIES, project_id, storage_dir)
        if project_id
        else read_records(RAW_CHANGE_SUMMARIES, storage_dir)
    )
    evidence_card_records = (
        read_records_by_project(EVIDENCE_CARDS, project_id, storage_dir)
        if project_id
        else read_records(EVIDENCE_CARDS, storage_dir)
    )
    capability_fact_records = (
        read_records_by_project(CAPABILITY_FACTS, project_id, storage_dir)
        if project_id
        else read_records(CAPABILITY_FACTS, storage_dir)
    )
    projects: dict[str, dict[str, Any]] = {}
    repos = set()
    raw_chars = 0
    chunk_repos = {str(record.get("chunk_id") or ""): _string(record.get("repo")) for record in chunk_records}

    for record in records:
        current_project_id = _string(record.get("project_id"))
        repo = _string(record.get("repo"))
        text = _string(record.get("raw_text"))
        current_chars = len(text)
        raw_chars += current_chars
        if repo:
            repos.add(repo)
        key = f"{current_project_id}\n{repo}"
        current = projects.setdefault(
            key,
            {
                "project_id": current_project_id,
                "repo": repo,
                "raw_sources": 0,
                "raw_chars": 0,
                "chunks": 0,
                "raw_change_summaries": 0,
                "evidence_cards": 0,
                "capability_facts": 0,
            },
        )
        current["raw_sources"] += 1
        current["raw_chars"] += current_chars

    for record in chunk_records:
        current_project_id = _string(record.get("project_id"))
        repo = _string(record.get("repo"))
        if repo:
            repos.add(repo)
        key = f"{current_project_id}\n{repo}"
        current = projects.setdefault(
            key,
            {
                "project_id": current_project_id,
                "repo": repo,
                "raw_sources": 0,
                "raw_chars": 0,
                "chunks": 0,
                "raw_change_summaries": 0,
                "evidence_cards": 0,
                "capability_facts": 0,
            },
        )
        current["chunks"] += 1

    for record in summary_records:
        current_project_id = _string(record.get("project_id"))
        source_chunk_ids = record.get("source_chunk_ids") if isinstance(record.get("source_chunk_ids"), list) else []
        repo = ""
        for chunk_id in source_chunk_ids:
            repo = chunk_repos.get(str(chunk_id), "")
            if repo:
                break
        if repo:
            repos.add(repo)
        key = f"{current_project_id}\n{repo}"
        current = projects.setdefault(
            key,
            {
                "project_id": current_project_id,
                "repo": repo,
                "raw_sources": 0,
                "raw_chars": 0,
                "chunks": 0,
                "raw_change_summaries": 0,
                "evidence_cards": 0,
                "capability_facts": 0,
            },
        )
        current["raw_change_summaries"] += 1

    for record in evidence_card_records:
        current_project_id = _string(record.get("project_id"))
        source_chunk_ids = record.get("source_chunk_ids") if isinstance(record.get("source_chunk_ids"), list) else []
        repo = ""
        for chunk_id in source_chunk_ids:
            repo = chunk_repos.get(str(chunk_id), "")
            if repo:
                break
        if repo:
            repos.add(repo)
        key = f"{current_project_id}\n{repo}"
        current = projects.setdefault(
            key,
            {
                "project_id": current_project_id,
                "repo": repo,
                "raw_sources": 0,
                "raw_chars": 0,
                "chunks": 0,
                "raw_change_summaries": 0,
                "evidence_cards": 0,
                "capability_facts": 0,
            },
        )
        current["evidence_cards"] += 1

    for record in capability_fact_records:
        current_project_id = _string(record.get("project_id"))
        key = next(
            (
                existing_key
                for existing_key, existing in projects.items()
                if _string(existing.get("project_id")) == current_project_id
            ),
            f"{current_project_id}\n",
        )
        current = projects.setdefault(
            key,
            {
                "project_id": current_project_id,
                "repo": "",
                "raw_sources": 0,
                "raw_chars": 0,
                "chunks": 0,
                "raw_change_summaries": 0,
                "evidence_cards": 0,
                "capability_facts": 0,
            },
        )
        current["capability_facts"] += 1

    return {
        "raw_chars": raw_chars,
        "repos_count": len(repos),
        "projects": sorted(
            projects.values(),
            key=lambda item: (str(item.get("project_id") or "").lower(), str(item.get("repo") or "").lower()),
        ),
    }


def make_github_raw_source(
    *,
    project_id: str,
    repo: str,
    source_type: str = "unknown",
    path: str = "",
    commit_sha: str = "",
    raw_text: str = "",
    source_id: str = "",
    created_at: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> GithubRawSource:
    raw_text_value = _string(raw_text)
    raw_hash = stable_hash(raw_text_value)
    source_type_value = _enum(source_type, ALLOWED_SOURCE_TYPES, "unknown")
    source_id_value = _string(source_id) or stable_record_id(
        "github_raw_source",
        [_string(project_id), _string(repo), source_type_value, _string(path), _string(commit_sha), raw_hash],
    )
    return {
        "source_id": source_id_value,
        "project_id": _string(project_id),
        "repo": _string(repo),
        "source_type": source_type_value,
        "path": _string(path),
        "commit_sha": _string(commit_sha),
        "raw_text": raw_text_value,
        "raw_hash": raw_hash,
        "created_at": _string(created_at) or utc_now_iso(),
        "metadata": _dict(metadata),
    }


def normalize_github_raw_source(record: Mapping[str, Any]) -> GithubRawSource:
    return make_github_raw_source(
        source_id=_string(record.get("source_id")),
        project_id=_string(record.get("project_id")),
        repo=_string(record.get("repo")),
        source_type=_string(record.get("source_type") or "unknown"),
        path=_string(record.get("path")),
        commit_sha=_string(record.get("commit_sha")),
        raw_text=_string(record.get("raw_text")),
        created_at=_string(record.get("created_at")),
        metadata=_dict(record.get("metadata")),
    )


def upsert_github_raw_source(
    record: Mapping[str, Any],
    storage_dir: str | Path | None = None,
) -> GithubRawSource:
    normalized = normalize_github_raw_source(record)
    return upsert_jsonl_record(get_record_path(GITHUB_RAW_SOURCES, storage_dir), normalized, "source_id")


def make_evidence_chunk(
    *,
    source_id: str,
    project_id: str,
    repo: str = "",
    path: str = "",
    symbol: str = "",
    chunk_type: str = "unknown",
    text: str = "",
    summary: str = "",
    keywords: list[Any] | None = None,
    technical_tags: list[Any] | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    chunk_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> EvidenceChunk:
    chunk_type_value = _enum(chunk_type, ALLOWED_CHUNK_TYPES, "unknown")
    hash_value = stable_hash(
        _canonical_json(
            {
                "project_id": _string(project_id),
                "source_id": _string(source_id),
                "repo": _string(repo),
                "path": _string(path),
                "symbol": _string(symbol),
                "chunk_type": chunk_type_value,
                "start_line": start_line,
                "end_line": end_line,
                "text": _string(text),
            }
        )
    )
    chunk_id_value = _string(chunk_id) or stable_record_id(
        "evidence_chunk",
        [_string(source_id), _string(project_id), _string(path), _string(symbol), chunk_type_value, start_line, end_line, hash_value],
    )
    return {
        "chunk_id": chunk_id_value,
        "source_id": _string(source_id),
        "project_id": _string(project_id),
        "repo": _string(repo),
        "path": _string(path),
        "symbol": _string(symbol),
        "chunk_type": chunk_type_value,
        "text": _string(text),
        "summary": _string(summary),
        "keywords": _list(keywords),
        "technical_tags": _list(technical_tags),
        "start_line": start_line,
        "end_line": end_line,
        "hash": hash_value,
        "metadata": _dict(metadata),
    }


def normalize_evidence_chunk(record: Mapping[str, Any]) -> EvidenceChunk:
    return make_evidence_chunk(
        chunk_id=_string(record.get("chunk_id")),
        source_id=_string(record.get("source_id")),
        project_id=_string(record.get("project_id")),
        repo=_string(record.get("repo")),
        path=_string(record.get("path")),
        symbol=_string(record.get("symbol")),
        chunk_type=_string(record.get("chunk_type") or "unknown"),
        text=_string(record.get("text")),
        summary=_string(record.get("summary")),
        keywords=_list(record.get("keywords")),
        technical_tags=_list(record.get("technical_tags")),
        start_line=_optional_int(record.get("start_line")),
        end_line=_optional_int(record.get("end_line")),
        metadata=_dict(record.get("metadata")),
    )


def upsert_evidence_chunk(
    record: Mapping[str, Any],
    storage_dir: str | Path | None = None,
) -> EvidenceChunk:
    normalized = normalize_evidence_chunk(record)
    return upsert_jsonl_record(get_record_path(EVIDENCE_CHUNKS, storage_dir), normalized, "chunk_id")


def make_raw_change_summary(
    *,
    project_id: str,
    source_chunk_ids: list[Any] | None = None,
    files_changed: list[Any] | None = None,
    symbols_changed: list[Any] | None = None,
    raw_change_type: list[Any] | None = None,
    what_changed: str = "",
    direct_code_evidence: list[Any] | None = None,
    uncertain_intent: list[Any] | None = None,
    change_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> RawChangeSummary:
    source_ids = _list(source_chunk_ids)
    files = _list(files_changed)
    symbols = _list(symbols_changed)
    change_types = _list(raw_change_type)
    change_id_value = _string(change_id) or stable_record_id(
        "raw_change_summary",
        [_string(project_id), source_ids, files, symbols, change_types, _string(what_changed)],
    )
    return {
        "change_id": change_id_value,
        "project_id": _string(project_id),
        "source_chunk_ids": source_ids,
        "files_changed": files,
        "symbols_changed": symbols,
        "raw_change_type": change_types,
        "what_changed": _string(what_changed),
        "direct_code_evidence": _list(direct_code_evidence),
        "uncertain_intent": _list(uncertain_intent),
        "metadata": _dict(metadata),
    }


def normalize_raw_change_summary(record: Mapping[str, Any]) -> RawChangeSummary:
    return make_raw_change_summary(
        change_id=_string(record.get("change_id")),
        project_id=_string(record.get("project_id")),
        source_chunk_ids=_list(record.get("source_chunk_ids")),
        files_changed=_list(record.get("files_changed")),
        symbols_changed=_list(record.get("symbols_changed")),
        raw_change_type=_list(record.get("raw_change_type")),
        what_changed=_string(record.get("what_changed")),
        direct_code_evidence=_list(record.get("direct_code_evidence")),
        uncertain_intent=_list(record.get("uncertain_intent")),
        metadata=_dict(record.get("metadata")),
    )


def upsert_raw_change_summary(
    record: Mapping[str, Any],
    storage_dir: str | Path | None = None,
) -> RawChangeSummary:
    normalized = normalize_raw_change_summary(record)
    return upsert_jsonl_record(get_record_path(RAW_CHANGE_SUMMARIES, storage_dir), normalized, "change_id")


def make_evidence_card(
    *,
    project_id: str,
    source_chunk_ids: list[Any] | None = None,
    problem: str = "",
    mechanism: str = "",
    implementation_details: list[Any] | None = None,
    safe_impact: str = "",
    resume_angle: str = "",
    confidence: str = "low",
    metric_support: str = "none",
    allowed_claims: list[Any] | None = None,
    forbidden_claims: list[Any] | None = None,
    evidence_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> EvidenceCard:
    source_ids = _list(source_chunk_ids)
    confidence_value = _enum(confidence, ALLOWED_CONFIDENCE, "low")
    metric_support_value = _enum(metric_support, ALLOWED_METRIC_SUPPORT, "none")
    evidence_id_value = _string(evidence_id) or stable_record_id(
        "evidence_card",
        [_string(project_id), source_ids, _string(problem), _string(mechanism), _string(safe_impact), _string(resume_angle)],
    )
    return {
        "evidence_id": evidence_id_value,
        "project_id": _string(project_id),
        "source_chunk_ids": source_ids,
        "problem": _string(problem),
        "mechanism": _string(mechanism),
        "implementation_details": _list(implementation_details),
        "safe_impact": _string(safe_impact),
        "resume_angle": _string(resume_angle),
        "confidence": confidence_value,
        "metric_support": metric_support_value,
        "allowed_claims": _list(allowed_claims),
        "forbidden_claims": _list(forbidden_claims),
        "metadata": _dict(metadata),
    }


def normalize_evidence_card(record: Mapping[str, Any]) -> EvidenceCard:
    return make_evidence_card(
        evidence_id=_string(record.get("evidence_id")),
        project_id=_string(record.get("project_id")),
        source_chunk_ids=_list(record.get("source_chunk_ids")),
        problem=_string(record.get("problem")),
        mechanism=_string(record.get("mechanism")),
        implementation_details=_list(record.get("implementation_details")),
        safe_impact=_string(record.get("safe_impact")),
        resume_angle=_string(record.get("resume_angle")),
        confidence=_string(record.get("confidence") or "low"),
        metric_support=_string(record.get("metric_support") or "none"),
        allowed_claims=_list(record.get("allowed_claims")),
        forbidden_claims=_list(record.get("forbidden_claims")),
        metadata=_dict(record.get("metadata")),
    )


def upsert_evidence_card(
    record: Mapping[str, Any],
    storage_dir: str | Path | None = None,
) -> EvidenceCard:
    normalized = normalize_evidence_card(record)
    return upsert_jsonl_record(get_record_path(EVIDENCE_CARDS, storage_dir), normalized, "evidence_id")


def make_capability_fact(
    *,
    project_id: str,
    capability_type: str,
    present: bool = True,
    confidence: str = "low",
    mechanisms: list[Any] | None = None,
    source_evidence_ids: list[Any] | None = None,
    allowed_resume_claims: list[Any] | None = None,
    forbidden_claims: list[Any] | None = None,
    metric_support: str = "none",
    capability_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> CapabilityFact:
    project_id_value = _string(project_id)
    capability_type_value = _string(capability_type)
    capability_id_value = _string(capability_id) or stable_record_id(
        "capability_fact",
        [project_id_value, capability_type_value],
    )
    return {
        "capability_id": capability_id_value,
        "project_id": project_id_value,
        "capability_type": capability_type_value,
        "present": bool(present),
        "confidence": _enum(confidence, ALLOWED_CONFIDENCE, "low"),
        "mechanisms": _list(mechanisms),
        "source_evidence_ids": _list(source_evidence_ids),
        "allowed_resume_claims": _list(allowed_resume_claims),
        "forbidden_claims": _list(forbidden_claims),
        "metric_support": _enum(metric_support, ALLOWED_METRIC_SUPPORT, "none"),
        "metadata": _dict(metadata),
    }


def normalize_capability_fact(record: Mapping[str, Any]) -> CapabilityFact:
    return make_capability_fact(
        capability_id=_string(record.get("capability_id")),
        project_id=_string(record.get("project_id")),
        capability_type=_string(record.get("capability_type")),
        present=bool(record.get("present", True)),
        confidence=_string(record.get("confidence") or "low"),
        mechanisms=_list(record.get("mechanisms")),
        source_evidence_ids=_list(record.get("source_evidence_ids")),
        allowed_resume_claims=_list(record.get("allowed_resume_claims")),
        forbidden_claims=_list(record.get("forbidden_claims")),
        metric_support=_string(record.get("metric_support") or "none"),
        metadata=_dict(record.get("metadata")),
    )


def upsert_capability_fact(
    record: Mapping[str, Any],
    storage_dir: str | Path | None = None,
) -> CapabilityFact:
    normalized = normalize_capability_fact(record)
    path = get_record_path(CAPABILITY_FACTS, storage_dir)
    records = read_jsonl(path)
    merged_record: dict[str, Any] = dict(normalized)
    retained_records: list[dict[str, Any]] = []

    for existing in records:
        same_id = str(existing.get("capability_id") or "") == normalized["capability_id"]
        same_capability = (
            str(existing.get("project_id") or "") == normalized["project_id"]
            and str(existing.get("capability_type") or "") == normalized["capability_type"]
        )
        if same_id or same_capability:
            merged_record = _merge_capability_facts(existing, merged_record)
        else:
            retained_records.append(existing)

    retained_records.append(merged_record)
    write_jsonl(path, retained_records)
    return merged_record


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _enum(value: Any, allowed: set[str], default: str) -> Any:
    value_string = _string(value)
    return value_string if value_string in allowed else default


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _merge_capability_facts(existing: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    merged.update(dict(new))
    merged["capability_id"] = str(existing.get("capability_id") or new.get("capability_id") or "")
    merged["mechanisms"] = _merge_lists(existing.get("mechanisms"), new.get("mechanisms"))
    merged["source_evidence_ids"] = _merge_lists(existing.get("source_evidence_ids"), new.get("source_evidence_ids"))
    merged["allowed_resume_claims"] = _merge_lists(
        existing.get("allowed_resume_claims"),
        new.get("allowed_resume_claims"),
    )
    merged["forbidden_claims"] = _merge_lists(existing.get("forbidden_claims"), new.get("forbidden_claims"))
    merged["metadata"] = {**_dict(existing.get("metadata")), **_dict(new.get("metadata"))}
    return merged


def _merge_lists(*values: Any) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for value in values:
        for item in _list(value):
            key = _canonical_json(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged
