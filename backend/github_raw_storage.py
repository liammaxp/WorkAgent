"""Backend-only storage and redaction for raw GitHub source material."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, TypedDict

from backend import evidence_memory


DEFAULT_GITHUB_RAW_SOURCES_PATH = (
    Path(__file__).resolve().parents[1] / "information" / "github_raw_sources.jsonl"
)
ALLOWED_SOURCE_TYPES = frozenset(
    {"commit_patch", "file_snapshot", "readme", "issue", "pr", "log", "unknown"}
)
_SENSITIVE_METADATA_TERMS = (
    "body",
    "content",
    "diff",
    "file",
    "hunk",
    "patch",
    "raw",
    "text",
)
_MAX_SAFE_METADATA_ITEMS = 16
_MAX_SAFE_METADATA_VALUE_CHARS = 256


class GithubRawSourceRecord(TypedDict):
    source_id: str
    project_id: str
    repo: str
    source_type: str
    path: str
    commit_sha: str
    raw_hash: str
    raw_chars: int
    raw_text: str
    created_at: str
    metadata: dict[str, Any]


def _optional_string(value: Any, field_name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    return value


def build_github_raw_source_record(
    *,
    project_id: str = "",
    repo: str = "",
    source_type: str = "unknown",
    path: str = "",
    commit_sha: str = "",
    raw_text: str = "",
    created_at: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> GithubRawSourceRecord:
    """Build a deterministic raw record; timestamps are caller-supplied data."""

    project_id_value = _optional_string(project_id, "project_id").strip()
    repo_value = _optional_string(repo, "repo").strip()
    source_type_value = _optional_string(source_type, "source_type").strip().casefold()
    if source_type_value not in ALLOWED_SOURCE_TYPES:
        source_type_value = "unknown"
    path_value = _optional_string(path, "path").strip()
    commit_sha_value = _optional_string(commit_sha, "commit_sha").strip()
    raw_text_value = _optional_string(raw_text, "raw_text")
    created_at_value = _optional_string(created_at, "created_at").strip()
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")
    metadata_value = dict(metadata or {})
    raw_hash = evidence_memory.stable_hash(raw_text_value)
    source_id = evidence_memory.stable_record_id(
        "raw",
        [
            project_id_value,
            repo_value,
            source_type_value,
            path_value,
            commit_sha_value,
            raw_hash,
        ],
    )
    return {
        "source_id": source_id,
        "project_id": project_id_value,
        "repo": repo_value,
        "source_type": source_type_value,
        "path": path_value,
        "commit_sha": commit_sha_value,
        "raw_hash": raw_hash,
        "raw_chars": len(raw_text_value),
        "raw_text": raw_text_value,
        "created_at": created_at_value,
        "metadata": metadata_value,
    }


def _normalize_record(record: Mapping[str, Any]) -> GithubRawSourceRecord:
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    return build_github_raw_source_record(
        project_id=record.get("project_id"),
        repo=record.get("repo"),
        source_type=record.get("source_type", "unknown"),
        path=record.get("path"),
        commit_sha=record.get("commit_sha"),
        raw_text=record.get("raw_text"),
        created_at=record.get("created_at"),
        metadata=record.get("metadata"),
    )


def append_github_raw_source_record(
    record: Mapping[str, Any],
    artifact_path: str | Path = DEFAULT_GITHUB_RAW_SOURCES_PATH,
) -> GithubRawSourceRecord:
    """Persist one normalized record, idempotently replacing the same source ID."""

    normalized = _normalize_record(record)
    evidence_memory.upsert_jsonl_record(artifact_path, normalized, "source_id")
    return normalized


def load_github_raw_source_records(
    artifact_path: str | Path = DEFAULT_GITHUB_RAW_SOURCES_PATH,
) -> list[GithubRawSourceRecord]:
    """Load and validate backend-only records; malformed JSONL fails closed."""

    return [_normalize_record(record) for record in evidence_memory.read_jsonl(artifact_path)]


def safe_metadata_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    summary: dict[str, Any] = {}
    for key in sorted(value, key=lambda item: str(item).casefold()):
        key_value = str(key)
        normalized_key = key_value.casefold()
        if any(term in normalized_key for term in _SENSITIVE_METADATA_TERMS):
            continue
        item = value[key]
        if isinstance(item, bool) or item is None or isinstance(item, (int, float)):
            summary[key_value] = item
        elif isinstance(item, str) and len(item) <= _MAX_SAFE_METADATA_VALUE_CHARS:
            summary[key_value] = item
        if len(summary) >= _MAX_SAFE_METADATA_ITEMS:
            break
    return summary


def redact_github_raw_source_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a bounded summary that cannot contain source bodies or patches."""

    normalized = _normalize_record(record)
    return {
        "source_id": normalized["source_id"],
        "project_id": normalized["project_id"],
        "repo": normalized["repo"],
        "source_type": normalized["source_type"],
        "path": normalized["path"],
        "commit_sha": normalized["commit_sha"],
        "raw_hash": normalized["raw_hash"],
        "raw_chars": normalized["raw_chars"],
        "raw_available": bool(normalized["raw_text"]),
        "metadata": safe_metadata_summary(normalized["metadata"]),
    }
