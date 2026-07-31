"""Conservative GitHub evidence raw change summary extraction."""

from __future__ import annotations

import os
import re
from typing import Any

import evidence_memory


GITHUB_EVIDENCE_MEMORY_ENV = "USE_GITHUB_EVIDENCE_MEMORY"
ENABLED_VALUES = {"1", "true", "yes", "on"}
SUPPORTED_CHUNK_TYPES = {
    "diff_hunk",
    "function",
    "class",
    "endpoint",
    "config",
    "log_entry",
    "unknown",
    "readme_section",
}
DIRECT_EVIDENCE_LIMIT = 6
DIRECT_EVIDENCE_CHARS = 200
SUMMARY_CHARS = 240

UNSUPPORTED_CLAIM_PATTERNS = [
    r"hallucination reduction",
    r"reduced hallucinations",
    r"eliminated hallucinations",
    r"ATS score",
    r"interview success",
    r"guaranteed",
    r"\bperfect\b",
    r"reduced by",
    r"improved by",
    r"%",
    r"latency reduction",
    r"cost reduction",
    r"business impact",
]

CHANGE_TYPE_RULES = [
    ("merge_logic_update", ["merge", "final_bullets", "bullet_depth_profile"]),
    ("validation_rule_update", ["validation", "quality_gate", "blocker", "guard", "template pollution"]),
    ("storage_update", ["JSONL", "storage", "upsert", "file write", "persist"]),
    ("schema_update", ["TypedDict", "schema", "dataclass", "fields"]),
    ("api_route_update", ["route", "endpoint", "GET", "POST", "api"]),
    ("frontend_display_update", ["frontend", "React", "page", "component"]),
    ("test_update", ["test", "unittest", "pytest"]),
    ("retrieval_logic_update", ["retrieve", "retrieval", "Chroma", "query", "search"]),
    ("ranking_logic_update", ["rank", "rerank", "score"]),
    ("chunking_update", ["chunk", "chunker", "split"]),
    ("configuration_update", ["env", "flag", "config"]),
    ("prompt_update", ["prompt", "system prompt"]),
    ("documentation_update", ["README", "docs", "markdown"]),
    ("logging_update", ["log", "logging"]),
    ("error_handling_update", ["try", "except", "fallback", "error"]),
]

GENERIC_BOILERPLATE_PATTERNS = [
    r"^\s*$",
    r"^[{}\[\],;]+$",
    r"^[-+ ]*$",
]


def github_evidence_enabled() -> bool:
    return str(os.getenv(GITHUB_EVIDENCE_MEMORY_ENV, "1")).strip().lower() in ENABLED_VALUES


def extract_raw_change_summaries_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for chunk in chunks:
        summary = extract_raw_change_summary_from_chunk(chunk)
        if summary is not None:
            summaries.append(summary)
    return summaries


def extract_raw_change_summary_from_chunk(chunk: dict[str, Any]) -> dict[str, Any] | None:
    text = normalize_text(chunk.get("text"))
    if not should_process_chunk(chunk, text):
        return None

    project_id = str(chunk.get("project_id") or "")
    chunk_id = str(chunk.get("chunk_id") or "")
    path = str(chunk.get("path") or "").strip()
    symbol = str(chunk.get("symbol") or "").strip()
    keywords = safe_list(chunk.get("keywords"))
    change_types = classify_change_types(chunk, text, keywords)
    direct_evidence, source_claim_text = extract_direct_code_evidence(text)
    if not direct_evidence and keywords:
        direct_evidence = [truncate(f"Observed technical terms: {', '.join(keywords[:4])}", DIRECT_EVIDENCE_CHARS)]

    if not direct_evidence and not path and not symbol:
        return None

    what_changed = unsupported_claim_guard(
        generate_what_changed(
            path=path,
            symbol=symbol,
            change_types=change_types,
            keywords=keywords,
            chunk=chunk,
        )
    )
    direct_evidence = [unsupported_claim_guard(item) for item in direct_evidence]
    direct_evidence = [item for item in direct_evidence if item]
    if not what_changed:
        return None

    files_changed = [path] if path else []
    symbols_changed = [symbol] if symbol else []
    uncertain_intent = uncertain_intent_for_change_types(change_types)
    metadata = {
        "source": "github_evidence_raw_change_summary",
        "chunk_type": str(chunk.get("chunk_type") or ""),
        "source_id": str(chunk.get("source_id") or ""),
        "repo": str(chunk.get("repo") or ""),
        "chunk_summary": str(chunk.get("summary") or ""),
    }
    if source_claim_text:
        metadata["source_claim_text"] = source_claim_text[:DIRECT_EVIDENCE_LIMIT]

    change_id = evidence_memory.stable_record_id(
        "raw_change_summary",
        [
            project_id,
            [chunk_id],
            files_changed,
            symbols_changed,
            change_types,
            evidence_memory.stable_hash(what_changed + "\n" + "\n".join(direct_evidence)),
        ],
    )
    return evidence_memory.make_raw_change_summary(
        change_id=change_id,
        project_id=project_id,
        source_chunk_ids=[chunk_id],
        files_changed=files_changed,
        symbols_changed=symbols_changed,
        raw_change_type=change_types,
        what_changed=what_changed,
        direct_code_evidence=direct_evidence,
        uncertain_intent=uncertain_intent,
        metadata=metadata,
    )


def build_github_evidence_raw_change_summaries(
    project_id: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    if not github_evidence_enabled():
        return {
            "enabled": False,
            "memory_type": "github_evidence",
            "project_id": project_id or None,
            "processed_chunks": 0,
            "created_summaries": 0,
            "updated_summaries": 0,
            "created_or_updated_summaries": 0,
            "raw_change_summaries_count": 0,
            "message": "GitHub context evidence memory is disabled.",
            "errors": [],
        }

    chunks = (
        evidence_memory.read_records_by_project(evidence_memory.EVIDENCE_CHUNKS, project_id)
        if project_id
        else evidence_memory.read_records(evidence_memory.EVIDENCE_CHUNKS)
    )
    if limit is not None:
        chunks = chunks[: max(0, int(limit))]

    existing_ids = {
        str(record.get("change_id") or "")
        for record in evidence_memory.read_records(evidence_memory.RAW_CHANGE_SUMMARIES)
    }
    created_summaries = 0
    updated_summaries = 0
    unchanged_summaries = 0
    skipped_chunks = []
    created_or_updated = 0
    summaries_preview: list[dict[str, Any]] = []

    for chunk in chunks:
        summary = extract_raw_change_summary_from_chunk(chunk)
        if summary is None:
            skipped_chunks.append({
                "chunk_id": str(chunk.get("chunk_id") or ""),
                "project_id": str(chunk.get("project_id") or ""),
                "reason": "chunk produced no qualifying change summary",
            })
            continue
        change_id = str(summary.get("change_id") or "")
        _, write_status = evidence_memory.upsert_raw_change_summary_with_status(summary)
        if write_status == "updated":
            updated_summaries += 1
        elif write_status == "created":
            created_summaries += 1
            existing_ids.add(change_id)
        else:
            unchanged_summaries += 1
        if write_status != "unchanged":
            created_or_updated += 1
        summaries_preview.append(
            {
                "change_id": change_id,
                "project_id": summary["project_id"],
                "what_changed": summary["what_changed"],
                "raw_change_type": summary["raw_change_type"],
            }
        )

    counts = evidence_memory.get_github_evidence_memory_counts(project_id=project_id)
    return {
        "enabled": True,
        "memory_type": "github_evidence",
        "project_id": project_id or None,
        "processed_chunks": len(chunks),
        "created_summaries": created_summaries,
        "updated_summaries": updated_summaries,
        "unchanged_summaries": unchanged_summaries,
        "skipped_chunks": len(skipped_chunks),
        "created_or_updated_summaries": created_or_updated,
        "raw_change_summaries_count": counts["raw_change_summaries_count"],
        "message": "GitHub evidence raw change summaries built successfully.",
        "summaries": summaries_preview,
        "skips": skipped_chunks,
    }


def should_process_chunk(chunk: dict[str, Any], text: str) -> bool:
    if not text:
        return False
    chunk_type = str(chunk.get("chunk_type") or "unknown")
    if chunk_type not in SUPPORTED_CHUNK_TYPES:
        return False
    if any(re.match(pattern, text) for pattern in GENERIC_BOILERPLATE_PATTERNS):
        return False
    keywords = safe_list(chunk.get("keywords"))
    if len(text) < 80 and not keywords and not str(chunk.get("path") or ""):
        return False
    if chunk_type == "unknown" and not keywords and not classify_change_types(chunk, text, keywords):
        return False
    return True


def classify_change_types(chunk: dict[str, Any], text: str, keywords: list[str]) -> list[str]:
    haystack = " ".join(
        [
            text,
            str(chunk.get("summary") or ""),
            str(chunk.get("path") or ""),
            str(chunk.get("symbol") or ""),
            " ".join(keywords),
            " ".join(safe_list(chunk.get("technical_tags"))),
        ]
    )
    lowered = haystack.lower()
    change_types = []
    for change_type, terms in CHANGE_TYPE_RULES:
        if any(term.lower() in lowered for term in terms):
            change_types.append(change_type)
    return change_types or ["unknown_update"]


def extract_direct_code_evidence(text: str) -> tuple[list[str], list[str]]:
    candidates = changed_lines(text) or important_lines(text)
    evidence: list[str] = []
    source_claim_text: list[str] = []
    seen = set()
    for line in candidates:
        clean = clean_evidence_line(line)
        if not clean:
            continue
        if contains_unsupported_claim(clean):
            source_claim_text.append(truncate(clean, DIRECT_EVIDENCE_CHARS))
            continue
        guarded = unsupported_claim_guard(clean)
        key = guarded.lower()
        if not guarded or key in seen:
            continue
        seen.add(key)
        evidence.append(truncate(guarded, DIRECT_EVIDENCE_CHARS))
        if len(evidence) >= DIRECT_EVIDENCE_LIMIT:
            break
    return evidence, source_claim_text


def changed_lines(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def important_lines(text: str) -> list[str]:
    important_terms = [
        "def ",
        "class ",
        "function ",
        "const ",
        "route",
        "endpoint",
        "GET ",
        "POST ",
        "TypedDict",
        "schema",
        "upsert",
        "JSONL",
        "validation",
        "quality_gate",
        "bullet_depth_profile",
        "final_bullets",
        "chunk",
        "fallback",
        "try:",
        "except ",
    ]
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(term.lower() in stripped.lower() for term in important_terms):
            lines.append(stripped)
    return lines


def generate_what_changed(
    *,
    path: str,
    symbol: str,
    change_types: list[str],
    keywords: list[str],
    chunk: dict[str, Any],
) -> str:
    primary_type = change_types[0] if change_types else "unknown_update"
    if primary_type == "documentation_update":
        heading = first_heading(str(chunk.get("summary") or "")) or first_heading(str(chunk.get("text") or ""))
        if heading:
            return truncate(f"Updated documentation section related to {heading}.", SUMMARY_CHARS)
    if path and symbol:
        return truncate(f"Updated {symbol} in {path} for {primary_type}.", SUMMARY_CHARS)
    if path:
        return truncate(f"Updated {path} for {primary_type}.", SUMMARY_CHARS)
    if keywords:
        return truncate(f"Updated code related to {', '.join(keywords[:3])}.", SUMMARY_CHARS)
    return truncate(f"Updated code for {primary_type}.", SUMMARY_CHARS)


def uncertain_intent_for_change_types(change_types: list[str]) -> list[str]:
    intents = []
    if "storage_update" in change_types or "schema_update" in change_types:
        intents.append("May support more traceable evidence memory.")
    if "api_route_update" in change_types:
        intents.append("May support safer GitHub evidence debugging.")
    if "chunking_update" in change_types:
        intents.append("May support raw GitHub context inspection.")
    return intents[:3]


def unsupported_claim_guard(text: str) -> str:
    guarded = str(text or "")
    for pattern in UNSUPPORTED_CLAIM_PATTERNS:
        guarded = re.sub(pattern, "", guarded, flags=re.IGNORECASE)
    guarded = re.sub(r"\s+", " ", guarded).strip(" .,-")
    return truncate(guarded + "." if guarded and not guarded.endswith(".") else guarded, SUMMARY_CHARS)


def contains_unsupported_claim(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in UNSUPPORTED_CLAIM_PATTERNS)


def clean_evidence_line(line: str) -> str:
    clean = line.strip()
    if clean.startswith("+"):
        clean = clean[1:].strip()
    clean = re.sub(r"\s+", " ", clean)
    if not clean or clean in {"{", "}", ")", "(", "]", "["}:
        return ""
    return clean


def normalize_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def safe_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def first_heading(text: str) -> str:
    stripped = str(text or "").strip()
    stripped = re.sub(r"^#{1,6}\s+", "", stripped)
    return truncate(stripped, 80)


def truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
