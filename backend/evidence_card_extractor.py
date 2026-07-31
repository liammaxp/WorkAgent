"""Deterministic GitHub evidence evidence card extraction."""

from __future__ import annotations

import os
import re
from typing import Any

import evidence_memory


GITHUB_EVIDENCE_MEMORY_ENV = "USE_GITHUB_EVIDENCE_MEMORY"
ENABLED_VALUES = {"1", "true", "yes", "on"}
DETAIL_LIMIT = 6
DETAIL_CHARS = 220

UNSUPPORTED_CLAIM_PATTERNS = [
    r"reduced hallucinations",
    r"eliminated hallucinations",
    r"hallucination reduction",
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
    r"\brevenue\b",
    r"\busers\b",
]

PROBLEM_BY_TYPE = {
    "merge_logic_update": "Generic or lower-value generated content could displace stronger project-specific evidence during final merge.",
    "validation_rule_update": "Invalid, polluted, or unsupported generated content could pass into downstream output without a deterministic guard.",
    "storage_update": "Raw GitHub context could be difficult to audit or reuse without structured persistence.",
    "schema_update": "Project evidence could be difficult to normalize, deduplicate, or inspect without a consistent schema.",
    "api_route_update": "GitHub evidence evidence state could be difficult to inspect without safe read-only or debug endpoints.",
    "chunking_update": "Repo-level GitHub context was too coarse for precise downstream evidence extraction.",
    "test_update": "GitHub evidence behavior could regress without targeted tests.",
    "configuration_update": "New GitHub evidence behavior needed a safe feature flag boundary to avoid affecting the old pipeline.",
    "retrieval_logic_update": "Retrieved evidence could be difficult to trace without explicit source boundaries.",
    "ranking_logic_update": "Project or evidence ordering could be difficult to reason about without deterministic ranking logic.",
    "documentation_update": "GitHub evidence behavior could be difficult to inspect or operate without clear documentation.",
    "error_handling_update": "GitHub evidence processing needed guarded fallback behavior for unexpected local state.",
}

MECHANISM_BY_TYPE = {
    "merge_logic_update": "Updated merge logic to preserve mechanism-rich evidence before generic generated content.",
    "validation_rule_update": "Added deterministic validation or guard logic around generated output.",
    "storage_update": "Added JSONL-backed storage helpers with stable IDs, hashes, and upsert behavior.",
    "schema_update": "Added structured schema fields for normalizing and inspecting evidence records.",
    "api_route_update": "Added explicit GitHub evidence backend endpoint for safe status, preview, or manual build operations.",
    "chunking_update": "Split raw GitHub context into bounded, source-traceable evidence chunks.",
    "test_update": "Added targeted regression tests for GitHub evidence behavior.",
    "configuration_update": "Added feature-flag gating around GitHub evidence behavior.",
    "retrieval_logic_update": "Updated evidence lookup logic while preserving source traceability.",
    "ranking_logic_update": "Updated deterministic ordering or scoring logic.",
    "documentation_update": "Updated documentation or markdown content for GitHub evidence behavior.",
    "error_handling_update": "Added guarded error handling or fallback behavior.",
}

SAFE_IMPACT_BY_TYPE = {
    "merge_logic_update": "Improved technical specificity and ordering reliability of generated project evidence before final output.",
    "validation_rule_update": "Improved deterministic guarding of generated or stored evidence before downstream use.",
    "storage_update": "Improved traceability and repeatability of GitHub evidence evidence memory.",
    "schema_update": "Improved consistency of evidence records for later inspection and reuse.",
    "api_route_update": "Improved debuggability and safe inspection of GitHub evidence evidence state.",
    "chunking_update": "Improved source traceability by converting coarse raw GitHub context into bounded evidence chunks.",
    "test_update": "Improved regression coverage for GitHub evidence evidence memory behavior.",
    "configuration_update": "Reduced risk of GitHub evidence changes affecting the existing pipeline by keeping behavior behind a feature flag.",
    "retrieval_logic_update": "Improved traceability of evidence lookup behavior.",
    "ranking_logic_update": "Improved determinism of evidence or project ordering.",
    "documentation_update": "Improved inspectability of GitHub evidence behavior for local debugging.",
    "error_handling_update": "Improved resilience of local GitHub evidence processing around unexpected state.",
}

RESUME_ANGLE_BY_TYPE = {
    "storage_update": "schema_and_storage_design",
    "schema_update": "schema_and_storage_design",
    "api_route_update": "api_inspection",
    "chunking_update": "chunking_and_index_preparation",
    "validation_rule_update": "validation_and_guardrails",
    "merge_logic_update": "merge_quality_control",
    "test_update": "testing_and_regression_safety",
    "configuration_update": "configuration_safety",
    "documentation_update": "debuggability",
    "retrieval_logic_update": "source_traceability",
    "ranking_logic_update": "source_traceability",
    "error_handling_update": "debuggability",
}

ALLOWED_CLAIM_BY_TYPE = {
    "merge_logic_update": "preserved mechanism-rich project evidence during deterministic merge ordering",
    "validation_rule_update": "added deterministic guards for template pollution or unsupported generated content",
    "storage_update": "implemented JSONL-backed GitHub evidence evidence memory with stable IDs and upsert behavior",
    "schema_update": "defined structured GitHub evidence evidence records for inspection and reuse",
    "api_route_update": "added safe GitHub evidence status and preview endpoints without exposing full raw content",
    "chunking_update": "converted raw GitHub context into source-traceable evidence chunks",
    "test_update": "added regression tests for GitHub evidence evidence memory behavior",
    "configuration_update": "kept GitHub evidence behavior behind a feature flag boundary",
    "retrieval_logic_update": "preserved source traceability in evidence lookup logic",
    "ranking_logic_update": "added deterministic ordering logic for evidence handling",
    "documentation_update": "documented GitHub evidence evidence memory behavior for local inspection",
    "error_handling_update": "added guarded fallback behavior for GitHub evidence evidence processing",
}

GENERIC_FORBIDDEN_CLAIMS = [
    "do not claim quantified improvement without explicit metric evidence",
    "do not claim ATS score improvement",
    "do not claim interview success",
    "do not claim guaranteed factual correctness",
    "do not claim hallucinations were eliminated",
]


def github_evidence_enabled() -> bool:
    return str(os.getenv(GITHUB_EVIDENCE_MEMORY_ENV, "1")).strip().lower() in ENABLED_VALUES


def build_evidence_cards_from_change_summaries(
    summaries: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for summary in summaries:
        card = build_evidence_card_from_change_summary(summary, chunks_by_id=chunks_by_id)
        if card is not None:
            cards.append(card)
    return cards


def build_evidence_card_from_change_summary(
    summary: dict[str, Any],
    related_chunks: list[dict[str, Any]] | None = None,
    chunks_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    source_chunk_ids = safe_list(summary.get("source_chunk_ids"))
    if related_chunks is None:
        chunks_by_id = chunks_by_id or {}
        related_chunks = [chunks_by_id[chunk_id] for chunk_id in source_chunk_ids if chunk_id in chunks_by_id]

    raw_change_types = safe_list(summary.get("raw_change_type")) or ["unknown_update"]
    primary_type = first_supported_type(raw_change_types)
    project_id = str(summary.get("project_id") or "")
    direct_evidence = sanitized_list(summary.get("direct_code_evidence"))
    unsupported_claims = unsupported_source_claims(summary)

    problem = sanitize_fact(PROBLEM_BY_TYPE.get(primary_type) or "The change addressed a code/documentation area that needed more structured handling.")
    mechanism = sanitize_fact(mechanism_for_summary(summary, primary_type))
    implementation_details = implementation_details_for_summary(summary, related_chunks)
    safe_impact = sanitize_fact(SAFE_IMPACT_BY_TYPE.get(primary_type) or "Improved structural handling of GitHub evidence evidence memory.")
    resume_angle = RESUME_ANGLE_BY_TYPE.get(primary_type, "unknown")
    metric_support = metric_support_for_summary(summary)
    allowed_claims = allowed_claims_for_summary(summary, primary_type)
    forbidden_claims = forbidden_claims_for_card(raw_change_types, metric_support)
    confidence = confidence_for_summary(summary, primary_type, direct_evidence, implementation_details)

    quality_score, skip_reasons = evidence_card_quality_score(
        problem=problem,
        mechanism=mechanism,
        source_chunk_ids=source_chunk_ids,
        implementation_details=implementation_details,
        safe_impact=safe_impact,
        resume_angle=resume_angle,
        allowed_claims=allowed_claims,
    )
    if quality_score < 5:
        return None

    metadata = {
        "source": "github_evidence_card_extractor",
        "raw_change_type": raw_change_types,
        "source_change_id": str(summary.get("change_id") or ""),
        "files_changed": safe_list(summary.get("files_changed")),
        "symbols_changed": safe_list(summary.get("symbols_changed")),
        "quality_score": quality_score,
        "skip_reasons": skip_reasons,
    }
    if unsupported_claims:
        metadata["unsupported_source_claims"] = unsupported_claims

    evidence_id = evidence_memory.stable_record_id(
        "evidence_card",
        [
            project_id,
            source_chunk_ids,
            problem,
            mechanism,
            evidence_memory.stable_hash("\n".join(allowed_claims) + "\n" + safe_impact),
        ],
    )
    return evidence_memory.make_evidence_card(
        evidence_id=evidence_id,
        project_id=project_id,
        source_chunk_ids=source_chunk_ids,
        problem=problem,
        mechanism=mechanism,
        implementation_details=implementation_details,
        safe_impact=safe_impact,
        resume_angle=resume_angle,
        confidence=confidence,
        metric_support=metric_support,
        allowed_claims=allowed_claims,
        forbidden_claims=forbidden_claims,
        metadata=metadata,
    )


def build_github_evidence_cards(
    project_id: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    if not github_evidence_enabled():
        return {
            "enabled": False,
            "memory_type": "github_evidence",
            "project_id": project_id or None,
            "processed_summaries": 0,
            "created_or_updated_evidence_cards": 0,
            "skipped_summaries": 0,
            "evidence_cards_count": 0,
            "message": "GitHub context evidence memory is disabled.",
            "errors": [],
        }

    summaries = (
        evidence_memory.read_records_by_project(evidence_memory.RAW_CHANGE_SUMMARIES, project_id)
        if project_id
        else evidence_memory.read_records(evidence_memory.RAW_CHANGE_SUMMARIES)
    )
    if limit is not None:
        summaries = summaries[: max(0, int(limit))]

    chunks = evidence_memory.read_records(evidence_memory.EVIDENCE_CHUNKS)
    chunks_by_id = {str(chunk.get("chunk_id") or ""): chunk for chunk in chunks}
    existing_ids = {
        str(record.get("evidence_id") or "")
        for record in evidence_memory.read_records(evidence_memory.EVIDENCE_CARDS)
    }
    created_cards = 0
    updated_cards = 0
    unchanged_cards = 0
    created_or_updated = 0
    skipped = []
    cards_preview = []

    for summary in summaries:
        card = build_evidence_card_from_change_summary(summary, chunks_by_id=chunks_by_id)
        if card is None:
            skipped.append(
                {
                    "change_id": str(summary.get("change_id") or ""),
                    "project_id": str(summary.get("project_id") or ""),
                    "reason": "summary did not meet evidence card quality threshold",
                }
            )
            continue
        evidence_id = str(card.get("evidence_id") or "")
        _, write_status = evidence_memory.upsert_evidence_card_with_status(card)
        if write_status == "updated":
            updated_cards += 1
        elif write_status == "created":
            created_cards += 1
            existing_ids.add(evidence_id)
        else:
            unchanged_cards += 1
        if write_status != "unchanged":
            created_or_updated += 1
        cards_preview.append(
            {
                "evidence_id": evidence_id,
                "project_id": card["project_id"],
                "resume_angle": card["resume_angle"],
                "confidence": card["confidence"],
            }
        )

    counts = evidence_memory.get_github_evidence_memory_counts(project_id=project_id)
    return {
        "enabled": True,
        "memory_type": "github_evidence",
        "project_id": project_id or None,
        "processed_summaries": len(summaries),
        "created_evidence_cards": created_cards,
        "updated_evidence_cards": updated_cards,
        "unchanged_evidence_cards": unchanged_cards,
        "created_or_updated_evidence_cards": created_or_updated,
        "skipped_summaries": len(skipped),
        "evidence_cards_count": counts["evidence_cards_count"],
        "message": "GitHub evidence evidence cards built successfully.",
        "cards": cards_preview,
        "skips": skipped,
        "errors": [],
    }


def first_supported_type(raw_change_types: list[str]) -> str:
    for change_type in raw_change_types:
        if change_type in PROBLEM_BY_TYPE or change_type in MECHANISM_BY_TYPE:
            return change_type
    return raw_change_types[0] if raw_change_types else "unknown_update"


def mechanism_for_summary(summary: dict[str, Any], primary_type: str) -> str:
    mechanism = MECHANISM_BY_TYPE.get(primary_type)
    files = safe_list(summary.get("files_changed"))
    symbols = safe_list(summary.get("symbols_changed"))
    if mechanism:
        if symbols and files:
            return f"{mechanism} Source change touched {symbols[0]} in {files[0]}."
        if files:
            return f"{mechanism} Source change touched {files[0]}."
        return mechanism
    what_changed = str(summary.get("what_changed") or "").strip()
    return what_changed or "Updated source-backed evidence handling."


def implementation_details_for_summary(summary: dict[str, Any], related_chunks: list[dict[str, Any]]) -> list[str]:
    details: list[str] = []
    for item in sanitized_list(summary.get("direct_code_evidence")):
        append_detail(details, item)
    for file_path in safe_list(summary.get("files_changed")):
        append_detail(details, f"Changed file: {file_path}")
    for symbol in safe_list(summary.get("symbols_changed")):
        append_detail(details, f"Changed symbol: {symbol}")
    for change_type in safe_list(summary.get("raw_change_type")):
        append_detail(details, f"Change category: {change_type}")
    for chunk in related_chunks:
        keywords = safe_list(chunk.get("keywords"))
        tags = safe_list(chunk.get("technical_tags"))
        if keywords:
            append_detail(details, f"Related keywords: {', '.join(keywords[:4])}")
        if tags:
            append_detail(details, f"Technical tags: {', '.join(tags[:4])}")
    return details[:DETAIL_LIMIT]


def append_detail(details: list[str], value: str) -> None:
    clean = sanitize_fact(value, limit=DETAIL_CHARS)
    if clean and clean.lower() not in {detail.lower() for detail in details}:
        details.append(clean)


def allowed_claims_for_summary(summary: dict[str, Any], primary_type: str) -> list[str]:
    claims = []
    for change_type in safe_list(summary.get("raw_change_type")):
        claim = ALLOWED_CLAIM_BY_TYPE.get(change_type)
        if claim:
            claims.append(claim)
    if not claims and primary_type in ALLOWED_CLAIM_BY_TYPE:
        claims.append(ALLOWED_CLAIM_BY_TYPE[primary_type])
    if not claims:
        claims.append("maintained source-traceable GitHub evidence evidence records")
    sanitized = []
    for claim in claims:
        clean = sanitize_fact(claim, terminal_period=False)
        if clean and clean not in sanitized:
            sanitized.append(clean)
    return sanitized[:4]


def forbidden_claims_for_card(raw_change_types: list[str], metric_support: str) -> list[str]:
    claims = list(GENERIC_FORBIDDEN_CLAIMS)
    if metric_support == "none":
        claims.append("do not use percentages or before/after numeric improvements")
    if any(change_type in {"validation_rule_update", "storage_update", "schema_update", "merge_logic_update"} for change_type in raw_change_types):
        claims.append("do not claim hallucination reduction unless explicit evaluation evidence exists")
    return dedupe(claims)


def confidence_for_summary(
    summary: dict[str, Any],
    primary_type: str,
    direct_evidence: list[str],
    implementation_details: list[str],
) -> str:
    has_source = bool(safe_list(summary.get("source_chunk_ids")))
    has_file_or_symbol = bool(safe_list(summary.get("files_changed")) or safe_list(summary.get("symbols_changed")))
    known_type = primary_type != "unknown_update"
    if has_source and has_file_or_symbol and direct_evidence and known_type:
        return "high"
    if has_source and str(summary.get("what_changed") or "").strip() and (has_file_or_symbol or direct_evidence or implementation_details):
        return "medium"
    return "low"


def metric_support_for_summary(summary: dict[str, Any]) -> str:
    metadata = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
    declared = str(metadata.get("metric_support") or "").lower()
    if declared in {"explicit", "approximate"}:
        return declared
    return "none"


def evidence_card_quality_score(
    *,
    problem: str,
    mechanism: str,
    source_chunk_ids: list[str],
    implementation_details: list[str],
    safe_impact: str,
    resume_angle: str,
    allowed_claims: list[str],
) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    if problem and not is_generic_problem(problem):
        score += 2
    else:
        reasons.append("problem is weak or generic")
    if mechanism and not is_generic_mechanism(mechanism):
        score += 2
    else:
        score -= 3
        reasons.append("mechanism is weak or generic")
    if source_chunk_ids:
        score += 2
    else:
        reasons.append("missing source_chunk_ids")
    if implementation_details:
        score += 1
    else:
        reasons.append("missing implementation details")
    if safe_impact and not is_generic_impact(safe_impact):
        score += 1
    else:
        score -= 3
        reasons.append("safe impact is weak or generic")
    if resume_angle != "unknown":
        score += 1
    else:
        reasons.append("resume angle is unknown")
    if any(contains_unsupported_claim(value) for value in [problem, mechanism, safe_impact, *allowed_claims]):
        score -= 5
        reasons.append("unsupported metric or impact claim detected")
    return score, reasons


def unsupported_source_claims(summary: dict[str, Any]) -> list[str]:
    claims = []
    fields = [
        summary.get("what_changed"),
        *safe_list(summary.get("direct_code_evidence")),
        *safe_list(summary.get("uncertain_intent")),
    ]
    metadata = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
    fields.extend(safe_list(metadata.get("source_claim_text")))
    for value in fields:
        text = str(value or "")
        if contains_unsupported_claim(text):
            claims.append(truncate(text, DETAIL_CHARS))
    return dedupe(claims)


def sanitized_list(value: Any) -> list[str]:
    result = []
    for item in safe_list(value):
        if contains_unsupported_claim(item):
            continue
        clean = sanitize_fact(item, limit=DETAIL_CHARS)
        if clean:
            result.append(clean)
    return dedupe(result)


def sanitize_fact(value: str, limit: int = 260, terminal_period: bool = True) -> str:
    text = str(value or "")
    for pattern in UNSUPPORTED_CLAIM_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .,-")
    text = truncate(text, limit)
    if terminal_period and text and not text.endswith("."):
        text += "."
    return text


def contains_unsupported_claim(text: str) -> bool:
    return any(re.search(pattern, str(text or ""), flags=re.IGNORECASE) for pattern in UNSUPPORTED_CLAIM_PATTERNS)


def is_generic_problem(problem: str) -> bool:
    return problem.lower() in {
        "the change addressed a code/documentation area that needed more structured handling.",
        "",
    }


def is_generic_mechanism(mechanism: str) -> bool:
    return mechanism.lower() in {"updated source-backed evidence handling.", ""}


def is_generic_impact(safe_impact: str) -> bool:
    return safe_impact.lower() in {"improved structural handling of GitHub evidence evidence memory.", ""}


def safe_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def dedupe(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
