"""Deterministic Phase 2 capability fact extraction."""

from __future__ import annotations

import os
import re
from typing import Any

import evidence_memory


PHASE2_FLAG_ENV = "USE_GITHUB_CONTEXT_PHASE2"
ENABLED_VALUES = {"1", "true", "yes", "on"}
DETAIL_CHARS = 220
MAX_MECHANISMS = 8
MAX_ALLOWED_CLAIMS = 8

CAPABILITY_TYPES = {
    "evidence_memory",
    "github_context_persistence",
    "source_traceability",
    "debuggability",
    "validation_and_guardrails",
    "merge_quality_control",
    "chunking_and_index_preparation",
    "schema_and_storage_design",
    "testing_and_regression_safety",
    "configuration_safety",
    "api_inspection",
    "storage_idempotency",
    "unsupported_claim_boundary",
    "unknown",
}

RESUME_ANGLE_TO_CAPABILITY = {
    "evidence_memory": "evidence_memory",
    "github_context_persistence": "github_context_persistence",
    "source_traceability": "source_traceability",
    "debuggability": "debuggability",
    "validation_and_guardrails": "validation_and_guardrails",
    "merge_quality_control": "merge_quality_control",
    "chunking_and_index_preparation": "chunking_and_index_preparation",
    "schema_and_storage_design": "schema_and_storage_design",
    "testing_and_regression_safety": "testing_and_regression_safety",
    "configuration_safety": "configuration_safety",
    "api_inspection": "api_inspection",
}

DEFAULT_ALLOWED_CLAIM_BY_CAPABILITY = {
    "evidence_memory": "maintained source-traceable Phase 2 evidence records",
    "github_context_persistence": "persisted raw GitHub context into source-traceable records",
    "source_traceability": "preserved source evidence identifiers across Phase 2 evidence records",
    "debuggability": "added safe inspection surfaces for Phase 2 evidence state",
    "validation_and_guardrails": "added deterministic guards around unsupported or exaggerated claims",
    "merge_quality_control": "preserved mechanism-rich project evidence during deterministic merge ordering",
    "chunking_and_index_preparation": "converted raw GitHub context into bounded evidence chunks",
    "schema_and_storage_design": "implemented structured Phase 2 evidence memory records",
    "testing_and_regression_safety": "added targeted regression tests for Phase 2 memory behavior",
    "configuration_safety": "kept Phase 2 behavior behind a feature flag boundary",
    "api_inspection": "added safe Phase 2 status and preview endpoints",
    "storage_idempotency": "implemented stable IDs and upsert behavior for Phase 2 evidence records",
    "unsupported_claim_boundary": "preserved boundaries between allowed and forbidden evidence-backed claims",
}

GENERIC_FORBIDDEN_CLAIMS = [
    "do not claim quantified improvement without explicit metric evidence",
    "do not claim ATS score improvement",
    "do not claim interview success",
    "do not claim guaranteed factual correctness",
    "do not claim hallucinations were eliminated",
    "do not claim business impact without explicit evidence",
]

UNSUPPORTED_FACTUAL_PATTERNS = [
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


def phase2_enabled() -> bool:
    return str(os.getenv(PHASE2_FLAG_ENV, "1")).strip().lower() in ENABLED_VALUES


def extract_capability_facts_from_evidence_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_cards_by_capability(cards)
    facts: list[dict[str, Any]] = []
    for (project_id, capability_type), group_cards in grouped.items():
        fact = build_capability_fact_from_cards(project_id, capability_type, group_cards)
        if fact is not None:
            facts.append(fact)
    return facts


def build_capability_fact_from_cards(
    project_id: str,
    capability_type: str,
    cards: list[dict[str, Any]],
) -> dict[str, Any] | None:
    capability_type = capability_type if capability_type in CAPABILITY_TYPES else "unknown"
    source_evidence_ids = [
        str(card.get("evidence_id") or "")
        for card in cards
        if str(card.get("evidence_id") or "")
    ]
    source_evidence_ids = dedupe(source_evidence_ids)
    if not source_evidence_ids:
        return None

    mechanisms, unsupported_claims = mechanisms_from_cards(cards)
    allowed_resume_claims, unsupported_allowed = allowed_claims_from_cards(cards)
    if not allowed_resume_claims and capability_type in DEFAULT_ALLOWED_CLAIM_BY_CAPABILITY:
        allowed_resume_claims = [DEFAULT_ALLOWED_CLAIM_BY_CAPABILITY[capability_type]]
    unsupported_claims.extend(unsupported_allowed)
    unsupported_claims.extend(unsupported_source_claims_from_cards(cards))
    forbidden_claims = forbidden_claims_from_cards(cards, capability_type)
    metric_support = aggregate_metric_support(cards)
    if metric_support == "none":
        forbidden_claims.extend(
            [
                "do not use percentages or before/after numeric improvements",
                "do not claim quantified impact without explicit metric evidence",
            ]
        )
    if capability_type in {
        "validation_and_guardrails",
        "unsupported_claim_boundary",
        "evidence_memory",
        "source_traceability",
    }:
        forbidden_claims.append("do not claim hallucination reduction unless explicit evaluation evidence exists")

    confidence = confidence_for_cards(cards, source_evidence_ids, mechanisms, allowed_resume_claims)
    quality_score, skip_reasons = capability_quality_score(
        capability_type=capability_type,
        source_evidence_ids=source_evidence_ids,
        mechanisms=mechanisms,
        allowed_resume_claims=allowed_resume_claims,
        confidence=confidence,
    )
    if quality_score < 5:
        return None

    metadata = {
        "source": "phase2_capability_extractor",
        "quality_score": quality_score,
        "skip_reasons": skip_reasons,
        "supporting_evidence_cards": len(cards),
    }
    unsupported_claims = dedupe([claim for claim in unsupported_claims if claim])
    if unsupported_claims:
        metadata["unsupported_source_claims"] = unsupported_claims

    return evidence_memory.make_capability_fact(
        capability_id=evidence_memory.stable_record_id("capability_fact", [project_id, capability_type]),
        project_id=project_id,
        capability_type=capability_type,
        present=True,
        confidence=confidence,
        mechanisms=mechanisms,
        source_evidence_ids=source_evidence_ids,
        allowed_resume_claims=allowed_resume_claims,
        forbidden_claims=dedupe(forbidden_claims),
        metric_support=metric_support,
        metadata=metadata,
    )


def build_phase2_capability_facts(
    project_id: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    if not phase2_enabled():
        return {
            "enabled": False,
            "phase": "phase2",
            "project_id": project_id or None,
            "processed_evidence_cards": 0,
            "created_or_updated_capability_facts": 0,
            "skipped_groups": 0,
            "capability_facts_count": 0,
            "message": "GitHub context evidence memory is disabled.",
            "errors": [],
        }

    cards = (
        evidence_memory.read_records_by_project(evidence_memory.EVIDENCE_CARDS, project_id)
        if project_id
        else evidence_memory.read_records(evidence_memory.EVIDENCE_CARDS)
    )
    if limit is not None:
        cards = cards[: max(0, int(limit))]

    grouped = group_cards_by_capability(cards)
    skipped = []
    existing_ids = {
        str(record.get("capability_id") or "")
        for record in evidence_memory.read_records(evidence_memory.CAPABILITY_FACTS)
    }
    created_facts = 0
    updated_facts = 0
    created_or_updated = 0
    preview = []

    mapped_card_ids = {
        str(card.get("evidence_id") or "")
        for group_cards in grouped.values()
        for card in group_cards
    }
    for card in cards:
        card_id = str(card.get("evidence_id") or "")
        if card_id and card_id not in mapped_card_ids:
            skipped.append(
                {
                    "evidence_id": card_id,
                    "project_id": str(card.get("project_id") or ""),
                    "reason": "card did not map to a supported capability type",
                }
            )

    for (current_project_id, capability_type), group_cards in grouped.items():
        fact = build_capability_fact_from_cards(current_project_id, capability_type, group_cards)
        if fact is None:
            skipped.append(
                {
                    "project_id": current_project_id,
                    "capability_type": capability_type,
                    "reason": "capability group did not meet quality threshold",
                }
            )
            continue
        capability_id = str(fact.get("capability_id") or "")
        evidence_memory.upsert_capability_fact(fact)
        if capability_id in existing_ids:
            updated_facts += 1
        else:
            created_facts += 1
            existing_ids.add(capability_id)
        created_or_updated += 1
        preview.append(
            {
                "capability_id": capability_id,
                "project_id": fact["project_id"],
                "capability_type": fact["capability_type"],
                "confidence": fact["confidence"],
            }
        )

    counts = evidence_memory.get_phase2_memory_counts(project_id=project_id)
    return {
        "enabled": True,
        "phase": "phase2",
        "project_id": project_id or None,
        "processed_evidence_cards": len(cards),
        "created_capability_facts": created_facts,
        "updated_capability_facts": updated_facts,
        "created_or_updated_capability_facts": created_or_updated,
        "skipped_groups": len(skipped),
        "capability_facts_count": counts["capability_facts_count"],
        "message": "Phase 2 capability facts built successfully.",
        "capability_facts": preview,
        "skips": skipped,
        "errors": [],
    }


def group_cards_by_capability(cards: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for card in cards:
        project_id = str(card.get("project_id") or "")
        if not project_id:
            continue
        for capability_type in capability_types_for_card(card):
            grouped.setdefault((project_id, capability_type), []).append(card)
    return grouped


def capability_types_for_card(card: dict[str, Any]) -> list[str]:
    capability_types: list[str] = []
    resume_angle = str(card.get("resume_angle") or "")
    mapped = RESUME_ANGLE_TO_CAPABILITY.get(resume_angle)
    if mapped:
        capability_types.append(mapped)

    haystack = card_text(card).lower()
    if any(term in haystack for term in ["stable id", "stable ids", "hash", "upsert", "jsonl"]):
        capability_types.append("storage_idempotency")
    if "unsupported" in haystack or "forbidden claim" in haystack or "do not claim hallucination" in haystack:
        capability_types.append("unsupported_claim_boundary")
    if "traceability" in haystack or "source-traceable" in haystack or "source traceability" in haystack:
        capability_types.append("source_traceability")
    if "debug" in haystack or "inspectability" in haystack or "inspection" in haystack:
        capability_types.append("debuggability")

    return [capability for capability in dedupe(capability_types) if capability != "unknown"]


def card_text(card: dict[str, Any]) -> str:
    parts = [
        card.get("problem"),
        card.get("mechanism"),
        card.get("safe_impact"),
        card.get("resume_angle"),
        *safe_list(card.get("implementation_details")),
        *safe_list(card.get("allowed_claims")),
        *safe_list(card.get("forbidden_claims")),
    ]
    return " ".join(str(part or "") for part in parts)


def mechanisms_from_cards(cards: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    mechanisms: list[str] = []
    unsupported_claims: list[str] = []
    for card in cards:
        values = [card.get("mechanism"), *safe_list(card.get("implementation_details")), *safe_list(card.get("allowed_claims"))]
        for value in values:
            text = str(value or "")
            if contains_unsupported_factual_claim(text):
                unsupported_claims.append(truncate(text, DETAIL_CHARS))
                continue
            sanitized = sanitize_factual_claim(text)
            if sanitized and not is_generic_text(sanitized):
                mechanisms.append(sanitized)
    return dedupe(mechanisms)[:MAX_MECHANISMS], unsupported_claims


def allowed_claims_from_cards(cards: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    allowed: list[str] = []
    unsupported_claims: list[str] = []
    for card in cards:
        for claim in safe_list(card.get("allowed_claims")):
            if contains_unsupported_factual_claim(claim):
                unsupported_claims.append(truncate(claim, DETAIL_CHARS))
                continue
            sanitized = sanitize_factual_claim(claim, terminal_period=False)
            if sanitized and not is_generic_text(sanitized):
                allowed.append(sanitized)
    return dedupe(allowed)[:MAX_ALLOWED_CLAIMS], unsupported_claims


def forbidden_claims_from_cards(cards: list[dict[str, Any]], capability_type: str) -> list[str]:
    claims = list(GENERIC_FORBIDDEN_CLAIMS)
    for card in cards:
        claims.extend(safe_list(card.get("forbidden_claims")))
    if capability_type == "unsupported_claim_boundary":
        claims.append("do not convert forbidden claim boundaries into factual resume claims")
    return dedupe([claim for claim in claims if claim])


def aggregate_metric_support(cards: list[dict[str, Any]]) -> str:
    supports = {str(card.get("metric_support") or "none").lower() for card in cards}
    if "explicit" in supports:
        return "explicit"
    if "approximate" in supports:
        return "approximate"
    return "none"


def confidence_for_cards(
    cards: list[dict[str, Any]],
    source_evidence_ids: list[str],
    mechanisms: list[str],
    allowed_resume_claims: list[str],
) -> str:
    confidences = [str(card.get("confidence") or "low").lower() for card in cards]
    high_count = sum(1 for confidence in confidences if confidence == "high")
    has_source = bool(source_evidence_ids)
    if has_source and high_count >= 2:
        return "high"
    if has_source and high_count >= 1 and mechanisms and allowed_resume_claims:
        return "high"
    if has_source and any(confidence in {"medium", "high"} for confidence in confidences) and mechanisms:
        return "medium"
    return "low"


def capability_quality_score(
    *,
    capability_type: str,
    source_evidence_ids: list[str],
    mechanisms: list[str],
    allowed_resume_claims: list[str],
    confidence: str,
) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    if source_evidence_ids:
        score += 2
    else:
        reasons.append("missing source evidence ids")
    if mechanisms and not all(is_generic_text(mechanism) for mechanism in mechanisms):
        score += 2
    else:
        score -= 3
        reasons.append("mechanisms are generic or missing")
    if allowed_resume_claims and not all(is_generic_text(claim) for claim in allowed_resume_claims):
        score += 1
    else:
        score -= 3
        reasons.append("allowed claims are generic or missing")
    if confidence in {"medium", "high"}:
        score += 1
    else:
        reasons.append("confidence is low")
    if capability_type != "unknown":
        score += 1
    else:
        reasons.append("capability type is unknown")
    if any(contains_unsupported_factual_claim(value) for value in [*mechanisms, *allowed_resume_claims]):
        score -= 5
        reasons.append("unsupported factual metric or impact claim detected")
    return score, reasons


def unsupported_source_claims_from_cards(cards: list[dict[str, Any]]) -> list[str]:
    claims: list[str] = []
    for card in cards:
        metadata = card.get("metadata") if isinstance(card.get("metadata"), dict) else {}
        for value in [
            card.get("mechanism"),
            card.get("safe_impact"),
            *safe_list(card.get("allowed_claims")),
            *safe_list(metadata.get("unsupported_source_claims")),
        ]:
            text = str(value or "")
            if contains_unsupported_factual_claim(text):
                claims.append(truncate(text, DETAIL_CHARS))
    return dedupe(claims)


def sanitize_factual_claim(value: str, terminal_period: bool = True) -> str:
    text = str(value or "")
    for pattern in UNSUPPORTED_FACTUAL_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .,-")
    text = truncate(text, DETAIL_CHARS)
    if terminal_period and text and not text.endswith("."):
        text += "."
    return text


def contains_unsupported_factual_claim(text: str) -> bool:
    return any(re.search(pattern, str(text or ""), flags=re.IGNORECASE) for pattern in UNSUPPORTED_FACTUAL_PATTERNS)


def is_generic_text(text: str) -> bool:
    normalized = str(text or "").strip().lower().rstrip(".")
    return normalized in {
        "",
        "updated code",
        "updated source-backed evidence handling",
        "maintained source-traceable phase 2 evidence records",
        "unknown",
    }


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
