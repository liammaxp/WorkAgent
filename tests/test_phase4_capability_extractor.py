from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

from backend.phase4_capability_extractor import (
    MAX_DECISION_SAMPLES,
    SIGNAL_EXTRACTION_RULES,
    Phase4CapabilityExtractionReport,
    Phase4SignalExtractionRule,
    classify_phase4_source_category,
    extract_phase4_capabilities_by_project,
    extract_phase4_fact_signals,
    extract_phase4_fact_signals_many,
    extract_phase4_project_capabilities,
    list_phase4_signal_extraction_rules,
    validate_phase4_signal_extraction_rules,
)
from backend.phase4_capability_taxonomy import (
    CAPABILITY_ALIASES,
    CAPABILITY_TAXONOMY,
    list_phase4_signal_identifiers,
)
from backend.phase4_evidence_normalizer import dedupe_phase4_inputs
from backend.phase4_evidence_scoring import score_phase4_evidence_facts
from backend.phase4_evidence_synthesizer import synthesize_phase4_evidence_facts
from backend.phase4_input_adapter import load_phase4_inputs
from backend.phase4_models import (
    Confidence,
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    Phase4CapabilityFact,
    Phase4EvidenceFact,
    Phase4SourceRef,
)


def source(
    source_id: str = "src-1",
    *,
    project_id: str = "workagent",
    source_type: str = "phase3_evidence_card",
    content_seed: str | None = None,
    metadata: dict | None = None,
    file_path: str | None = "backend/module.py",
    symbol: str | None = "build_result",
) -> Phase4SourceRef:
    seed = content_seed if content_seed is not None else source_id
    return Phase4SourceRef(
        source_type=source_type,
        source_id=source_id,
        project_id=project_id,
        content_hash=hashlib.sha256(seed.encode("utf-8")).hexdigest(),
        file_path=file_path,
        symbol=symbol,
        metadata=metadata or {},
    )


def scored_fact(
    mechanism: str,
    *,
    project_id: str = "workagent",
    evidence_type: EvidenceType = EvidenceType.ARCHITECTURE,
    source_type: str = "phase3_evidence_card",
    source_id: str = "src-1",
    content_seed: str | None = None,
    status: EvidenceStatus = EvidenceStatus.ACCEPTED,
    quality_score: int = 80,
    confidence: Confidence = Confidence.MEDIUM,
    metric_support: MetricSupport = MetricSupport.NONE,
    implementation: list[str] | None = None,
    safe_impact: list[str] | None = None,
    technical_tags: list[str] | None = None,
    metadata: dict | None = None,
    evidence_fact_id: str = "",
) -> Phase4EvidenceFact:
    return Phase4EvidenceFact(
        project_id=project_id,
        mechanism=mechanism,
        implementation=implementation or ["backend/module.py::build_result"],
        safe_impact=safe_impact or [],
        evidence_type=evidence_type,
        confidence=confidence,
        metric_support=metric_support,
        technical_tags=technical_tags or [],
        status=status,
        quality_score=quality_score,
        quality_breakdown={
            "recommended_status": status.value,
            "recommended_quality_band": "high_value" if quality_score >= 80 else "supporting_value",
        },
        source_refs=[source(
            source_id,
            project_id=project_id,
            source_type=source_type,
            content_seed=content_seed,
            metadata=metadata,
        )],
        evidence_fact_id=evidence_fact_id,
    )


def signals(item: Phase4EvidenceFact) -> set[str]:
    return set(extract_phase4_fact_signals(item).signals)


def capability_types(items: list[Phase4CapabilityFact]) -> set[str]:
    return {item.capability_type for item in items}


def extract_for_project(*facts: Phase4EvidenceFact):
    return extract_phase4_project_capabilities("workagent", facts)


@pytest.mark.parametrize(
    ("mechanism", "expected"),
    [
        ("reranking candidates", "reranking"),
        ("candidate filtering", "candidate_filtering"),
        ("canonical JSON", "canonical_serialization"),
        ("stable ID", "stable_identity"),
        ("exact duplicate detection", "exact_deduplication"),
        ("integrity conflict detection", "integrity_conflict_detection"),
        ("atomic write", "atomic_persistence"),
        ("schema versioning", "schema_versioning"),
        ("diff processing", "diff_processing"),
        ("cache reuse", "cache_reuse"),
        ("prompt constraint", "prompt_constraint"),
        ("RAG", "rag_terminology"),
        ("unsupported claim blocking", "unsupported_claim_blocking"),
        ("LaTeX compilation check", "latex_compile_check"),
        ("LaTeX repair", "latex_repair"),
        ("template pollution detection", "template_pollution_detection"),
        ("safe degradation", "safe_degradation"),
        ("regression test", "regression_testing"),
    ],
)
def test_explicit_named_signal_rules(mechanism, expected):
    extraction = extract_phase4_fact_signals(scored_fact(mechanism))
    assert expected in extraction.signals
    assert all(binding.rule_id for binding in extraction.bindings)


def test_retrieval_evidence_type_emits_only_base_retrieval_signal():
    extraction = extract_phase4_fact_signals(scored_fact("retrieval", evidence_type=EvidenceType.RETRIEVAL))
    assert extraction.signals == ("retrieval",)
    assert extraction.bindings[0].matched_field == "evidence_type"
    assert "reranking" not in extraction.signals


@pytest.mark.parametrize(
    ("mechanism", "forbidden"),
    [
        ("diff", {"diff_processing", "diff_only_analysis", "repeated_input_avoidance"}),
        ("cache", {"cache_reuse", "repeated_input_avoidance"}),
        ("validation", {"claim_validation", "latex_validation", "schema_validation"}),
        ("memory", {"project_memory_read", "project_memory_write", "persistent_storage"}),
        ("sort", {"deterministic_ordering"}),
        ("prompt constraint", {"claim_validation", "factuality_evaluation", "source_grounding"}),
        ("RAG", {"source_grounding", "unsupported_claim_blocking"}),
    ],
)
def test_loose_or_contextual_words_do_not_emit_unrelated_signals(mechanism, forbidden):
    assert not (signals(scored_fact(mechanism)) & forbidden)


def test_generic_technology_tags_do_not_become_signals():
    item = scored_fact(
        "implemented a bounded component",
        technical_tags=["Python", "React", "FastAPI", "SQLite", "Chroma", "LaTeX", "LLM", "backend"],
    )
    assert not signals(item)


def test_exact_registered_technical_tag_may_emit_only_that_signal():
    item = scored_fact("implemented a bounded component", technical_tags=["candidate_filtering"])
    assert extract_phase4_fact_signals(item).signals == ("candidate_filtering",)


def test_declared_signal_metadata_is_bounded_to_registry():
    valid = scored_fact("implemented a bounded component", metadata={"signals": ["retrieval"]})
    invalid = scored_fact("implemented a bounded component", metadata={"signals": ["unknown_signal"]})
    assert "retrieval" in signals(valid)
    assert "unsupported_signal_candidate" in extract_phase4_fact_signals(invalid).rejected_candidates


def test_schema_version_provenance_metadata_emits_only_contextual_schema_signal():
    item = scored_fact(
        "implemented a bounded component",
        source_type="project_memory",
        status=EvidenceStatus.SUPPORTING,
        evidence_type=EvidenceType.UNKNOWN,
        metadata={"schema_version": 1},
    )
    extraction = extract_phase4_fact_signals(item)
    assert extraction.signals == ("schema_versioning",)
    assert extraction.bindings[0].matched_field == "source_metadata"
    capabilities, _ = extract_for_project(item)
    assert not capabilities


@pytest.mark.parametrize("mechanism", ["API route", "api_route_update", "api-route-update"])
def test_explicit_api_route_variants_emit_backend_route(mechanism):
    extraction = extract_phase4_fact_signals(scored_fact(mechanism))
    assert extraction.signals == ("backend_route",)
    assert extraction.bindings[0].matched_field == "mechanism"


def test_non_registry_api_route_tag_alone_does_not_become_backend_route():
    item = scored_fact("implemented a bounded component", technical_tags=["api_route_update"])
    assert "backend_route" not in signals(item)


def test_backend_route_alone_does_not_prove_frontend_backend_integration():
    capabilities, _ = extract_for_project(scored_fact("api_route_update"))
    assert "frontend_backend_integration" not in capability_types(capabilities)


def test_unknown_rule_signal_is_rejected_before_extraction():
    rule = Phase4SignalExtractionRule(
        rule_id="unknown_signal_rule",
        signal="not_in_taxonomy",
        allowed_fields=("mechanism",),
        exact_values=("anything",),
    )
    assert validate_phase4_signal_extraction_rules((rule,)) == ("rule[0]:unknown_signal",)
    with pytest.raises(ValueError, match="unknown_signal"):
        extract_phase4_fact_signals(scored_fact("anything"), rules=(rule,))


def test_rule_table_is_immutable_complete_and_deterministically_ordered():
    rules = list_phase4_signal_extraction_rules()
    assert rules is SIGNAL_EXTRACTION_RULES
    assert validate_phase4_signal_extraction_rules() == ()
    assert [item.rule_id for item in rules] == sorted(item.rule_id for item in rules)
    assert {item.signal for item in rules} == set(list_phase4_signal_identifiers())
    with pytest.raises(Exception):
        rules[0].signal = "retrieval"


def test_fact_signal_extraction_is_deterministic_and_does_not_mutate_fact():
    item = scored_fact("canonical serialization and exact deduplication")
    before = item.to_json()
    first = extract_phase4_fact_signals(item)
    second = extract_phase4_fact_signals(item)
    assert first == second
    assert item.to_json() == before


def test_many_signal_extraction_order_and_report_are_input_order_independent():
    first = scored_fact("canonical serialization", source_id="a")
    second = scored_fact("exact deduplication", source_id="b")
    output_a, report_a = extract_phase4_fact_signals_many([first, second])
    output_b, report_b = extract_phase4_fact_signals_many([second, first])
    assert output_a == output_b
    assert report_a == report_b
    assert report_a.fact_count == 2
    assert report_a.signal_binding_count == 2


@pytest.mark.parametrize(
    "phrase",
    [
        "changed file detection",
        "prior state comparison",
        "metric support validation",
        "source attribution",
        "response schema",
        "frontend state handling",
        "failure case coverage",
        "stage IO contract",
    ],
)
def test_additional_canonical_signal_phrases_are_explicitly_supported(phrase):
    extraction = extract_phase4_fact_signals(scored_fact(phrase))
    assert len(extraction.signals) == 1


@pytest.mark.parametrize(
    ("source_type", "status", "expected"),
    [
        ("phase2_evidence_chunk", EvidenceStatus.ACCEPTED, "direct_evidence"),
        ("phase3_evidence_card", EvidenceStatus.ACCEPTED, "direct_evidence"),
        ("project_memory", EvidenceStatus.SUPPORTING, "project_context"),
        ("phase2_capability_fact", EvidenceStatus.WEAK, "capability_context"),
        ("unknown_safe_source", EvidenceStatus.ACCEPTED, "other"),
    ],
)
def test_source_category_classification(source_type, status, expected):
    item = scored_fact("stable identity", source_type=source_type, status=status)
    assert classify_phase4_source_category(item) == expected


@pytest.mark.parametrize("status", [EvidenceStatus.WEAK, EvidenceStatus.REJECTED])
def test_weak_or_rejected_fact_cannot_satisfy_required_group(status):
    item = scored_fact(
        "canonical serialization with exact deduplication",
        status=status,
    )
    capabilities, _ = extract_for_project(item)
    assert "deterministic_evidence_normalization" not in capability_types(capabilities)


def test_low_quality_fact_cannot_satisfy_taxonomy_threshold():
    item = scored_fact("canonical serialization with exact deduplication", quality_score=59)
    capabilities, report = extract_for_project(item)
    assert "deterministic_evidence_normalization" not in capability_types(capabilities)
    assert report.insufficient_quality_count > 0


def test_project_context_cannot_satisfy_direct_fact_minimum():
    item = scored_fact(
        "canonical serialization with exact deduplication",
        source_type="project_memory",
        status=EvidenceStatus.SUPPORTING,
        evidence_type=EvidenceType.UNKNOWN,
    )
    capabilities, report = extract_for_project(item)
    assert not capabilities
    assert report.contextual_only_rejected_count > 0
    assert report.insufficient_direct_fact_count > 0


def test_legacy_capability_context_does_not_count_as_direct():
    item = scored_fact(
        "canonical serialization with exact deduplication",
        source_type="phase3_capability_fact",
        status=EvidenceStatus.WEAK,
        technical_tags=["testing_and_regression_safety"],
    )
    capabilities, _ = extract_for_project(item)
    assert not capabilities
    assert "legacy_capability_alias" in extract_phase4_fact_signals(item).rejected_candidates


def test_duplicate_lineage_is_not_two_independent_direct_facts():
    shared = source("same")
    first = replace(
        scored_fact("source grounding and unsupported claim blocking", evidence_type=EvidenceType.VALIDATION),
        source_refs=[shared],
    )
    second = replace(
        scored_fact("structured output validation", source_id="different", evidence_type=EvidenceType.VALIDATION),
        source_refs=[shared],
    )
    capabilities, report = extract_for_project(first, second)
    assert "llm_reliability" not in capability_types(capabilities)
    assert report.insufficient_direct_fact_count > 0


def test_duplicate_source_refs_do_not_create_independent_direct_facts():
    shared = source("same-ref")
    first = scored_fact(
        "source grounding with unsupported claim blocking",
        evidence_type=EvidenceType.VALIDATION,
        quality_score=100,
    )
    second = scored_fact(
        "structured output validation",
        evidence_type=EvidenceType.VALIDATION,
        source_id="second",
        quality_score=100,
    )
    first = replace(first, source_refs=[shared, shared])
    second = replace(second, source_refs=[shared])
    capabilities, report = extract_for_project(first, second)
    assert "llm_reliability" not in capability_types(capabilities)
    assert report.insufficient_direct_fact_count > 0


def test_and_of_or_requires_every_group():
    item = scored_fact("canonical serialization")
    capabilities, report = extract_for_project(item)
    assert "deterministic_evidence_normalization" not in capability_types(capabilities)
    decision = next(item for item in report.decisions if item.capability_type == "deterministic_evidence_normalization")
    assert decision.missing_required_group_indexes == (1,)


def test_one_or_alternative_per_group_is_sufficient():
    item = scored_fact("stable identity with field normalization")
    capabilities, _ = extract_for_project(item)
    assert "deterministic_evidence_normalization" in capability_types(capabilities)


def test_supporting_signal_does_not_replace_required_group():
    item = scored_fact("integrity conflict detection with schema validation")
    capabilities, _ = extract_for_project(item)
    assert "deterministic_evidence_normalization" not in capability_types(capabilities)


def test_one_fact_can_explicitly_satisfy_multiple_groups_but_counts_once():
    item = scored_fact("canonical serialization with exact deduplication")
    capabilities, _ = extract_for_project(item)
    capability = next(item for item in capabilities if item.capability_type == "deterministic_evidence_normalization")
    assert capability.source_evidence_fact_ids == [item.evidence_fact_id]
    assert capability.confidence is Confidence.MEDIUM


def test_group_proof_retains_exact_supporting_fact_ids():
    retrieval = scored_fact("retrieval", evidence_type=EvidenceType.RETRIEVAL, source_id="retrieval")
    filtering = scored_fact("candidate filtering", evidence_type=EvidenceType.RETRIEVAL, source_id="filtering")
    capabilities, _ = extract_for_project(retrieval, filtering)
    capability = next(item for item in capabilities if item.capability_type == "retrieval_and_reranking")
    assert set(capability.source_evidence_fact_ids) == {
        retrieval.evidence_fact_id,
        filtering.evidence_fact_id,
    }


@pytest.mark.parametrize(
    ("expected", "facts"),
    [
        ("structured_evidence_extraction", [
            scored_fact("structured extraction with schema validation", source_id="structured"),
        ]),
        ("deterministic_evidence_normalization", [
            scored_fact("canonical serialization with exact deduplication", source_id="normalization"),
        ]),
        ("retrieval_and_reranking", [
            scored_fact("candidate filtering", evidence_type=EvidenceType.RETRIEVAL, source_id="retrieval"),
        ]),
        ("frontend_backend_integration", [
            scored_fact("backend route with response schema", evidence_type=EvidenceType.INTEGRATION, source_id="backend"),
            scored_fact("frontend API call with frontend state handling", evidence_type=EvidenceType.INTEGRATION, source_id="frontend"),
        ]),
        ("failure_recovery", [
            scored_fact("failure detection with retry", evidence_type=EvidenceType.FAILURE_RECOVERY, source_id="recovery"),
        ]),
        ("workflow_orchestration", [
            scored_fact("state transition with branching gate", evidence_type=EvidenceType.WORKFLOW, source_id="workflow"),
        ]),
        ("test_and_regression_hardening", [
            scored_fact("regression testing with failure case coverage", evidence_type=EvidenceType.TESTING, source_id="testing"),
        ]),
    ],
)
def test_qualifying_capability_fixtures_emit_canonical_type(expected, facts):
    capabilities, _ = extract_for_project(*facts)
    assert expected in capability_types(capabilities)
    assert all(item.capability_type in CAPABILITY_TAXONOMY for item in capabilities)


@pytest.mark.parametrize(
    ("mechanism", "evidence_type", "forbidden"),
    [
        ("retrieval", EvidenceType.RETRIEVAL, "retrieval_and_reranking"),
        ("persistent storage with schema versioning", EvidenceType.DATA_PERSISTENCE, "project_memory_management"),
        ("structured document assembly with section ordering", EvidenceType.WORKFLOW, "latex_validation_and_repair"),
        ("quality dimensions", EvidenceType.VALIDATION, "claim_validation"),
        ("source grounding with unsupported claim blocking", EvidenceType.VALIDATION, "llm_reliability"),
        ("failure detection", EvidenceType.FAILURE_RECOVERY, "failure_recovery"),
        ("workflow staging", EvidenceType.WORKFLOW, "workflow_orchestration"),
        ("regression testing", EvidenceType.TESTING, "test_and_regression_hardening"),
    ],
)
def test_single_partial_behavior_does_not_imply_broader_capability(mechanism, evidence_type, forbidden):
    capabilities, _ = extract_for_project(scored_fact(mechanism, evidence_type=evidence_type))
    assert forbidden not in capability_types(capabilities)


def test_alias_presence_never_bypasses_canonical_evidence_requirements():
    item = scored_fact(
        "implemented a bounded component",
        source_type="phase2_capability_fact",
        status=EvidenceStatus.WEAK,
        technical_tags=["token_or_cost_reduction"],
    )
    capabilities, _ = extract_for_project(item)
    assert "token_or_context_efficiency" not in capability_types(capabilities)
    assert all(item.capability_type not in CAPABILITY_ALIASES for item in capabilities)


def test_project_id_is_preserved_exactly_and_aliases_are_not_inferred():
    first = scored_fact("canonical serialization with exact deduplication", project_id="example/workagent", source_id="a")
    second = scored_fact("canonical serialization with exact deduplication", project_id="liammaxp/WorkAgent", source_id="b")
    grouped, report = extract_phase4_capabilities_by_project([first, second])
    assert set(grouped) == {"example/workagent", "liammaxp/WorkAgent"}
    assert all(item.project_id == project for project, items in grouped.items() for item in items)
    assert report.project_count == 2


def test_project_extractor_reports_and_ignores_mismatched_project():
    matching = scored_fact("canonical serialization with exact deduplication", project_id="workagent")
    other = scored_fact("candidate filtering", project_id="other", evidence_type=EvidenceType.RETRIEVAL)
    capabilities, report = extract_phase4_project_capabilities("workagent", [matching, other])
    assert capability_types(capabilities) == {"deterministic_evidence_normalization"}
    assert report.project_mismatch_count == 1


def test_token_efficiency_requires_explicit_reuse_and_avoidance():
    allowed = scored_fact(
        "cache reuse with repeated input avoidance",
        evidence_type=EvidenceType.OPTIMIZATION,
        quality_score=70,
    )
    capabilities, _ = extract_for_project(allowed)
    assert "token_or_context_efficiency" in capability_types(capabilities)


@pytest.mark.parametrize("mechanism", ["diff processing", "cache", "chunking", "retrieval", "stable identity"])
def test_token_efficiency_is_not_emitted_from_loose_or_partial_evidence(mechanism):
    capabilities, _ = extract_for_project(scored_fact(
        mechanism,
        evidence_type=EvidenceType.OPTIMIZATION,
        quality_score=100,
    ))
    assert "token_or_context_efficiency" not in capability_types(capabilities)


def test_token_efficiency_threshold_is_not_lowered():
    item = scored_fact(
        "cache reuse with repeated input avoidance",
        evidence_type=EvidenceType.OPTIMIZATION,
        quality_score=69,
    )
    capabilities, report = extract_for_project(item)
    assert "token_or_context_efficiency" not in capability_types(capabilities)
    assert report.insufficient_quality_count > 0


def test_llm_reliability_requires_three_control_groups_and_two_independent_facts():
    grounding = scored_fact(
        "source grounding with unsupported claim blocking",
        evidence_type=EvidenceType.VALIDATION,
        source_id="grounding",
        quality_score=80,
    )
    safeguard = scored_fact(
        "structured output validation",
        evidence_type=EvidenceType.VALIDATION,
        source_id="safeguard",
        quality_score=80,
    )
    capabilities, _ = extract_for_project(grounding, safeguard)
    assert "llm_reliability" in capability_types(capabilities)


@pytest.mark.parametrize(
    "facts",
    [
        [scored_fact("retrieval", evidence_type=EvidenceType.RETRIEVAL)],
        [scored_fact("prompt constraint", evidence_type=EvidenceType.VALIDATION)],
        [scored_fact("structured output validation", evidence_type=EvidenceType.VALIDATION)],
        [scored_fact("source grounding with unsupported claim blocking", evidence_type=EvidenceType.VALIDATION)],
    ],
)
def test_partial_llm_controls_do_not_emit_llm_reliability(facts):
    capabilities, _ = extract_for_project(*facts)
    assert "llm_reliability" not in capability_types(capabilities)


def test_contextual_evidence_cannot_prove_high_risk_capability():
    token = scored_fact(
        "cache reuse with repeated input avoidance",
        source_type="project_memory",
        status=EvidenceStatus.SUPPORTING,
        evidence_type=EvidenceType.UNKNOWN,
        quality_score=100,
    )
    capabilities, report = extract_for_project(token)
    assert "token_or_context_efficiency" not in capability_types(capabilities)
    assert report.high_risk_blocked_count > 0


def test_minimum_proof_is_medium_and_two_strong_independent_facts_may_be_high():
    minimum = scored_fact("canonical serialization with exact deduplication", quality_score=80)
    capabilities, _ = extract_for_project(minimum)
    normal = next(item for item in capabilities if item.capability_type == "deterministic_evidence_normalization")
    assert normal.confidence is Confidence.MEDIUM

    second = scored_fact("stable identity with field normalization", source_id="second", quality_score=80)
    capabilities, _ = extract_for_project(minimum, second)
    normal = next(item for item in capabilities if item.capability_type == "deterministic_evidence_normalization")
    assert normal.confidence is Confidence.HIGH
    assert set(normal.source_evidence_fact_ids) == {minimum.evidence_fact_id, second.evidence_fact_id}


def test_repeated_fact_or_duplicate_refs_do_not_raise_confidence():
    item = scored_fact("canonical serialization with exact deduplication", quality_score=100)
    capabilities, report = extract_for_project(item, item)
    normal = next(item for item in capabilities if item.capability_type == "deterministic_evidence_normalization")
    assert normal.confidence is Confidence.MEDIUM
    assert report.duplicate_fact_binding_count == 1


def test_capability_fields_are_conservative_and_use_only_selected_facts():
    retrieval = scored_fact(
        "candidate filtering",
        evidence_type=EvidenceType.RETRIEVAL,
        technical_tags=["Python", "candidate_filtering"],
        metric_support=MetricSupport.EXPLICIT,
        safe_impact=["Reduced candidate input by 20%."],
        source_id="retrieval",
    )
    unrelated = scored_fact(
        "schema versioning",
        evidence_type=EvidenceType.DATA_PERSISTENCE,
        technical_tags=["React"],
        source_id="unrelated",
    )
    capabilities, _ = extract_for_project(retrieval, unrelated)
    result = next(item for item in capabilities if item.capability_type == "retrieval_and_reranking")
    assert result.present is True
    assert result.allowed_resume_claims == []
    assert result.forbidden_claims == []
    assert result.source_evidence_fact_ids == [retrieval.evidence_fact_id]
    assert result.technical_tags == ["Python"]
    assert result.metric_support is MetricSupport.EXPLICIT


def test_metric_support_uses_weakest_selected_core_fact():
    first = scored_fact(
        "retrieval",
        evidence_type=EvidenceType.RETRIEVAL,
        metric_support=MetricSupport.EXPLICIT,
        safe_impact=["Reduced candidate input by 20%."],
        source_id="first",
    )
    second = scored_fact(
        "candidate filtering",
        evidence_type=EvidenceType.RETRIEVAL,
        metric_support=MetricSupport.NONE,
        safe_impact=["Reduced candidate processing by 10%."],
        source_id="second",
    )
    capabilities, _ = extract_for_project(first, second)
    result = next(item for item in capabilities if item.capability_type == "retrieval_and_reranking")
    assert result.metric_support is MetricSupport.NONE


def test_capability_id_is_stable_and_input_order_independent():
    first = scored_fact("retrieval", evidence_type=EvidenceType.RETRIEVAL, source_id="first")
    second = scored_fact("candidate filtering", evidence_type=EvidenceType.RETRIEVAL, source_id="second")
    caps_a, report_a = extract_for_project(first, second)
    caps_b, report_b = extract_for_project(second, first)
    assert [item.to_dict() for item in caps_a] == [item.to_dict() for item in caps_b]
    assert report_a == report_b


def test_repeated_capability_extraction_is_idempotent():
    facts = [
        scored_fact("retrieval", evidence_type=EvidenceType.RETRIEVAL, source_id="first"),
        scored_fact("candidate filtering", evidence_type=EvidenceType.RETRIEVAL, source_id="second"),
    ]
    first = extract_phase4_capabilities_by_project(facts)
    second = extract_phase4_capabilities_by_project(facts)
    assert first == second


def test_failed_candidates_are_reported_but_not_emitted_as_absent_facts():
    capabilities, report = extract_for_project(scored_fact("retrieval", evidence_type=EvidenceType.RETRIEVAL))
    assert all(item.present for item in capabilities)
    decision = next(item for item in report.decisions if item.capability_type == "retrieval_and_reranking")
    assert decision.decision_code == "missing_required_groups"


def test_shared_evidence_can_support_overlaps_only_when_each_definition_passes():
    item = scored_fact(
        "structured extraction with schema validation and canonical serialization with exact deduplication",
        quality_score=80,
    )
    capabilities, report = extract_for_project(item)
    assert {
        "structured_evidence_extraction",
        "deterministic_evidence_normalization",
    } <= capability_types(capabilities)
    assert report.overlap_count == 1
    assert report.overlaps[0].shared_evidence_fact_ids == (item.evidence_fact_id,)


def test_one_valid_capability_never_automatically_implies_overlap_partner():
    item = scored_fact("structured extraction with schema validation")
    capabilities, _ = extract_for_project(item)
    assert "structured_evidence_extraction" in capability_types(capabilities)
    assert "deterministic_evidence_normalization" not in capability_types(capabilities)


def test_same_fact_id_with_different_safe_payload_blocks_affected_capability():
    first = scored_fact(
        "canonical serialization",
        source_id="first",
        evidence_fact_id="p4ef_conflict",
    )
    second = scored_fact(
        "exact deduplication",
        source_id="second",
        evidence_fact_id="p4ef_conflict",
    )
    capabilities, report = extract_for_project(first, second)
    assert "deterministic_evidence_normalization" not in capability_types(capabilities)
    assert report.conflict_count == 1
    assert report.conflicts[0].conflict_code == "same_fact_id_different_payload"


def test_same_logical_lineage_with_different_hash_and_signals_is_conflict():
    first = scored_fact(
        "canonical serialization",
        source_id="same",
        content_seed="version-a",
    )
    second = scored_fact(
        "exact deduplication",
        source_id="same",
        content_seed="version-b",
    )
    capabilities, report = extract_for_project(first, second)
    assert "deterministic_evidence_normalization" not in capability_types(capabilities)
    assert any(item.conflict_code == "same_lineage_conflicting_signals" for item in report.conflicts)


def test_duplicate_fact_id_across_projects_is_reported_and_never_combined():
    first = scored_fact(
        "canonical serialization with exact deduplication",
        project_id="one",
        evidence_fact_id="p4ef_shared",
    )
    second = scored_fact(
        "canonical serialization with exact deduplication",
        project_id="two",
        evidence_fact_id="p4ef_shared",
    )
    grouped, report = extract_phase4_capabilities_by_project([first, second])
    assert set(grouped) == {"one", "two"}
    assert report.conflict_count == 2
    assert all(not values for values in grouped.values())


def test_capability_extraction_does_not_mutate_evidence_facts():
    facts = [
        scored_fact("retrieval", evidence_type=EvidenceType.RETRIEVAL, source_id="first"),
        scored_fact("candidate filtering", evidence_type=EvidenceType.RETRIEVAL, source_id="second"),
    ]
    before = [item.to_json() for item in facts]
    extract_phase4_capabilities_by_project(facts)
    assert [item.to_json() for item in facts] == before


def test_reports_are_bounded_deterministic_and_contain_no_evidence_body_text():
    marker = "RAW_EVIDENCE_SENTINEL"
    facts = [
        scored_fact(f"{marker} canonical serialization", project_id=f"project-{index}", source_id=str(index))
        for index in range(12)
    ]
    _grouped, report = extract_phase4_capabilities_by_project(reversed(facts))
    encoded = json.dumps(report.to_dict(), sort_keys=True)
    assert marker not in encoded
    assert len(report.decisions) <= MAX_DECISION_SAMPLES
    assert report.project_count == 12


def test_bounded_decision_sampling_is_input_order_independent():
    facts = [
        scored_fact("canonical serialization", project_id=f"project-{index:03d}", source_id=str(index))
        for index in range(12)
    ]
    _first_grouped, first = extract_phase4_capabilities_by_project(facts)
    _second_grouped, second = extract_phase4_capabilities_by_project(reversed(facts))
    assert first.decisions == second.decisions
    assert first.grouped_decision_counts == second.grouped_decision_counts
    assert sum(item.count for item in first.grouped_decision_counts) == 12 * len(CAPABILITY_TAXONOMY)


def test_blank_project_id_is_rejected_without_alias_inference():
    with pytest.raises(ValueError, match="blank"):
        extract_phase4_project_capabilities("  ", [])


def test_empty_input_has_empty_valid_report():
    grouped, report = extract_phase4_capabilities_by_project([])
    assert grouped == {}
    assert isinstance(report, Phase4CapabilityExtractionReport)
    assert report.project_count == report.fact_count == report.capabilities_emitted == 0


def test_grouped_decision_counts_cover_every_evaluated_candidate():
    _capabilities, report = extract_for_project(scored_fact("retrieval", evidence_type=EvidenceType.RETRIEVAL))
    assert sum(item.count for item in report.grouped_decision_counts) == report.capability_candidates_evaluated


def test_calls_write_no_files(tmp_path):
    before = list(tmp_path.rglob("*"))
    extract_for_project(scored_fact("canonical serialization with exact deduplication"))
    assert list(tmp_path.rglob("*")) == before


def test_import_has_no_external_or_runtime_side_effects():
    before = set(sys.modules)
    module = importlib.import_module("backend.phase4_capability_extractor")
    importlib.reload(module)
    newly_loaded = set(sys.modules) - before
    assert not any(name.startswith(("chromadb", "openai", "requests", "sqlite3")) for name in newly_loaded)


def test_forbidden_raw_or_secret_fields_remain_model_rejected():
    payload = source().to_dict()
    payload["metadata"] = {"raw_patch": "secret"}
    with pytest.raises(ValueError, match="forbidden"):
        Phase4SourceRef.from_dict(payload)


def test_step5_scored_facts_pass_directly_and_all_results_validate():
    inputs, _ = load_phase4_inputs()
    normalized, _ = dedupe_phase4_inputs(inputs)
    synthesized, _ = synthesize_phase4_evidence_facts(normalized)
    scored, _ = score_phase4_evidence_facts(synthesized)
    grouped, report = extract_phase4_capabilities_by_project(scored)
    source_ids = {item.evidence_fact_id for item in scored}
    assert len(scored) == 283
    assert report.fact_count == 283
    assert all(isinstance(item, Phase4CapabilityFact) for values in grouped.values() for item in values)
    assert all(item.capability_type in CAPABILITY_TAXONOMY for values in grouped.values() for item in values)
    assert all(set(item.source_evidence_fact_ids) <= source_ids for values in grouped.values() for item in values)


def test_real_data_audit_is_read_only_and_high_risk_is_conservative():
    information = Path(__file__).resolve().parents[1] / "information"
    before = {path: path.stat().st_mtime_ns for path in information.rglob("*") if path.is_file()}
    inputs, _ = load_phase4_inputs()
    normalized, _ = dedupe_phase4_inputs(inputs)
    synthesized, _ = synthesize_phase4_evidence_facts(normalized)
    scored, _ = score_phase4_evidence_facts(synthesized)
    grouped, _ = extract_phase4_capabilities_by_project(scored)
    after = {path: path.stat().st_mtime_ns for path in information.rglob("*") if path.is_file()}
    capability_context = [
        fact for fact in scored if classify_phase4_source_category(fact) == "capability_context"
    ]
    project_context = [
        fact for fact in scored if classify_phase4_source_category(fact) == "project_context"
    ]
    context_only_caps, _ = extract_phase4_capabilities_by_project([*capability_context, *project_context])
    all_types = capability_types([item for values in grouped.values() for item in values])
    assert len(capability_context) == 43
    assert len(project_context) == 5
    assert all(not values for values in context_only_caps.values())
    assert "token_or_context_efficiency" not in all_types
    assert "llm_reliability" not in all_types
    assert before == after
