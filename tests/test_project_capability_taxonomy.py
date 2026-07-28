from dataclasses import FrozenInstanceError, replace
import importlib
import json
from pathlib import Path
import re
import sys

import pytest

from backend.project_capability_taxonomy import (
    CAPABILITY_ALIASES,
    CAPABILITY_DEFINITIONS,
    CAPABILITY_OVERLAP_RULES,
    CAPABILITY_TAXONOMY,
    FORBIDDEN_INFERENCE_REGISTRY,
    SIGNAL_REGISTRY,
    ProjectCapabilityDefinition,
    capability_taxonomy_to_dict,
    get_capability_rule,
    get_capability_taxonomy,
    get_capability_types_for_signal,
    get_project_capability_alias_target,
    get_project_capability_definition,
    is_project_capability_supported,
    list_project_capability_definitions,
    list_project_capability_overlap_rules,
    list_project_evidence_signal_identifiers,
    normalize_capability_label,
    resolve_capability_type,
    validate_project_capability_taxonomy,
    validate_project_capability_type,
)
from backend.project_evidence_normalizer import dedupe_project_evidence_inputs
from backend.project_evidence_scoring import score_project_evidence_facts
from backend.project_evidence_synthesizer import synthesize_project_evidence_facts
from backend.project_evidence_input import load_project_evidence_inputs
from backend.project_evidence_models import EvidenceType, ProjectCapabilityFact, ProjectEvidenceFact
from backend.project_evidence_models import MetricSupport
from backend.project_evidence_memory import load_project_evidence_memory


REQUIRED_CAPABILITIES = {
    "claim_validation",
    "data_persistence",
    "deterministic_document_generation",
    "deterministic_evidence_normalization",
    "evidence_grounded_generation",
    "failure_recovery",
    "frontend_backend_integration",
    "incremental_change_processing",
    "latex_validation_and_repair",
    "llm_reliability",
    "output_quality_control",
    "project_memory_management",
    "retrieval_and_reranking",
    "structured_evidence_extraction",
    "template_pollution_blocking",
    "test_and_regression_hardening",
    "token_or_context_efficiency",
    "workflow_orchestration",
}
IDENTIFIER_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def definition(name: str) -> ProjectCapabilityDefinition:
    return get_project_capability_definition(name)


def required_signals(name: str) -> tuple[frozenset[str], ...]:
    return tuple(frozenset(group) for group in definition(name).required_signal_groups)


def validate_fixture(item: ProjectCapabilityDefinition, **changes):
    return validate_project_capability_taxonomy(
        (replace(item, **changes),),
        aliases=(),
        overlap_rules=(),
    )


def test_taxonomy_contains_complete_required_initial_set():
    assert {item.capability_type for item in CAPABILITY_DEFINITIONS} == REQUIRED_CAPABILITIES


def test_capability_ids_are_unique_valid_lowercase_snake_case():
    identifiers = [item.capability_type for item in CAPABILITY_DEFINITIONS]
    assert len(identifiers) == len(set(identifiers))
    assert all(IDENTIFIER_RE.fullmatch(value) for value in identifiers)


def test_definitions_have_descriptions_and_nonempty_required_groups():
    assert all(item.description.strip() for item in CAPABILITY_DEFINITIONS)
    assert all(item.required_signal_groups for item in CAPABILITY_DEFINITIONS)
    assert all(group for item in CAPABILITY_DEFINITIONS for group in item.required_signal_groups)


def test_all_definition_signals_exist_and_are_not_duplicated():
    registry = set(SIGNAL_REGISTRY)
    for item in CAPABILITY_DEFINITIONS:
        values = [signal for group in item.required_signal_groups for signal in group]
        values.extend(item.supporting_signals)
        assert set(values) <= registry
        assert len(values) == len(set(values))


def test_evidence_type_policies_are_valid_complete_partitions():
    expected = {item.value for item in EvidenceType}
    for item in CAPABILITY_DEFINITIONS:
        policy = (
            *item.accepted_evidence_types,
            *item.contextual_evidence_types,
            *item.unsupported_evidence_types,
        )
        assert len(policy) == len(set(policy))
        assert set(policy) == expected


def test_quality_and_fact_count_policies_are_bounded():
    for item in CAPABILITY_DEFINITIONS:
        assert isinstance(item.minimum_quality_score, int)
        assert 0 <= item.minimum_quality_score <= 100
        assert 0 <= item.minimum_direct_fact_count <= item.minimum_total_fact_count
        assert item.requires_direct_provenance


def test_high_risk_capabilities_use_stricter_requirements():
    token = definition("token_or_context_efficiency")
    reliability = definition("llm_reliability")
    assert token.high_risk and token.minimum_quality_score >= 70
    assert reliability.high_risk and reliability.minimum_quality_score >= 70
    assert reliability.minimum_direct_fact_count >= 2
    assert reliability.minimum_total_fact_count >= 2
    assert not token.allows_contextual_support
    assert not reliability.allows_contextual_support


def test_taxonomy_and_signal_order_are_deterministic():
    assert [item.capability_type for item in CAPABILITY_DEFINITIONS] == sorted(REQUIRED_CAPABILITIES)
    assert SIGNAL_REGISTRY == tuple(sorted(SIGNAL_REGISTRY))
    assert list_project_capability_definitions() == list_project_capability_definitions()


def test_valid_lookup_and_repeated_lookup_succeed():
    first = get_project_capability_definition("claim_validation")
    second = get_project_capability_definition("claim_validation")
    assert first == second
    assert first.capability_type == "claim_validation"


@pytest.mark.parametrize("value", ["unknown_capability", "Claim_Validation", " bad", "bad__name", ""])
def test_unknown_or_invalid_lookup_fails_safely(value):
    with pytest.raises(ValueError):
        get_project_capability_definition(value)
    assert not is_project_capability_supported(value)


def test_non_string_lookup_fails_safely():
    with pytest.raises(TypeError):
        validate_project_capability_type(None)  # type: ignore[arg-type]
    assert not is_project_capability_supported(None)  # type: ignore[arg-type]


def test_supported_check_accepts_canonical_and_declared_alias():
    assert is_project_capability_supported("claim_validation")
    assert is_project_capability_supported("unsupported_claim_boundary")
    assert validate_project_capability_type("unsupported_claim_boundary") == "claim_validation"


def test_list_and_lookup_results_are_immutable():
    listed = list_project_capability_definitions()
    with pytest.raises(AttributeError):
        listed.append(listed[0])  # type: ignore[attr-defined]
    with pytest.raises(FrozenInstanceError):
        listed[0].description = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        CAPABILITY_TAXONOMY["new"] = listed[0]  # type: ignore[index]
    with pytest.raises(TypeError):
        CAPABILITY_ALIASES["new"] = "claim_validation"  # type: ignore[index]


@pytest.mark.parametrize(
    ("alias", "target"),
    [
        ("deterministic_latex_validation", "latex_validation_and_repair"),
        ("local_project_memory", "project_memory_management"),
        ("testing_and_regression_safety", "test_and_regression_hardening"),
        ("token_or_cost_reduction", "token_or_context_efficiency"),
        ("unsupported_claim_boundary", "claim_validation"),
    ],
)
def test_declared_legacy_aliases_resolve_to_canonical_definition(alias, target):
    assert get_project_capability_alias_target(alias) == target
    assert get_project_capability_definition(alias) == definition(target)


def test_unknown_alias_does_not_resolve():
    assert get_project_capability_alias_target("not_declared") is None
    assert get_project_capability_alias_target("Bad Alias") is None


def test_alias_cycle_is_rejected():
    report = validate_project_capability_taxonomy(aliases=(("legacy_a", "legacy_b"), ("legacy_b", "legacy_a")))
    assert any("alias_cycle" in error for error in report.errors)


def test_alias_chain_is_rejected():
    report = validate_project_capability_taxonomy(
        aliases=(("legacy_a", "legacy_b"), ("legacy_b", "claim_validation"))
    )
    assert any("alias_chain" in error for error in report.errors)


def test_alias_target_must_exist():
    report = validate_project_capability_taxonomy(aliases=(("legacy_a", "missing_target"),))
    assert any("missing_target" in error for error in report.errors)


def test_canonical_capability_cannot_be_alias():
    report = validate_project_capability_taxonomy(aliases=(("claim_validation", "data_persistence"),))
    assert any("canonical_name_used_as_alias" in error for error in report.errors)


def test_duplicate_alias_is_rejected_and_aliases_do_not_duplicate_definitions():
    report = validate_project_capability_taxonomy(
        aliases=(("legacy_a", "claim_validation"), ("legacy_a", "data_persistence"))
    )
    assert any("duplicate_alias" in error for error in report.errors)
    assert len(list_project_capability_definitions()) == len(REQUIRED_CAPABILITIES)


@pytest.mark.parametrize(
    ("capability", "required_group_alternatives"),
    [
        ("structured_evidence_extraction", ({"structured_extraction", "evidence_card_generation"}, {"schema_validation", "source_attribution"})),
        ("evidence_grounded_generation", ({"retrieval", "source_grounding"}, {"unsupported_claim_blocking", "output_evidence_validation"})),
        ("retrieval_and_reranking", ({"retrieval", "query_operation"}, {"reranking", "candidate_filtering"})),
        ("project_memory_management", ({"project_memory_write", "persistent_storage"}, {"project_memory_read", "load_validate_write_lifecycle"})),
        ("incremental_change_processing", ({"diff_processing", "changed_file_detection"}, {"incremental_update", "exact_deduplication"})),
        ("deterministic_evidence_normalization", ({"stable_identity", "canonical_serialization"}, {"field_normalization", "exact_deduplication"})),
        ("output_quality_control", ({"quality_dimensions", "validation_gate"},)),
        ("claim_validation", ({"claim_validation", "unsupported_claim_blocking"}, {"source_grounding", "output_evidence_validation"})),
        ("deterministic_document_generation", ({"document_merge", "structured_document_assembly"}, {"deterministic_ordering", "template_constraint"})),
        ("latex_validation_and_repair", ({"latex_validation", "latex_compile_check"}, {"latex_repair", "invalid_section_detection"})),
        ("template_pollution_blocking", ({"template_pollution_detection", "source_ownership_validation"}, {"unsupported_template_filtering", "validation_gate"})),
        ("failure_recovery", ({"failure_detection", "error_state_handling"}, {"fallback", "retry", "repair"})),
        ("data_persistence", ({"persistent_storage", "atomic_persistence"}, {"schema_versioning", "integrity_hash"})),
        ("workflow_orchestration", ({"workflow_staging", "state_transition"}, {"stage_io_contract", "failure_propagation"})),
        ("frontend_backend_integration", ({"backend_route", "response_schema"}, {"frontend_api_call", "frontend_state_handling"})),
        ("test_and_regression_hardening", ({"regression_testing", "invariant_check"}, {"targeted_testing", "failure_case_coverage"})),
        ("token_or_context_efficiency", ({"context_compression", "cache_reuse"}, {"measured_input_reduction", "repeated_input_avoidance"})),
        ("llm_reliability", ({"source_grounding", "source_attribution"}, {"claim_validation", "unsupported_claim_blocking"}, {"low_evidence_refusal", "factuality_evaluation"})),
    ],
)
def test_required_capability_semantics_use_distinct_and_groups(capability, required_group_alternatives):
    groups = required_signals(capability)
    assert len(groups) >= len(required_group_alternatives)
    for expected, actual in zip(required_group_alternatives, groups):
        assert expected <= actual


@pytest.mark.parametrize(
    ("capability", "forbidden_code"),
    [
        ("retrieval_and_reranking", "retrieval_implies_hallucination_reduction"),
        ("incremental_change_processing", "reduced_token_usage"),
        ("data_persistence", "improved_latency"),
        ("output_quality_control", "validation_implies_production_reliability"),
        ("test_and_regression_hardening", "tests_imply_bug_free"),
        ("frontend_backend_integration", "production_deployment"),
        ("data_persistence", "persistence_implies_scalability"),
        ("latex_validation_and_repair", "latex_export_proves_validation_repair"),
        ("llm_reliability", "rag_term_proves_llm_reliability"),
        ("llm_reliability", "prompt_constraints_alone_prove_llm_reliability"),
    ],
)
def test_forbidden_inference_rules_block_unsafe_shortcuts(capability, forbidden_code):
    assert forbidden_code in definition(capability).forbidden_inferences


def test_sqlite_or_storage_does_not_imply_latency():
    assert "improved_latency" in definition("incremental_change_processing").forbidden_inferences
    assert "persistence_implies_scalability" in definition("data_persistence").forbidden_inferences


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("output_quality_control", "claim_validation"),
        ("claim_validation", "llm_reliability"),
        ("evidence_grounded_generation", "llm_reliability"),
        ("deterministic_document_generation", "latex_validation_and_repair"),
        ("structured_evidence_extraction", "deterministic_evidence_normalization"),
        ("data_persistence", "project_memory_management"),
    ],
)
def test_overlapping_capabilities_do_not_imply_each_other(left, right):
    assert required_signals(left) != required_signals(right)
    assert get_project_capability_alias_target(left) is None
    assert validate_project_capability_type(left) == left
    assert validate_project_capability_type(right) == right


def test_overlap_rules_are_explicit_immutable_and_distinguishing():
    rules = list_project_capability_overlap_rules()
    assert rules == CAPABILITY_OVERLAP_RULES
    assert all(rule.left_distinguishing_signals for rule in rules)
    assert all(rule.right_distinguishing_signals for rule in rules)
    assert all(rule.non_implication_rule for rule in rules)


def test_safe_templates_have_placeholders_and_no_unsafe_metrics_or_guarantees():
    unsafe = re.compile(r"(?:\d+(?:\.\d+)?\s*%|eliminated|guaranteed|production[- ]scale|enterprise[- ]grade)", re.I)
    for item in CAPABILITY_DEFINITIONS:
        for template in item.safe_claim_templates:
            assert "{" in template and "}" in template
            assert not unsafe.search(template)


def test_safe_templates_do_not_assert_project_technologies_or_create_concrete_project_claims():
    project_technology = re.compile(r"\b(?:Python|React|FastAPI|SQLite|Chroma|LaTeX|OpenAI)\b", re.I)
    for item in CAPABILITY_DEFINITIONS:
        for template in item.safe_claim_templates:
            assert not project_technology.search(template)
            assert "{" in template


def test_high_risk_forbidden_lists_include_explicit_overclaims():
    assert "eliminated_hallucinations" in definition("llm_reliability").forbidden_inferences
    efficiency = definition("token_or_context_efficiency").forbidden_inferences
    assert {"reduced_token_usage", "reduced_api_cost", "improved_latency"} <= set(efficiency)


def test_taxonomy_self_validation_succeeds():
    report = validate_project_capability_taxonomy()
    assert report.valid, report.errors
    assert report.capability_count == 18
    assert report.signal_count == len(SIGNAL_REGISTRY)
    assert report.alias_count == len(CAPABILITY_ALIASES)


def test_duplicate_definition_is_detected():
    items = (CAPABILITY_DEFINITIONS[0], CAPABILITY_DEFINITIONS[0])
    report = validate_project_capability_taxonomy(items, aliases=(), overlap_rules=())
    assert any("duplicate_capability_id" in error for error in report.errors)


def test_unknown_signal_is_detected():
    report = validate_fixture(
        CAPABILITY_DEFINITIONS[0],
        required_signal_groups=(("unknown_signal_for_fixture",),),
        supporting_signals=(),
    )
    assert any("unknown_signal" in error for error in report.errors)


def test_invalid_evidence_type_is_detected():
    report = validate_fixture(
        CAPABILITY_DEFINITIONS[0],
        accepted_evidence_types=("not_an_evidence_type",),
    )
    assert any("invalid_evidence_type" in error for error in report.errors)


@pytest.mark.parametrize("threshold", [-1, 101])
def test_invalid_quality_threshold_is_detected(threshold):
    report = validate_fixture(CAPABILITY_DEFINITIONS[0], minimum_quality_score=threshold)
    assert any("invalid_quality_threshold" in error for error in report.errors)


def test_direct_count_above_total_is_detected():
    report = validate_fixture(
        CAPABILITY_DEFINITIONS[0],
        minimum_direct_fact_count=2,
        minimum_total_fact_count=1,
    )
    assert any("direct_count_exceeds_total" in error for error in report.errors)


def test_empty_required_group_is_detected():
    report = validate_fixture(CAPABILITY_DEFINITIONS[0], required_signal_groups=((),))
    assert any("empty_required_group" in error for error in report.errors)


def test_unsafe_template_is_detected():
    report = validate_fixture(
        CAPABILITY_DEFINITIONS[0],
        safe_claim_templates=("Guaranteed 50% improvement with Python.",),
    )
    assert any("unsafe_template" in error for error in report.errors)


def test_unsupported_forbidden_inference_is_detected_without_echoing_fixture_content():
    sentinel = "RAW_SOURCE_SENTINEL"
    report = validate_fixture(
        CAPABILITY_DEFINITIONS[0],
        forbidden_inferences=(sentinel,),
        description=sentinel,
    )
    serialized = json.dumps(report.__dict__)
    assert "unsupported_forbidden_inference" in serialized
    assert sentinel not in serialized


def test_signal_and_definition_collection_apis_are_immutable():
    assert list_project_evidence_signal_identifiers() is SIGNAL_REGISTRY
    with pytest.raises(AttributeError):
        list_project_evidence_signal_identifiers().append("new")  # type: ignore[attr-defined]
    assert set(FORBIDDEN_INFERENCE_REGISTRY)


def test_import_has_no_runtime_file_or_external_service_side_effects(monkeypatch):
    module_name = "backend.project_capability_taxonomy"
    sys.modules.pop(module_name, None)
    before = set(sys.modules)
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: pytest.fail("unexpected read"))
    monkeypatch.setattr(Path, "write_text", lambda *_args, **_kwargs: pytest.fail("unexpected write"))
    importlib.import_module(module_name)
    newly_loaded = set(sys.modules) - before
    assert not any(name.startswith(("chromadb", "openai", "sqlite3", "requests")) for name in newly_loaded)


def test_real_taxonomy_audit_is_read_only_and_creates_no_capability_facts():
    information = Path(__file__).resolve().parents[1] / "information"
    before_mtimes = {path: path.stat().st_mtime_ns for path in information.rglob("*") if path.is_file()}
    inputs, _ = load_project_evidence_inputs()
    normalized, _ = dedupe_project_evidence_inputs(inputs)
    facts, _ = synthesize_project_evidence_facts(normalized)
    scored, _ = score_project_evidence_facts(facts)
    before_facts = [item.to_json() for item in scored]
    report = validate_project_capability_taxonomy()
    after_mtimes = {path: path.stat().st_mtime_ns for path in information.rglob("*") if path.is_file()}
    assert report.valid
    assert len(scored) == 283
    assert all(isinstance(item, ProjectEvidenceFact) for item in scored)
    assert not any(isinstance(item, ProjectCapabilityFact) for item in scored)
    assert before_facts == [item.to_json() for item in scored]
    assert before_mtimes == after_mtimes


def test_canonical_registry_exposes_explicit_future_proof_minimums():
    registry = get_capability_taxonomy()
    assert registry is CAPABILITY_TAXONOMY
    for rule in registry.values():
        assert rule.minimum_total_fact_count >= 1
        assert rule.minimum_direct_fact_count >= 1
        assert 1 <= rule.minimum_distinct_signal_group_count <= len(rule.required_signal_groups)
        assert rule.minimum_mechanism_count >= 1
        assert get_capability_rule(rule.capability_type) is rule


def test_normalized_alias_lookup_is_deterministic_and_not_fuzzy():
    assert normalize_capability_label("  Local-Project  Memory ") == "local_project_memory"
    assert resolve_capability_type(" local-project-memory ") == "project_memory_management"
    assert resolve_capability_type("Project Memory Management") == "project_memory_management"
    assert get_capability_rule("LOCAL PROJECT MEMORY") is definition("project_memory_management")
    assert resolve_capability_type("project_memory") is None
    assert resolve_capability_type("retriev") is None


def test_normalized_alias_collision_is_rejected():
    report = validate_project_capability_taxonomy(
        aliases=(
            ("shared-alias", "claim_validation"),
            ("shared_alias", "data_persistence"),
        ),
    )
    assert any("normalized_alias_collision" in error for error in report.errors)


def test_registered_signal_mapping_is_exact_and_deterministic():
    first = get_capability_types_for_signal("source-grounding")
    assert first == tuple(sorted(first))
    assert "evidence_grounded_generation" in first
    assert get_capability_types_for_signal("source grounding") == first
    assert get_capability_types_for_signal("ground") == ()


def test_registry_and_rules_cannot_be_mutated_by_callers():
    registry = get_capability_taxonomy()
    with pytest.raises(TypeError):
        registry["new_capability"] = CAPABILITY_DEFINITIONS[0]
    with pytest.raises(FrozenInstanceError):
        CAPABILITY_DEFINITIONS[0].display_name = "Changed"


def test_registry_serialization_is_stable_and_json_compatible():
    first = capability_taxonomy_to_dict()
    second = capability_taxonomy_to_dict()
    assert first == second
    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second, ensure_ascii=False, sort_keys=True
    )
    assert list(first["definitions"]) == sorted(CAPABILITY_TAXONOMY)


def test_taxonomy_uses_authoritative_metric_support_values():
    assert {item.value for item in MetricSupport} == {"none", "approximate", "explicit"}


def test_real_capability_diagnostics_remain_unchanged_and_read_only():
    artifact = Path(__file__).resolve().parents[1] / "information" / "project_evidence_memory.json"
    before = artifact.read_bytes()
    before_mtime = artifact.stat().st_mtime_ns
    loaded = load_project_evidence_memory(artifact)
    assert loaded.status == "ready" and loaded.snapshot is not None
    assert loaded.snapshot.diagnostics.capability_fact_count == 0
    assert loaded.snapshot.diagnostics.unsupported_capability_blocked_count == 11
    assert artifact.read_bytes() == before
    assert artifact.stat().st_mtime_ns == before_mtime
