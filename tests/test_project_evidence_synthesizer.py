import importlib
import json
from pathlib import Path
import sys

import pytest

from backend.project_evidence_normalizer import dedupe_project_evidence_inputs
from backend.project_evidence_synthesizer import (
    build_project_evidence_bundles,
    synthesize_project_evidence_fact,
    synthesize_project_evidence_facts,
)
from backend.project_evidence_input import load_project_evidence_inputs
from backend.project_evidence_models import (
    Confidence,
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    ProjectEvidenceInput,
    EvidenceSourceRef,
)


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def source(source_id="chunk-1", **changes):
    values = dict(
        source_type="github_evidence_chunk",
        source_id=source_id,
        project_id="workagent",
        content_hash=DIGEST_A,
        repo="owner/WorkAgent",
        commit_sha="abc123",
        file_path="backend/synthesizer.py",
        symbol="synthesize_fact",
    )
    values.update(changes)
    return EvidenceSourceRef(**values)


def evidence(**changes):
    values = dict(
        project_id="workagent",
        input_type="github_evidence_card",
        title="Structured evidence",
        summary="Structured evidence input.",
        problem_signal="Duplicate evidence required normalization.",
        mechanism_signals=["Validated normalized evidence inputs"],
        implementation_signals=["Added a deterministic synthesis function"],
        impact_signals=["Preserved explicit source provenance"],
        technical_tags=["Python", "validation"],
        source_refs=[source()],
        content_hash=DIGEST_B,
    )
    values.update(changes)
    return ProjectEvidenceInput(**values)


def test_valid_evidence_card_produces_accepted_fact():
    fact = synthesize_project_evidence_fact(evidence())
    assert fact is not None
    assert fact.status is EvidenceStatus.ACCEPTED
    assert fact.confidence is Confidence.MEDIUM


def test_valid_raw_change_produces_accepted_fact_and_explicit_type():
    item = evidence(
        input_type="github_evidence_raw_change_summary",
        problem_signal=None,
        mechanism_signals=["validation_logic_update"],
        implementation_signals=["backend/validator.py", "validate_output"],
        impact_signals=[],
    )
    fact = synthesize_project_evidence_fact(item)
    assert fact is not None
    assert fact.status is EvidenceStatus.ACCEPTED
    assert fact.evidence_type is EvidenceType.VALIDATION


def test_concrete_capability_is_at_most_supporting():
    item = evidence(
        input_type="github_evidence_capability_fact",
        mechanism_signals=["Applied deterministic output validation"],
        implementation_signals=["Validation state machine"],
    )
    fact = synthesize_project_evidence_fact(item)
    assert fact is not None
    assert fact.status is EvidenceStatus.SUPPORTING
    assert fact.confidence is Confidence.LOW


def test_concrete_project_memory_is_at_most_supporting():
    fact = synthesize_project_evidence_fact(evidence(input_type="project_memory"))
    assert fact is not None
    assert fact.status is EvidenceStatus.SUPPORTING
    assert fact.confidence is Confidence.LOW


@pytest.mark.parametrize(
    "generic",
    [
        "Built an AI-powered system",
        "Improved the pipeline",
        "Enhanced retrieval",
        "Optimized performance",
        "Developed backend functionality",
    ],
)
def test_generic_project_memory_positioning_is_rejected(generic):
    item = evidence(input_type="project_memory", mechanism_signals=[generic])
    assert synthesize_project_evidence_fact(item) is None


def test_missing_problem_is_allowed_for_valid_feature():
    fact = synthesize_project_evidence_fact(evidence(problem_signal=None))
    assert fact is not None
    assert fact.problem == ""


def test_missing_safe_impact_is_allowed():
    fact = synthesize_project_evidence_fact(evidence(impact_signals=[]))
    assert fact is not None
    assert fact.safe_impact == []


def test_missing_mechanism_rejects_candidate():
    assert synthesize_project_evidence_fact(evidence(mechanism_signals=[])) is None


def test_missing_implementation_is_weak_not_upgraded():
    ref = source(file_path=None, symbol=None)
    fact = synthesize_project_evidence_fact(evidence(implementation_signals=[], source_refs=[ref]))
    assert fact is not None
    assert fact.status is EvidenceStatus.WEAK
    assert fact.implementation == []


def test_generic_implementation_does_not_satisfy_structural_acceptance():
    ref = source(file_path=None, symbol=None)
    fact = synthesize_project_evidence_fact(
        evidence(
            implementation_signals=["Developed backend functionality"],
            source_refs=[ref],
        )
    )
    assert fact is not None
    assert fact.status is EvidenceStatus.WEAK


@pytest.mark.parametrize(
    "file_path,symbol,expected",
    [
        ("backend/a.py", "validate", "backend/a.py::validate"),
        ("backend/a.py", None, "backend/a.py"),
        (None, "validate", "validate"),
    ],
)
def test_provenance_fallback_builds_only_concrete_reference(file_path, symbol, expected):
    fact = synthesize_project_evidence_fact(
        evidence(implementation_signals=[], source_refs=[source(file_path=file_path, symbol=symbol)])
    )
    assert fact is not None
    assert fact.implementation == [expected]


def test_missing_source_refs_rejects_candidate():
    item = evidence()
    item.source_refs.clear()
    assert synthesize_project_evidence_fact(item) is None


def test_fields_are_copied_only_from_explicit_signals_in_order():
    refs = [source("second", content_hash=DIGEST_B), source("first", content_hash=DIGEST_A)]
    item = evidence(
        project_id="WorkAgent",
        problem_signal="Explicit problem.",
        mechanism_signals=["retrieve", "validate"],
        implementation_signals=["reader", "validator"],
        impact_signals=["Explicit bounded impact."],
        technical_tags=["retrieval", "Python"],
        source_refs=[EvidenceSourceRef.from_dict({**ref.to_dict(), "project_id": "WorkAgent"}) for ref in refs],
    )
    fact = synthesize_project_evidence_fact(item)
    assert fact is not None
    assert fact.project_id == "WorkAgent"
    assert fact.problem == "Explicit problem."
    assert fact.mechanism == "retrieve; validate"
    assert fact.implementation == ["reader", "validator"]
    assert fact.safe_impact == ["Explicit bounded impact."]
    assert fact.technical_tags == ["Python", "retrieval"]
    assert [ref.source_id for ref in fact.source_refs] == ["second", "first"]


def test_duplicate_source_refs_removed_only_when_fully_equal():
    item = evidence(source_refs=[source("same"), source("other")])
    item.source_refs.append(EvidenceSourceRef.from_dict(item.source_refs[0].to_dict()))
    fact = synthesize_project_evidence_fact(item)
    assert fact is not None
    assert [ref.source_id for ref in fact.source_refs] == ["same", "other"]


def test_defaults_do_not_generate_scoring_metrics_or_claim_boundaries():
    fact = synthesize_project_evidence_fact(evidence())
    assert fact is not None
    assert fact.metric_support is MetricSupport.NONE
    assert fact.quality_score is None
    assert fact.quality_breakdown == {}
    assert fact.allowed_claims == []
    assert fact.forbidden_claims == []


@pytest.mark.parametrize(
    "mechanism,forbidden_impact",
    [
        ("retrieval", "reduced hallucinations"),
        ("diff processing", "reduced token usage"),
        ("SQLite persistence", "latency improvement"),
        ("validation", "production reliability"),
        ("prompt constraints", "factual correctness"),
    ],
)
def test_technical_mechanism_does_not_infer_impact(mechanism, forbidden_impact):
    fact = synthesize_project_evidence_fact(evidence(mechanism_signals=[mechanism], impact_signals=[]))
    assert fact is not None
    assert forbidden_impact not in " ".join(fact.safe_impact).lower()


@pytest.mark.parametrize(
    "mechanism,implementation",
    [
        ("test_update", ["Executed 42 tests"]),
        ("validation_logic_update", ["Changed 150 lines"]),
        ("retrieval_logic_update", ["issue 12345"]),
    ],
)
def test_numbers_do_not_infer_metrics(mechanism, implementation):
    fact = synthesize_project_evidence_fact(
        evidence(input_type="github_evidence_raw_change_summary", mechanism_signals=[mechanism], implementation_signals=implementation)
    )
    assert fact is not None
    assert fact.metric_support is MetricSupport.NONE
    assert fact.safe_impact == ["Preserved explicit source provenance"]


def test_exact_provenance_card_and_change_bundle_safely():
    card = evidence(input_type="github_evidence_card")
    change = evidence(
        input_type="github_evidence_raw_change_summary",
        problem_signal=None,
        mechanism_signals=["validation_logic_update"],
        implementation_signals=["backend/synthesizer.py"],
        impact_signals=[],
    )
    bundles = build_project_evidence_bundles([change, card])
    facts, report = synthesize_project_evidence_facts([change, card])
    assert len(bundles) == 1
    assert len(facts) == 1
    assert report.bundle_count == 1
    assert facts[0].mechanism == "Validated normalized evidence inputs; validation_logic_update"


def test_similar_wording_without_exact_provenance_remains_separate():
    one = evidence(source_refs=[source("one")], summary="added validation")
    two = evidence(source_refs=[source("two")], summary="validated output")
    facts, _ = synthesize_project_evidence_facts([one, two])
    assert len(facts) == 2


@pytest.mark.parametrize(
    "first_ref,second_ref",
    [
        (source(project_id="example/workagent"), source(project_id="liammaxp/WorkAgent")),
        (source(commit_sha="one"), source(commit_sha="two")),
        (source(file_path="one.py"), source(file_path="two.py")),
        (source(symbol="one"), source(symbol="two")),
    ],
)
def test_distinct_project_or_provenance_never_bundles(first_ref, second_ref):
    first = evidence(project_id=first_ref.project_id, source_refs=[first_ref])
    second = evidence(project_id=second_ref.project_id, source_refs=[second_ref])
    assert len(build_project_evidence_bundles([first, second])) == 2


def test_project_memory_does_not_bundle_with_code_evidence():
    memory = evidence(input_type="project_memory")
    card = evidence(input_type="github_evidence_card")
    assert len(build_project_evidence_bundles([memory, card])) == 2


def test_conflicting_exact_provenance_inputs_are_retained_separately():
    first = evidence(mechanism_signals=["First concrete mechanism"])
    second = evidence(mechanism_signals=["Second concrete mechanism"])
    facts, report = synthesize_project_evidence_facts([first, second])
    assert len(facts) == 2
    assert report.provenance_conflict_count == 1


def test_direct_explicit_high_confidence_may_be_preserved():
    ref = source(metadata={"confidence": "high"})
    fact = synthesize_project_evidence_fact(evidence(source_refs=[ref]))
    assert fact is not None
    assert fact.confidence is Confidence.HIGH


def test_explicit_metric_support_may_be_preserved():
    ref = source(metadata={"metric_support": "explicit"})
    fact = synthesize_project_evidence_fact(
        evidence(source_refs=[ref], impact_signals=["Reduced processing time by 20%"])
    )
    assert fact is not None
    assert fact.metric_support is MetricSupport.EXPLICIT
    assert fact.safe_impact == ["Reduced processing time by 20%"]


def test_repetition_and_tag_count_do_not_promote_confidence():
    item = evidence(
        input_type="github_evidence_capability_fact",
        technical_tags=[f"tag-{index}" for index in range(20)],
    )
    facts, _ = synthesize_project_evidence_facts([item, item])
    assert all(fact.confidence is Confidence.LOW for fact in facts)


def test_status_is_structural_not_score_based():
    fact = synthesize_project_evidence_fact(evidence(impact_signals=[], technical_tags=[]))
    assert fact is not None
    assert fact.status is EvidenceStatus.ACCEPTED
    assert fact.quality_score is None


@pytest.mark.parametrize(
    "unsafe",
    [
        "Improved accuracy by 50%",
        "Guaranteed factual correctness",
        "Eliminated all failures",
        "Fully prevented invalid output",
        "Reduced hallucinations",
        "Improved ATS success",
    ],
)
def test_unsupported_numeric_or_absolute_impact_is_dropped_without_losing_fact(unsafe):
    facts, report = synthesize_project_evidence_facts([evidence(impact_signals=[unsafe])])
    assert len(facts) == 1
    assert facts[0].safe_impact == []
    assert report.unsafe_impact_dropped_count == 1


def test_forbidden_raw_fields_and_secrets_cannot_enter_output():
    payload = source().to_dict()
    payload["raw_text"] = "RAW_SENTINEL"
    with pytest.raises(ValueError, match="unknown"):
        EvidenceSourceRef.from_dict(payload)
    with pytest.raises(ValueError, match="forbidden"):
        source(metadata={"secret": "SECRET_SENTINEL"})


def test_oversized_invalid_content_is_not_truncated():
    item = evidence()
    item.mechanism_signals.append("x" * 2001)
    with pytest.raises(ValueError, match="maximum length"):
        synthesize_project_evidence_fact(item)


def test_decisions_contain_no_source_content():
    item = evidence(summary="RAW_SOURCE_SENTINEL", mechanism_signals=[])
    _, report = synthesize_project_evidence_facts([item])
    serialized = json.dumps(report.to_dict())
    assert "RAW_SOURCE_SENTINEL" not in serialized


def test_inputs_are_not_mutated():
    item = evidence()
    before = item.to_json()
    synthesize_project_evidence_facts([item])
    assert item.to_json() == before


def test_no_files_are_written(tmp_path):
    marker = tmp_path / "marker.json"
    marker.write_text("{}", encoding="utf-8")
    before = {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()}
    synthesize_project_evidence_facts([evidence()])
    after = {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()}
    assert before == after


def test_import_has_no_external_or_runtime_side_effects():
    module_name = "backend.project_evidence_synthesizer"
    sys.modules.pop(module_name, None)
    before = set(sys.modules)
    importlib.import_module(module_name)
    newly_loaded = set(sys.modules) - before
    assert "backend.api_server" not in newly_loaded
    assert "backend.project_change_pipeline" not in newly_loaded
    assert not any(name.startswith(("chromadb", "openai", "sqlite3", "requests")) for name in newly_loaded)


def test_equivalent_input_produces_same_fact_id_and_repeated_output():
    one = synthesize_project_evidence_fact(evidence())
    two = synthesize_project_evidence_fact(evidence())
    assert one is not None and two is not None
    assert one.evidence_fact_id == two.evidence_fact_id
    assert one.to_json() == two.to_json()


def test_materially_different_mechanism_changes_fact_id():
    one = synthesize_project_evidence_fact(evidence(mechanism_signals=["mechanism one"]))
    two = synthesize_project_evidence_fact(evidence(mechanism_signals=["mechanism two"]))
    assert one is not None and two is not None
    assert one.evidence_fact_id != two.evidence_fact_id


def test_implementation_order_changes_fact_id():
    one = synthesize_project_evidence_fact(evidence(implementation_signals=["read", "validate"]))
    two = synthesize_project_evidence_fact(evidence(implementation_signals=["validate", "read"]))
    assert one is not None and two is not None
    assert one.evidence_fact_id != two.evidence_fact_id


def test_tag_order_does_not_change_fact_id():
    one = synthesize_project_evidence_fact(evidence(technical_tags=["Python", "validation"]))
    two = synthesize_project_evidence_fact(evidence(technical_tags=["validation", "Python"]))
    assert one is not None and two is not None
    assert one.evidence_fact_id == two.evidence_fact_id


def test_output_order_and_report_are_deterministic():
    alpha = evidence(project_id="alpha", source_refs=[source(project_id="alpha")])
    zeta = evidence(project_id="zeta", source_refs=[source(project_id="zeta")])
    first, first_report = synthesize_project_evidence_facts([zeta, alpha])
    second, second_report = synthesize_project_evidence_facts([alpha, zeta])
    assert [fact.to_json() for fact in first] == [fact.to_json() for fact in second]
    assert first_report.to_dict() == second_report.to_dict()


def test_report_counters_are_structural():
    card = evidence()
    change = evidence(input_type="github_evidence_raw_change_summary", problem_signal=None, mechanism_signals=["validation_logic_update"], impact_signals=[])
    capability = evidence(input_type="github_evidence_capability_fact", implementation_signals=[], source_refs=[source(file_path=None, symbol=None)])
    rejected = evidence(input_type="project_memory", mechanism_signals=["Improved the pipeline"])
    facts, report = synthesize_project_evidence_facts([card, change, capability, rejected])
    assert len(facts) == 2
    assert report.input_count == 4
    assert report.bundle_count == 3
    assert report.evidence_fact_count == 2
    assert report.accepted_count == 1
    assert report.weak_count == 1
    assert report.rejected_count == 1
    assert report.missing_mechanism_count == 1
    assert report.missing_implementation_count == 1


def test_real_step2_step3_output_passes_stepe_read_only():
    information = Path(__file__).resolve().parents[1] / "information"
    before = {path: path.stat().st_mtime_ns for path in information.rglob("*") if path.is_file()}
    inputs, _ = load_project_evidence_inputs()
    normalized, _ = dedupe_project_evidence_inputs(inputs)
    facts, report = synthesize_project_evidence_facts(normalized)
    after = {path: path.stat().st_mtime_ns for path in information.rglob("*") if path.is_file()}
    assert len(normalized) == 449
    assert len(facts) == 283
    assert report.bundle_count == 283
    assert before == after
