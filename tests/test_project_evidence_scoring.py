import importlib
import json
from pathlib import Path
import sys

import pytest

from backend.project_evidence_normalizer import dedupe_project_evidence_inputs
from backend.project_evidence_scoring import (
    DIMENSION_MAXIMUMS,
    evaluate_project_evidence_quality,
    score_project_evidence_fact,
    score_project_evidence_facts,
)
from backend.project_evidence_synthesizer import synthesize_project_evidence_facts
from backend.project_evidence_input import load_project_evidence_inputs
from backend.project_evidence_models import (
    Confidence,
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    ProjectEvidenceFact,
    EvidenceSourceRef,
)


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def source(source_id="source-1", **changes):
    values = dict(
        source_type="github_evidence_chunk",
        source_id=source_id,
        project_id="workagent",
        content_hash=DIGEST_A,
        repo="owner/WorkAgent",
        commit_sha="abc123",
        file_path="backend/scoring.py",
        symbol="score_fact",
    )
    values.update(changes)
    return EvidenceSourceRef(**values)


def fact(**changes):
    values = dict(
        project_id="workagent",
        problem="Duplicate source records caused a validation conflict in EvidenceStore.",
        mechanism="Applied deterministic merge ordering; validated source_bound_record() before persistence",
        implementation=[
            "backend/scoring.py::score_fact",
            "backend/models.py::EvidenceStore",
            "normalize source identity",
            "validate project binding",
            "persist deterministic result",
        ],
        safe_impact=["Blocked unsupported source-state transitions before persistence"],
        evidence_type=EvidenceType.VALIDATION,
        source_refs=[source("one", content_hash=DIGEST_A), source("two", content_hash=DIGEST_B)],
        confidence=Confidence.MEDIUM,
        metric_support=MetricSupport.NONE,
        allowed_claims=[],
        forbidden_claims=[],
        technical_tags=["Python", "validation", "persistence"],
        status=EvidenceStatus.ACCEPTED,
        quality_score=None,
        quality_breakdown={},
    )
    values.update(changes)
    return ProjectEvidenceFact(**values)


def test_maximum_quality_direct_fact_is_high_value():
    evaluation = evaluate_project_evidence_quality(fact())
    assert 80 <= evaluation.score <= 100
    assert evaluation.quality_band == "high_value"


def test_concrete_mechanism_scores_higher_than_generic():
    concrete = evaluate_project_evidence_quality(fact())
    generic = evaluate_project_evidence_quality(fact(mechanism="Improved the pipeline"))
    assert concrete.breakdown["mechanism_specificity"] > generic.breakdown["mechanism_specificity"]
    assert generic.generic_content_penalty


def test_concrete_implementation_scores_higher_than_generic():
    concrete = evaluate_project_evidence_quality(fact())
    generic = evaluate_project_evidence_quality(fact(implementation=["Developed features"]))
    assert concrete.breakdown["implementation_specificity"] > generic.breakdown["implementation_specificity"]


def test_direct_provenance_scores_higher_than_project_context():
    direct = evaluate_project_evidence_quality(fact(source_refs=[source()]))
    contextual_ref = source(source_type="project_memory", file_path=None, symbol=None)
    contextual = evaluate_project_evidence_quality(fact(source_refs=[contextual_ref]))
    assert direct.breakdown["provenance_strength"] > contextual.breakdown["provenance_strength"]


def test_explicit_safe_impact_scores_higher_than_absent_impact():
    present = evaluate_project_evidence_quality(fact())
    absent = evaluate_project_evidence_quality(fact(safe_impact=[]))
    assert present.breakdown["safe_impact_quality"] > absent.breakdown["safe_impact_quality"]


def test_absent_impact_does_not_prevent_high_value_score():
    evaluation = evaluate_project_evidence_quality(fact(safe_impact=[]))
    assert evaluation.quality_band == "high_value"


def test_absent_problem_does_not_prevent_high_value_score():
    evaluation = evaluate_project_evidence_quality(fact(problem=""))
    assert evaluation.quality_band == "high_value"


def test_coherent_technical_signals_beat_unrelated_tag_quantity():
    coherent = evaluate_project_evidence_quality(fact())
    unrelated = evaluate_project_evidence_quality(
        fact(
            mechanism="processed records",
            implementation=["component"],
            evidence_type=EvidenceType.UNKNOWN,
            technical_tags=[f"tag-{index}" for index in range(30)],
        )
    )
    assert coherent.breakdown["technical_specificity"] > unrelated.breakdown["technical_specificity"]


def test_empty_mechanism_is_a_blocker():
    item = fact()
    object.__setattr__(item, "mechanism", "")
    evaluation = evaluate_project_evidence_quality(item)
    assert "missing_mechanism" in evaluation.blockers
    assert evaluation.quality_band == "rejected_value"


def test_empty_implementation_materially_lowers_score():
    complete = evaluate_project_evidence_quality(fact())
    incomplete = evaluate_project_evidence_quality(fact(implementation=[], status=EvidenceStatus.WEAK))
    assert complete.score - incomplete.score >= 15
    assert incomplete.breakdown["penalties"]["missing_implementation"] == -15


def test_missing_provenance_is_a_blocker():
    item = fact()
    item.source_refs.clear()
    evaluation = evaluate_project_evidence_quality(item)
    assert "missing_provenance" in evaluation.blockers
    assert evaluation.score <= 39


def test_unknown_evidence_type_can_still_score_highly():
    evaluation = evaluate_project_evidence_quality(fact(evidence_type=EvidenceType.UNKNOWN))
    assert evaluation.quality_band == "high_value"


@pytest.mark.parametrize(
    "generic",
    [
        "Built an AI-powered system",
        "Enhanced functionality",
        "Optimized performance",
        "Worked on backend",
        "Used advanced technologies",
    ],
)
def test_generic_mechanism_receives_penalty(generic):
    evaluation = evaluate_project_evidence_quality(fact(mechanism=generic))
    assert evaluation.generic_content_penalty
    assert evaluation.breakdown["penalties"]["generic_content"] < 0


def test_generic_impact_receives_limited_credit():
    evaluation = evaluate_project_evidence_quality(fact(safe_impact=["Improved reliability"]))
    assert evaluation.breakdown["safe_impact_quality"] <= 2


def test_generic_phrase_plus_concrete_mechanism_is_not_rejected():
    evaluation = evaluate_project_evidence_quality(
        fact(mechanism="Improved the pipeline; validated source_bound_record()")
    )
    assert "marketing_only_content" not in evaluation.blockers
    assert evaluation.score >= 40


def test_marketing_only_fact_is_rejected():
    evaluation = evaluate_project_evidence_quality(
        fact(mechanism="Improved the pipeline", implementation=["Developed features"])
    )
    assert evaluation.quality_band == "rejected_value"
    assert "marketing_only_content" in evaluation.blockers


def test_short_technical_identifier_is_not_generic():
    evaluation = evaluate_project_evidence_quality(fact(mechanism="SHA256"))
    assert not evaluation.generic_content_penalty
    assert evaluation.breakdown["mechanism_specificity"] >= 12


@pytest.mark.parametrize(
    "claim",
    [
        "Improved throughput by 50%",
        "Processed data 3 times faster",
        "Eliminated all invalid output",
        "Guaranteed reliability",
        "Reduced hallucinations",
        "Improved ATS success",
        "Provided enterprise-grade production-scale processing",
    ],
)
def test_unsupported_claims_receive_penalty(claim):
    evaluation = evaluate_project_evidence_quality(fact(safe_impact=[claim]))
    assert evaluation.unsupported_claim_penalty
    assert evaluation.breakdown["penalties"]["unsupported_claim"] == -20
    assert evaluation.recommended_status == EvidenceStatus.REJECTED.value


@pytest.mark.parametrize(
    "technical_value",
    [
        "Supported Python 3.12",
        "Handled HTTP 500 responses",
        "Validated SHA abc123 at line 150",
        "Used API v2",
    ],
)
def test_technical_numbers_do_not_trigger_metric_penalty(technical_value):
    evaluation = evaluate_project_evidence_quality(fact(implementation=[technical_value]))
    assert not evaluation.unsupported_claim_penalty


def test_explicit_metric_support_prevents_numeric_penalty():
    evaluation = evaluate_project_evidence_quality(
        fact(
            safe_impact=["Reduced processing time by 20%"],
            metric_support=MetricSupport.EXPLICIT,
        )
    )
    assert not evaluation.unsupported_claim_penalty


def test_approximate_metric_requires_approximate_language():
    supported = evaluate_project_evidence_quality(
        fact(safe_impact=["Approximately 20% reduction"], metric_support=MetricSupport.APPROXIMATE)
    )
    unsupported = evaluate_project_evidence_quality(
        fact(safe_impact=["Reduced processing time by 20%"], metric_support=MetricSupport.APPROXIMATE)
    )
    assert not supported.unsupported_claim_penalty
    assert unsupported.unsupported_claim_penalty


def test_metric_none_does_not_block_non_numeric_fact():
    evaluation = evaluate_project_evidence_quality(fact(metric_support=MetricSupport.NONE))
    assert not evaluation.unsupported_claim_penalty


def test_accepted_high_quality_fact_remains_accepted():
    scored = score_project_evidence_fact(fact(status=EvidenceStatus.ACCEPTED))
    assert scored.status is EvidenceStatus.ACCEPTED


def test_supporting_high_quality_fact_is_not_upgraded():
    scored = score_project_evidence_fact(fact(status=EvidenceStatus.SUPPORTING))
    assert scored.status is EvidenceStatus.SUPPORTING


def test_weak_high_score_fact_is_not_upgraded():
    scored = score_project_evidence_fact(fact(status=EvidenceStatus.WEAK))
    assert scored.status is EvidenceStatus.WEAK
    assert scored.quality_score >= 80


def test_hard_blocker_may_downgrade_accepted_fact():
    scored = score_project_evidence_fact(
        fact(status=EvidenceStatus.ACCEPTED, safe_impact=["Guaranteed reliability"])
    )
    assert scored.status is EvidenceStatus.REJECTED
    assert scored.quality_score <= 39


def test_confidence_is_not_rewritten_or_derived_from_score():
    original = fact(confidence=Confidence.LOW)
    scored = score_project_evidence_fact(original)
    assert scored.confidence is Confidence.LOW
    assert scored.quality_score != scored.confidence.value


def test_breakdown_sums_and_contains_required_dimensions():
    scored = score_project_evidence_fact(fact())
    breakdown = scored.quality_breakdown
    assert set(DIMENSION_MAXIMUMS).issubset(breakdown)
    calculated = (
        sum(breakdown[key] for key in DIMENSION_MAXIMUMS)
        + sum(breakdown["penalties"].values())
        + breakdown["blocker_adjustment"]
    )
    assert calculated == scored.quality_score


def test_penalties_and_quality_band_are_deterministic():
    scored = score_project_evidence_fact(fact(implementation=[] , status=EvidenceStatus.WEAK))
    assert scored.quality_breakdown["penalties"] == {
        "generic_content": 0,
        "unsupported_claim": 0,
        "missing_implementation": -15,
    }
    score = scored.quality_score
    expected = "high_value" if score >= 80 else "supporting_value" if score >= 60 else "weak_value" if score >= 40 else "rejected_value"
    assert scored.quality_breakdown["recommended_quality_band"] == expected


def test_score_is_integer_and_model_range_is_zero_to_one_hundred():
    scored = score_project_evidence_fact(fact())
    assert isinstance(scored.quality_score, int)
    ProjectEvidenceFact.from_dict({**scored.to_dict(), "quality_score": 100})
    with pytest.raises(ValueError, match="between 0 and 100"):
        ProjectEvidenceFact.from_dict({**scored.to_dict(), "quality_score": 101})


def test_breakdown_contains_no_source_content():
    scored = score_project_evidence_fact(
        fact(mechanism="validated source RAW_CONTENT_SENTINEL")
    )
    assert "RAW_CONTENT_SENTINEL" not in json.dumps(scored.quality_breakdown)


def test_scoring_does_not_mutate_input():
    original = fact()
    before = original.to_json()
    score_project_evidence_fact(original)
    assert original.to_json() == before


def test_equivalent_facts_have_identical_score_and_breakdown():
    one = score_project_evidence_fact(fact())
    two = score_project_evidence_fact(fact())
    assert one.quality_score == two.quality_score
    assert one.quality_breakdown == two.quality_breakdown


def test_input_order_does_not_affect_scores_or_output_order():
    alpha = fact(project_id="alpha", source_refs=[source(project_id="alpha")])
    zeta = fact(project_id="zeta", source_refs=[source(project_id="zeta")])
    first, first_report = score_project_evidence_facts([zeta, alpha])
    second, second_report = score_project_evidence_facts([alpha, zeta])
    assert [item.to_json() for item in first] == [item.to_json() for item in second]
    assert first_report.to_dict() == second_report.to_dict()


def test_bounded_decision_sample_is_input_order_independent():
    items = [
        fact(
            project_id=f"project-{index:03d}",
            source_refs=[source(project_id=f"project-{index:03d}")],
        )
        for index in range(105)
    ]
    _, first_report = score_project_evidence_facts(items)
    _, second_report = score_project_evidence_facts(reversed(items))
    assert len(first_report.decisions) == 100
    assert first_report.decisions == second_report.decisions


def test_grouped_structural_status_preserves_stepe_status_on_downgrade():
    item = fact(
        safe_impact=["Guaranteed reliability"],
        status=EvidenceStatus.ACCEPTED,
    )
    _, report = score_project_evidence_facts([item])
    assert report.status_changed_count == 1
    assert report.grouped_counts[0].structural_status == EvidenceStatus.ACCEPTED.value


def test_repeated_scoring_is_idempotent():
    once = score_project_evidence_fact(fact())
    twice = score_project_evidence_fact(once)
    assert once.to_json() == twice.to_json()


def test_machine_paths_timestamps_and_tag_order_do_not_affect_score():
    one_ref = source(file_path="C:/machine-one/repo/a.py", metadata={"created_at": "2026-01-01"})
    two_ref = source(file_path="D:/machine-two/repo/a.py", metadata={"created_at": "2027-01-01"})
    one = evaluate_project_evidence_quality(fact(source_refs=[one_ref], technical_tags=["Python", "validation"]))
    two = evaluate_project_evidence_quality(fact(source_refs=[two_ref], technical_tags=["validation", "Python"]))
    assert one.score == two.score
    assert one.breakdown == two.breakdown


def test_implementation_order_is_preserved_and_scoring_is_stable():
    first = fact(implementation=["read", "validate", "write"])
    second = fact(implementation=["write", "validate", "read"])
    first_scored = score_project_evidence_fact(first)
    second_scored = score_project_evidence_fact(second)
    assert first_scored.implementation == ["read", "validate", "write"]
    assert second_scored.implementation == ["write", "validate", "read"]
    assert first_scored.quality_score == second_scored.quality_score


def test_raw_change_direct_provenance_scores_conservatively():
    evaluation = evaluate_project_evidence_quality(
        fact(source_refs=[source(source_type="github_evidence_raw_change_summary")])
    )
    assert evaluation.breakdown["provenance_strength"] == 11


def test_capability_weak_fact_remains_weak():
    capability_ref = source(source_type="github_evidence_capability_fact", file_path=None, symbol=None)
    scored = score_project_evidence_fact(
        fact(source_refs=[capability_ref], implementation=[], status=EvidenceStatus.WEAK)
    )
    assert scored.status is EvidenceStatus.WEAK


def test_duplicate_refs_do_not_inflate_provenance():
    ref = source()
    one = evaluate_project_evidence_quality(fact(source_refs=[ref]))
    duplicated = evaluate_project_evidence_quality(fact(source_refs=[ref, EvidenceSourceRef.from_dict(ref.to_dict())]))
    assert one.breakdown["provenance_strength"] == duplicated.breakdown["provenance_strength"]


def test_multiple_independent_direct_refs_may_improve_provenance():
    one = evaluate_project_evidence_quality(fact(source_refs=[source("one")]))
    two = evaluate_project_evidence_quality(
        fact(source_refs=[source("one", content_hash=DIGEST_A), source("two", content_hash=DIGEST_B)])
    )
    assert two.breakdown["provenance_strength"] > one.breakdown["provenance_strength"]


def test_cross_project_source_mismatch_is_blocked():
    item = fact()
    object.__setattr__(item.source_refs[0], "project_id", "other")
    evaluation = evaluate_project_evidence_quality(item)
    assert "project_provenance_mismatch" in evaluation.blockers


def test_forbidden_raw_fields_and_secret_metadata_remain_rejected():
    payload = source().to_dict()
    payload["raw_patch"] = "RAW"
    with pytest.raises(ValueError, match="unknown"):
        EvidenceSourceRef.from_dict(payload)
    with pytest.raises(ValueError, match="forbidden"):
        source(metadata={"api_key": "SECRET"})


def test_oversized_invalid_fact_is_not_truncated():
    item = fact()
    item.implementation.append("x" * 2001)
    with pytest.raises(ValueError, match="maximum length"):
        score_project_evidence_fact(item)


def test_no_files_are_written(tmp_path):
    marker = tmp_path / "marker.json"
    marker.write_text("{}", encoding="utf-8")
    before = {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()}
    score_project_evidence_facts([fact()])
    after = {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()}
    assert before == after


def test_import_has_no_external_or_runtime_side_effects():
    module_name = "backend.project_evidence_scoring"
    sys.modules.pop(module_name, None)
    before = set(sys.modules)
    importlib.import_module(module_name)
    newly_loaded = set(sys.modules) - before
    assert "backend.api_server" not in newly_loaded
    assert "backend.project_change_pipeline" not in newly_loaded
    assert not any(name.startswith(("chromadb", "openai", "sqlite3", "requests")) for name in newly_loaded)


def test_scoring_decisions_contain_no_complete_evidence_content():
    item = fact(mechanism="validated RAW_EVIDENCE_SENTINEL")
    _, report = score_project_evidence_facts([item])
    assert "RAW_EVIDENCE_SENTINEL" not in json.dumps(report.to_dict())


def test_stepe_output_scores_to_valid_populated_models():
    inputs, _ = load_project_evidence_inputs()
    normalized, _ = dedupe_project_evidence_inputs(inputs)
    facts, _ = synthesize_project_evidence_facts(normalized)
    scored, report = score_project_evidence_facts(facts)
    assert len(scored) == report.output_count == 283
    assert all(isinstance(item, ProjectEvidenceFact) for item in scored)
    assert all(isinstance(item.quality_score, int) for item in scored)
    assert all(item.quality_breakdown for item in scored)


def test_real_data_audit_is_read_only_and_band_totals_match(tmp_path):
    information = Path(__file__).resolve().parents[1] / "information"
    before = {path: path.stat().st_mtime_ns for path in information.rglob("*") if path.is_file()}
    inputs, _ = load_project_evidence_inputs()
    normalized, _ = dedupe_project_evidence_inputs(inputs)
    facts, _ = synthesize_project_evidence_facts(normalized)
    scored, report = score_project_evidence_facts(facts)
    after = {path: path.stat().st_mtime_ns for path in information.rglob("*") if path.is_file()}
    assert report.high_value_count + report.supporting_value_count + report.weak_value_count + report.rejected_value_count == len(scored)
    assert before == after


def test_real_weak_and_project_context_are_not_promoted():
    inputs, _ = load_project_evidence_inputs()
    normalized, _ = dedupe_project_evidence_inputs(inputs)
    facts, _ = synthesize_project_evidence_facts(normalized)
    scored, _ = score_project_evidence_facts(facts)
    assert all(item.status is EvidenceStatus.WEAK for item in scored if item.source_refs[0].source_type == "github_evidence_capability_fact")
    assert all(item.status is EvidenceStatus.SUPPORTING for item in scored if item.source_refs[0].source_type == "project_memory")


def test_real_missing_impact_direct_facts_can_retain_useful_quality():
    inputs, _ = load_project_evidence_inputs()
    normalized, _ = dedupe_project_evidence_inputs(inputs)
    facts, _ = synthesize_project_evidence_facts(normalized)
    scored, _ = score_project_evidence_facts(facts)
    no_impact_direct = [item for item in scored if item.status is EvidenceStatus.ACCEPTED and not item.safe_impact]
    assert no_impact_direct
    assert max(item.quality_score for item in no_impact_direct) >= 60
