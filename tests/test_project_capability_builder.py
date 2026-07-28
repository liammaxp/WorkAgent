from dataclasses import FrozenInstanceError, replace
import hashlib
from pathlib import Path

import pytest

import backend.project_capability_builder as builder_module
from backend.project_capability_boundaries import (
    CapabilityClaimPolicy,
    inherit_capability_claim_policy,
    inherit_project_capability_claim_policies,
)
from backend.project_capability_builder import (
    PROJECT_CAPABILITY_FACT_BUILD_STATUSES,
    ProjectCapabilityFactBuildResult,
    build_project_capability_fact,
    build_project_capability_facts,
)
from backend.project_capability_grouping import group_project_evidence_facts
from backend.project_capability_memory import CapabilityCandidate
from backend.project_capability_scoring import (
    assess_capability_candidate_support,
    assess_project_capability_candidates,
)
from backend.project_claim_boundaries import build_project_evidence_claim_boundary
from backend.project_evidence_memory import load_project_evidence_memory
from backend.project_evidence_models import (
    Confidence,
    EvidenceSourceRef,
    EvidenceType,
    MetricSupport,
    ProjectCapabilityFact,
    ProjectEvidenceFact,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "information" / "project_evidence_memory.json"
CAPABILITY_ARTIFACT = ROOT / "information" / "project_capability_memory.json"
EXPECTED_CONTENT_HASH = "37967289816ec13638b4b30e31a74f52688acc9bc08ff6c6faf760b2c6180fd3"
EXPECTED_FILE_HASH = "95750df456d1fb3dea56cf40891593834a52731414a882896d99aa5a51b3f106"


def _fact(
    evidence_id: str,
    *,
    project_id: str = "project-a",
    mechanism: str = "stable hash-based evidence identity",
    signals: tuple[str, ...] = ("quality_dimensions",),
    technical_tags: tuple[str, ...] = ("FastAPI",),
    quality: float = 90,
    metric_support: MetricSupport = MetricSupport.NONE,
    safe_impact: tuple[str, ...] = (),
    problem: str = "",
    evidence_type: EvidenceType = EvidenceType.VALIDATION,
) -> ProjectEvidenceFact:
    return ProjectEvidenceFact(
        project_id=project_id,
        problem=problem,
        mechanism=mechanism,
        implementation=["Applied deterministic structured validation."],
        source_refs=[EvidenceSourceRef(
            source_type="github_evidence_card",
            source_id=f"source-{evidence_id}",
            project_id=project_id,
            content_hash=hashlib.sha256(f"{project_id}:{evidence_id}".encode()).hexdigest(),
        )],
        evidence_type=evidence_type,
        metric_support=metric_support,
        safe_impact=list(safe_impact),
        technical_tags=[*signals, *technical_tags],
        quality_score=quality,
        evidence_fact_id=evidence_id,
    )


def _candidate(
    facts: tuple[ProjectEvidenceFact, ...],
    *,
    capability_type: str = "output_quality_control",
    signals: tuple[str, ...] = ("quality_dimensions",),
) -> CapabilityCandidate:
    return CapabilityCandidate(
        project_id=facts[0].project_id,
        capability_type=capability_type,
        supporting_evidence_ids=tuple(fact.evidence_fact_id for fact in facts),
        supporting_signals=signals,
        conflicting_signals=(),
        candidate_score=0.0,
        metadata={"evaluation_state": "unscored"},
    )


def _verified(
    facts: tuple[ProjectEvidenceFact, ...] | None = None,
    *,
    capability_type: str = "output_quality_control",
    signals: tuple[str, ...] = ("quality_dimensions",),
) -> tuple[CapabilityCandidate, object, CapabilityClaimPolicy, tuple[ProjectEvidenceFact, ...]]:
    facts = facts or (_fact("pef_verified"),)
    candidate = _candidate(facts, capability_type=capability_type, signals=signals)
    evidence_index = {fact.evidence_fact_id: fact for fact in facts}
    assessment = assess_capability_candidate_support(
        candidate=candidate, evidence_index=evidence_index
    )
    boundaries = []
    for fact in facts:
        boundary = build_project_evidence_claim_boundary(fact)
        assert boundary is not None
        boundaries.append(boundary)
    policy = inherit_capability_claim_policy(
        candidate=candidate,
        assessment=assessment,
        evidence_index=evidence_index,
        claim_boundaries=boundaries,
    )
    assert assessment.eligibility_status == "eligible"
    assert policy.policy_status == "eligible"
    return candidate, assessment, policy, facts


def _build(verified=None):
    candidate, assessment, policy, facts = verified or _verified()
    return build_project_capability_fact(
        candidate=candidate,
        assessment=assessment,
        policy=policy,
        evidence_index={fact.evidence_fact_id: fact for fact in facts},
    )


def test_builds_authoritative_project_capability_fact_from_verified_inputs():
    candidate, _assessment, policy, facts = _verified()
    result = _build((candidate, _assessment, policy, facts))
    assert result.build_status == "built"
    assert isinstance(result.fact, ProjectCapabilityFact)
    assert result.fact.present is True
    assert tuple(result.fact.source_evidence_fact_ids) == candidate.supporting_evidence_ids
    assert result.fact.mechanisms == [facts[0].mechanism]
    assert result.fact.allowed_resume_claims == list(policy.allowed_claims)
    assert result.fact.forbidden_claims == list(policy.forbidden_claims)
    assert result.fact.metric_support.value == policy.metric_support
    assert result.fact.capability_id.startswith("pcf_")
    assert not CAPABILITY_ARTIFACT.exists()


def test_ineligible_support_assessment_cannot_build_fact():
    candidate, assessment, policy, facts = _verified()
    result = _build((candidate, replace(assessment, eligibility_status="insufficient_evidence"), policy, facts))
    assert result.build_status == "ineligible_support" and result.fact is None


def test_ineligible_claim_policy_cannot_build_fact():
    candidate, assessment, policy, facts = _verified()
    result = _build((candidate, assessment, replace(policy, policy_status="missing_boundaries"), facts))
    assert result.build_status == "ineligible_claim_policy" and result.fact is None


def test_high_support_score_cannot_bypass_ineligible_status():
    candidate, assessment, policy, facts = _verified()
    blocked = replace(assessment, support_score=1.0, eligibility_status="insufficient_signal_coverage")
    result = _build((candidate, blocked, policy, facts))
    assert result.build_status == "ineligible_support" and result.fact is None


def test_candidate_and_assessment_identity_mismatch_fails_closed():
    candidate, assessment, policy, facts = _verified()
    result = _build((candidate, replace(assessment, project_id="project-b"), policy, facts))
    assert result.build_status == "identity_mismatch"
    assert result.reasons == ("assessment_project_mismatch",)


def test_candidate_and_policy_identity_mismatch_fails_closed():
    candidate, assessment, policy, facts = _verified()
    result = _build((candidate, assessment, replace(policy, capability_type="failure_recovery"), facts))
    assert result.build_status == "identity_mismatch"
    assert result.reasons == ("policy_capability_mismatch",)


def test_candidate_assessment_policy_evidence_mismatch_fails_closed():
    candidate, assessment, policy, facts = _verified()
    result = _build((candidate, assessment, replace(policy, supporting_evidence_ids=("pef_other",)), facts))
    assert result.build_status == "evidence_mismatch" and result.fact is None


def test_cross_project_evidence_cannot_build_capability_fact():
    candidate, assessment, policy, facts = _verified()
    foreign = _fact(facts[0].evidence_fact_id, project_id="project-b")
    result = build_project_capability_fact(
        candidate=candidate,
        assessment=assessment,
        policy=policy,
        evidence_index={foreign.evidence_fact_id: foreign},
    )
    assert result.build_status == "evidence_mismatch"
    assert result.reasons == ("cross_project_evidence",)


def test_fact_mechanisms_are_derived_only_from_supporting_evidence():
    facts = (
        _fact("pef_mech_a", mechanism="stable hash-based evidence identity"),
        _fact("pef_mech_b", mechanism=" stable  hash-based evidence identity "),
    )
    candidate, assessment, policy, facts = _verified(facts)
    unrelated = _fact("pef_unrelated", mechanism="taxonomy template architecture")
    result = build_project_capability_fact(
        candidate=candidate,
        assessment=assessment,
        policy=policy,
        evidence_index={**{fact.evidence_fact_id: fact for fact in facts}, unrelated.evidence_fact_id: unrelated},
    )
    assert result.fact is not None
    assert result.fact.mechanisms == ["stable hash-based evidence identity"]
    assert unrelated.mechanism not in result.fact.mechanisms


def test_missing_concrete_mechanisms_fail_closed():
    candidate, assessment, policy, facts = _verified()
    generic = replace(facts[0], mechanism="Implemented")
    result = build_project_capability_fact(
        candidate=candidate,
        assessment=assessment,
        policy=policy,
        evidence_index={generic.evidence_fact_id: generic},
    )
    assert result.build_status == "missing_mechanisms" and result.fact is None


def test_fact_technical_tags_cannot_leak_from_other_projects_or_taxonomy():
    supporting = _fact(
        "pef_tech",
        technical_tags=("FastAPI", "jd:Kubernetes", "taxonomy-derived Elasticsearch"),
    )
    candidate, assessment, policy, facts = _verified((supporting,))
    compatible_but_unsupported = _fact("pef_taxonomy_tech", technical_tags=("Elasticsearch",))
    foreign = _fact("pef_foreign_tech", project_id="project-b", technical_tags=("React",))
    result = build_project_capability_fact(
        candidate=candidate,
        assessment=assessment,
        policy=policy,
        evidence_index={
            supporting.evidence_fact_id: supporting,
            compatible_but_unsupported.evidence_fact_id: compatible_but_unsupported,
            foreign.evidence_fact_id: foreign,
        },
    )
    assert result.fact is not None
    assert result.fact.technical_tags == ["FastAPI"]


def test_fact_allowed_claims_are_copied_only_from_eligible_policy():
    candidate, assessment, policy, facts = _verified()
    result = _build((candidate, assessment, policy, facts))
    assert result.fact is not None
    assert result.fact.allowed_resume_claims == list(policy.allowed_claims)
    assert not set(policy.taxonomy_allowed_claims) - set(policy.allowed_claims) & set(result.fact.allowed_resume_claims)


def test_fact_preserves_all_policy_forbidden_claims():
    candidate, assessment, policy, facts = _verified()
    result = _build((candidate, assessment, policy, facts))
    assert result.fact is not None
    assert set(result.fact.forbidden_claims) == set(policy.forbidden_claims)


def test_allowed_forbidden_claim_collision_blocks_fact():
    candidate, assessment, policy, facts = _verified()
    collision = replace(policy, forbidden_claims=(*policy.forbidden_claims, policy.allowed_claims[0]))
    result = _build((candidate, assessment, collision, facts))
    assert result.build_status == "ineligible_claim_policy"
    assert result.reasons == ("allowed_forbidden_claim_collision",)


def test_fact_preserves_policy_metric_support_without_escalation():
    candidate, assessment, policy, facts = _verified()
    approximate = replace(policy, metric_support="approximate")
    result = _build((candidate, assessment, approximate, facts))
    assert result.fact is not None
    assert result.fact.metric_support is MetricSupport.APPROXIMATE


def test_numeric_claim_incompatible_with_metric_support_blocks_fact():
    candidate, assessment, policy, facts = _verified()
    inconsistent = replace(policy, allowed_claims=("metric:Reduced processing time by 25%",))
    result = _build((candidate, assessment, inconsistent, facts))
    assert result.build_status == "invalid_metric_support" and result.fact is None


def test_fact_confidence_is_deterministic_and_uses_authoritative_rules():
    facts = (
        _fact("pef_conf_a", quality=100),
        _fact("pef_conf_b", mechanism="atomic artifact replacement", quality=100),
    )
    verified = _verified(facts)
    first = _build(verified)
    second = _build(verified)
    assert first.fact is not None and second.fact is not None
    assert first.fact.confidence is Confidence.HIGH
    assert first.fact.confidence == second.fact.confidence
    assert first.diagnostics["confidence_derivation"] == "authoritative_extractor_proof_rule"


def test_builder_does_not_emit_low_confidence_fact_for_ineligible_candidate():
    candidate, assessment, policy, facts = _verified()
    blocked = replace(
        assessment,
        eligibility_status="insufficient_evidence",
        meets_evidence_minimum=False,
    )
    result = _build((candidate, blocked, policy, facts))
    assert result.fact is None and result.build_status == "ineligible_support"


def test_fact_uses_authoritative_stable_pcf_identity():
    facts = (
        _fact("pef_id_z"),
        _fact("pef_id_a", mechanism="atomic artifact replacement"),
    )
    candidate, assessment, policy, facts = _verified(tuple(reversed(facts)))
    first = _build((candidate, assessment, policy, facts))
    second = build_project_capability_fact(
        candidate=candidate,
        assessment=assessment,
        policy=policy,
        evidence_index={fact.evidence_fact_id: fact for fact in reversed(facts)},
    )
    assert first.fact is not None and second.fact is not None
    expected = ProjectCapabilityFact(
        project_id=first.fact.project_id,
        capability_type=first.fact.capability_type,
        present=True,
        source_evidence_fact_ids=list(first.fact.source_evidence_fact_ids),
        confidence=first.fact.confidence,
        mechanisms=list(first.fact.mechanisms),
        allowed_resume_claims=list(first.fact.allowed_resume_claims),
        forbidden_claims=list(first.fact.forbidden_claims),
        metric_support=first.fact.metric_support,
        technical_tags=list(first.fact.technical_tags),
    )
    assert first.fact.capability_id == second.fact.capability_id == expected.capability_id
    assert first.fact.capability_id.startswith("pcf_") and len(first.fact.capability_id) == 28


def test_builder_returns_existing_project_capability_fact_model():
    result = _build()
    assert type(result.fact) is ProjectCapabilityFact
    assert not hasattr(builder_module, "CapabilityFact")


def test_duplicate_lifecycle_records_are_deduplicated_or_rejected_safely():
    candidate, assessment, policy, facts = _verified()
    results = build_project_capability_facts(
        project_id="project-a",
        candidates=[candidate, candidate],
        assessments=[assessment, assessment],
        policies=[policy, policy],
        evidence_facts=[facts[0], facts[0]],
    )
    assert len(results) == 1 and results[0].build_status == "built"
    with pytest.raises(ValueError, match="conflicting assessments"):
        build_project_capability_facts(
            project_id="project-a",
            candidates=[candidate],
            assessments=[assessment, replace(assessment, support_score=0.123456)],
            policies=[policy],
            evidence_facts=list(facts),
        )


def test_batch_builder_requires_complete_candidate_assessment_policy_pairs():
    candidate, assessment, _policy, facts = _verified()
    with pytest.raises(ValueError, match="complete candidate"):
        build_project_capability_facts(
            project_id="project-a",
            candidates=[candidate],
            assessments=[assessment],
            policies=[],
            evidence_facts=list(facts),
        )


def test_batch_fact_building_is_input_order_independent():
    first = _verified((_fact("pef_batch_quality"),))
    recovery_fact = _fact(
        "pef_batch_recovery",
        mechanism="deterministic fallback recovery",
        signals=("error_state_handling", "fallback"),
        evidence_type=EvidenceType.FAILURE_RECOVERY,
    )
    second = _verified(
        (recovery_fact,),
        capability_type="failure_recovery",
        signals=("error_state_handling", "fallback"),
    )
    candidates = [first[0], second[0]]
    assessments = [first[1], second[1]]
    policies = [first[2], second[2]]
    facts = [*first[3], *second[3]]
    forward = build_project_capability_facts(
        project_id="project-a",
        candidates=candidates,
        assessments=assessments,
        policies=policies,
        evidence_facts=facts,
    )
    reverse = build_project_capability_facts(
        project_id="project-a",
        candidates=list(reversed(candidates)),
        assessments=list(reversed(assessments)),
        policies=list(reversed(policies)),
        evidence_facts=list(reversed(facts)),
    )
    assert tuple(item.to_json() for item in forward) == tuple(item.to_json() for item in reverse)


def test_fact_builder_does_not_mutate_inputs():
    candidate, assessment, policy, facts = _verified()
    before = (candidate.to_dict(), assessment.to_dict(), policy.to_dict(), tuple(fact.to_json() for fact in facts))
    result = _build((candidate, assessment, policy, facts))
    after = (candidate.to_dict(), assessment.to_dict(), policy.to_dict(), tuple(fact.to_json() for fact in facts))
    assert before == after
    with pytest.raises(FrozenInstanceError):
        result.build_status = "invalid_fact"
    with pytest.raises(TypeError):
        result.diagnostics["changed"] = True


def test_fact_build_result_serialization_is_deterministic_and_safe():
    candidate, assessment, policy, facts = _verified()
    sentinel = "raw_patch=PRIVATE_GITHUB_SENTINEL"
    private_fact = replace(facts[0], problem=sentinel)
    result = build_project_capability_fact(
        candidate=candidate,
        assessment=assessment,
        policy=policy,
        evidence_index={private_fact.evidence_fact_id: private_fact},
    )
    serialized = result.to_json()
    assert serialized == result.to_json()
    assert sentinel not in serialized
    assert not {"raw", "patch", "diff", "source_code", "github_context"} & set(result.to_dict())
    unsafe_policy = replace(policy, allowed_claims=(sentinel,))
    blocked = _build((candidate, assessment, unsafe_policy, facts))
    assert blocked.build_status == "ineligible_claim_policy" and blocked.fact is None
    assert sentinel not in blocked.to_json()


def test_fact_builder_does_not_persist_capability_memory():
    before = ARTIFACT.read_bytes()
    assert not CAPABILITY_ARTIFACT.exists()
    assert _build().build_status == "built"
    assert ARTIFACT.read_bytes() == before
    assert not CAPABILITY_ARTIFACT.exists()


def test_real_ineligible_lifecycle_builds_zero_capability_facts_read_only():
    before = ARTIFACT.read_bytes()
    before_mtime = ARTIFACT.stat().st_mtime_ns
    loaded = load_project_evidence_memory(ARTIFACT)
    assert loaded.status == "ready" and loaded.snapshot is not None
    candidates = []
    assessments = []
    policies = []
    for project in loaded.snapshot.projects:
        grouping = group_project_evidence_facts(
            project_id=project.project_id, evidence_facts=project.evidence_facts
        )
        project_assessments = assess_project_capability_candidates(
            project_id=project.project_id,
            candidates=grouping.candidates,
            evidence_facts=project.evidence_facts,
        )
        project_policies = inherit_project_capability_claim_policies(
            project_id=project.project_id,
            candidates=grouping.candidates,
            assessments=project_assessments,
            evidence_facts=project.evidence_facts,
            claim_boundaries=project.claim_boundaries,
        )
        candidates.extend(grouping.candidates)
        assessments.extend(project_assessments)
        policies.extend(project_policies)
    assert len(loaded.snapshot.projects) == 11
    assert len(candidates) == len(assessments) == 57
    assert not any(item.eligibility_status == "eligible" for item in assessments)
    assert policies == []
    assert sum(len(project.capability_facts) for project in loaded.snapshot.projects) == 0
    assert loaded.snapshot.content_hash == EXPECTED_CONTENT_HASH
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == EXPECTED_FILE_HASH
    assert ARTIFACT.read_bytes() == before
    assert ARTIFACT.stat().st_mtime_ns == before_mtime
    assert not CAPABILITY_ARTIFACT.exists()


def test_project_capability_builder_uses_semantic_naming():
    source = (ROOT / "backend" / "project_capability_builder.py").read_text(encoding="utf-8").casefold()
    forbidden = (
        "phase" + "5",
        "phase_" + "5",
        "project_memory_" + "phase" + "5",
        "project_capability_" + "phase" + "5",
        "use_" + "phase" + "5",
        "phase" + "5.v1",
    )
    assert not any(token in source for token in forbidden)
    assert PROJECT_CAPABILITY_FACT_BUILD_STATUSES == {
        "built", "ineligible_support", "ineligible_claim_policy", "identity_mismatch",
        "evidence_mismatch", "missing_mechanisms", "missing_allowed_claims",
        "invalid_metric_support", "invalid_fact",
    }
    assert isinstance(ProjectCapabilityFactBuildResult.__dataclass_fields__, dict)
