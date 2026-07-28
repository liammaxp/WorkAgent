from dataclasses import FrozenInstanceError, replace
import hashlib
from pathlib import Path

import pytest

from backend.project_capability_boundaries import (
    CAPABILITY_CLAIM_POLICY_STATUSES,
    CapabilityClaimPolicy,
    inherit_capability_claim_policy,
    inherit_project_capability_claim_policies,
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
    ClaimSubjectType,
    EvidenceSourceRef,
    EvidenceType,
    MetricSupport,
    ProjectCapabilityFact,
    ProjectClaimBoundary,
    ProjectEvidenceFact,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "information" / "project_evidence_memory.json"


def _fact(
    evidence_id: str,
    *,
    project_id: str = "project-a",
    mechanism: str = "stable hash-based identity",
    metric_support: MetricSupport = MetricSupport.NONE,
    safe_impact: tuple[str, ...] = (),
    forbidden_claims: tuple[str, ...] = (),
    problem: str = "",
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
            content_hash=hashlib.sha256(evidence_id.encode()).hexdigest(),
        )],
        evidence_type=EvidenceType.VALIDATION,
        metric_support=metric_support,
        safe_impact=list(safe_impact),
        forbidden_claims=list(forbidden_claims),
        technical_tags=["quality_dimensions"],
        quality_score=90,
        evidence_fact_id=evidence_id,
    )


def _candidate_and_assessment(facts: tuple[ProjectEvidenceFact, ...]):
    candidate = CapabilityCandidate(
        project_id=facts[0].project_id,
        capability_type="output_quality_control",
        supporting_evidence_ids=tuple(fact.evidence_fact_id for fact in facts),
        supporting_signals=("quality_dimensions",),
        conflicting_signals=(),
        candidate_score=0.0,
        metadata={"evaluation_state": "unscored"},
    )
    assessment = assess_capability_candidate_support(
        candidate=candidate,
        evidence_index={fact.evidence_fact_id: fact for fact in facts},
    )
    return candidate, assessment


def _boundary(fact: ProjectEvidenceFact) -> ProjectClaimBoundary:
    boundary = build_project_evidence_claim_boundary(fact)
    assert boundary is not None
    return boundary


def _manual_boundary(
    fact: ProjectEvidenceFact,
    claim: str,
    *,
    metric_support: MetricSupport = MetricSupport.NONE,
    boundary_id: str | None = None,
) -> ProjectClaimBoundary:
    claim_hash = hashlib.sha256(claim.encode()).hexdigest()[:16]
    note = (
        f"claim_meta|{claim_hash}|high|{metric_support.value}|direct_evidence|"
        f"synthetic_policy_test|e={fact.evidence_fact_id}|c="
    )
    return ProjectClaimBoundary(
        project_id=fact.project_id,
        subject_type=ClaimSubjectType.EVIDENCE_FACT,
        subject_id=fact.evidence_fact_id,
        allowed_claims=[claim],
        forbidden_claims=[],
        metric_support=metric_support,
        notes=[note],
        boundary_id=boundary_id or f"pcb_manual_{fact.evidence_fact_id}",
    )


def _inherit(
    facts: tuple[ProjectEvidenceFact, ...],
    boundaries: list[ProjectClaimBoundary],
):
    candidate, assessment = _candidate_and_assessment(facts)
    policy = inherit_capability_claim_policy(
        candidate=candidate,
        assessment=assessment,
        evidence_index={fact.evidence_fact_id: fact for fact in facts},
        claim_boundaries=boundaries,
    )
    return candidate, assessment, policy


def test_eligible_candidate_inherits_project_claim_boundaries():
    fact = _fact("pef_eligible")
    boundary = _boundary(fact)
    candidate, assessment, policy = _inherit((fact,), [boundary])

    assert assessment.eligibility_status == "eligible"
    assert policy.policy_status == "eligible"
    assert policy.supporting_evidence_ids == candidate.supporting_evidence_ids
    assert policy.inherited_boundary_ids == (boundary.boundary_id,)
    assert policy.allowed_claims
    assert set(boundary.forbidden_claims) <= set(policy.forbidden_claims)
    assert not isinstance(policy, ProjectCapabilityFact)


def test_ineligible_support_assessment_cannot_inherit_claim_policy():
    fact = _fact("pef_ineligible")
    candidate = CapabilityCandidate(
        project_id=fact.project_id,
        capability_type="llm_reliability",
        supporting_evidence_ids=(fact.evidence_fact_id,),
        supporting_signals=("source_grounding", "claim_validation", "structured_output_validation"),
        conflicting_signals=(),
        candidate_score=0.0,
        metadata={"evaluation_state": "unscored"},
    )
    tagged = replace(
        fact,
        technical_tags=["source_grounding", "claim_validation", "structured_output_validation"],
    )
    assessment = assess_capability_candidate_support(
        candidate=candidate, evidence_index={tagged.evidence_fact_id: tagged}
    )
    policy = inherit_capability_claim_policy(
        candidate=candidate,
        assessment=assessment,
        evidence_index={},
        claim_boundaries=[],
    )
    assert assessment.eligibility_status == "insufficient_evidence"
    assert policy.policy_status == "ineligible_support"
    assert policy.allowed_claims == ()
    assert policy.inherited_boundary_ids == ()


def test_missing_claim_boundaries_fail_closed():
    fact = _fact("pef_missing")
    _candidate, _assessment, policy = _inherit((fact,), [])
    assert policy.policy_status == "missing_boundaries"
    assert policy.allowed_claims == ()
    assert policy.uncovered_evidence_ids == (fact.evidence_fact_id,)


def test_cross_project_claim_boundary_is_rejected():
    fact = _fact("pef_project_a")
    other = _fact("pef_project_b", project_id="project-b")
    with pytest.raises(ValueError, match="cross-project Claim Boundary"):
        _inherit((fact,), [_boundary(other)])


def test_cross_project_evidence_index_is_rejected():
    fact = _fact("pef_cross_evidence")
    candidate, assessment = _candidate_and_assessment((fact,))
    other = _fact("pef_cross_evidence", project_id="project-b")
    with pytest.raises(ValueError, match="cross-project Evidence Fact"):
        inherit_capability_claim_policy(
            candidate=candidate,
            assessment=assessment,
            evidence_index={other.evidence_fact_id: other},
            claim_boundaries=[],
        )


def test_claim_boundary_referencing_missing_evidence_fails_closed():
    fact = _fact("pef_candidate")
    candidate, assessment = _candidate_and_assessment((fact,))
    missing = _fact("pef_missing_boundary_subject")
    with pytest.raises(ValueError, match="invalid Claim Boundary structure"):
        inherit_capability_claim_policy(
            candidate=candidate,
            assessment=assessment,
            evidence_index={fact.evidence_fact_id: fact},
            claim_boundaries=[_boundary(missing)],
        )


def test_uncovered_supporting_evidence_is_reported_and_blocks_policy():
    first = _fact("pef_covered")
    second = _fact("pef_uncovered", mechanism="atomic artifact replacement")
    _candidate, _assessment, policy = _inherit((first, second), [_boundary(first)])
    assert policy.policy_status == "missing_boundaries"
    assert policy.covered_evidence_count == 1
    assert policy.uncovered_evidence_ids == (second.evidence_fact_id,)


def test_allowed_claims_are_normalized_deduplicated_and_deterministic():
    fact = _fact("pef_normalized", mechanism="  stable   hash-based identity  ")
    boundary = _boundary(fact)
    first = _inherit((fact,), [boundary, boundary])[2]
    second = _inherit((fact,), [boundary])[2]
    assert first == second
    assert first.allowed_claims == tuple(sorted(set(first.allowed_claims), key=lambda x: (x.casefold(), x)))
    assert "mechanism:stable hash-based identity" in first.allowed_claims


def test_forbidden_claims_include_boundary_and_taxonomy_restrictions():
    fact = _fact("pef_forbidden")
    boundary = _boundary(fact)
    policy = _inherit((fact,), [boundary])[2]
    assert set(boundary.forbidden_claims) <= set(policy.forbidden_claims)
    assert set(policy.taxonomy_forbidden_claims) <= set(policy.forbidden_claims)


def test_allowed_and_forbidden_claim_collision_is_conflicted():
    first = _fact("pef_allow", mechanism="stable hash-based identity")
    second = _fact(
        "pef_forbid",
        mechanism="atomic artifact replacement",
        forbidden_claims=("mechanism:stable hash-based identity",),
    )
    policy = _inherit((first, second), [_boundary(first), _boundary(second)])[2]
    assert policy.policy_status == "boundary_conflict"
    assert policy.has_boundary_conflict
    assert "allowed_forbidden_claim_collision" in policy.reasons
    assert "mechanism:stable hash-based identity" not in policy.allowed_claims


def test_taxonomy_allowed_claims_do_not_replace_project_boundaries():
    fact = _fact("pef_taxonomy_allowed")
    policy = _inherit((fact,), [])[2]
    assert policy.taxonomy_allowed_claims
    assert policy.allowed_claims == ()
    assert policy.policy_status == "missing_boundaries"


def test_taxonomy_forbidden_claims_constrain_project_policy():
    fact = _fact("pef_taxonomy_forbidden")
    policy = _inherit((fact,), [_boundary(fact)])[2]
    assert policy.taxonomy_forbidden_claims
    assert set(policy.taxonomy_forbidden_claims) <= set(policy.forbidden_claims)


def test_numeric_allowed_claim_requires_authoritative_metric_support():
    fact = _fact("pef_metric_none")
    boundary = _manual_boundary(fact, "metric:Reduced processing time by 25%")
    policy = _inherit((fact,), [boundary])[2]
    assert policy.policy_status == "metric_conflict"
    assert policy.metric_support == "none"
    assert "numeric_claim_without_metric_support" in policy.reasons
    assert not any("25%" in claim for claim in policy.allowed_claims)


def test_approximate_metric_support_does_not_authorize_explicit_claim():
    fact = _fact("pef_metric_approx", metric_support=MetricSupport.APPROXIMATE)
    boundary = _manual_boundary(
        fact,
        "metric:Reduced processing time by 25%",
        metric_support=MetricSupport.APPROXIMATE,
    )
    policy = _inherit((fact,), [boundary])[2]
    assert policy.policy_status == "metric_conflict"
    assert policy.metric_support == "none"


def test_explicit_metric_support_is_preserved_only_for_exact_supported_claim():
    claim = "Reduced processing time by 25%"
    fact = _fact(
        "pef_metric_explicit",
        metric_support=MetricSupport.EXPLICIT,
        safe_impact=(claim,),
    )
    policy = _inherit((fact,), [_boundary(fact)])[2]
    assert policy.policy_status == "eligible"
    assert policy.metric_support == "explicit"
    assert f"metric:{claim}" in policy.allowed_claims


def test_conflicting_metric_boundaries_fail_closed():
    claim = "Approximately 25% lower processing time"
    approximate = _fact(
        "pef_metric_a", metric_support=MetricSupport.APPROXIMATE, safe_impact=(claim,)
    )
    explicit = _fact(
        "pef_metric_b",
        mechanism="atomic artifact replacement",
        metric_support=MetricSupport.EXPLICIT,
        safe_impact=(claim,),
    )
    policy = _inherit(
        (approximate, explicit), [_boundary(approximate), _boundary(explicit)]
    )[2]
    assert policy.policy_status == "metric_conflict"
    assert policy.has_metric_conflict
    assert "metric_support_conflict" in policy.reasons


def test_absolute_guarantee_claim_is_blocked_by_taxonomy_policy():
    fact = _fact("pef_absolute")
    boundary = _manual_boundary(fact, "impact:Guaranteed factual correctness")
    policy = _inherit((fact,), [boundary])[2]
    assert policy.policy_status == "boundary_conflict"
    assert policy.has_boundary_conflict
    assert "taxonomy_forbidden_claim_collision" in policy.reasons
    assert not any("Guaranteed" in claim for claim in policy.allowed_claims)


def test_duplicate_boundaries_are_deduplicated_or_rejected_safely():
    first = _fact("pef_duplicate_a")
    second = _fact("pef_duplicate_b", mechanism="atomic artifact replacement")
    first_boundary = _boundary(first)
    identical_policy = _inherit((first,), [first_boundary, first_boundary])[2]
    assert identical_policy.boundary_count == 1

    conflicting = replace(_boundary(second), boundary_id=first_boundary.boundary_id)
    with pytest.raises(ValueError, match="conflicting semantic content"):
        _inherit((first, second), [first_boundary, conflicting])


def test_conflicting_duplicate_evidence_facts_fail_closed():
    first = _fact("pef_duplicate_fact")
    conflicting = _fact("pef_duplicate_fact", mechanism="atomic artifact replacement")
    candidate, assessment = _candidate_and_assessment((first,))
    with pytest.raises(ValueError, match="conflicting semantic content"):
        inherit_project_capability_claim_policies(
            project_id="project-a",
            candidates=[candidate],
            assessments=[assessment],
            evidence_facts=[first, conflicting],
            claim_boundaries=[_boundary(first)],
        )


def test_assessment_evidence_mismatch_fails_closed():
    first = _fact("pef_assessment_a")
    second = _fact("pef_assessment_b")
    candidate, assessment = _candidate_and_assessment((first,))
    mismatched = replace(assessment, supporting_evidence_ids=(second.evidence_fact_id,))
    with pytest.raises(ValueError, match="Evidence IDs"):
        inherit_capability_claim_policy(
            candidate=candidate,
            assessment=mismatched,
            evidence_index={first.evidence_fact_id: first},
            claim_boundaries=[_boundary(first)],
        )


def test_claim_policy_inheritance_does_not_mutate_inputs():
    fact = _fact("pef_immutable")
    boundary = _boundary(fact)
    candidate, assessment = _candidate_and_assessment((fact,))
    candidate_before = candidate.to_dict()
    assessment_before = assessment.to_dict()
    fact_before = fact.to_json()
    boundary_before = boundary.to_json()
    policy = inherit_capability_claim_policy(
        candidate=candidate,
        assessment=assessment,
        evidence_index={fact.evidence_fact_id: fact},
        claim_boundaries=[boundary],
    )
    assert candidate.to_dict() == candidate_before
    assert assessment.to_dict() == assessment_before
    assert fact.to_json() == fact_before
    assert boundary.to_json() == boundary_before
    with pytest.raises(FrozenInstanceError):
        policy.policy_status = "missing_boundaries"
    with pytest.raises(TypeError):
        policy.diagnostics["changed"] = True


def test_claim_policy_inheritance_is_input_order_independent():
    first = _fact("pef_order_z")
    second = _fact("pef_order_a", mechanism="atomic artifact replacement")
    candidate, assessment = _candidate_and_assessment((first, second))
    forward = inherit_capability_claim_policy(
        candidate=candidate,
        assessment=assessment,
        evidence_index={first.evidence_fact_id: first, second.evidence_fact_id: second},
        claim_boundaries=[_boundary(first), _boundary(second)],
    )
    reverse = inherit_capability_claim_policy(
        candidate=candidate,
        assessment=assessment,
        evidence_index={second.evidence_fact_id: second, first.evidence_fact_id: first},
        claim_boundaries=[_boundary(second), _boundary(first)],
    )
    assert forward == reverse
    assert forward.to_json() == reverse.to_json()


def test_capability_claim_policy_serialization_is_safe_and_deterministic():
    sentinel = "raw_patch=PRIVATE_GITHUB_SENTINEL"
    fact = _fact("pef_safe", problem=sentinel)
    policy = _inherit((fact,), [_boundary(fact)])[2]
    serialized = policy.to_json()
    assert serialized == policy.to_json()
    assert sentinel not in serialized
    assert not {"raw", "patch", "diff", "source_code", "github_context"} & set(policy.to_dict())


def test_claim_policy_inheritance_does_not_generate_project_capability_fact():
    fact = _fact("pef_no_fact")
    policy = _inherit((fact,), [_boundary(fact)])[2]
    assert isinstance(policy, CapabilityClaimPolicy)
    assert not isinstance(policy, ProjectCapabilityFact)
    assert "capability_id" not in policy.to_dict()
    assert CAPABILITY_CLAIM_POLICY_STATUSES == {
        "eligible", "ineligible_support", "missing_boundaries",
        "insufficient_allowed_claims", "boundary_conflict", "metric_conflict",
    }


def test_real_ineligible_candidates_do_not_enter_boundary_inheritance():
    before = ARTIFACT.read_bytes()
    before_mtime = ARTIFACT.stat().st_mtime_ns
    loaded = load_project_evidence_memory(ARTIFACT)
    assert loaded.status == "ready" and loaded.snapshot is not None
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
        assessments.extend(project_assessments)
        policies.extend(inherit_project_capability_claim_policies(
            project_id=project.project_id,
            candidates=grouping.candidates,
            assessments=project_assessments,
            evidence_facts=project.evidence_facts,
            claim_boundaries=project.claim_boundaries,
        ))
    assert len(loaded.snapshot.projects) == 11
    assert len(assessments) == 57
    assert not any(item.eligibility_status == "eligible" for item in assessments)
    assert policies == []
    assert sum(len(project.capability_facts) for project in loaded.snapshot.projects) == 0
    assert ARTIFACT.read_bytes() == before
    assert ARTIFACT.stat().st_mtime_ns == before_mtime
    assert not (ROOT / "information" / "project_capability_memory.json").exists()


def test_project_capability_boundaries_use_semantic_naming():
    source = (ROOT / "backend" / "project_capability_boundaries.py").read_text(encoding="utf-8").casefold()
    forbidden = ("phase" + "5", "phase_" + "5", "project_memory_" + "phase" + "5", "use_" + "phase" + "5")
    assert not any(token in source for token in forbidden)
