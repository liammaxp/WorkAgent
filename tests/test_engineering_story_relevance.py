from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import inspect
from pathlib import Path

import pytest

from backend.engineering_story_memory_service import EngineeringStoryView
from backend.engineering_story_models import (
    ClaimSufficiency,
    EngineeringStory,
    EngineeringStoryField,
    EngineeringStoryFieldName,
    EngineeringStoryLifecycle,
    EngineeringStoryStatus,
    EngineeringStoryType,
    StoryContextGap,
    StoryFieldEvidenceState,
    StoryOpportunity,
    StoryOpportunityLevel,
    StoryOpportunitySignal,
    StorySufficiency,
    SufficiencyLevel,
)
from backend.engineering_story_relevance import (
    MAX_STORY_RELEVANCE_BATCH,
    MAX_STORY_RELEVANCE_FEATURES,
    MAX_STORY_RELEVANCE_REASONS,
    StoryHiringRelevance,
    StoryRelevanceComponents,
    StoryRelevanceEvaluationError,
    StoryRelevanceEvaluationErrorCode,
    StoryRelevanceFeature,
    StoryRelevanceReason,
    StoryRelevanceWeights,
    evaluate_engineering_story_relevance,
    rank_engineering_stories_for_hiring_context,
)
from backend.hiring_context_intelligence import build_hiring_context_profile
from backend.hiring_context_models import (
    HiringContextConfidence,
    HiringContextProfile,
    HiringContextSignalKind,
    RoleFamily,
)
from backend.hiring_context_organization import (
    DEFAULT_ORGANIZATION_CONTEXT_REGISTRY,
    OrganizationContextRegistry,
    OrganizationContextRule,
    OrganizationDomainSignal,
)


PROJECT_ID = "story_relevance"
EVIDENCE_ID = "pef_story_relevance"
CAPABILITY_ID = "pcf_story_relevance"
STORY_FIELD_NAMES = tuple(item.value for item in EngineeringStoryFieldName)


def positive(
    value: str,
    state: StoryFieldEvidenceState = StoryFieldEvidenceState.CONFIRMED,
) -> EngineeringStoryField:
    return EngineeringStoryField(
        value=value,
        evidence_state=state,
        evidence_fact_ids=(EVIDENCE_ID,),
    )


def absent(
    state: StoryFieldEvidenceState = StoryFieldEvidenceState.UNSUPPORTED,
) -> EngineeringStoryField:
    return EngineeringStoryField(value=None, evidence_state=state)


def assessment(
    cls,
    level: SufficiencyLevel,
    positive_names: tuple[EngineeringStoryFieldName, ...],
    missing_names: tuple[EngineeringStoryFieldName, ...],
):
    if level is SufficiencyLevel.UNASSESSED:
        return cls(level=level)
    return cls(
        level=level,
        supported_fields=positive_names,
        missing_fields=missing_names[:3] if level is not SufficiencyLevel.HIGH else (),
    )


def story_view(
    *,
    suffix: str = "alpha",
    project_id: str = PROJECT_ID,
    story_type: EngineeringStoryType = EngineeringStoryType.OTHER,
    positive_fields: dict[
        str,
        str | tuple[str, StoryFieldEvidenceState],
    ] | None = None,
    field_states: dict[str, StoryFieldEvidenceState] | None = None,
    claim_level: SufficiencyLevel = SufficiencyLevel.HIGH,
    story_level: SufficiencyLevel = SufficiencyLevel.HIGH,
    status: EngineeringStoryStatus = EngineeringStoryStatus.ACTIVE,
    requires_revalidation: bool = False,
    opportunity_level: StoryOpportunityLevel = StoryOpportunityLevel.NONE,
) -> EngineeringStoryView:
    identity_suffix = hashlib.sha256(suffix.encode("utf-8")).hexdigest()[:24]
    canonical_id = f"engineering_story_{identity_suffix}"
    values = positive_fields or {
        "mechanism": "Implemented a bounded software change",
    }
    states = field_states or {}
    story_fields = {}
    positive_names = []
    missing_names = []
    for name in STORY_FIELD_NAMES:
        if name in values:
            item = values[name]
            if isinstance(item, tuple):
                value, state = item
            else:
                value, state = item, StoryFieldEvidenceState.CONFIRMED
            story_fields[name] = positive(value, state)
            positive_names.append(EngineeringStoryFieldName(name))
        else:
            state = states.get(name, StoryFieldEvidenceState.UNSUPPORTED)
            story_fields[name] = absent(state)
            missing_names.append(EngineeringStoryFieldName(name))
    positive_tuple = tuple(positive_names)
    missing_tuple = tuple(missing_names)
    lifecycle = EngineeringStoryLifecycle(
        status=status,
        requires_revalidation=requires_revalidation,
        superseded_by_story_id=(
            "engineering_story_successor"
            if status is EngineeringStoryStatus.SUPERSEDED
            else None
        ),
    )
    claim = assessment(
        ClaimSufficiency,
        claim_level,
        positive_tuple,
        missing_tuple,
    )
    sufficiency = assessment(
        StorySufficiency,
        story_level,
        positive_tuple,
        missing_tuple,
    )
    opportunity = (
        StoryOpportunity(level=StoryOpportunityLevel.NONE)
        if opportunity_level is StoryOpportunityLevel.NONE
        else StoryOpportunity(
            level=opportunity_level,
            signals=(StoryOpportunitySignal.MAJOR_DESIGN_DECISION,),
            missing_context=(StoryContextGap.DECISION_REASON,),
        )
    )
    story = EngineeringStory(
        story_id=canonical_id,
        project_id=project_id,
        story_type=story_type,
        **story_fields,
        evidence_fact_ids=(EVIDENCE_ID,),
        capability_fact_ids=(CAPABILITY_ID,),
        claim_boundary_ids=(),
        lifecycle=lifecycle,
        claim_sufficiency=claim,
        story_sufficiency=sufficiency,
        opportunity=opportunity,
    )
    fingerprint = hashlib.sha256(canonical_id.encode("utf-8")).hexdigest()
    return EngineeringStoryView(
        canonical_story_id=canonical_id,
        project_id=project_id,
        current_story=story,
        claim_sufficiency=claim,
        story_sufficiency=sufficiency,
        opportunity=opportunity,
        lifecycle=lifecycle,
        current_revision_id=f"engineering_story_revision_{identity_suffix}",
        evidence_fact_ids=story.evidence_fact_ids,
        capability_fact_ids=story.capability_fact_ids,
        claim_boundary_ids=story.claim_boundary_ids,
        provenance_fingerprint=fingerprint,
        source_lineage_fingerprints=(fingerprint,),
    )


def context(
    *,
    title: str = "Software Engineer",
    company: str = "Unregistered Systems",
    required: tuple[str, ...] = (),
    responsibilities: tuple[str, ...] = (),
    registry: OrganizationContextRegistry = DEFAULT_ORGANIZATION_CONTEXT_REGISTRY,
) -> HiringContextProfile:
    job_context = {}
    if required:
        job_context["required_qualifications"] = required
    if responsibilities:
        job_context["core_responsibilities"] = responsibilities
    return build_hiring_context_profile(
        company=company,
        team=None,
        parent_organization=None,
        role_title=title,
        normalized_job_context=job_context,
        organization_registry=registry,
    )


def evaluate(
    view: EngineeringStoryView,
    profile: HiringContextProfile | None = None,
) -> StoryHiringRelevance:
    return evaluate_engineering_story_relevance(
        hiring_context=profile or context(),
        story_view=view,
    )


def test_minimal_active_valid_story_evaluates():
    result = evaluate(story_view())
    assert isinstance(result, StoryHiringRelevance)
    assert result.lifecycle_status is EngineeringStoryStatus.ACTIVE
    assert 0.0 <= result.total_relevance_score <= 1.0


def test_story_project_revision_and_context_identity_are_preserved():
    profile = context(required=("API design",))
    view = story_view(suffix="identity")
    result = evaluate(view, profile)
    assert result.project_id == view.project_id
    assert result.canonical_story_id == view.canonical_story_id
    assert result.current_revision_id == view.current_revision_id
    assert result.hiring_context_profile_id == profile.profile_id
    assert result.hiring_context_fingerprint == profile.fingerprint
    assert result.story_provenance_fingerprint == view.provenance_fingerprint


def test_exact_explicit_jd_match_increases_explicit_component():
    view = story_view(positive_fields={
        "mechanism": "Built reliable REST API services",
    })
    matching = evaluate(
        view,
        context(required=("Build reliable REST API services",)),
    )
    unrelated = evaluate(view, context(required=("Embedded firmware",)))
    assert matching.components.explicit_jd_relevance > 0.0
    assert matching.components.explicit_jd_relevance > unrelated.components.explicit_jd_relevance
    assert StoryRelevanceReason.EXPLICIT_JD_ALIGNMENT in matching.reasons


def test_no_explicit_signal_match_produces_zero_explicit_component():
    result = evaluate(
        story_view(positive_fields={"mechanism": "Built a REST API"}),
        context(required=("Embedded firmware",)),
    )
    assert result.components.explicit_jd_relevance == 0.0


@pytest.mark.parametrize(
    ("title", "text", "feature"),
    (
        ("Backend Engineer", "Designed backend API storage service", StoryRelevanceFeature.BACKEND),
        ("Frontend Engineer", "Built React frontend browser UI", StoryRelevanceFeature.FRONTEND),
        ("Data Engineer", "Built an ETL data pipeline and warehouse", StoryRelevanceFeature.DATA_ENGINEERING),
        ("Data Analyst", "Created an analytics dashboard with decision support metrics", StoryRelevanceFeature.ANALYTICS),
        ("Game Developer", "Implemented gameplay game state in a real-time frame loop", StoryRelevanceFeature.GAME_DEVELOPMENT),
        ("DevOps Engineer", "Hardened cloud Kubernetes deployment infrastructure", StoryRelevanceFeature.DEVOPS_CLOUD),
        ("Security Engineer", "Implemented authentication encryption and access control", StoryRelevanceFeature.SECURITY),
    ),
)
def test_role_family_alignment_requires_supported_story_semantics(title, text, feature):
    result = evaluate(
        story_view(positive_fields={"mechanism": text}),
        context(title=title),
    )
    assert result.components.role_family_relevance > 0.0
    assert feature in result.semantic_features
    assert StoryRelevanceReason.PRIMARY_ROLE_ALIGNMENT in result.reasons


@pytest.mark.parametrize(
    "title",
    ("Game Developer", "Security Engineer", "DevOps Engineer", "Data Analyst"),
)
def test_domain_specific_role_does_not_match_generic_validation(title):
    result = evaluate(
        story_view(positive_fields={"validation": "Validated a generic pipeline"}),
        context(title=title),
    )
    assert result.components.role_family_relevance == 0.0


def test_coalition_context_boosts_supported_game_story_domain_relevance():
    profile = context(company="The Coalition")
    result = evaluate(
        story_view(positive_fields={
            "implementation": "Built gameplay and game state systems",
        }),
        profile,
    )
    assert result.components.organization_domain_relevance > 0.0
    assert StoryRelevanceReason.ORGANIZATION_DOMAIN_ALIGNMENT in result.reasons
    assert any(
        source.reference_id in {item.reference_id for item in profile.source_refs}
        for source in result.hiring_context_source_refs
    )


def test_coalition_context_does_not_turn_generic_api_story_into_game_story():
    result = evaluate(
        story_view(
            story_type=EngineeringStoryType.RELIABILITY_HARDENING,
            positive_fields={
                "mechanism": "Built a reliable CRUD API with retry handling",
            },
        ),
        context(company="The Coalition"),
    )
    assert result.components.organization_domain_relevance == 0.0
    assert StoryRelevanceFeature.GAME_DEVELOPMENT not in result.semantic_features
    assert result.components.transferable_engineering_relevance > 0.5


def test_generic_reliability_story_retains_transferable_value_in_game_context():
    result = evaluate(
        story_view(
            story_type=EngineeringStoryType.RELIABILITY_HARDENING,
            positive_fields={
                "mechanism": "Added retry and failover for a backend service",
                "validation": "Added regression tests and health checks",
            },
        ),
        context(company="The Coalition"),
    )
    assert result.components.organization_domain_relevance == 0.0
    assert result.components.transferable_engineering_relevance >= 0.7
    assert StoryRelevanceReason.TRANSFERABLE_RELIABILITY in result.reasons


def test_generic_saas_context_values_backend_story_without_domain_invention():
    result = evaluate(
        story_view(positive_fields={"mechanism": "Designed a backend API and database"}),
        context(title="Backend Engineer", company="Unknown SaaS Company"),
    )
    assert result.components.role_family_relevance > 0.0
    assert result.components.organization_domain_relevance == 0.0


def test_analytics_organization_context_values_supported_analysis_story():
    registry = OrganizationContextRegistry(rules=(
        OrganizationContextRule(
            canonical_name="Example Analytics",
            domains=(OrganizationDomainSignal(
                value="Data analytics",
                confidence=HiringContextConfidence.HIGH,
            ),),
        ),
    ))
    result = evaluate(
        story_view(positive_fields={
            "observable_outcome": "Delivered analytics and decision support metrics",
        }),
        context(
            title="Data Analyst",
            company="Example Analytics",
            registry=registry,
        ),
    )
    assert result.components.organization_domain_relevance > 0.0
    assert result.components.role_family_relevance > 0.0


def test_confirmed_field_contributes_more_than_supported_field():
    profile = context(required=("Reliable API",))
    confirmed = evaluate(
        story_view(positive_fields={"mechanism": "Built a reliable API"}),
        profile,
    )
    supported = evaluate(
        story_view(positive_fields={
            "mechanism": (
                "Built a reliable API",
                StoryFieldEvidenceState.SUPPORTED,
            ),
        }),
        profile,
    )
    assert confirmed.components.explicit_jd_relevance > supported.components.explicit_jd_relevance > 0.0


@pytest.mark.parametrize(
    "state",
    (StoryFieldEvidenceState.PLAUSIBLE_MISSING, StoryFieldEvidenceState.UNSUPPORTED),
)
def test_non_positive_field_states_never_create_factual_relevance(state):
    view = story_view(
        positive_fields={"mechanism": "Built a generic pipeline"},
        field_states={"decision": state},
    )
    result = evaluate(view, context(required=("Privacy engineering",)))
    assert view.current_story.decision.value is None
    assert result.components.explicit_jd_relevance == 0.0
    assert StoryRelevanceFeature.SECURITY not in result.semantic_features


def test_claim_high_story_low_remains_relevant_and_incomplete():
    profile = context(required=("Reliable API",))
    complete = evaluate(
        story_view(
            positive_fields={"mechanism": "Built a reliable API"},
            claim_level=SufficiencyLevel.HIGH,
            story_level=SufficiencyLevel.HIGH,
        ),
        profile,
    )
    incomplete = evaluate(
        story_view(
            suffix="incomplete",
            positive_fields={"mechanism": "Built a reliable API"},
            claim_level=SufficiencyLevel.HIGH,
            story_level=SufficiencyLevel.LOW,
        ),
        profile,
    )
    assert incomplete.raw_relevance_score == complete.raw_relevance_score
    assert incomplete.total_relevance_score == complete.total_relevance_score
    assert incomplete.components.story_completeness < complete.components.story_completeness
    assert StoryRelevanceReason.STORY_INCOMPLETE in incomplete.reasons


def test_story_high_claim_low_remains_matched_but_risk_limited():
    profile = context(required=("Reliable API",))
    safe = evaluate(
        story_view(
            positive_fields={"mechanism": "Built a reliable API"},
            claim_level=SufficiencyLevel.HIGH,
        ),
        profile,
    )
    weak = evaluate(
        story_view(
            suffix="weak",
            positive_fields={"mechanism": "Built a reliable API"},
            claim_level=SufficiencyLevel.LOW,
            story_level=SufficiencyLevel.HIGH,
        ),
        profile,
    )
    assert weak.raw_relevance_score == safe.raw_relevance_score
    assert weak.total_relevance_score < safe.total_relevance_score
    assert weak.evidence_risk_adjustment > 0.0
    assert weak.claim_sufficiency is SufficiencyLevel.LOW
    assert StoryRelevanceReason.CLAIM_EVIDENCE_RISK in weak.reasons


@pytest.mark.parametrize(
    ("status", "requires_revalidation", "expected"),
    (
        (EngineeringStoryStatus.STALE, True, StoryRelevanceEvaluationErrorCode.INACTIVE_STORY),
        (EngineeringStoryStatus.CONFLICTED, True, StoryRelevanceEvaluationErrorCode.INACTIVE_STORY),
        (EngineeringStoryStatus.SUPERSEDED, False, StoryRelevanceEvaluationErrorCode.INACTIVE_STORY),
        (EngineeringStoryStatus.ACTIVE, True, StoryRelevanceEvaluationErrorCode.REVALIDATION_REQUIRED),
    ),
)
def test_non_rankable_lifecycle_states_fail_closed(status, requires_revalidation, expected):
    view = story_view(status=status, requires_revalidation=requires_revalidation)
    with pytest.raises(StoryRelevanceEvaluationError) as raised:
        evaluate(view)
    assert raised.value.code is expected


@pytest.mark.parametrize(
    "attribute",
    ("evidence_fact_ids", "capability_fact_ids", "claim_boundary_ids"),
)
def test_view_authority_mismatch_fails_closed(attribute):
    view = story_view()
    replacement = {
        "evidence_fact_ids": ("pef_forged",),
        "capability_fact_ids": ("pcf_forged",),
        "claim_boundary_ids": ("pcb_forged",),
    }[attribute]
    forged = replace(view, **{attribute: replacement})
    with pytest.raises(StoryRelevanceEvaluationError) as raised:
        evaluate(forged)
    assert raised.value.code is StoryRelevanceEvaluationErrorCode.INVALID_INPUT


def test_hiring_context_confidence_never_becomes_candidate_evidence_safety():
    view = story_view(claim_level=SufficiencyLevel.MEDIUM)
    low_context = context(title="Software Engineer")
    high_context = context(
        title="Backend Engineer",
        responsibilities=("Build backend REST API services",),
    )
    assert low_context.confidence is HiringContextConfidence.LOW
    assert high_context.confidence is HiringContextConfidence.HIGH
    low = evaluate(view, low_context)
    high = evaluate(view, high_context)
    assert low.components.evidence_claim_safety == high.components.evidence_claim_safety
    assert low.claim_sufficiency == high.claim_sufficiency


@pytest.mark.parametrize(
    ("profile", "forbidden"),
    (
        (context(company="Microsoft"), {"c#", "azure", ".net"}),
        (context(company="The Coalition"), {"unity", "unreal", "c++"}),
        (context(company="Amazon"), {"aws"}),
    ),
)
def test_company_context_cannot_create_story_technology(profile, forbidden):
    view = story_view(positive_fields={"mechanism": "Built a generic backend API"})
    before = view.current_story.to_json()
    result = evaluate(view, profile)
    assert view.current_story.to_json() == before
    assert forbidden.isdisjoint({item.value for item in result.semantic_features})


def test_explicit_aws_context_matches_only_independently_supported_aws_story_text():
    profile = context(required=("AWS",))
    generic = evaluate(
        story_view(positive_fields={"mechanism": "Hardened cloud deployment"}),
        profile,
    )
    supported = evaluate(
        story_view(suffix="aws", positive_fields={"mechanism": "Deployed services on AWS"}),
        profile,
    )
    assert generic.components.explicit_jd_relevance == 0.0
    assert supported.components.explicit_jd_relevance > 0.0


@pytest.mark.parametrize(
    ("context_value", "story_value"),
    (
        ("Java", "Built a JavaScript frontend"),
        ("JavaScript", "Built a Java service"),
        ("C", "Built a C++ service"),
        ("C", "Built a C# service"),
        ("C++", "Built a C service"),
        ("C#", "Built a C service"),
    ),
)
def test_technology_tokens_do_not_match_by_substring(context_value, story_value):
    result = evaluate(
        story_view(positive_fields={"implementation": story_value}),
        context(required=(context_value,)),
    )
    assert result.components.explicit_jd_relevance == 0.0


def test_exact_atomic_technology_token_can_match_supported_story_text():
    result = evaluate(
        story_view(positive_fields={"implementation": "Built a C++ service"}),
        context(required=("C++",)),
    )
    assert result.components.explicit_jd_relevance > 0.0


@pytest.mark.parametrize(
    ("title", "required", "forbidden_feature"),
    (
        ("Software Engineer", "Privacy engineering", StoryRelevanceFeature.SECURITY),
        ("Security Engineer", "Cyber governance", StoryRelevanceFeature.SECURITY),
        ("Strategy Consultant", "Technology consulting", StoryRelevanceFeature.ANALYTICS),
    ),
)
def test_context_cannot_create_professional_persona_or_domain_fact(title, required, forbidden_feature):
    result = evaluate(
        story_view(positive_fields={"validation": "Validated a generic pipeline"}),
        context(title=title, required=(required,)),
    )
    assert forbidden_feature not in result.semantic_features
    assert result.components.organization_domain_relevance == 0.0
    assert not any("persona" in name or "candidate" in name for name in result.to_dict())


def test_story_profile_and_view_remain_unchanged_after_evaluation():
    profile = context(company="The Coalition", required=("Testing",))
    view = story_view(positive_fields={"validation": "Added gameplay regression testing"})
    profile_before = profile.to_json()
    view_before = view.to_json()
    story_before = view.current_story.to_json()
    evaluate(view, profile)
    assert profile.to_json() == profile_before
    assert view.to_json() == view_before
    assert view.current_story.to_json() == story_before


@pytest.mark.parametrize("target", ("profile", "view", "story", "result"))
def test_inputs_and_result_are_immutable(target):
    profile = context()
    view = story_view()
    result = evaluate(view, profile)
    value = {
        "profile": profile,
        "view": view,
        "story": view.current_story,
        "result": result,
    }[target]
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        value.project_id = "changed"


def test_deterministic_output_components_reasons_and_sources():
    profile = context(company="The Coalition", required=("Testing",))
    view = story_view(positive_fields={"validation": "Added gameplay regression testing"})
    first = evaluate(view, profile)
    second = evaluate(view, profile)
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.components == second.components
    assert first.reasons == second.reasons
    assert first.hiring_context_source_refs == second.hiring_context_source_refs
    assert len(first.reasons) <= MAX_STORY_RELEVANCE_REASONS
    assert len(first.semantic_features) <= MAX_STORY_RELEVANCE_FEATURES


def test_equivalent_registered_company_alias_produces_equivalent_result():
    view = story_view(positive_fields={"mechanism": "Built gameplay game state"})
    canonical = evaluate(view, context(company="The Coalition"))
    alias = evaluate(view, context(company="  coalition  "))
    assert canonical == alias


def test_material_jd_context_change_can_change_story_relevance():
    view = story_view(positive_fields={"mechanism": "Built a reliable API"})
    matching = evaluate(view, context(required=("Reliable API",)))
    unrelated = evaluate(view, context(required=("Embedded firmware",)))
    assert matching.total_relevance_score > unrelated.total_relevance_score


def test_material_company_domain_change_can_change_domain_relevance():
    view = story_view(positive_fields={"mechanism": "Built gameplay game state"})
    unknown = evaluate(view, context(company="Unknown Studio"))
    coalition = evaluate(view, context(company="The Coalition"))
    assert unknown.components.organization_domain_relevance == 0.0
    assert coalition.components.organization_domain_relevance > 0.0


def test_role_family_change_can_change_role_relevance():
    view = story_view(positive_fields={"mechanism": "Built a backend API and database"})
    backend = evaluate(view, context(title="Backend Engineer"))
    frontend = evaluate(view, context(title="Frontend Engineer"))
    assert backend.components.role_family_relevance > frontend.components.role_family_relevance


def test_arbitrary_substring_does_not_match_domain_or_explicit_context():
    result = evaluate(
        story_view(positive_fields={"mechanism": "Integrated a GameStop catalog API"}),
        context(required=("Game",), company="The Coalition"),
    )
    assert result.components.explicit_jd_relevance == 0.0
    assert result.components.organization_domain_relevance == 0.0
    assert StoryRelevanceFeature.GAME_DEVELOPMENT not in result.semantic_features


def test_high_domain_relevance_alone_cannot_force_total_to_one():
    result = evaluate(
        story_view(positive_fields={
            "mechanism": "Built gameplay game systems and game state",
        }),
        context(company="The Coalition"),
    )
    assert result.components.organization_domain_relevance > 0.0
    assert result.weights.organization_domain <= 0.25
    assert result.total_relevance_score < 1.0


def test_strong_cross_domain_engineering_can_remain_competitive():
    profile = context(company="The Coalition")
    game = evaluate(
        story_view(suffix="game", positive_fields={"mechanism": "Built gameplay systems"}),
        profile,
    )
    reliability = evaluate(
        story_view(
            suffix="reliability",
            story_type=EngineeringStoryType.RELIABILITY_HARDENING,
            positive_fields={
                "mechanism": "Redesigned reliable backend failover and recovery",
                "validation": "Added regression tests and monitoring",
            },
        ),
        profile,
    )
    assert game.components.organization_domain_relevance > reliability.components.organization_domain_relevance
    assert reliability.components.transferable_engineering_relevance > game.components.transferable_engineering_relevance
    assert reliability.total_relevance_score > 0.0


def test_story_opportunity_is_consumed_only_as_completion_hint():
    profile = context(required=("Reliable API",))
    plain = evaluate(
        story_view(positive_fields={"mechanism": "Built a reliable API"}),
        profile,
    )
    opportunity = evaluate(
        story_view(
            suffix="opportunity",
            positive_fields={"mechanism": "Built a reliable API"},
            opportunity_level=StoryOpportunityLevel.HIGH,
        ),
        profile,
    )
    assert opportunity.components == plain.components
    assert opportunity.raw_relevance_score == plain.raw_relevance_score
    assert opportunity.total_relevance_score == plain.total_relevance_score
    assert opportunity.clarification_value_hint == 1.0
    assert StoryRelevanceReason.STORY_COMPLETION_OPPORTUNITY in opportunity.reasons


def test_batch_ranking_is_stable_under_input_permutation():
    profile = context(required=("Reliable API",))
    matched = story_view(
        suffix="matched",
        positive_fields={"mechanism": "Built a reliable API"},
    )
    unrelated = story_view(
        suffix="unrelated",
        positive_fields={"mechanism": "Built embedded firmware"},
    )
    first = rank_engineering_stories_for_hiring_context(
        hiring_context=profile,
        story_views=(unrelated, matched),
    )
    second = rank_engineering_stories_for_hiring_context(
        hiring_context=profile,
        story_views=(matched, unrelated),
    )
    assert first == second
    assert first[0].canonical_story_id == matched.canonical_story_id
    assert {item.project_id for item in first} == {PROJECT_ID}


def test_batch_ties_use_canonical_story_id():
    profile = context()
    alpha = story_view(suffix="alpha")
    beta = story_view(suffix="beta")
    ranked = rank_engineering_stories_for_hiring_context(
        hiring_context=profile,
        story_views=(beta, alpha),
    )
    assert ranked[0].total_relevance_score == ranked[1].total_relevance_score
    assert [item.canonical_story_id for item in ranked] == sorted(
        (alpha.canonical_story_id, beta.canonical_story_id)
    )


def test_two_stories_in_one_project_receive_individual_relevance():
    profile = context(company="The Coalition")
    game = story_view(suffix="game", positive_fields={"mechanism": "Built gameplay systems"})
    api = story_view(suffix="api", positive_fields={"mechanism": "Built a generic API"})
    ranked = rank_engineering_stories_for_hiring_context(
        hiring_context=profile,
        story_views=(game, api),
    )
    by_id = {item.canonical_story_id: item for item in ranked}
    assert by_id[game.canonical_story_id].components.organization_domain_relevance > 0.0
    assert by_id[api.canonical_story_id].components.organization_domain_relevance == 0.0


def test_duplicate_story_identity_fails_closed():
    view = story_view()
    with pytest.raises(StoryRelevanceEvaluationError) as raised:
        rank_engineering_stories_for_hiring_context(
            hiring_context=context(),
            story_views=(view, view),
        )
    assert raised.value.code is StoryRelevanceEvaluationErrorCode.DUPLICATE_STORY


def test_batch_bound_fails_before_evaluation():
    with pytest.raises(StoryRelevanceEvaluationError) as raised:
        rank_engineering_stories_for_hiring_context(
            hiring_context=context(),
            story_views=(story_view(),) * (MAX_STORY_RELEVANCE_BATCH + 1),
        )
    assert raised.value.code is StoryRelevanceEvaluationErrorCode.BOUND_EXCEEDED


def test_scores_components_and_weights_are_bounded_and_explicit():
    result = evaluate(story_view())
    for value in result.components.to_dict().values():
        assert 0.0 <= value <= 1.0
    assert sum(result.weights.to_dict().values()) == pytest.approx(1.0)
    assert result.total_relevance_score == pytest.approx(
        result.raw_relevance_score - result.evidence_risk_adjustment,
        abs=0.000002,
    )


def test_result_schema_contains_no_project_rank_budget_resume_or_question_output():
    names = {item.name for item in fields(StoryHiringRelevance)}
    forbidden = {
        "project_rank",
        "project_score",
        "bullet_budget",
        "line_budget",
        "resume_wording",
        "question",
        "questions",
        "candidate_identity",
        "candidate_persona",
    }
    assert names.isdisjoint(forbidden)


def test_public_api_has_no_candidate_context_or_runtime_dependency_inputs():
    signature = inspect.signature(evaluate_engineering_story_relevance)
    assert set(signature.parameters) == {"hiring_context", "story_view"}
    assert not any(
        term in name
        for name in signature.parameters
        for term in ("candidate", "retrieval", "vector", "chroma", "web", "llm")
    )


def test_module_imports_only_story_view_models_and_hiring_context_models():
    path = Path(__file__).resolve().parents[1] / "backend" / "engineering_story_relevance.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imports & {
        "backend.engineering_story_memory_service",
        "backend.engineering_story_models",
        "backend.hiring_context_models",
    } == {
        "backend.engineering_story_memory_service",
        "backend.engineering_story_models",
        "backend.hiring_context_models",
    }
    assert not any(
        term in module
        for module in imports
        for term in (
            "reconstruction",
            "project_evidence",
            "capability",
            "retrieval",
            "chroma",
            "api_server",
        )
    )


def test_module_has_no_runtime_io_environment_network_model_or_persistence_calls():
    path = Path(__file__).resolve().parents[1] / "backend" / "engineering_story_relevance.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_names = {
        "open",
        "connect",
        "request",
        "urlopen",
        "getenv",
        "putenv",
        "load_authoritative_engineering_story_memory",
        "write_authoritative_engineering_story_memory",
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert calls.isdisjoint(forbidden_names)
    source = path.read_text(encoding="utf-8").casefold()
    assert "rank_projects_for_resume" not in source
    assert "select_staged_projects_with_ranking" not in source
    assert "tailor_resume_staged" not in source
    assert "engineeringstory(" not in source.replace(" ", "")


def test_result_contract_validation_is_fail_closed():
    with pytest.raises(ValueError, match="sum to 1"):
        StoryRelevanceWeights(0.5, 0.5, 0.5, 0.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        StoryRelevanceComponents(1.1, 0.0, 0.0, 0.0, 1.0, 1.0)
