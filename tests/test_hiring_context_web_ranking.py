from __future__ import annotations

import ast
from dataclasses import fields, replace
import inspect
from pathlib import Path

import pytest

from backend.engineering_story_models import (
    EngineeringStoryStatus,
    StoryOpportunityLevel,
    SufficiencyLevel,
)
from backend.engineering_story_relevance import (
    StoryHiringRelevance,
    StoryRelevanceComponents,
    StoryRelevanceFeature,
    StoryRelevanceWeights,
)
from backend.hiring_context_intelligence import build_hiring_context_profile
from backend.hiring_context_models import (
    HiringContextConfidence,
    HiringContextSignalKind,
)
from backend.hiring_context_official_sources import (
    OfficialHiringContextSourceKind,
    OfficialSourceSpecificity,
    OfficialSourceTrust,
)
from backend.hiring_context_web_enrichment import (
    HiringContextEnrichment,
    HiringContextEnrichmentStatus,
    HiringContextOfficialSourceRef,
    HiringContextWebSignal,
)
from backend.hiring_context_web_ranking import (
    MAX_WEB_CONTEXT_ADJUSTMENT,
    OFFICIAL_HIRING_CONTEXT_RANKING_INFLUENCE_FLAG,
    OFFICIAL_HIRING_CONTEXT_WEB_RANKING_POLICY_ID,
    WebAdjustedStoryHiringRelevance,
    WebEnrichmentChangeDecision,
    WebAwarePartialRerankDecision,
    WebAwareProjectStoryRankingState,
    WebRankingInfluenceError,
    WebRankingInfluenceErrorCode,
    apply_hiring_context_web_influence,
    build_web_aware_project_story_ranking_state,
    calculate_story_web_context_adjustment,
    classify_web_aware_partial_rerank,
    classify_web_enrichment_change,
    rank_projects_with_hiring_context_web_influence,
    refresh_web_aware_story_ranking,
)
from backend.project_story_ranking_state import (
    ProjectStoryRankingStateError,
    build_project_story_ranking_state,
)


def _profile():
    return build_hiring_context_profile(
        company="The Coalition",
        team="The Coalition",
        role_title="Software Engineer",
        normalized_job_context={"required_qualifications": ["software engineering"]},
    )


def _official_ref(
    *,
    specificity: OfficialSourceSpecificity = OfficialSourceSpecificity.TEAM_PRODUCT,
    suffix: str = "team",
) -> HiringContextOfficialSourceRef:
    kinds = {
        OfficialSourceSpecificity.TEAM_PRODUCT: (
            OfficialHiringContextSourceKind.OFFICIAL_TEAM_PAGE
        ),
        OfficialSourceSpecificity.COMPANY_ENGINEERING: (
            OfficialHiringContextSourceKind.OFFICIAL_ENGINEERING_PAGE
        ),
        OfficialSourceSpecificity.CAREERS: (
            OfficialHiringContextSourceKind.OFFICIAL_CAREERS_PAGE
        ),
        OfficialSourceSpecificity.COMPANY: (
            OfficialHiringContextSourceKind.OFFICIAL_COMPANY_PAGE
        ),
        OfficialSourceSpecificity.PARENT_ORGANIZATION: (
            OfficialHiringContextSourceKind.OFFICIAL_PARENT_ORG_PAGE
        ),
        OfficialSourceSpecificity.ORGANIZATION_GITHUB: (
            OfficialHiringContextSourceKind.OFFICIAL_ORG_GITHUB
        ),
    }
    return HiringContextOfficialSourceRef(
        normalized_url=f"https://example.com/{suffix}",
        source_kind=kinds[specificity],
        canonical_organization="The Coalition",
        content_fingerprint=("a" if suffix == "team" else "b") * 64,
        specificity=specificity,
        trust=OfficialSourceTrust.VERIFIED_OFFICIAL,
    )


def _signal(
    ref: HiringContextOfficialSourceRef,
    value: str = "Game development",
    *,
    confidence: HiringContextConfidence = HiringContextConfidence.HIGH,
    offline: tuple = (),
) -> HiringContextWebSignal:
    return HiringContextWebSignal(
        value=value,
        kind=HiringContextSignalKind.TEAM_DOMAIN,
        confidence=confidence,
        official_source_refs=(ref,),
        offline_source_refs=offline,
    )


def _enrichment(
    profile,
    *,
    status=HiringContextEnrichmentStatus.READY,
    signals=(),
    refs=(),
    conflicts=(),
) -> HiringContextEnrichment:
    return HiringContextEnrichment(
        offline_profile_id=profile.profile_id,
        offline_profile_fingerprint=profile.fingerprint,
        status=status,
        official_source_refs=tuple(refs),
        signals=tuple(signals),
        conflicts=tuple(conflicts),
        company=profile.company,
        team=profile.team,
        parent_organization=profile.parent_organization,
    )


def _story(
    profile,
    *,
    features=(),
    score=0.42,
    story_id="story",
    project_id="project",
    revision_id: str | None = None,
    raw_score: float | None = None,
    risk_adjustment=0.0,
    claim_sufficiency=SufficiencyLevel.HIGH,
    story_sufficiency=SufficiencyLevel.HIGH,
    claim_safety=1.0,
):
    raw = score + risk_adjustment if raw_score is None else raw_score
    return StoryHiringRelevance(
        project_id=project_id,
        canonical_story_id=story_id,
        current_revision_id=revision_id or f"revision_{story_id}",
        hiring_context_profile_id=profile.profile_id,
        hiring_context_fingerprint=profile.fingerprint,
        story_provenance_fingerprint=f"provenance_{project_id}_{story_id}",
        lifecycle_status=EngineeringStoryStatus.ACTIVE,
        claim_sufficiency=claim_sufficiency,
        story_sufficiency=story_sufficiency,
        story_opportunity=StoryOpportunityLevel.NONE,
        components=StoryRelevanceComponents(.4, .4, .4, .4, claim_safety, 1.0),
        weights=StoryRelevanceWeights(.3, .2, .2, .3),
        raw_relevance_score=raw,
        evidence_risk_adjustment=risk_adjustment,
        total_relevance_score=score,
        clarification_value_hint=0.0,
        semantic_features=tuple(features),
        reasons=(),
        hiring_context_source_refs=(),
    )


def test_feature_flag_off_is_exactly_offline_and_preserves_base_result():
    profile = _profile()
    ref = _official_ref()
    enrichment = _enrichment(profile, signals=(_signal(ref),), refs=(ref,))
    story = _story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,))

    result = apply_hiring_context_web_influence(
        hiring_context=profile,
        enrichment=enrichment,
        story_relevance=(story,),
        enabled=False,
    )

    assert result == (story,)
    assert result[0] is story


def test_environment_flag_can_enable_influence_without_enabling_acquisition():
    profile = _profile()
    ref = _official_ref()
    enrichment = _enrichment(profile, signals=(_signal(ref),), refs=(ref,))
    story = _story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,))

    result = apply_hiring_context_web_influence(
        hiring_context=profile,
        enrichment=enrichment,
        story_relevance=(story,),
        environ={OFFICIAL_HIRING_CONTEXT_RANKING_INFLUENCE_FLAG: "true"},
    )

    assert isinstance(result[0], WebAdjustedStoryHiringRelevance)
    assert result[0].base_relevance == story
    assert result[0].base_total_relevance_score == story.total_relevance_score


@pytest.mark.parametrize(
    "status",
    (
        HiringContextEnrichmentStatus.DISABLED,
        HiringContextEnrichmentStatus.NO_OFFICIAL_SOURCE,
        HiringContextEnrichmentStatus.UNAVAILABLE,
    ),
)
def test_empty_enrichment_statuses_have_zero_adjustment(status):
    profile = _profile()
    story = _story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,))
    enrichment = _enrichment(profile, status=status)
    adjustment = calculate_story_web_context_adjustment(
        hiring_context=profile,
        enrichment=enrichment,
        story_relevance=story,
    )
    assert adjustment.adjustment == 0.0


def test_conflicted_enrichment_fails_closed():
    profile = _profile()
    ref = _official_ref()
    enrichment = _enrichment(
        profile,
        status=HiringContextEnrichmentStatus.CONFLICTED,
        refs=(ref,),
        conflicts=("primary_domain:game|platform",),
    )
    story = _story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,))
    adjustment = calculate_story_web_context_adjustment(
        hiring_context=profile,
        enrichment=enrichment,
        story_relevance=story,
    )
    assert adjustment.adjustment == 0.0


def test_both_sides_are_required_and_plausible_or_unknown_traits_do_not_match():
    profile = _profile()
    ref = _official_ref()
    enrichment = _enrichment(profile, signals=(_signal(ref),), refs=(ref,))
    generic = _story(profile, features=(StoryRelevanceFeature.API_SYSTEM_DESIGN,))
    missing = _story(profile, features=())

    assert calculate_story_web_context_adjustment(
        hiring_context=profile, enrichment=enrichment, story_relevance=generic
    ).adjustment == 0.0
    assert calculate_story_web_context_adjustment(
        hiring_context=profile, enrichment=enrichment, story_relevance=missing
    ).adjustment == 0.0


def test_confidence_specificity_and_offline_corroboration_are_bounded():
    profile = _profile()
    team = _official_ref(suffix="team")
    parent = _official_ref(
        specificity=OfficialSourceSpecificity.PARENT_ORGANIZATION,
        suffix="parent",
    )
    story = _story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,))
    high = _enrichment(profile, signals=(_signal(team),), refs=(team,))
    medium = _enrichment(
        profile,
        signals=(_signal(parent, confidence=HiringContextConfidence.MEDIUM),),
        refs=(parent,),
    )
    corroborated = _enrichment(
        profile,
        signals=(_signal(team, offline=(profile.source_refs[0],)),),
        refs=(team,),
    )

    high_value = calculate_story_web_context_adjustment(
        hiring_context=profile, enrichment=high, story_relevance=story
    ).adjustment
    medium_value = calculate_story_web_context_adjustment(
        hiring_context=profile, enrichment=medium, story_relevance=story
    ).adjustment
    corroborated_value = calculate_story_web_context_adjustment(
        hiring_context=profile, enrichment=corroborated, story_relevance=story
    ).adjustment

    assert high_value > medium_value > 0.0
    assert high_value > corroborated_value > 0.0
    assert high_value <= MAX_WEB_CONTEXT_ADJUSTMENT


def test_duplicate_semantic_signal_does_not_stack_unboundedly():
    profile = _profile()
    team = _official_ref(suffix="team")
    parent = _official_ref(
        specificity=OfficialSourceSpecificity.PARENT_ORGANIZATION,
        suffix="parent",
    )
    story = _story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,))
    enrichment = _enrichment(
        profile,
        signals=(_signal(team), _signal(parent)),
        refs=(team, parent),
    )
    adjustment = calculate_story_web_context_adjustment(
        hiring_context=profile, enrichment=enrichment, story_relevance=story
    )
    assert adjustment.matched_signal_values == ("Game development",)
    assert adjustment.adjustment == MAX_WEB_CONTEXT_ADJUSTMENT


def test_base_components_truth_sufficiency_and_footprint_are_not_rewritten():
    profile = _profile()
    ref = _official_ref()
    enrichment = _enrichment(profile, signals=(_signal(ref),), refs=(ref,))
    story = _story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,))
    result = rank_projects_with_hiring_context_web_influence(
        hiring_context=profile,
        enrichment=enrichment,
        story_relevance=(story,),
        enabled=True,
    )
    adjusted = result.story_relevance[0]
    assert isinstance(adjusted, WebAdjustedStoryHiringRelevance)
    assert adjusted.base_relevance.components == story.components
    assert adjusted.claim_sufficiency == story.claim_sufficiency
    assert adjusted.story_sufficiency == story.story_sufficiency
    assert adjusted.story_opportunity == story.story_opportunity
    assert adjusted.semantic_features == story.semantic_features
    assert result.portfolio is not None
    footprint = result.portfolio.ranked_projects[0].semantic_footprint
    assert tuple(item.feature for item in footprint.traits) == (
        StoryRelevanceFeature.GAME_DEVELOPMENT,
    )


def test_partial_enrichment_uses_accepted_signals_and_no_web_technology_inference():
    profile = _profile()
    ref = _official_ref()
    enrichment = _enrichment(
        profile,
        status=HiringContextEnrichmentStatus.PARTIAL,
        signals=(_signal(ref, value="Security engineering"),),
        refs=(ref,),
    )
    security = _story(profile, features=(StoryRelevanceFeature.SECURITY,))
    result = apply_hiring_context_web_influence(
        hiring_context=profile,
        enrichment=enrichment,
        story_relevance=(security,),
        enabled=True,
    )
    assert isinstance(result[0], WebAdjustedStoryHiringRelevance)
    assert not any(
        feature.value in {"aws", "azure", "gcp", "c++", "unreal", "unity"}
        for feature in result[0].semantic_features
    )


def test_ranking_influence_state_binds_enrichment_and_policy_and_refreshes_full_on_change():
    profile = _profile()
    ref = _official_ref()
    enrichment = _enrichment(profile, signals=(_signal(ref),), refs=(ref,))
    story = _story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,))
    result = rank_projects_with_hiring_context_web_influence(
        hiring_context=profile,
        enrichment=enrichment,
        story_relevance=(story,),
        enabled=True,
    )
    assert isinstance(result.state, WebAwareProjectStoryRankingState)
    state = result.state
    assert state.enrichment_fingerprint == enrichment.fingerprint
    assert state.web_influence_policy_id == OFFICIAL_HIRING_CONTEXT_WEB_RANKING_POLICY_ID
    assert state.hiring_context_fingerprint == profile.fingerprint
    assert classify_web_aware_partial_rerank(
        prior_state=state, hiring_context=profile, enrichment=enrichment
    ) is WebAwarePartialRerankDecision.STORY_ONLY_ALLOWED

    changed = _enrichment(
        profile,
        signals=(_signal(ref, value="Real-time interactive software"),),
        refs=(ref,),
    )
    assert classify_web_aware_partial_rerank(
        prior_state=state, hiring_context=profile, enrichment=changed
    ) is WebAwarePartialRerankDecision.CONTEXT_CHANGED_REQUIRES_FULL_RERANK


def test_mismatched_enrichment_is_rejected_and_state_has_no_timestamps():
    profile = _profile()
    other = build_hiring_context_profile(
        company="Microsoft",
        role_title="Software Engineer",
        normalized_job_context={"required_qualifications": ["software engineering"]},
    )
    ref = _official_ref()
    enrichment = _enrichment(other, signals=(_signal(ref),), refs=(ref,))
    story = _story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,))
    with pytest.raises(WebRankingInfluenceError) as captured:
        calculate_story_web_context_adjustment(
            hiring_context=profile,
            enrichment=enrichment,
            story_relevance=story,
        )
    assert captured.value.code is WebRankingInfluenceErrorCode.ENRICHMENT_CONTEXT_MISMATCH
    assert {item.name for item in fields(WebAwareProjectStoryRankingState)}.isdisjoint(
        {"timestamp", "created_at", "updated_at"}
    )


def test_web_ranking_module_is_pure_policy_without_forbidden_io_or_services():
    source = Path(__file__).parents[1] / "backend" / "hiring_context_web_ranking.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    forbidden = {
        "fetch_official_hiring_context_sources",
        "parse_html",
        "retrieve_project_evidence",
        "search_project_evidence",
        "call_llm",
        "chromadb",
    }
    assert not forbidden.intersection(
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    )
    assert "requests" not in source.read_text(encoding="utf-8")
    assert "fetch_official_hiring_context_sources" not in inspect.getsource(
        calculate_story_web_context_adjustment
    )


def test_positive_adjustment_has_a_complete_source_signal_candidate_trace():
    profile = _profile()
    ref = _official_ref()
    signal = _signal(ref)
    enrichment = _enrichment(profile, signals=(signal,), refs=(ref,))
    story = _story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,))

    adjustment = calculate_story_web_context_adjustment(
        hiring_context=profile,
        enrichment=enrichment,
        story_relevance=story,
    )

    assert adjustment.adjustment > 0.0
    assert len(adjustment.traces) == 1
    trace = adjustment.traces[0]
    assert trace.signal_id == signal.signal_id
    assert trace.official_source_refs == (ref,)
    assert trace.candidate_features == (StoryRelevanceFeature.GAME_DEVELOPMENT,)
    assert trace.context_confidence is HiringContextConfidence.HIGH
    assert trace.raw_contribution == 1.0


def test_trace_cannot_be_tampered_or_used_as_candidate_truth():
    profile = _profile()
    ref = _official_ref()
    enrichment = _enrichment(profile, signals=(_signal(ref),), refs=(ref,))
    story = _story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,))
    trace = calculate_story_web_context_adjustment(
        hiring_context=profile,
        enrichment=enrichment,
        story_relevance=story,
    ).traces[0]

    with pytest.raises(ValueError):
        replace(trace, raw_contribution=0.5, trace_id="")
    assert not {
        "candidate_evidence",
        "story_provenance",
        "evidence_fact",
        "capability_fact",
        "candidate_technologies",
        "persona",
    } & {item.name for item in fields(type(trace))}


@pytest.mark.parametrize(
    ("status", "signals_usable"),
    (
        (HiringContextEnrichmentStatus.DISABLED, False),
        (HiringContextEnrichmentStatus.NO_OFFICIAL_SOURCE, False),
        (HiringContextEnrichmentStatus.UNAVAILABLE, False),
        (HiringContextEnrichmentStatus.CONFLICTED, False),
        (HiringContextEnrichmentStatus.PARTIAL, True),
        (HiringContextEnrichmentStatus.READY, True),
    ),
)
def test_every_enrichment_status_has_deterministic_ranking_semantics(status, signals_usable):
    profile = _profile()
    ref = _official_ref()
    story = _story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,))
    if status is HiringContextEnrichmentStatus.CONFLICTED:
        enrichment = _enrichment(
            profile,
            status=status,
            refs=(ref,),
            conflicts=("primary_domain:game|platform",),
        )
    elif signals_usable:
        enrichment = _enrichment(
            profile,
            status=status,
            signals=(_signal(ref),),
            refs=(ref,),
        )
    else:
        enrichment = _enrichment(profile, status=status)

    result = rank_projects_with_hiring_context_web_influence(
        hiring_context=profile,
        enrichment=enrichment,
        story_relevance=(story,),
        enabled=True,
    )

    assert result.web_influence_applied is signals_usable
    assert isinstance(result.state, WebAwareProjectStoryRankingState) is signals_usable


def test_ready_or_partial_irrelevant_context_reuses_exact_offline_ranking_state():
    profile = _profile()
    ref = _official_ref()
    story = _story(profile, features=(StoryRelevanceFeature.API_SYSTEM_DESIGN,))
    offline = rank_projects_with_hiring_context_web_influence(
        hiring_context=profile,
        enrichment=_enrichment(profile, status=HiringContextEnrichmentStatus.DISABLED),
        story_relevance=(story,),
        enabled=False,
    )
    for status in (
        HiringContextEnrichmentStatus.READY,
        HiringContextEnrichmentStatus.PARTIAL,
    ):
        web = rank_projects_with_hiring_context_web_influence(
            hiring_context=profile,
            enrichment=_enrichment(
                profile,
                status=status,
                signals=(_signal(ref),),
                refs=(ref,),
            ),
            story_relevance=(story,),
            enabled=True,
        )
        assert web.web_influence_applied is False
        assert web.story_relevance == offline.story_relevance
        assert web.projects == offline.projects
        assert web.portfolio == offline.portfolio
        assert web.state == offline.state


@pytest.mark.parametrize(
    "specificity",
    tuple(OfficialSourceSpecificity),
)
def test_each_source_specificity_class_has_its_exact_relative_weight(specificity):
    profile = _profile()
    ref = _official_ref(specificity=specificity, suffix=f"specificity-{int(specificity)}")
    adjustment = calculate_story_web_context_adjustment(
        hiring_context=profile,
        enrichment=_enrichment(profile, signals=(_signal(ref),), refs=(ref,)),
        story_relevance=_story(
            profile,
            features=(StoryRelevanceFeature.GAME_DEVELOPMENT,),
        ),
    )
    expected = {
        OfficialSourceSpecificity.TEAM_PRODUCT: 1.0,
        OfficialSourceSpecificity.COMPANY_ENGINEERING: 0.75,
        OfficialSourceSpecificity.CAREERS: 0.55,
        OfficialSourceSpecificity.COMPANY: 0.40,
        OfficialSourceSpecificity.PARENT_ORGANIZATION: 0.25,
        OfficialSourceSpecificity.ORGANIZATION_GITHUB: 0.20,
    }[specificity]
    assert adjustment.adjustment == pytest.approx(
        MAX_WEB_CONTEXT_ADJUSTMENT * expected
    )
    assert adjustment.adjustment <= MAX_WEB_CONTEXT_ADJUSTMENT


@pytest.mark.parametrize(
    ("confidence", "expected"),
    (
        (HiringContextConfidence.LOW, 0.35),
        (HiringContextConfidence.MEDIUM, 0.65),
        (HiringContextConfidence.HIGH, 1.0),
    ),
)
def test_context_confidence_weights_are_exact_and_context_only(confidence, expected):
    profile = _profile()
    ref = _official_ref()
    story = _story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,))
    adjustment = calculate_story_web_context_adjustment(
        hiring_context=profile,
        enrichment=_enrichment(
            profile,
            signals=(_signal(ref, confidence=confidence),),
            refs=(ref,),
        ),
        story_relevance=story,
    )
    assert adjustment.adjustment == pytest.approx(
        MAX_WEB_CONTEXT_ADJUSTMENT * expected
    )
    assert story.claim_sufficiency is SufficiencyLevel.HIGH
    assert story.components.evidence_claim_safety == 1.0


def test_duplicate_semantic_signals_keep_the_strongest_actual_policy_contribution():
    profile = _profile()
    team = _official_ref(specificity=OfficialSourceSpecificity.TEAM_PRODUCT, suffix="team-low")
    parent = _official_ref(
        specificity=OfficialSourceSpecificity.PARENT_ORGANIZATION,
        suffix="parent-high",
    )
    story = _story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,))
    adjustment = calculate_story_web_context_adjustment(
        hiring_context=profile,
        enrichment=_enrichment(
            profile,
            signals=(
                _signal(team, confidence=HiringContextConfidence.LOW),
                _signal(parent, confidence=HiringContextConfidence.HIGH),
            ),
            refs=(team, parent),
        ),
        story_relevance=story,
    )
    assert adjustment.adjustment == pytest.approx(MAX_WEB_CONTEXT_ADJUSTMENT * 0.35)
    assert adjustment.traces[0].official_source_refs == (team,)


def test_distinct_signals_can_saturate_but_duplicates_cannot_accelerate_saturation():
    profile = _profile()
    team = _official_ref(suffix="saturation")
    features = (
        StoryRelevanceFeature.GAME_DEVELOPMENT,
        StoryRelevanceFeature.REAL_TIME_SYSTEMS,
        StoryRelevanceFeature.PLATFORM_ENGINEERING,
        StoryRelevanceFeature.DATA_ENGINEERING,
        StoryRelevanceFeature.SECURITY,
    )
    signals = tuple(
        _signal(team, value=value, confidence=HiringContextConfidence.MEDIUM)
        for value in (
            "Game development",
            "Real-time interactive software",
            "Developer platform",
            "Data infrastructure",
            "Security engineering",
        )
    )
    distinct = calculate_story_web_context_adjustment(
        hiring_context=profile,
        enrichment=_enrichment(profile, signals=signals, refs=(team,)),
        story_relevance=_story(profile, features=features),
    )
    duplicates = calculate_story_web_context_adjustment(
        hiring_context=profile,
        enrichment=_enrichment(
            profile,
            signals=(
                _signal(team, confidence=HiringContextConfidence.MEDIUM),
                _signal(team, confidence=HiringContextConfidence.MEDIUM),
            ),
            refs=(team,),
        ),
        story_relevance=_story(
            profile,
            features=(StoryRelevanceFeature.GAME_DEVELOPMENT,),
        ),
    )
    assert distinct.adjustment == MAX_WEB_CONTEXT_ADJUSTMENT
    assert len(distinct.traces) == 5
    assert duplicates.adjustment == MAX_WEB_CONTEXT_ADJUSTMENT * 0.65
    assert len(duplicates.traces) == 1


def test_many_parent_signals_remain_bounded_and_cannot_rescue_a_weak_story():
    profile = _profile()
    parent = _official_ref(
        specificity=OfficialSourceSpecificity.PARENT_ORGANIZATION,
        suffix="weak-parent",
    )
    weak = _story(
        profile,
        score=0.10,
        features=(
            StoryRelevanceFeature.GAME_DEVELOPMENT,
            StoryRelevanceFeature.REAL_TIME_SYSTEMS,
            StoryRelevanceFeature.PLATFORM_ENGINEERING,
            StoryRelevanceFeature.DATA_ENGINEERING,
            StoryRelevanceFeature.SECURITY,
        ),
    )
    result = apply_hiring_context_web_influence(
        hiring_context=profile,
        enrichment=_enrichment(
            profile,
            signals=tuple(
                _signal(parent, value=value)
                for value in (
                    "Game development",
                    "Real-time interactive software",
                    "Developer platform",
                    "Data infrastructure",
                    "Security engineering",
                )
            ),
            refs=(parent,),
        ),
        story_relevance=(weak,),
        enabled=True,
    )
    assert result[0].total_relevance_score == 0.15
    assert result[0].total_relevance_score < 0.50


def test_low_claim_safety_remains_low_after_web_adjustment_and_stays_below_raw_score():
    profile = _profile()
    ref = _official_ref()
    risky = _story(
        profile,
        score=0.60,
        raw_score=0.80,
        risk_adjustment=0.20,
        claim_sufficiency=SufficiencyLevel.LOW,
        claim_safety=0.75,
        features=(StoryRelevanceFeature.GAME_DEVELOPMENT,),
    )
    adjusted = apply_hiring_context_web_influence(
        hiring_context=profile,
        enrichment=_enrichment(profile, signals=(_signal(ref),), refs=(ref,)),
        story_relevance=(risky,),
        enabled=True,
    )[0]
    assert adjusted.claim_sufficiency is SufficiencyLevel.LOW
    assert adjusted.components.evidence_claim_safety == 0.75
    assert adjusted.base_relevance.evidence_risk_adjustment == 0.20
    assert adjusted.total_relevance_score == 0.6375
    assert adjusted.total_relevance_score < adjusted.raw_relevance_score


def test_web_context_cannot_improve_story_sufficiency_or_candidate_feature_set():
    profile = _profile()
    ref = _official_ref()
    incomplete = _story(
        profile,
        story_sufficiency=SufficiencyLevel.LOW,
        features=(StoryRelevanceFeature.GAME_DEVELOPMENT,),
    )
    adjusted = apply_hiring_context_web_influence(
        hiring_context=profile,
        enrichment=_enrichment(profile, signals=(_signal(ref),), refs=(ref,)),
        story_relevance=(incomplete,),
        enabled=True,
    )[0]
    assert adjusted.story_sufficiency is SufficiencyLevel.LOW
    assert adjusted.semantic_features == incomplete.semantic_features


def test_equivalent_content_has_no_timestamp_identity_churn_or_enrichment_change():
    profile = _profile()
    ref = _official_ref()
    signal = _signal(ref)
    first = _enrichment(profile, signals=(signal,), refs=(ref,))
    later = _enrichment(profile, signals=(signal,), refs=(ref,))
    assert first == later
    assert first.fingerprint == later.fingerprint
    assert classify_web_enrichment_change(
        prior_enrichment=first,
        current_enrichment=later,
    ) is WebEnrichmentChangeDecision.UNCHANGED
    assert not {"timestamp", "fetched_at", "retrieved_at"} & {
        item.name for item in fields(HiringContextEnrichment)
    }


@pytest.mark.parametrize("change", ("content", "confidence", "status"))
def test_material_enrichment_changes_require_full_rerank(change):
    profile = _profile()
    ref = _official_ref()
    prior = _enrichment(profile, signals=(_signal(ref),), refs=(ref,))
    if change == "content":
        changed_ref = HiringContextOfficialSourceRef(
            normalized_url=ref.normalized_url,
            source_kind=ref.source_kind,
            canonical_organization=ref.canonical_organization,
            content_fingerprint="c" * 64,
            specificity=ref.specificity,
            trust=ref.trust,
        )
        current = _enrichment(
            profile,
            signals=(_signal(changed_ref),),
            refs=(changed_ref,),
        )
    elif change == "confidence":
        current = _enrichment(
            profile,
            signals=(_signal(ref, confidence=HiringContextConfidence.MEDIUM),),
            refs=(ref,),
        )
    else:
        current = _enrichment(profile, status=HiringContextEnrichmentStatus.UNAVAILABLE)
    assert classify_web_enrichment_change(
        prior_enrichment=prior,
        current_enrichment=current,
    ) is WebEnrichmentChangeDecision.CONTEXT_CHANGED_REQUIRES_FULL_RERANK


def test_source_and_signal_order_are_deterministic():
    profile = _profile()
    team = _official_ref(suffix="ordered-team")
    company = _official_ref(
        specificity=OfficialSourceSpecificity.COMPANY,
        suffix="ordered-company",
    )
    first = _enrichment(
        profile,
        signals=(
            _signal(team, value="Game development"),
            _signal(company, value="Developer platform"),
        ),
        refs=(team, company),
    )
    second = _enrichment(
        profile,
        signals=(
            _signal(company, value="Developer platform"),
            _signal(team, value="Game development"),
        ),
        refs=(company, team),
    )
    story = _story(
        profile,
        features=(
            StoryRelevanceFeature.GAME_DEVELOPMENT,
            StoryRelevanceFeature.PLATFORM_ENGINEERING,
        ),
    )
    assert first == second
    assert calculate_story_web_context_adjustment(
        hiring_context=profile, enrichment=first, story_relevance=story
    ) == calculate_story_web_context_adjustment(
        hiring_context=profile, enrichment=second, story_relevance=story
    )


def test_web_aware_state_rejects_adjusted_snapshots_from_another_enrichment_generation():
    profile = _profile()
    ref = _official_ref()
    enrichment = _enrichment(profile, signals=(_signal(ref),), refs=(ref,))
    result = rank_projects_with_hiring_context_web_influence(
        hiring_context=profile,
        enrichment=enrichment,
        story_relevance=(_story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,)),),
        enabled=True,
    )
    assert isinstance(result.state, WebAwareProjectStoryRankingState)
    changed_ref = HiringContextOfficialSourceRef(
        normalized_url=ref.normalized_url,
        source_kind=ref.source_kind,
        canonical_organization=ref.canonical_organization,
        content_fingerprint="d" * 64,
        specificity=ref.specificity,
        trust=ref.trust,
    )
    changed = _enrichment(
        profile,
        signals=(_signal(changed_ref),),
        refs=(changed_ref,),
    )
    with pytest.raises(WebRankingInfluenceError) as captured:
        build_web_aware_project_story_ranking_state(
            hiring_context=profile,
            enrichment=changed,
            portfolio=result.portfolio,
            project_snapshots=result.state.project_snapshots,
        )
    assert captured.value.code is WebRankingInfluenceErrorCode.MIXED_WEB_RANKING_GENERATION


def test_offline_state_builder_rejects_web_adjusted_snapshots():
    profile = _profile()
    ref = _official_ref()
    enrichment = _enrichment(profile, signals=(_signal(ref),), refs=(ref,))
    result = rank_projects_with_hiring_context_web_influence(
        hiring_context=profile,
        enrichment=enrichment,
        story_relevance=(
            _story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,)),
        ),
        enabled=True,
    )
    assert isinstance(result.state, WebAwareProjectStoryRankingState)
    with pytest.raises(ProjectStoryRankingStateError):
        build_project_story_ranking_state(
            hiring_context=profile,
            portfolio=result.portfolio,
            project_snapshots=result.state.project_snapshots,
        )


def test_policy_change_and_enrichment_change_reject_story_only_rerank():
    profile = _profile()
    ref = _official_ref()
    enrichment = _enrichment(profile, signals=(_signal(ref),), refs=(ref,))
    result = rank_projects_with_hiring_context_web_influence(
        hiring_context=profile,
        enrichment=enrichment,
        story_relevance=(_story(profile, features=(StoryRelevanceFeature.GAME_DEVELOPMENT,)),),
        enabled=True,
    )
    assert isinstance(result.state, WebAwareProjectStoryRankingState)
    assert classify_web_aware_partial_rerank(
        prior_state=result.state,
        hiring_context=profile,
        enrichment=enrichment,
        policy_id="hiring_context_web_ranking.bounded_context_adjustment.v1",
    ) is WebAwarePartialRerankDecision.CONTEXT_CHANGED_REQUIRES_FULL_RERANK


def test_web_aware_partial_rerank_equals_a_clean_full_rerank():
    profile = _profile()
    ref = _official_ref()
    enrichment = _enrichment(profile, signals=(_signal(ref),), refs=(ref,))
    original = _story(
        profile,
        project_id="game-project",
        story_id="game",
        score=0.25,
        features=(StoryRelevanceFeature.GAME_DEVELOPMENT,),
    )
    other = _story(
        profile,
        project_id="backend-project",
        story_id="backend",
        score=0.32,
        features=(StoryRelevanceFeature.API_SYSTEM_DESIGN,),
    )
    prior = rank_projects_with_hiring_context_web_influence(
        hiring_context=profile,
        enrichment=enrichment,
        story_relevance=(original, other),
        enabled=True,
    )
    assert isinstance(prior.state, WebAwareProjectStoryRankingState)
    updated = _story(
        profile,
        project_id="game-project",
        story_id="game",
        revision_id="revision_game_r2",
        score=0.46,
        features=(StoryRelevanceFeature.GAME_DEVELOPMENT,),
    )
    partial = refresh_web_aware_story_ranking(
        prior_state=prior.state,
        hiring_context=profile,
        enrichment=enrichment,
        updated_story_relevance=updated,
        enabled=True,
    )
    clean = rank_projects_with_hiring_context_web_influence(
        hiring_context=profile,
        enrichment=enrichment,
        story_relevance=(updated, other),
        enabled=True,
    )
    assert partial.story_relevance == clean.story_relevance
    assert partial.projects == clean.projects
    assert partial.portfolio == clean.portfolio
    assert partial.state == clean.state


def test_product_review_and_preference_layers_remain_web_ranking_independent():
    root = Path(__file__).parents[1] / "backend"
    review = (root / "hiring_context_ranking_review.py").read_text(encoding="utf-8")
    preferences = (root / "resume_emphasis_preferences.py").read_text(encoding="utf-8")
    assert "hiring_context_web_ranking" not in review
    assert "hiring_context_web_ranking" not in preferences


@pytest.mark.parametrize("level", tuple(SufficiencyLevel))
@pytest.mark.parametrize("raw", (0.0, 0.70, 0.98, 1.0))
def test_web_semantics_pass_through_authoritative_candidate_gate(level, raw):
    from backend.engineering_story_relevance import _CLAIM_SAFETY

    profile = _profile()
    ref = _official_ref()
    safety = _CLAIM_SAFETY[level]
    total = round(raw * safety, 6)
    story = _story(
        profile, raw_score=raw, score=total,
        risk_adjustment=round(raw - total, 6), claim_safety=safety,
        claim_sufficiency=level, story_sufficiency=SufficiencyLevel.LOW,
        features=(StoryRelevanceFeature.GAME_DEVELOPMENT,),
    )
    before = story.to_dict()
    enrichment = _enrichment(profile, signals=(_signal(ref),), refs=(ref,))
    trace = calculate_story_web_context_adjustment(
        hiring_context=profile, enrichment=enrichment, story_relevance=story,
    )
    adjusted = apply_hiring_context_web_influence(
        hiring_context=profile, enrichment=enrichment,
        story_relevance=(story,), enabled=True,
    )[0]
    expected_raw = min(1.0, raw + 0.05)
    assert adjusted.adjusted_raw_relevance_score == expected_raw
    assert adjusted.total_relevance_score == round(expected_raw * safety, 6)
    assert adjusted.total_relevance_score <= expected_raw <= 1.0
    if raw == 0.70:
        assert adjusted.total_relevance_score - total == pytest.approx(0.05 * safety)
        assert adjusted.total_relevance_score > total
    assert adjusted.base_relevance is story
    assert story.to_dict() == before
    assert adjusted.components is story.components
    assert adjusted.claim_sufficiency is level
    assert adjusted.story_sufficiency is SufficiencyLevel.LOW
    assert adjusted.semantic_features == story.semantic_features
    assert adjusted.web_context_adjustment == trace
    assert adjusted.to_dict()["adjusted_raw_relevance_score"] == expected_raw
    with pytest.raises(ValueError):
        replace(adjusted.web_context_adjustment,
                policy_id="hiring_context_web_ranking.bounded_context_adjustment.v1")
