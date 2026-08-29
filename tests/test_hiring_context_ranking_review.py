"""Product-safety tests for the read-only tailoring review projection."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import backend.hiring_context_ranking_review as review_module
from backend.engineering_story_memory_service import (
    StoryMemoryArtifactStatus,
    StoryMemoryReadinessState,
)
from backend.engineering_story_models import StoryOpportunityLevel, SufficiencyLevel
from backend.engineering_story_relevance import rank_engineering_stories_for_hiring_context
from backend.hiring_context_intelligence import build_hiring_context_profile
from backend.project_portfolio_ranking import rank_project_portfolio
from backend.project_story_ranking import aggregate_project_story_relevance
from backend.project_story_ranking_refresh import (
    PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
    ProjectStoryRelevanceSnapshot,
)
from backend.project_story_ranking_state import build_project_story_ranking_state
from tests.test_project_story_ranking_refresh import _view


ALPHA = "review_alpha"
BETA = "review_beta"


def _views(*, both_gaps: bool = False, include_additional: bool = True):
    values = [
        _view(
            story_key="review_architecture",
            revision_key="review_architecture_r1",
            project_id=ALPHA,
            text="Designed backend API system architecture and reliable data flow",
        ),
        _view(
            story_key="review_reliability",
            revision_key="review_reliability_r1",
            project_id=ALPHA,
            text="Built reliability validation repair testing and recovery",
            claim_level=SufficiencyLevel.LOW,
            story_level=(SufficiencyLevel.LOW if both_gaps else SufficiencyLevel.HIGH),
            opportunity_level=StoryOpportunityLevel.HIGH,
        ),
        _view(
            story_key="review_debugging",
            revision_key="review_debugging_r1",
            project_id=ALPHA,
            text="Debugged performance concurrency algorithms with regression testing",
            story_level=SufficiencyLevel.LOW,
            opportunity_level=StoryOpportunityLevel.HIGH,
        ),
        _view(
            story_key="review_data",
            revision_key="review_data_r1",
            project_id=BETA,
            text="Built data engineering transformation analytics and decision support",
        ),
    ]
    if include_additional:
        values.insert(3, _view(
            story_key="review_automation",
            revision_key="review_automation_r1",
            project_id=ALPHA,
            text="Automated an integration workflow",
        ))
    return tuple(values)


def _state(*, company="Example Systems", role="Backend Engineer", both_gaps=False):
    profile = build_hiring_context_profile(
        company=company,
        role_title=role,
        team=None,
        normalized_job_context={
            "required_qualifications": [
                "Design reliable backend API architecture and validation",
            ],
        },
    )
    ranked = rank_engineering_stories_for_hiring_context(
        hiring_context=profile,
        story_views=_views(both_gaps=both_gaps),
    )
    by_project = {}
    for story in ranked:
        by_project.setdefault(story.project_id, []).append(story)
    projects = {
        project_id: aggregate_project_story_relevance(
            project_id=project_id,
            story_relevance=stories,
        )
        for project_id, stories in by_project.items()
    }
    portfolio = rank_project_portfolio(projects=tuple(projects.values()))
    assert portfolio is not None
    snapshots = tuple(
        ProjectStoryRelevanceSnapshot(
            snapshot_policy_id=PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
            source_portfolio_id=portfolio.portfolio_id,
            project_relevance=projects[ranking.project_id],
            story_relevance=tuple(by_project[ranking.project_id]),
        )
        for ranking in portfolio.ranked_projects
    )
    return profile, build_project_story_ranking_state(
        hiring_context=profile,
        portfolio=portfolio,
        project_snapshots=snapshots,
    )


def _review(*, project_memory=None, company="Example Systems", role="Backend Engineer", both_gaps=False, language="en"):
    profile, state = _state(company=company, role=role, both_gaps=both_gaps)
    return review_module.present_hiring_context_ranking_review(
        hiring_context=profile,
        ranking_state=state,
        project_memory=project_memory or {
            "projects": [
                {"project_id": ALPHA, "project_name": "Alpha Product"},
                {"project_id": BETA, "project_name": "Beta Analytics"},
            ],
        },
        language=language,
    )


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def _all_text(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _all_text(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_text(child)


def test_ready_review_uses_product_status():
    assert _review().status.value == "ready"


def test_context_exposes_company_role_and_optional_team():
    result = _review()
    assert result.hiring_context.company == "Example Systems"
    assert result.hiring_context.role_title == "Backend Engineer"
    assert result.hiring_context.team is None


def test_context_role_family_is_product_copy():
    assert _review().hiring_context.primary_role_family == "Backend engineering"


def test_context_signals_are_bounded():
    assert len(_review().hiring_context.context_signals) <= 8


def test_secondary_role_families_are_bounded():
    assert len(_review().hiring_context.secondary_role_families) <= 6


def test_project_order_is_authoritative_portfolio_order():
    profile, state = _state()
    result = review_module.present_hiring_context_ranking_review(
        hiring_context=profile,
        ranking_state=state,
        project_memory=None,
    )
    assert [item.project_id for item in result.projects] == [
        item.project_id for item in state.portfolio.ranked_projects
    ]


def test_project_display_name_uses_exact_identity():
    result = _review(project_memory={"projects": [{"project_id": ALPHA, "project_name": "Exact Alpha"}]})
    alpha = next(item for item in result.projects if item.project_id == ALPHA)
    assert alpha.display_name == "Exact Alpha"


def test_project_display_name_does_not_fuzzy_match():
    result = _review(project_memory={"projects": [{"project_id": ALPHA.upper(), "project_name": "Wrong Alpha"}]})
    alpha = next(item for item in result.projects if item.project_id == ALPHA)
    assert alpha.display_name == f"Project {alpha.position}"


def test_missing_project_name_uses_neutral_position_label():
    result = _review(project_memory={"projects": []})
    assert all(item.display_name == f"Project {item.position}" for item in result.projects)


def test_project_metadata_cannot_change_ranking_order():
    first = _review(project_memory={"projects": [{"project_id": ALPHA, "project_name": "Zulu"}]})
    second = _review(project_memory={"projects": [{"project_id": ALPHA, "project_name": "Aardvark"}]})
    assert [item.project_id for item in first.projects] == [item.project_id for item in second.projects]


def test_story_order_preserves_complete_snapshot_order():
    profile, state = _state()
    result = review_module.present_hiring_context_ranking_review(
        hiring_context=profile,
        ranking_state=state,
        project_memory=None,
    )
    for project in result.projects:
        snapshot = state.snapshot_for_project(project.project_id)
        assert snapshot is not None
        visible = project.strongest_stories + project.additional_stories
        assert [item.story_id for item in visible] == [
            item.canonical_story_id for item in snapshot.story_relevance
        ]


def test_strongest_story_membership_comes_from_contributions():
    profile, state = _state()
    result = review_module.present_hiring_context_ranking_review(
        hiring_context=profile,
        ranking_state=state,
        project_memory=None,
    )
    alpha = next(item for item in result.projects if item.project_id == ALPHA)
    snapshot = state.snapshot_for_project(ALPHA)
    assert snapshot is not None
    assert {item.story_id for item in alpha.strongest_stories} == {
        item.canonical_story_id for item in snapshot.project_relevance.contributions
    }


def test_noncontributing_story_is_present_as_additional():
    alpha = next(item for item in _review().projects if item.project_id == ALPHA)
    assert len(alpha.additional_stories) == 1


def test_story_labels_never_use_internal_story_identity():
    for project in _review().projects:
        for story in project.strongest_stories + project.additional_stories:
            assert story.story_id not in story.label
            assert "engineering_story_" not in story.label


def test_story_labels_are_deterministic():
    assert _review().to_dict() == _review().to_dict()


def test_story_reasons_are_bounded():
    for project in _review().projects:
        for story in project.strongest_stories + project.additional_stories:
            assert len(story.relevance_reasons) <= 3


def test_project_reasons_are_bounded():
    assert all(len(project.relevance_reasons) <= 3 for project in _review().projects)


def test_claim_gap_uses_product_safe_notice():
    notices = [
        notice
        for project in _review().projects
        for story in project.strongest_stories
        for notice in story.notices
    ]
    assert "Some claims may need confirmation" in notices


def test_story_gap_uses_product_safe_notice():
    notices = [
        notice
        for project in _review().projects
        for story in project.strongest_stories
        for notice in story.notices
    ]
    assert "More context could strengthen this story" in notices


def test_both_gaps_remain_independent_notices():
    assert any(
        len(story.notices) == 2
        for project in _review(both_gaps=True).projects
        for story in project.strongest_stories
    )


def test_presentation_excludes_numeric_ranking_fields():
    keys = set(_all_keys(_review().to_dict()))
    assert not any("score" in key for key in keys)
    assert not any("weight" in key for key in keys)
    assert not any("adjustment" in key for key in keys)


def test_presentation_excludes_integrity_and_policy_fields():
    keys = set(_all_keys(_review().to_dict()))
    assert not any(token in key for key in keys for token in ("policy", "fingerprint", "revision", "provenance"))


def test_presentation_excludes_raw_candidate_authority():
    keys = set(_all_keys(_review().to_dict()))
    assert not any(token in key for key in keys for token in ("evidence", "capability", "claim_boundary", "current_story"))


def test_unsupported_and_plausible_missing_never_enter_copy():
    text = " ".join(_all_text(_review().to_dict())).casefold()
    assert "plausible_missing" not in text
    assert "unsupported" not in text


def test_coalition_context_does_not_create_candidate_persona():
    text = " ".join(_all_text(_review(company="The Coalition", role="Software Engineering Intern").to_dict())).casefold()
    assert "candidate is a game developer" not in text
    assert "game developer candidate" not in text


def test_coalition_context_does_not_invent_candidate_technologies():
    text = " ".join(_all_text(_review(company="The Coalition", role="Software Engineering Intern").to_dict())).casefold()
    for technology in ("unity", "unreal", "c++", "directx"):
        assert technology not in text


def test_privacy_role_title_does_not_become_story_copy():
    result = _review(company="Advisory", role="Privacy Engineer")
    story_text = " ".join(
        item
        for project in result.projects
        for story in project.strongest_stories + project.additional_stories
        for item in (story.label, *story.relevance_reasons)
    ).casefold()
    assert "privacy" not in story_text
    assert "consultant" not in story_text


def test_correction_identity_stays_in_hiring_context():
    result = _review(company="Corrected Employer", role="Corrected Role")
    assert result.hiring_context.company == "Corrected Employer"
    project_text = " ".join(_all_text([item.to_dict() for item in result.projects]))
    assert "Corrected Employer" not in project_text
    assert "Corrected Role" not in project_text


def test_chinese_copy_is_deterministic_and_product_safe():
    result = _review(language="zh")
    assert result.hiring_context.primary_role_family == "后端工程"
    assert result.projects[0].relevance_reasons[0].startswith("突出实践：")


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (StoryMemoryArtifactStatus.EMPTY, "empty"),
        (StoryMemoryArtifactStatus.MISSING, "unavailable"),
        (StoryMemoryArtifactStatus.INVALID, "unavailable"),
        (StoryMemoryArtifactStatus.UNSUPPORTED_VERSION, "unavailable"),
        (StoryMemoryArtifactStatus.INTEGRITY_MISMATCH, "error"),
    ],
)
def test_story_memory_nonready_states_are_product_safe(monkeypatch, status, expected):
    monkeypatch.setattr(
        review_module,
        "load_authoritative_engineering_story_memory",
        lambda _path: SimpleNamespace(status=status, memory=None),
    )
    result = review_module.build_hiring_context_ranking_review(
        company="Example",
        team=None,
        role_title="Backend Engineer",
        normalized_job_context={},
    )
    assert result.status.value == expected
    assert result.projects == ()


def test_ranking_failure_returns_no_partial_projects(monkeypatch):
    monkeypatch.setattr(
        review_module,
        "load_authoritative_engineering_story_memory",
        lambda _path: SimpleNamespace(
            status=StoryMemoryArtifactStatus.READY,
            memory=SimpleNamespace(histories=()),
        ),
    )
    monkeypatch.setattr(review_module, "_active_story_views", lambda _memory: {ALPHA: _views()})
    monkeypatch.setattr(
        review_module,
        "inspect_authoritative_engineering_story_memory_readiness",
        lambda **_kwargs: SimpleNamespace(state=StoryMemoryReadinessState.READY),
    )
    monkeypatch.setattr(
        review_module,
        "rank_engineering_stories_for_hiring_context",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("integrity")),
    )
    result = review_module.build_hiring_context_ranking_review(
        company="Example",
        team=None,
        role_title="Backend Engineer",
        normalized_job_context={},
    )
    assert result.status.value == "error"
    assert result.projects == ()


@pytest.mark.parametrize(
    "state",
    [
        StoryMemoryReadinessState.MISSING,
        StoryMemoryReadinessState.INVALID,
        StoryMemoryReadinessState.STALE_OR_REVALIDATION_REQUIRED,
        StoryMemoryReadinessState.CONFLICTED,
    ],
)
def test_nonready_lifecycle_state_does_not_rank_partial_memory(monkeypatch, state):
    monkeypatch.setattr(
        review_module,
        "load_authoritative_engineering_story_memory",
        lambda _path: SimpleNamespace(
            status=StoryMemoryArtifactStatus.READY,
            memory=SimpleNamespace(histories=()),
        ),
    )
    monkeypatch.setattr(
        review_module,
        "inspect_authoritative_engineering_story_memory_readiness",
        lambda **_kwargs: SimpleNamespace(state=state),
    )
    monkeypatch.setattr(
        review_module,
        "rank_engineering_stories_for_hiring_context",
        lambda **_kwargs: pytest.fail("ranking must not run"),
    )
    result = review_module.build_hiring_context_ranking_review(
        company="Example",
        team=None,
        role_title="Backend Engineer",
        normalized_job_context={},
    )
    assert result.status.value == "unavailable"
    assert result.projects == ()


def test_authoritative_active_reader_is_used_per_exact_project(monkeypatch):
    calls = []
    memory = SimpleNamespace(histories=(
        SimpleNamespace(project_id=BETA),
        SimpleNamespace(project_id=ALPHA),
        SimpleNamespace(project_id=ALPHA),
    ))
    monkeypatch.setattr(
        review_module,
        "get_active_engineering_stories_for_project",
        lambda source, project_id: calls.append((source, project_id)) or (_views(include_additional=False)[0],),
    )
    review_module._active_story_views(memory)
    assert calls == [(memory, ALPHA), (memory, BETA)]


def test_service_source_has_no_legacy_web_model_retrieval_or_write_dependency():
    source = Path(review_module.__file__).read_text(encoding="utf-8").casefold()
    for forbidden in (
        "rank_projects_for_resume",
        "select_staged_projects_with_ranking",
        "project_bullet_budget",
        "tailor_resume_staged",
        "chroma",
        "openai",
        "requests.",
        "write_authoritative_engineering_story_memory",
        "project_retrieval",
    ):
        assert forbidden not in source
