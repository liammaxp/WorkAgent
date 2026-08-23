from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import api_server
from backend.project_evidence_coverage import (
    CoverageCategory,
    CoverageGap,
    CoverageReasonCode,
    CoverageState,
    GapPriority,
    GapPriorityReasonCode,
    PrioritizedCoverageGap,
)
from backend.project_evidence_followup_intents import build_followup_retrieval_intents
from backend import project_retrieval_v2


PROJECT = {"project_id": "WorkAgent", "project_name": "WorkAgent"}
OTHER_PROJECT_ID = "OtherProject"


def intent(project_id: str = PROJECT["project_id"]):
    prioritized = PrioritizedCoverageGap(
        gap=CoverageGap(
            category=CoverageCategory.VALIDATION_REPAIR,
            state=CoverageState.MISSING,
            reason_code=CoverageReasonCode.UNSUPPORTED,
        ),
        priority=GapPriority.HIGH,
        searchable=True,
        reason_code=GapPriorityReasonCode.HIGH_VALUE_MECHANISM_GAP,
    )
    return build_followup_retrieval_intents(
        project_id=project_id,
        prioritized_gaps=(prioritized,),
    )[0]


@pytest.mark.parametrize("argument", ("omitted", None, ()))
def test_wrapper_omitted_none_and_empty_preserve_strict_old_v2_contract(monkeypatch, argument):
    expected = [{"safe": "evidence"}]
    captured = {}

    def strict_v2(project, *, jd_targets, limit):
        captured.update({"project": project, "jd_targets": jd_targets, "limit": limit})
        return expected

    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", strict_v2)
    kwargs = {
        "jd_targets": {"technologies": ["Python"]},
        "retrieval_mode": api_server.RESUME_EVIDENCE_RETRIEVAL_MODE_V2,
    }
    if argument != "omitted":
        kwargs["retrieval_intents"] = argument

    result = api_server.retrieve_evidence_for_project_for_resume(PROJECT, **kwargs)

    assert result is expected
    assert captured == {
        "project": PROJECT,
        "jd_targets": {"technologies": ["Python"]},
        "limit": api_server.RESUME_PROJECT_EVIDENCE_LIMIT,
    }


def test_wrapper_explicit_v2_forwards_valid_intents_without_rereading_flag(monkeypatch):
    supplied = (intent(),)
    captured = {}
    monkeypatch.setattr(
        api_server,
        "resolve_resume_evidence_retrieval_mode",
        lambda: pytest.fail("explicit retrieval mode must not reread the feature flag"),
    )
    monkeypatch.setattr(
        api_server,
        "retrieve_evidence_for_project",
        lambda *_args, **_kwargs: pytest.fail("legacy retrieval must not run"),
    )

    def v2(project, **kwargs):
        captured.update({"project": project, **kwargs})
        return []

    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", v2)

    assert api_server.retrieve_evidence_for_project_for_resume(
        PROJECT,
        retrieval_mode=api_server.RESUME_EVIDENCE_RETRIEVAL_MODE_V2,
        retrieval_intents=supplied,
    ) == []
    assert captured["project"] is PROJECT
    assert captured["retrieval_intents"] == supplied
    assert captured["limit"] == api_server.RESUME_PROJECT_EVIDENCE_LIMIT


def test_legacy_valid_intents_are_validated_then_ignored_without_extra_retrieval(monkeypatch):
    expected = [{"legacy": True}]
    calls = []
    monkeypatch.setattr(
        project_retrieval_v2,
        "retrieve_evidence_for_project_v2",
        lambda *_args, **_kwargs: pytest.fail("v2 must not run in legacy mode"),
    )

    def legacy(project):
        calls.append(project)
        return expected

    monkeypatch.setattr(api_server, "retrieve_evidence_for_project", legacy)

    result = api_server.retrieve_evidence_for_project_for_resume(
        PROJECT,
        retrieval_mode=api_server.RESUME_EVIDENCE_RETRIEVAL_MODE_LEGACY,
        retrieval_intents=(intent(),),
    )

    assert result is expected
    assert calls == [PROJECT]


@pytest.mark.parametrize(
    ("project", "supplied"),
    (
        (PROJECT, lambda: (intent(OTHER_PROJECT_ID),)),
        (PROJECT, lambda: (intent(), intent(OTHER_PROJECT_ID))),
        ({**PROJECT, "project_id": " WorkAgent "}, lambda: (intent(),)),
        (PROJECT, lambda: "raw query"),
        (PROJECT, lambda: {"raw_query": "validation"}),
        (PROJECT, lambda: (item for item in (intent(),))),
        (PROJECT, lambda: (object(),)),
        (PROJECT, lambda: (intent(),) * 5),
    ),
)
def test_invalid_or_cross_project_intent_sequences_fail_before_all_retrieval(
    monkeypatch, project, supplied
):
    calls = []
    monkeypatch.setattr(
        api_server,
        "retrieve_evidence_for_project",
        lambda *_args, **_kwargs: calls.append("legacy"),
    )
    monkeypatch.setattr(
        project_retrieval_v2,
        "retrieve_evidence_for_project_v2",
        lambda *_args, **_kwargs: calls.append("v2"),
    )

    result = api_server.retrieve_evidence_for_project_for_resume(
        project,
        retrieval_mode=api_server.RESUME_EVIDENCE_RETRIEVAL_MODE_V2,
        retrieval_intents=supplied(),
    )

    assert result == []
    assert calls == []
