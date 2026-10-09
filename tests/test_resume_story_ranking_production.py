"""Behavioral coverage of the production Story ranking handoff, without I/O."""

from copy import deepcopy
from dataclasses import replace
import json
from unittest.mock import Mock, call

import pytest

from tests.test_resume_retrieval_v2_evidence_integration import (
    EvidenceBoundaryReached, api_server, prepare_pipeline_to_evidence_boundary,
)
from tests.test_resume_story_ranking_integration import (
    CONTEXT, FLAG, integration, matrix_plan, ready_memory,
)


def forbidden(*args, **kwargs):
    pytest.fail("Unexpected legacy, model, or artifact operation")


@pytest.fixture
def portfolio():
    plan = matrix_plan(CONTEXT)
    assert len(plan.projects) == 5
    cards = [{"project_id": p.project_id, "project_name": p.project_id}
             for p in reversed(plan.projects)]
    return plan, cards


def prepare_boundary(monkeypatch, cards):
    real_select = api_server.select_staged_projects_with_ranking
    prepare_pipeline_to_evidence_boundary(monkeypatch, cards, [])
    monkeypatch.setattr(api_server, "select_staged_projects_with_ranking", real_select)
    monkeypatch.setattr(api_server, "jd_requirements_for_prompt", lambda *_: CONTEXT)


@pytest.mark.parametrize("flag", [None, "0", "false", "invalid"])
@pytest.mark.parametrize("allow_selection", [False, True])
def test_disabled_selection_exact_legacy_call_and_order(monkeypatch, portfolio, flag, allow_selection):
    _, cards = portfolio
    prepare_boundary(monkeypatch, cards)
    if flag is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, flag)
    for name in ("prepare_resume_ranking", "load_authoritative_engineering_story_memory",
                 "inspect_authoritative_engineering_story_memory_readiness",
                 "load_project_repository_identity_authority"):
        monkeypatch.setattr(integration, name, forbidden)
    ranking = {"selected_projects": [
        {"project_id": c["project_id"], "project_name": c["project_name"],
         "rank": i, "legacy_extra": ["unchanged"]}
        for i, c in enumerate(reversed(cards[:3]), 1)], "omitted_projects": []}
    before = deepcopy(ranking)
    ranker = Mock(return_value=ranking)
    monkeypatch.setattr(api_server, "rank_projects_for_resume", ranker)
    constraints = {"length_mode": "one_page"}
    selected, returned = api_server.select_staged_projects_with_ranking(
        "job", "resume", {"projects": cards}, allow_selection, constraints)
    ranker.assert_called_once_with(
        [api_server.compact_project_for_prompt(c) for c in (cards if allow_selection else cards[:3])],
        CONTEXT, {"role_family": "software_engineering"},
        api_server.default_resume_constraints(constraints))
    assert selected == (list(reversed(cards[:3])) if allow_selection else cards[:3])
    assert returned is ranking
    assert returned == before


@pytest.mark.parametrize("flag", [None, "0"])
@pytest.mark.parametrize("allow_selection", [False, True])
def test_disabled_staged_selection_resume_evidence_candidate_boundary(monkeypatch, portfolio, flag, allow_selection):
    _, cards = portfolio
    prepare_boundary(monkeypatch, cards)
    if flag is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, flag)
    monkeypatch.setenv(api_server.project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "1")
    for name in ("prepare_resume_ranking", "load_authoritative_engineering_story_memory",
                 "inspect_authoritative_engineering_story_memory_readiness",
                 "load_project_repository_identity_authority", "build_hiring_context_profile",
                 "build_configured_hiring_context_ranking_state"):
        monkeypatch.setattr(integration, name, forbidden)
    ranking = {"selected_projects": [
        {**project, "rank": rank, "bullet_budget": budget, "legacy_extra": ["unchanged"]}
        for rank, (project, budget) in enumerate(zip(reversed(cards[:3]), [4, 3, 2]), 1)
    ], "omitted_projects": []}
    before = deepcopy(ranking)
    selected = list(reversed(cards[:3])) if allow_selection else cards[:3]
    evidence = {project["project_id"]: [{"project_id": project["project_id"], "chunk_id": "safe"}]
                for project in selected}
    ranker = Mock(return_value=ranking)
    provider = Mock(side_effect=lambda project, **_: evidence[project["project_id"]])
    resume_evidence = Mock(wraps=api_server.retrieve_evidence_for_project_for_resume)
    resolver = Mock(wraps=api_server.resolve_resume_evidence_retrieval_mode)
    candidate = Mock(side_effect=lambda *args: {"project_id": args[2]["project_id"], "final_bullets": []})
    monkeypatch.setattr(api_server, "rank_projects_for_resume", ranker)
    monkeypatch.setattr(api_server.project_retrieval_v2, "retrieve_evidence_for_project_v2", provider)
    monkeypatch.setattr(api_server, "retrieve_evidence_for_project_for_resume", resume_evidence)
    monkeypatch.setattr(api_server, "resolve_resume_evidence_retrieval_mode", resolver)
    monkeypatch.setattr(api_server, "retrieve_evidence_for_project", forbidden)
    monkeypatch.setattr(api_server, "build_project_resume_candidate", candidate)
    calls = Mock()
    for name, mock in (("rank", ranker), ("resume_evidence", resume_evidence), ("provider", provider), ("candidate", candidate)):
        calls.attach_mock(mock, name)
    body = api_server.TailorBody(allow_project_selection=allow_selection, length_mode="one_page", language="en")
    constraints = api_server.default_resume_constraints({"length_mode": "one_page"})
    with pytest.raises(EvidenceBoundaryReached):
        api_server.tailor_resume_staged(body)
    expected_calls = [call.rank(
        [api_server.compact_project_for_prompt(project) for project in (cards if allow_selection else cards[:3])],
        CONTEXT, {"role_family": "software_engineering"}, constraints)]
    for project in selected:
        entry = next(entry for entry in ranking["selected_projects"] if entry["project_id"] == project["project_id"])
        expected_calls.extend([
            call.resume_evidence(project, jd_targets=CONTEXT, retrieval_mode=api_server.RESUME_EVIDENCE_RETRIEVAL_MODE_V2),
            call.provider(project, jd_targets=CONTEXT, limit=api_server.RESUME_PROJECT_EVIDENCE_LIMIT),
            call.candidate("Python API validation role", "Existing resume", project,
                           evidence[project["project_id"]], "en", "", ranking, entry, constraints),
        ])
    assert calls.mock_calls == expected_calls
    resolver.assert_called_once_with()
    assert [item.args[7]["bullet_budget"] for item in candidate.call_args_list] == (
        [4, 3, 2] if allow_selection else [2, 3, 4])
    for item in candidate.call_args_list:
        assert item.args[3] is evidence[item.args[2]["project_id"]]
        assert item.args[6] is ranking
        assert "story_ranking" not in item.args[7]
    assert ranking == before


def test_real_candidate_writer_and_checkpoint_keep_tail_story_revision(monkeypatch, portfolio):
    plan, cards = portfolio
    real_builder = api_server.build_project_resume_candidate
    prepare_boundary(monkeypatch, cards)
    monkeypatch.setattr(api_server, "build_project_resume_candidate", real_builder)
    project = next(card for card in cards if card["project_id"] == plan.projects[0].project_id)
    ranked = plan.projects[0]
    # Nine Stories exceed every generic ranking list limit (4, 6, and 8).
    ranked = replace(ranked, stories=ranked.stories + tuple(
        replace(ranked.stories[0], canonical_story_id=f"tail-story-{index}",
                current_revision_id=f"tail-revision-{index}", contribution_member=False)
        for index in range(4)))
    entry = {**project, "rank": 1, "bullet_budget": 4, "story_ranking": ranked.story_handoff()}
    changed = deepcopy(entry)
    changed["story_ranking"]["stories"][-1]["current_revision_id"] = "tail-revision-changed"
    changed["story_ranking"]["current_revision_ids"][-1] = "tail-revision-changed"
    for string_limit, list_limit in ((450, 4), (450, 6), (500, 8)):
        assert api_server.compact_value_for_prompt(entry, string_limit, list_limit) == (
            api_server.compact_value_for_prompt(changed, string_limit, list_limit))
    before = deepcopy(entry)
    evidence = [{"project_id": project["project_id"], "chunk_id": "safe"}]
    constraints = api_server.default_resume_constraints({"length_mode": "one_page"})
    checkpoints = {}

    def persist(records):
        checkpoints.clear()
        checkpoints.update(deepcopy(records))

    monkeypatch.setattr(api_server, "read_resume_candidate_checkpoints", lambda: deepcopy(checkpoints))
    persisted = Mock(side_effect=persist)
    monkeypatch.setattr(api_server, "write_resume_candidate_checkpoints", persisted)
    lookup = Mock(wraps=api_server.get_completed_resume_candidate_checkpoint)
    monkeypatch.setattr(api_server, "get_completed_resume_candidate_checkpoint", lookup)
    writer = Mock(side_effect=lambda **_: {"final_bullets": []})
    monkeypatch.setattr(api_server, "run_resume_bullet_writer_tool", writer)

    def build(rank_entry):
        return api_server.build_project_resume_candidate(
            "job", "resume", project, evidence, "en", "guidance",
            {"selected_projects": [rank_entry], "ranking_generation_id": plan.ranking_generation_id}, rank_entry, constraints)

    token = api_server.current_agent_task_id.set("story-tail-checkpoint-test")
    try:
        first = build(entry)
        cached = build(deepcopy(entry))
        assert cached == first
        assert cached is not first
        assert writer.call_count == persisted.call_count == 1
        revised = build(changed)
        assert build(deepcopy(changed)) == revised
    finally:
        api_server.current_agent_task_id.reset(token)
    assert writer.call_count == persisted.call_count == 2
    assert lookup.call_count == 4
    hashes = [item.args[3] for item in lookup.call_args_list]
    assert hashes[0] == hashes[1]
    assert hashes[2] == hashes[3]
    assert hashes[0] != hashes[2]
    assert {record["source_hash"] for record in checkpoints.values()} == {hashes[0], hashes[2]}
    assert all(record["status"] == "done" for record in checkpoints.values())
    for item, rank_entry in zip(writer.call_args_list, (entry, changed)):
        assert not item.args
        kwargs = item.kwargs
        assert {key: value for key, value in kwargs.items() if key != "extra_rules"} == {
            "section_type": "project", "source_name": project["project_name"],
            "job_description": "job", "resume": "resume",
            "source_facts": api_server.compact_project_for_prompt(project),
            "evidence": evidence, "existing_bullets": [], "language": "en",
        }
        context, _ = json.JSONDecoder().raw_decode(kwargs["extra_rules"].split("Ranking context: ", 1)[1])
        assert context["story_ranking"] == rank_entry["story_ranking"]
        assert len(context["story_ranking"]["stories"]) == 9
        assert context["bullet_budget"] == 4
        assert context["resume_length_constraints"] == constraints
    assert entry == before


@pytest.mark.parametrize("limit, budgets", [(1, [5]), (2, [5, 3]), (3, [4, 3, 2]), (9, [4, 3, 2])])
def test_enabled_selection_portfolio_order_and_fixed_budgets(monkeypatch, portfolio, limit, budgets):
    plan, cards = portfolio
    prepare_boundary(monkeypatch, cards)
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(api_server, "rank_projects_for_resume", forbidden)

    def prepare(**kwargs):
        assert kwargs == {"normalized_job_context": CONTEXT, "project_cards": cards,
                          "project_limit": min(limit, 3)}
        return integration.ResumeRankingDecision(
            integration.ResumeRankingMode.STORY_AWARE_OFFLINE, replace(plan, project_limit=kwargs["project_limit"]))

    monkeypatch.setattr(integration, "prepare_resume_ranking", prepare)
    selected, ranking = api_server.select_staged_projects_with_ranking(
        "job", "resume", {"projects": cards}, True, {"maximum_project_count": limit})
    assert [c["project_id"] for c in selected] == [p.project_id for p in plan.projects[:len(budgets)]]
    assert [e["bullet_budget"] for e in ranking["selected_projects"]] == budgets
    assert [e["rank"] for e in ranking["selected_projects"]] == list(range(1, len(budgets) + 1))
    assert len(selected) <= 3


def test_actual_staged_candidates_receive_resume_evidence_and_full_story_metadata(monkeypatch, portfolio):
    plan, cards = portfolio
    prepare_boundary(monkeypatch, cards)
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(api_server, "rank_projects_for_resume", forbidden)
    monkeypatch.setattr(integration, "prepare_resume_ranking", Mock(return_value=
        integration.ResumeRankingDecision(integration.ResumeRankingMode.STORY_AWARE_OFFLINE, plan)))
    resume_evidence = Mock(side_effect=lambda project, **kwargs: [{"project_id": project["project_id"], "chunk_id": "safe"}])
    monkeypatch.setattr(api_server, "retrieve_evidence_for_project_for_resume", resume_evidence)
    monkeypatch.setattr(api_server, "retrieve_evidence_for_project", forbidden)
    generated = []
    entries = []

    def candidate(job, resume, project, evidence, language, guidance, ranking, entry, constraints):
        assert evidence == [{"project_id": project["project_id"], "chunk_id": "safe"}]
        entries.append(deepcopy(entry))
        result = {"project_id": "model-invented-id", "project_rank": 999,
                  "story_ranking": {"invented": True}, "final_bullets": [],
                  "ranking_generation_id": ranking["ranking_generation_id"]}
        generated.append(result)
        return result

    monkeypatch.setattr(api_server, "build_project_resume_candidate", candidate)
    captured = {}

    class StopBeforeSkills(RuntimeError):
        pass

    def attach(ranking, candidates):
        captured.update(ranking=ranking, candidates=deepcopy(candidates))
        raise StopBeforeSkills

    monkeypatch.setattr(api_server, "attach_candidate_claims_to_project_ranking", attach)
    with pytest.raises(StopBeforeSkills):
        api_server.tailor_resume_staged(api_server.TailorBody())
    assert resume_evidence.call_count == 3
    assert [call.args[0]["project_id"] for call in resume_evidence.call_args_list] == [p.project_id for p in plan.projects[:3]]
    assert all(call.kwargs["jd_targets"] == CONTEXT for call in resume_evidence.call_args_list)
    assert [e["bullet_budget"] for e in entries] == [4, 3, 2]
    assert len(captured["candidates"]) == 3
    for candidate, project in zip(captured["candidates"], plan.projects):
        assert candidate["project_id"] == project.project_id
        assert candidate["project_rank"] == project.portfolio_position
        assert candidate["story_ranking"] == project.story_handoff()
        assert len(candidate["story_ranking"]["stories"]) == 5


def test_story_identity_lookup_claims_and_candidate_matching_require_exact_id(portfolio):
    plan, _ = portfolio
    entry = {"project_id": plan.projects[0].project_id, "project_name": "Same Name",
             "story_ranking": plan.projects[0].story_handoff()}
    ranking = {"selected_projects": [entry]}
    wrong = {"project_id": "impostor", "project_name": "Same Name", "allowed_claims": ["wrong"]}
    right = {"project_id": entry["project_id"], "project_name": "Different Name",
             "allowed_claims": ["right"], "forbidden_claims": ["unsafe"], "final_bullets": ["bullet"]}
    assert api_server.project_ranking_entry_for_project(ranking, wrong) == {}
    assert api_server.project_ranking_entry_for_project(ranking, right) is entry
    assert not api_server.project_candidate_matches_entry(wrong, entry)
    assert api_server.project_candidate_matches_entry(right, entry)
    assert api_server.find_project_candidate_for_entry([wrong, right], entry) is right
    assert api_server.find_project_candidate_for_entry([wrong], entry) is None
    untouched = api_server.attach_candidate_claims_to_project_ranking(ranking, [wrong])
    assert untouched == ranking
    attached = api_server.attach_candidate_claims_to_project_ranking(ranking, [wrong, right])["selected_projects"][0]
    assert attached["allowed_claims"] == ["right"]
    assert attached["forbidden_claims"] == ["unsafe"]
    assert attached["candidate_bullet_count"] == 1
    assert attached["story_ranking"] == entry["story_ranking"]
    assert "allowed_claims" not in entry


@pytest.mark.parametrize("story", [False, True])
def test_summary_builder_and_reducer_order_against_generated_fit(monkeypatch, portfolio, story):
    plan, cards = portfolio
    real_summary_builder = api_server.build_summary_resume_candidate
    prepare_boundary(monkeypatch, cards)
    monkeypatch.setattr(api_server, "build_summary_resume_candidate", real_summary_builder)
    monkeypatch.setenv(FLAG, "1" if story else "0")
    monkeypatch.setattr(api_server, "jd_core_for_prompt", lambda *_: {})
    candidates = [{"project_id": p.project_id, "project_name": p.project_id,
                   "project_rank": i, "fit": fit, "final_bullets": ["Supported work"],
                   **({"story_ranking": p.story_handoff()} if story else {})}
                  for i, (p, fit) in enumerate(zip(plan.projects, ["low", "medium", "high"]), 1)]
    expected = [c["project_name"] for c in (candidates if story else list(reversed(candidates)))]
    payload = api_server.build_compact_summary_input("job", "resume", list(reversed(candidates)), {}, {}, "en")
    assert [p["projectName"] for p in payload["topProjectHighlights"]] == expected
    scrambled = deepcopy(payload)
    scrambled["topProjectHighlights"].reverse()
    reduced = api_server.reduce_compact_summary_payload_for_limit(scrambled, max_chars=1)
    assert [p["projectName"] for p in reduced["topProjectHighlights"]] == expected[:2]
    assert scrambled != reduced

    def model_boundary(**kwargs):
        built = kwargs["compact_builder"](max_chars=100000)["payload"]
        assert [p["projectName"] for p in built["topProjectHighlights"]] == expected
        return '{"recommended_summary": "Supported engineering work."}'

    model = Mock(side_effect=model_boundary)
    monkeypatch.setattr(api_server, "safe_model_call", model)
    result = api_server.build_summary_resume_candidate("job", "resume", {}, candidates, {}, {}, "en")
    assert result["recommended_summary"]
    model.assert_called_once()


def test_stale_story_integrity_returns_http409_without_legacy(monkeypatch, ready_memory, portfolio):
    _, cards = portfolio
    _, readiness = ready_memory
    prepare_boundary(monkeypatch, cards)
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(integration, "inspect_authoritative_engineering_story_memory_readiness",
                        Mock(return_value=replace(readiness,
                            state=integration.StoryMemoryReadinessState.STALE_OR_REVALIDATION_REQUIRED,
                            stale_story_count=1)))
    monkeypatch.setattr(api_server, "rank_projects_for_resume", forbidden)
    monkeypatch.setattr(api_server, "retrieve_evidence_for_project_for_resume", forbidden)
    with pytest.raises(api_server.HTTPException) as error:
        api_server.tailor_resume_staged(api_server.TailorBody())
    assert error.value.status_code == 409
    assert error.value.detail == "Project ranking needs revalidation before generation."
