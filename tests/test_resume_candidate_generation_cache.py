"""Actual candidate checkpoints consume the full accepted ranking generation."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json
from unittest.mock import Mock

import pytest

from backend import resume_story_ranking_integration as integration
from backend.hiring_context_web_enrichment import HiringContextEnrichmentStatus
from tests.test_hiring_context_cross_context_ranking import _candidate_views
from tests.test_hiring_context_web_ranking import _enrichment, _official_ref, _signal
from tests.test_resume_story_ranking_integration import CONTEXT, FLAG, matrix_plan
from tests.test_resume_story_ranking_layout import api_server


GENERATION = "ranking_generation_id"


def selection(plan):
    cards = [{"project_id": item.project_id, "project_name": item.project_id} for item in plan.projects]
    return integration.production_selection(plan, cards, lambda rank, count: (4, 3, 2)[rank - 1])


@pytest.fixture
def checkpoints(monkeypatch):
    records = {}
    monkeypatch.setattr(api_server, "read_resume_candidate_checkpoints", lambda: json.loads(json.dumps(records)))

    def persist(current):
        records.clear()
        records.update(json.loads(json.dumps(current)))

    save = Mock(side_effect=persist)
    monkeypatch.setattr(api_server, "write_resume_candidate_checkpoints", save)
    writer = Mock(side_effect=lambda **_: {"final_bullets": [{"bullet": "Implemented supported validation."}]})
    monkeypatch.setattr(api_server, "run_resume_bullet_writer_tool", writer)
    monkeypatch.setattr(api_server, "ensure_project_candidate_bullet_budget", lambda candidate, *_: candidate)
    token = api_server.current_agent_task_id.set("accepted-generation-checkpoint")
    try:
        yield records, writer, save
    finally:
        api_server.current_agent_task_id.reset(token)


def build(ranking, project=None):
    entry = ranking["selected_projects"][0] if project is None else next(
        item for item in ranking["selected_projects"] if item["project_id"] == project["project_id"]
    )
    project = project or {"project_id": entry["project_id"], "project_name": entry["project_name"]}
    return api_server.build_project_resume_candidate(
        "Unchanged job description", "Original resume", project,
        [{"project_id": project["project_id"], "chunk_id": "accepted-proof"}], "en", "guidance",
        ranking, entry, {"length_mode": "one_page"},
    )


def changed_revision(views, story):
    return tuple(
        replace(view, current_revision_id="engineering_story_revision_" + "d" * 24,
                provenance_fingerprint=hashlib.sha256(b"accepted-new-revision").hexdigest())
        if view.canonical_story_id == story.canonical_story_id else view
        for view in views
    )


def plans_for_change(kind):
    views = _candidate_views()
    before = matrix_plan(CONTEXT, views=views)
    if kind == "hidden":
        story = before.projects[0].stories[4]
        assert not story.contribution_member
        after = matrix_plan(CONTEXT, views=changed_revision(views, story))
    elif kind == "unselected":
        assert len(before.projects) > before.project_limit
        story = before.projects[before.project_limit].stories[0]
        after = matrix_plan(CONTEXT, views=changed_revision(views, story))
    elif kind == "selected":
        after = matrix_plan(CONTEXT, views=changed_revision(views, before.projects[0].stories[0]))
    elif kind == "context":
        after = matrix_plan({**CONTEXT, "team": "Analytics", "job_title": "Data Analyst"}, views=views)
    elif kind == "web":
        ref = _official_ref()
        enrichment = _enrichment(before.hiring_context, refs=(ref,), signals=(_signal(ref),))
        after = matrix_plan(CONTEXT, views=views, enrichment=enrichment)
    else:
        raise AssertionError(kind)
    assert before.state.state_id != after.state.state_id
    return before, after


def test_equivalent_complete_generation_resumes_actual_checkpoint(checkpoints):
    records, writer, saved = checkpoints
    before = matrix_plan(CONTEXT)
    reconstructed = matrix_plan(CONTEXT, views=tuple(reversed(_candidate_views())))
    assert before.ranking_generation_id == reconstructed.ranking_generation_id == before.state.state_id
    _, ranking = selection(before)
    _, reconstructed_ranking = selection(reconstructed)
    first = build(ranking)
    restored = build(reconstructed_ranking)
    assert restored == first and restored is not first
    assert first[GENERATION] == before.state.state_id
    assert writer.call_count == saved.call_count == len(records) == 1


@pytest.mark.parametrize("kind", ["hidden", "unselected", "selected", "context", "web"])
def test_actual_checkpoint_restarts_under_changed_complete_generation(checkpoints, kind):
    records, writer, saved = checkpoints
    before, after = plans_for_change(kind)
    projects, old_ranking = selection(before)
    new_projects, new_ranking = selection(after)
    common = {p["project_id"] for p in projects} & {p["project_id"] for p in new_projects}
    project = next(project for project in projects if project["project_id"] in common)
    if kind in {"hidden", "unselected"}:
        assert [p["project_id"] for p in projects] == [p["project_id"] for p in new_projects]
        assert before.projects[0].top_contributing_story_ids == after.projects[0].top_contributing_story_ids
        if kind == "hidden":
            assert before.state.portfolio == after.state.portfolio
        else:
            assert old_ranking["selected_projects"] == new_ranking["selected_projects"]
    first = build(old_ranking, project)
    assert build(deepcopy(old_ranking), project) == first
    second = build(new_ranking, project)
    assert build(deepcopy(new_ranking), project) == second
    assert first[GENERATION] != second[GENERATION]
    assert writer.call_count == saved.call_count == len(records) == 2
    assert len({record["source_hash"] for record in records.values()}) == 2


def test_upstream_policy_generation_control_invalidates_reuse_without_downstream_policy_logic(checkpoints):
    records, writer, _ = checkpoints
    _, ranking = selection(matrix_plan(CONTEXT))
    first = build(ranking)
    new = deepcopy(ranking)
    # Simulate the next accepted upstream identity; cache must not interpret policy versions.
    new[GENERATION] = "project_story_ranking_state_" + "e" * 24
    second = build(new)
    assert first[GENERATION] != second[GENERATION]
    assert len(records) == writer.call_count == 2


@pytest.mark.parametrize("status", [
    HiringContextEnrichmentStatus.DISABLED,
    HiringContextEnrichmentStatus.NO_OFFICIAL_SOURCE,
    HiringContextEnrichmentStatus.UNAVAILABLE,
    HiringContextEnrichmentStatus.CONFLICTED,
])
def test_optional_web_without_effect_preserves_actual_cache_hit(checkpoints, status):
    records, writer, _ = checkpoints
    before = matrix_plan(CONTEXT)
    enrichment = _enrichment(before.hiring_context, status=status,
        conflicts=("domain_conflict",) if status is HiringContextEnrichmentStatus.CONFLICTED else ())
    after = matrix_plan(CONTEXT, enrichment=enrichment)
    assert before.state == after.state
    _, old_ranking = selection(before)
    _, new_ranking = selection(after)
    assert build(old_ranking) == build(new_ranking)
    assert len(records) == writer.call_count == 1


@pytest.mark.parametrize("tamper", ["missing", "different"])
def test_matching_cache_key_cannot_restore_candidate_from_missing_or_other_generation(checkpoints, tamper):
    records, writer, saved = checkpoints
    _, ranking = selection(matrix_plan(CONTEXT))
    first = build(ranking)
    cached = next(iter(records.values()))["candidate_json"]
    if tamper == "missing":
        cached.pop(GENERATION)
    else:
        cached[GENERATION] = "project_story_ranking_state_" + "f" * 24
    restored = build(ranking)
    assert restored[GENERATION] == first[GENERATION]
    assert writer.call_count == saved.call_count == 2
    assert len(records) == 1


def test_missing_story_generation_fails_before_checkpoint_can_hide_it(monkeypatch, checkpoints):
    _, writer, _ = checkpoints
    _, ranking = selection(matrix_plan(CONTEXT))
    ranking.pop(GENERATION)
    lookup = Mock(return_value={"project_id": "stale", "final_bullets": []})
    monkeypatch.setattr(api_server, "get_completed_resume_candidate_checkpoint", lookup)
    with pytest.raises(api_server.HTTPException) as error:
        build(ranking)
    assert error.value.status_code == 409
    lookup.assert_not_called()
    writer.assert_not_called()


def test_generation_is_separate_control_input_and_never_writer_prompt_content(monkeypatch, checkpoints):
    records, writer, _ = checkpoints
    _, ranking = selection(matrix_plan(CONTEXT))
    captured = []
    real_hash = api_server.stable_hash

    def hashing(payload):
        captured.append(deepcopy(payload))
        return real_hash(payload)

    monkeypatch.setattr(api_server, "stable_hash", hashing)
    result = build(ranking)
    source = next(payload for payload in captured if "project_angle_strategy" in payload)
    assert source[GENERATION] == ranking[GENERATION]
    assert GENERATION not in json.dumps(source["ranking_context"])
    assert ranking[GENERATION] not in json.dumps(writer.call_args.kwargs)
    assert result[GENERATION] == ranking[GENERATION]
    assert len(records) == 1


def test_prompt_compaction_cannot_discard_generation_control_identity(monkeypatch, checkpoints):
    records, writer, _ = checkpoints
    before, after = plans_for_change("unselected")
    _, old_ranking = selection(before)
    _, new_ranking = selection(after)
    original_compactor = api_server.compact_value_for_prompt

    def compactor(value, *args, **kwargs):
        if isinstance(value, dict) and ("selected_projects" in value or "story_ranking" in value):
            return {"prompt_trimmed": True}
        return original_compactor(value, *args, **kwargs)

    monkeypatch.setattr(api_server, "compact_value_for_prompt", compactor)
    build(old_ranking)
    build(new_ranking)
    assert len(records) == writer.call_count == 2
    assert len({record["source_hash"] for record in records.values()}) == 2


@pytest.mark.parametrize("fallback", [False, True])
def test_legacy_off_and_allowed_fallback_keep_exact_existing_source_hash_inputs(monkeypatch, checkpoints, fallback):
    records, writer, _ = checkpoints
    monkeypatch.setenv(FLAG, "1" if fallback else "0")
    for name in ("load_authoritative_engineering_story_memory", "build_configured_hiring_context_ranking_state"):
        monkeypatch.setattr(integration, name, Mock(side_effect=AssertionError("No new Story reads for cache")))
    ranking = {"selected_projects": [{"project_id": "legacy", "project_name": "Legacy", "rank": 1, "bullet_budget": 4}]}
    project = {"project_id": "legacy", "project_name": "Legacy"}
    before = deepcopy(ranking)
    result = build(ranking, project)
    assert build(ranking, project) == result
    assert GENERATION not in result and GENERATION not in ranking
    assert ranking == before
    entry = ranking["selected_projects"][0]
    expected_source_hash = api_server.stable_hash({
        "job_description": "Unchanged job description",
        "project": api_server.compact_project_for_prompt(project),
        "evidence": [{"project_id": "legacy", "chunk_id": "accepted-proof"}],
        "language": "en", "progress_guidance": "guidance",
        "ranking_context": {
            "project_ranking": api_server.compact_value_for_prompt(ranking, 450, 4),
            "this_project_rank": api_server.compact_value_for_prompt(entry, 450, 6),
            "bullet_budget": 4,
            "resume_length_constraints": api_server.default_resume_constraints({"length_mode": "one_page"}),
        },
        "project_angle_strategy": "project_angle_profile_v1",
    })
    assert next(iter(records.values()))["source_hash"] == expected_source_hash
    assert len(records) == writer.call_count == 1


@pytest.mark.parametrize("defect", ["missing", "mixed", "unknown", "duplicate"])
def test_mixed_or_unbound_candidate_assembly_fails_closed(defect):
    _, ranking = selection(matrix_plan(CONTEXT))
    candidates = [{"project_id": entry["project_id"], GENERATION: ranking[GENERATION]} for entry in ranking["selected_projects"]]
    if defect == "missing":
        candidates[0].pop(GENERATION)
    elif defect == "mixed":
        candidates[0][GENERATION] = "other-generation"
    elif defect == "unknown":
        candidates[0]["project_id"] = "unknown-project"
    else:
        candidates[1]["project_id"] = candidates[0]["project_id"]
    before = deepcopy(candidates)
    with pytest.raises(api_server.HTTPException) as error:
        api_server.validate_resume_candidate_generation(ranking, candidates)
    assert error.value.status_code == 409
    assert ranking[GENERATION] not in str(error.value.detail)
    assert candidates == before


def test_coherent_candidate_set_and_legacy_set_validate_read_only():
    _, ranking = selection(matrix_plan(CONTEXT))
    candidates = [{"project_id": entry["project_id"], GENERATION: ranking[GENERATION]} for entry in ranking["selected_projects"]]
    before = deepcopy((ranking, candidates))
    api_server.validate_resume_candidate_generation(ranking, candidates)
    assert (ranking, candidates) == before
    api_server.validate_resume_candidate_generation({"selected_projects": [{"project_id": "legacy"}]}, [{"project_id": "legacy"}])


def test_plan_consumes_upstream_state_identity_without_reconstructing_a_ranking_hash():
    plan = matrix_plan(CONTEXT)
    assert plan.ranking_generation_id == plan.state.state_id
    getter = inspect.getsource(type(plan).ranking_generation_id.fget)
    assert "state.state_id" in getter
    assert "hash" not in getter and "json" not in getter


def test_mixed_generation_fails_before_any_final_merge_operation(monkeypatch, checkpoints):
    records, writer, _ = checkpoints
    before, after = plans_for_change("hidden")
    projects, ranking = selection(before)
    _, changed_ranking = selection(after)
    candidates = [build(ranking, project) for project in projects]
    assert {candidate[GENERATION] for candidate in candidates} == {before.state.state_id}
    assert [build(ranking, project) for project in projects] == candidates
    assert writer.call_count == len(records) == len(projects)
    api_server.validate_resume_candidate_generation(ranking, candidates)
    candidates[0] = build(changed_ranking, projects[0])
    operation = Mock(side_effect=AssertionError("Mixed generations must not reach downstream rendering"))
    monkeypatch.setattr(api_server, "enforce_project_section_allocation", operation)
    with pytest.raises(api_server.HTTPException) as error:
        api_server.merge_staged_resume(
            "job", "resume", candidates, {}, {}, {}, api_server.TailorBody(), ranking, {}
        )
    assert error.value.status_code == 409
    operation.assert_not_called()


def test_generic_prompt_and_merge_compactors_strip_nested_generation_control():
    control = "project_story_ranking_state_" + "a" * 24
    candidate = {
        "project_id": "project", GENERATION: control,
        "project_ranking_context": {GENERATION: control, "rank": 1},
        "final_bullets": [{"bullet": "Supported work.", GENERATION: control}],
    }
    before = deepcopy(candidate)
    for value in (
        api_server.compact_value_for_prompt(candidate, 100000, 100),
        api_server.candidate_for_prompt(candidate),
        api_server.compact_merge_candidate(candidate, 100000, 100),
    ):
        serialized = json.dumps(value)
        assert GENERATION not in serialized and control not in serialized
        assert "Supported work." in serialized
    assert candidate == before


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("github", [False, True])
def test_actual_staged_generation_binds_candidates_before_assembly_and_hides_product_control(monkeypatch, checkpoints, mixed, github):
    monkeypatch.setenv(FLAG, "1")
    forbidden_agent = Mock(side_effect=AssertionError("Accepted ranking must not enter the general agent"))
    monkeypatch.setattr(api_server, "run_agent_task", forbidden_agent)
    records, writer, _ = checkpoints
    plan = matrix_plan(CONTEXT)
    projects, ranking = selection(plan)
    document = "\\documentclass{article}\n\\begin{document}\n\\section{Summary}\nSupported work.\n\\end{document}\n"
    monkeypatch.setattr(api_server.agent, "read_job_description", lambda: "job")
    monkeypatch.setattr(api_server.agent, "read_resume", lambda: document)
    monkeypatch.setattr(api_server.agent, "read_project_memory", lambda: json.dumps({"projects": projects}))
    monkeypatch.setattr(api_server, "resolve_saved_application_hint", lambda *_: {"company": "", "role": ""})
    monkeypatch.setattr(api_server, "agent_progress_guidance_text", lambda *_: "")
    monkeypatch.setattr(api_server, "classify_role_family", lambda *_: {"role_family": "software_engineering"})
    monkeypatch.setattr(api_server, "jd_requirements_for_prompt", lambda *_: CONTEXT)
    monkeypatch.setattr(api_server, "enrich_jd_profile_with_tech_ontology", lambda context, *_: context)
    monkeypatch.setattr(api_server, "select_staged_projects_with_ranking", lambda *_: (projects, ranking))
    retrieval = Mock(return_value=[])
    if not github:
        retrieval.side_effect = AssertionError("GitHub retrieval was disabled by the request")
        monkeypatch.setattr(api_server, "resolve_resume_evidence_retrieval_mode", retrieval)
    monkeypatch.setattr(api_server, "retrieve_evidence_for_project_for_resume", retrieval)
    built = []
    real_builder = api_server.build_project_resume_candidate

    def candidate_builder(*args, **kwargs):
        candidate = real_builder(*args, **kwargs)
        built.append(candidate)
        if mixed and len(built) == 2:
            candidate[GENERATION] = "stale-generation"
        return candidate

    monkeypatch.setattr(api_server, "build_project_resume_candidate", candidate_builder)
    skills = Mock(return_value={})
    monkeypatch.setattr(api_server, "build_skills_resume_candidate", skills)
    monkeypatch.setattr(api_server, "build_experience_resume_candidate", lambda *_: {})
    monkeypatch.setattr(api_server, "build_summary_resume_candidate", lambda *_: {"recommended_summary": "Supported work."})
    monkeypatch.setattr(api_server, "build_resume_gap_report", lambda *_: {})
    monkeypatch.setattr(api_server, "merge_staged_resume", lambda *_: document)
    monkeypatch.setattr(api_server, "normalize_latex_for_resume_output", lambda value: value)
    monkeypatch.setattr(api_server.agent, "looks_like_latex_resume", lambda *_: True)
    monkeypatch.setattr(api_server, "prepare_tailored_resume_for_save", lambda value, **_: (value, {"valid": True}))
    monkeypatch.setattr(api_server, "validate_tailored_resume_language_quality_or_raise", lambda *_: None)
    monkeypatch.setattr(api_server, "validate_project_section_allocation", lambda *_: {"issues": []})
    save_artifact = Mock()
    monkeypatch.setattr(api_server.agent, "save_tailored_resume", save_artifact)
    monkeypatch.setattr(api_server, "list_output_files", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(api_server.agent, "latest_tailored_resume_path", lambda: "test-output.txt")
    monkeypatch.setattr(api_server.agent, "read_tailored_resume", lambda: document)
    if mixed:
        with pytest.raises(api_server.HTTPException) as error:
            api_server.tailor_resume_task(api_server.TailorBody(use_github_context=github))
        assert error.value.status_code == 409
        skills.assert_not_called()
        save_artifact.assert_not_called()
        assert len(built) == 2
    else:
        result = api_server.tailor_resume_task(api_server.TailorBody(use_github_context=github))
        assert len(built) == len(projects) == writer.call_count == len(records)
        assert all(candidate[GENERATION] == plan.state.state_id for candidate in built)
        serialized = json.dumps(result)
        assert GENERATION not in serialized
        assert plan.state.state_id not in serialized
        assert len(result["staged_project_candidates"]) == len(projects)
        for actual, authoritative in zip(result["project_ranking"]["selected_projects"], ranking["selected_projects"]):
            assert {key: actual[key] for key in authoritative} == authoritative
            assert actual["candidate_bullet_count"] == 1
        save_artifact.assert_called_once()
    forbidden_agent.assert_not_called()
    assert retrieval.call_count == (len(built) if github else 0)
