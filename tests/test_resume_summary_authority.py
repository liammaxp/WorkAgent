"""Summary realization retains the accepted project portfolio's priority."""

from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from tests.test_resume_story_ranking_integration import CONTEXT, FLAG, matrix_plan
from tests.test_resume_story_ranking_layout import api_server


@pytest.fixture
def candidates():
    plan = matrix_plan(CONTEXT)
    return [
        {
            "project_id": project.project_id,
            "project_name": name,
            "project_rank": position,
            "story_ranking": project.story_handoff(),
            "fit": fit,
            "final_bullets": ["Implemented supported engineering work."],
        }
        for position, (project, name, fit) in enumerate(
            zip(plan.projects, ("Alpha", "Beta"), ("low", "high")), 1
        )
    ]


def compact(candidates):
    return api_server.build_compact_summary_input(
        "Backend Engineer", "Original resume", candidates, {}, {}, "en"
    )


def generate(monkeypatch, candidates, text, evidence=(), language="en"):
    model = Mock(return_value=json.dumps({
        "recommended_summary": text,
        "evidence_basis": list(evidence),
        "keywords_to_include": [],
        "claims_to_avoid": [],
        "risks": [],
    }))
    monkeypatch.setattr(api_server, "safe_model_call", model)
    result = api_server.build_summary_resume_candidate(
        "Backend Engineer", "Original resume", {}, candidates, {}, {}, language
    )
    model.assert_called_once()
    return result, model.call_args.kwargs


def test_accepted_portfolio_overrides_card_order_fit_and_alphabetical_order(candidates):
    candidates[0]["project_name"] = "Zulu"
    candidates[1]["project_name"] = "Aardvark"
    before = deepcopy(candidates)
    payload = compact(list(reversed(candidates)))
    highlights = payload["topProjectHighlights"]
    assert [item["projectId"] for item in highlights] == [c["project_id"] for c in candidates]
    assert [item["projectRank"] for item in highlights] == [1, 2]
    assert [item["projectName"] for item in highlights] == ["Zulu", "Aardvark"]
    assert all("fitScore" not in item for item in highlights)
    assert candidates == before


def test_story_summary_never_calls_legacy_summary_score(monkeypatch, candidates):
    def forbidden(*_):
        pytest.fail("Story-aware summary must not derive another project score")
    monkeypatch.setattr(api_server, "summary_fit_score", forbidden)
    payload = compact(candidates)
    assert len(payload["topProjectHighlights"]) == 2


def test_reduction_retains_accepted_order_without_a_second_score(candidates):
    payload = compact(candidates)
    payload["topProjectHighlights"].reverse()
    before = deepcopy(payload)
    reduced = api_server.reduce_compact_summary_payload_for_limit(payload, max_chars=1)
    assert [item["projectId"] for item in reduced["topProjectHighlights"]] == [c["project_id"] for c in candidates]
    assert all("fitScore" not in item for item in reduced["topProjectHighlights"])
    assert payload == before


def test_initial_and_retry_prompts_use_the_same_accepted_priority(monkeypatch, candidates):
    result, kwargs = generate(monkeypatch, list(reversed(candidates)), "Supported backend work.")
    initial = kwargs["prompt"]
    retry = kwargs["compact_builder"](max_chars=100000)
    assert "fitScore" not in initial
    assert initial.index('"projectName": "Alpha"') < initial.index('"projectName": "Beta"')
    assert [p["projectId"] for p in retry["payload"]["topProjectHighlights"]] == [c["project_id"] for c in candidates]
    for prompt in (initial, retry["prompt"]):
        assert "rerank" in prompt.lower()
        assert "phase " + str(7) not in prompt.lower()
    assert result["recommended_summary"] == "Supported backend work."


@pytest.mark.parametrize("text", [
    "Strongest project: Beta. Next project: Alpha.",
    "Beta provides relevant engineering experience, complemented by Alpha.",
    "Beta and Alpha demonstrate supported work.",
])
def test_reversed_exact_project_references_cannot_establish_priority(monkeypatch, candidates, text):
    result, _ = generate(monkeypatch, candidates, text)
    assert result["recommended_summary"] == "Project experience includes Alpha; Beta."
    assert "Strongest" not in result["recommended_summary"]


@pytest.mark.parametrize("text", [
    "Implemented reliable backend workflows.",
    "Alpha and Beta demonstrate supported work.",
    "Beta demonstrates supported work.",
    "Alpha demonstrates supported work.",
    "Alphabetical tools and Betamax devices are unrelated references.",
])
def test_name_free_monotonic_or_single_project_prose_is_preserved(monkeypatch, candidates, text):
    result, _ = generate(monkeypatch, candidates, text)
    assert result["recommended_summary"] == text


def test_longest_literal_name_prevents_substring_reference_crossover(monkeypatch, candidates):
    candidates[0]["project_name"] = "WorkAgent API"
    candidates[1]["project_name"] = "WorkAgent"
    text = "WorkAgent API demonstrates supported work."
    result, _ = generate(monkeypatch, candidates, text)
    assert result["recommended_summary"] == text
    reversed_result, _ = generate(monkeypatch, candidates, "WorkAgent complements WorkAgent API.")
    assert reversed_result["recommended_summary"] == "Project experience includes WorkAgent API; WorkAgent."


def test_duplicate_display_names_do_not_guess_project_ownership(monkeypatch, candidates):
    for candidate in candidates:
        candidate["project_name"] = "Shared Project"
    text = "Shared Project demonstrates supported engineering work."
    result, _ = generate(monkeypatch, candidates, text)
    assert result["recommended_summary"] == text


def test_duplicate_longer_names_do_not_resolve_to_a_unique_short_name(monkeypatch, candidates):
    candidates[1]["project_name"] = "WorkAgent"
    for rank in (3, 4):
        duplicate = deepcopy(candidates[1])
        duplicate.update(project_id=f"duplicate-project-{rank}", project_rank=rank, project_name="WorkAgent API")
        candidates.append(duplicate)
    text = "WorkAgent API and Alpha provide supported work."
    result, _ = generate(monkeypatch, candidates, text)
    assert result["recommended_summary"] == text


def test_exact_project_ids_take_priority_over_other_projects_display_labels(monkeypatch, candidates):
    first_id, second_id = [candidate["project_id"] for candidate in candidates]
    candidates[0]["project_name"] = second_id
    candidates[1]["project_name"] = first_id
    result, _ = generate(monkeypatch, candidates, "Supported backend work.", [second_id, first_id])
    assert result["evidence_basis"] == [first_id, second_id]


@pytest.mark.parametrize("reference_key", ["project_name", "project_id"])
def test_exact_structured_evidence_references_preserve_priority(monkeypatch, candidates, reference_key):
    first, second = [candidate[reference_key] for candidate in candidates]
    result, _ = generate(
        monkeypatch, candidates, "Supported backend work.", [second, "Shared supporting source", first]
    )
    assert result["evidence_basis"] == [first, "Shared supporting source", second]


def test_evidence_descriptions_are_not_fuzzily_rebound(monkeypatch, candidates):
    evidence = ["Beta supporting source", "Alpha supporting source", "Other repository"]
    result, _ = generate(monkeypatch, candidates, "Supported backend work.", evidence)
    assert result["evidence_basis"] == evidence


def test_chinese_reversal_uses_neutral_localized_realization(monkeypatch, candidates):
    result, _ = generate(monkeypatch, candidates, "Beta 是最强项目，Alpha 是次要项目。", language="zh")
    assert result["recommended_summary"].index("Alpha") < result["recommended_summary"].index("Beta")
    assert "最强" not in result["recommended_summary"]


@pytest.mark.parametrize("legacy_kind", ["off", "accepted_fallback"])
def test_legacy_summary_retains_fit_order_full_prompt_and_model_output(monkeypatch, candidates, legacy_kind):
    for candidate in candidates:
        candidate.pop("story_ranking")
    monkeypatch.setenv(FLAG, "0" if legacy_kind == "off" else "1")
    payload = compact(candidates)
    assert [item["projectName"] for item in payload["topProjectHighlights"]] == ["Beta", "Alpha"]
    assert [item["fitScore"] for item in payload["topProjectHighlights"]] == [95, 35]
    text = "Strongest project: Beta. Next project: Alpha."
    result, kwargs = generate(monkeypatch, candidates, text, ["Beta", "Alpha"])
    assert result["recommended_summary"] == text
    assert result["evidence_basis"] == ["Beta", "Alpha"]
    assert "Original resume:" in kwargs["prompt"]
    assert "Project Memory:" in kwargs["prompt"]


def test_summary_realization_preserves_handoff_story_order_and_candidate_truth(monkeypatch, candidates):
    before = deepcopy(candidates)
    compact(candidates)
    generate(monkeypatch, candidates, "Beta then Alpha.")
    assert candidates == before
    for original, current in zip(before, candidates):
        assert current["story_ranking"]["ordered_story_ids"] == original["story_ranking"]["ordered_story_ids"]
        assert current["story_ranking"]["current_revision_ids"] == original["story_ranking"]["current_revision_ids"]


@pytest.mark.parametrize("existing", [False, True])
def test_summary_section_renderer_preserves_exact_safe_prose_and_other_sections(existing):
    summary_section = "\\section{Profile}\nStale model summary.\n" if existing else ""
    original = (
        "\\documentclass{article}\n\\begin{document}\n"
        + summary_section
        + "\\section{Experience}\nSupported experience.\n\\end{document}\n"
    )
    result = api_server.apply_summary_section_candidate(
        original, {"recommended_summary": "Project experience includes Alpha; Beta."}
    )
    assert "Project experience includes Alpha; Beta." in result
    assert "Stale model summary." not in result
    assert "\\section{Experience}\nSupported experience." in result
    assert result.count("\\end{document}") == 1
    assert api_server.find_latex_section(result, "summary") is not None


def test_summary_section_renderer_escapes_model_text_as_plain_latex_content():
    original = "\\documentclass{article}\n\\begin{document}\n\\section{Summary}\nOld summary.\n\\end{document}\n"
    result = api_server.apply_summary_section_candidate(
        original, {"recommended_summary": "Built C# tools & validated 50% of test_cases with $5 input {values}."}
    )
    for escaped in (r"C\#", r"\&", r"50\%", r"test\_cases", r"\$5", r"\{values\}"):
        assert escaped in result


@pytest.mark.parametrize("candidate", [{}, {"recommended_summary": ""}, {"recommended_summary": "   "}])
def test_empty_summary_candidate_preserves_existing_presentation(candidate):
    original = "\\documentclass{article}\n\\begin{document}\n\\section{Summary}\nExisting summary.\n\\end{document}\n"
    assert api_server.apply_summary_section_candidate(original, candidate) == original


@pytest.mark.parametrize("story_aware", [False, True])
def test_final_merge_uses_deterministic_summary_only_for_accepted_story_authority(monkeypatch, candidates, story_aware):
    selected = [{
        "project_id": candidate["project_id"], "rank": candidate["project_rank"],
        **({"story_ranking": candidate["story_ranking"]} if story_aware else {}),
    } for candidate in candidates]
    original = "\\documentclass{article}\n\\begin{document}\n\\section{Summary}\nOld summary.\n\\end{document}\n"
    monkeypatch.setattr(api_server, "enforce_project_section_allocation", lambda current, *_: current)
    monkeypatch.setattr(api_server, "validate_complete_resume_or_raise", lambda current, *_: current)
    monkeypatch.setattr(api_server, "apply_skills_section_candidate", lambda current, *_: current)
    monkeypatch.setattr(api_server, "repair_final_resume_policy_once", lambda current, *_, **__: current)
    calls = []

    def section_editor(section_name, _jd, current, _candidate, _body):
        calls.append(section_name)
        if section_name == "Summary/Profile-section":
            return current.replace("Old summary.", "Legacy editor output.")
        return current

    monkeypatch.setattr(api_server, "apply_resume_section_candidate", section_editor)
    ranking = {"selected_projects": selected}
    if story_aware:
        generation = matrix_plan(CONTEXT).ranking_generation_id
        ranking["ranking_generation_id"] = generation
        for candidate in candidates:
            candidate["ranking_generation_id"] = generation
    result = api_server.merge_staged_resume(
        "Backend Engineer", original, candidates, {}, {},
        {"recommended_summary": "Project experience includes Alpha; Beta."},
        api_server.TailorBody(), ranking, {},
    )
    assert calls.count("Summary/Profile-section") == (0 if story_aware else 1)
    assert "Experience-section" in calls
    if story_aware:
        assert "Project experience includes Alpha; Beta." in result
        assert "Legacy editor output." not in result
    else:
        assert "Legacy editor output." in result
