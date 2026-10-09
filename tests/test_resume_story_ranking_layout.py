"""Final Story ranking layout preserves candidate identity and legacy behavior."""

from copy import deepcopy

import pytest

from tests.test_resume_retrieval_v2_evidence_integration import api_server


def entry(project_id, name, rank=1, *, story=True):
    return {
        "project_id": project_id, "project_name": name, "rank": rank,
        "bullet_budget": 1, **({"story_ranking": {"ordered_story_ids": []}} if story else {}),
    }


def candidate(project_id, name, technology, bullet):
    return {
        "project_id": project_id, "project_name": name,
        "skills_to_emphasize": [technology], "final_bullets": [{"bullet": bullet}],
        "project_presentation": {"project_id": project_id, "display_name": name,
                                 "technologies": [technology], "url": ""},
    }


def block(name, technology="OldTech", link="https://old.example/project", bullet="Old evidence."):
    return (
        "\\resumeProjectHeading\n"
        f"  {{\\textbf{{{name}}} $|$ \\emph{{{technology}}}}}{{\\href{{{link}}}{{Code}}}}\n"
        "\\resumeItemListStart\n"
        f"  \\resumeItem{{{bullet}}}\n"
        "\\resumeItemListEnd"
    )


def resume(*blocks):
    return (
        "\\documentclass{article}\n\\begin{document}\n"
        "\\section{Projects}\n\\resumeSubHeadingListStart\n"
        + "\n\n".join(blocks)
        + "\n\\resumeSubHeadingListEnd\n\\end{document}\n"
    )


def rendered_blocks(value):
    section = api_server.find_latex_section(value, "projects")
    return api_server.project_blocks_for_validation(section["text"])


@pytest.mark.parametrize("names", [("API", "API Gateway"), ("API Gateway", "API")])
def test_overlapping_names_rebuild_in_rank_order_from_exact_candidates(names):
    entries = [entry("first", names[0]), entry("second", names[1], 2)]
    ranking = {"selected_projects": list(reversed(entries)), "omitted_projects": []}
    candidates = [
        candidate("impostor", names[0], "WrongTech", "Wrong candidate evidence."),
        candidate("second", "Model invented second name", "Rust", "Implemented gateway routing."),
        candidate("first", "Model invented first name", "Python", "Implemented request validation."),
    ]
    before = deepcopy((ranking, candidates))
    original = resume(block("API Gateway"), block("API"))
    result = api_server.enforce_project_section_allocation(original, ranking, candidates)
    blocks = rendered_blocks(result)
    assert [api_server.project_block_heading(value) for value in blocks] == list(names)
    assert "Python" in blocks[0] and "request validation" in blocks[0]
    assert "Rust" in blocks[1] and "gateway routing" in blocks[1]
    for stale in ("OldTech", "old.example", "WrongTech", "Wrong candidate", "Model invented"):
        assert stale not in result
    assert (ranking, candidates) == before
    ordered_ranking = {**ranking, "selected_projects": entries}
    report = api_server.validate_project_section_allocation(result, ordered_ranking)
    assert report["allocation_matches_ranking"]
    assert report["bullet_counts"] == {names[0]: 1, names[1]: 1}
    assert api_server.enforce_project_section_allocation(result, ranking, candidates) == result


def test_missing_exact_heading_does_not_reuse_substring_block():
    ranking = {"selected_projects": [entry("api", "API")], "omitted_projects": []}
    candidates = [candidate("api", "API", "Python", "Implemented request validation.")]
    result = api_server.enforce_project_section_allocation(resume(block("API Gateway")), ranking, candidates)
    assert [api_server.project_block_heading(value) for value in rendered_blocks(result)] == ["API"]
    assert "old.example" not in result and "OldTech" not in result


def test_duplicate_existing_headings_are_rebuilt_from_candidate():
    ranking = {"selected_projects": [entry("api", "API")], "omitted_projects": []}
    candidates = [candidate("api", "API", "Python", "Implemented request validation.")]
    result = api_server.enforce_project_section_allocation(resume(block("API"), block("API")), ranking, candidates)
    assert len(rendered_blocks(result)) == 1
    assert "old.example" not in result and "OldTech" not in result


def test_unambiguous_heading_cannot_supply_an_unowned_link():
    ranking = {"selected_projects": [entry("api", "API")], "omitted_projects": []}
    candidates = [candidate("api", "API", "Python", "Implemented request validation.")]
    result = api_server.enforce_project_section_allocation(resume(block("API", "Python")), ranking, candidates)
    assert "old.example/project" not in result
    assert "Implemented request validation." in result
    assert "Old evidence." not in result


def test_story_layout_cannot_substitute_same_name_candidate_with_wrong_id():
    ranking = {"selected_projects": [entry("api", "API")], "omitted_projects": []}
    wrong = candidate("other", "API", "WrongTech", "Wrong evidence.")
    with pytest.raises(api_server.HTTPException) as error:
        api_server.enforce_project_section_allocation(resume(block("API")), ranking, [wrong])
    assert error.value.status_code == 409


@pytest.mark.parametrize("selected_name, omitted_name", [
    ("API", "API Gateway"), ("API Gateway", "API"), ("API", "API"),
])
def test_selected_heading_is_not_falsely_reported_as_omitted(selected_name, omitted_name):
    ranking = {
        "selected_projects": [entry("selected", selected_name)],
        "omitted_projects": [{"project_id": "omitted", "project_name": omitted_name}],
    }
    report = api_server.validate_project_section_allocation(resume(block(selected_name)), ranking)
    assert report["omitted_projects_readded"] == []
    assert not any("Omitted projects were re-added" in issue for issue in report["issues"])


def test_validator_still_detects_actually_readded_omitted_project():
    ranking = {
        "selected_projects": [entry("api", "API")],
        "omitted_projects": [{"project_id": "gateway", "project_name": "API Gateway"}],
    }
    report = api_server.validate_project_section_allocation(resume(block("API"), block("API Gateway")), ranking)
    assert report["omitted_projects_readded"] == ["API Gateway"]


@pytest.mark.parametrize("headings", [("API Gateway", "API"), ("Unrelated", "Different")])
def test_validator_rejects_wrong_story_project_order_or_headings(headings):
    ranking = {"selected_projects": [entry("api", "API"), entry("gateway", "API Gateway", 2)]}
    report = api_server.validate_project_section_allocation(resume(*(block(name) for name in headings)), ranking)
    assert "Project order does not follow the ranking." in report["issues"]


@pytest.mark.parametrize("flag", [None, "0", "invalid"])
def test_legacy_layout_and_fuzzy_validation_remain_unchanged(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("USE_STORY_AWARE_RESUME_RANKING", raising=False)
    else:
        monkeypatch.setenv("USE_STORY_AWARE_RESUME_RANKING", flag)
    ranking = {
        "selected_projects": [entry("api", "API", story=False)],
        "omitted_projects": [{"project_id": "gateway", "project_name": "API Gateway"}],
    }
    original = resume(block("API Gateway"))
    candidates = [candidate("different-id", "API", "Python", "Implemented request validation.")]
    result = api_server.enforce_project_section_allocation(original, ranking, candidates)
    assert result == resume(block("API Gateway", bullet="Implemented request validation."))
    report = api_server.validate_project_section_allocation(result, ranking)
    assert report["omitted_projects_readded"] == ["API Gateway"]
    assert report["allocation_matches_ranking"]
