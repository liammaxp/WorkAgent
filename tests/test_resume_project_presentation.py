"""Exact project-owned headers remain separate from generated resume prose."""

from copy import deepcopy

import pytest

from tests.test_resume_story_ranking_layout import (
    api_server, block, candidate, entry, rendered_blocks, resume,
)


def owned(project_id, name, technology, bullet):
    result = candidate(project_id, name, "GeneratedTechnology", bullet)
    result["project_presentation"] = {
        "project_id": project_id, "display_name": name,
        "technologies": [technology], "url": f"https://example.test/{project_id}",
    }
    return result


@pytest.mark.parametrize("corruption", ["technology", "link", "both", "missing", "extra"])
def test_generated_headers_cannot_replace_project_owned_metadata(corruption):
    ranking = {"selected_projects": [entry("alpha", "Alpha"), entry("beta", "Beta", 2)]}
    candidates = [owned("alpha", "Alpha", "Python", "Implemented Alpha validation."),
                  owned("beta", "Beta", "Rust", "Implemented Beta recovery.")]
    first_tech = "Rust" if corruption in {"technology", "both"} else "Python"
    second_tech = "Python" if corruption in {"technology", "both"} else "Rust"
    first_link = "https://example.test/beta" if corruption in {"link", "both"} else "https://example.test/alpha"
    second_link = "https://example.test/alpha" if corruption in {"link", "both"} else "https://example.test/beta"
    if corruption == "missing":
        first_tech = second_tech = first_link = second_link = ""
    if corruption == "extra":
        first_tech = second_tech = "JDTechnology EmployerTechnology SummaryTechnology"
    original = resume(block("Alpha", first_tech, first_link, "Wrong Beta body."),
                      block("Beta", second_tech, second_link, "Wrong Alpha body."))
    before = deepcopy((ranking, candidates))
    result = api_server.enforce_project_section_allocation(original, ranking, candidates)
    blocks = rendered_blocks(result)
    assert "Python" in blocks[0] and "Rust" not in blocks[0]
    assert "Rust" in blocks[1] and "Python" not in blocks[1]
    assert "https://example.test/alpha" in blocks[0] and "https://example.test/beta" not in blocks[0]
    assert "https://example.test/beta" in blocks[1] and "https://example.test/alpha" not in blocks[1]
    assert "Alpha validation" in blocks[0] and "Beta recovery" in blocks[1]
    assert "Wrong" not in result and "GeneratedTechnology" not in result
    assert "JDTechnology" not in result and "EmployerTechnology" not in result
    assert "SummaryTechnology" not in result
    assert (ranking, candidates) == before
    assert api_server.enforce_project_section_allocation(result, ranking, candidates) == result


@pytest.mark.parametrize("names", [
    ("WorkAgent", "Work Agent", "WorkAgent API"),
    ("Student Management", "Student Management System", "Student Management"),
    ("Same", "Same", "Same"),
])
@pytest.mark.parametrize("model_missing", [False, True])
def test_exact_ids_bind_similar_or_duplicate_names_and_restore_portfolio_order(names, model_missing):
    entries = [entry("a", names[0], 1), entry("c", names[1], 2), entry("b", names[2], 3)]
    ranking = {"selected_projects": list(reversed(entries))}
    candidates = [owned(value["project_id"], value["project_name"], tech, f"Implemented {value['project_id']} mechanism.")
                  for value, tech in zip(entries, ["Python", "Rust", "Java"])]
    supplied = [block(names[2], "WrongTech", "https://wrong.test/b"),
                block(names[0], "WrongTech", "https://wrong.test/a")]
    if not model_missing:
        supplied.append(block(names[1], "WrongTech", "https://wrong.test/c"))
    result = api_server.enforce_project_section_allocation(resume(*supplied), ranking, list(reversed(candidates)))
    blocks = rendered_blocks(result)
    assert len(blocks) == 3
    assert [api_server.project_block_heading(value) for value in blocks] == list(names)
    for value, project_id, tech in zip(blocks, ["a", "c", "b"], ["Python", "Rust", "Java"]):
        assert f"https://example.test/{project_id}" in value
        assert tech in value and f"{project_id} mechanism" in value
        for other in {"a", "c", "b"} - {project_id}:
            assert f"https://example.test/{other}" not in value
            assert f"{other} mechanism" not in value
    assert "WrongTech" not in result and "wrong.test" not in result


def test_absent_authoritative_metadata_uses_safe_header_without_generated_technology_or_link():
    ranking = {"selected_projects": [entry("alpha", "Alpha")]}
    generated = candidate("alpha", "Generated title", "JDTechnology", "Implemented validation.")
    generated.pop("project_presentation", None)
    generated["evidence_card"] = {"technologies": ["EmployerTechnology"]}
    generated["url"] = "https://model.test/other"
    result = api_server.enforce_project_section_allocation(
        resume(block("Alpha", "OtherProjectTechnology", "https://other.test/project")), ranking, [generated])
    assert "Alpha" in result and "Implemented validation" in result
    for value in ("Generated title", "JDTechnology", "EmployerTechnology", "OtherProjectTechnology", "https://"):
        assert value not in result


def test_mismatched_presentation_id_is_rejected_instead_of_borrowed():
    value = owned("alpha", "Alpha", "Python", "Implemented validation.")
    value["project_presentation"]["project_id"] = "beta"
    with pytest.raises(api_server.HTTPException) as error:
        api_server.enforce_project_section_allocation(
            resume(block("Alpha")), {"selected_projects": [entry("alpha", "Alpha")]}, [value])
    assert error.value.status_code == 409


@pytest.mark.parametrize("story_fallback", [False, True])
def test_legacy_and_accepted_legacy_fallback_keep_existing_header_semantics(story_fallback):
    # Fallback projections intentionally contain no accepted Story ranking entry.
    ranking = {"selected_projects": [entry("alpha", "Alpha", story=False)]}
    if story_fallback:
        ranking["fallback_reason"] = "empty_story_memory"
    generated = owned("alpha", "Alpha", "Python", "Implemented validation.")
    original = resume(block("Alpha", "LegacyTech", "https://legacy.test/project"))
    result = api_server.enforce_project_section_allocation(original, ranking, [generated])
    assert result == resume(block("Alpha", "LegacyTech", "https://legacy.test/project", "Implemented validation."))


def test_server_binding_uses_accepted_project_metadata_not_generated_candidate_metadata():
    project = {"project_id": "alpha", "project_name": "Authoritative Alpha",
               "tech_stack": ["Python"], "tools": ["SQLite"], "technologies": ["Redis"],
               "github_url": "https://github.com/owner/alpha"}
    generated = candidate("beta", "Wrong name", "JDTechnology", "Implemented validation.")
    generated["evidence_card"] = {"technologies": ["EmployerTechnology"]}
    generated["project_presentation"] = {"project_id": "beta", "technologies": ["OtherProjectTechnology"]}
    ranked = entry("alpha", "Authoritative Alpha", 2)
    before = deepcopy((project, generated, ranked))
    bound = api_server.bind_story_ranked_project_candidate(project, generated, ranked)
    assert bound["project_id"] == "alpha"
    assert bound["project_name"] == "Authoritative Alpha"
    assert bound["project_rank"] == 2
    assert bound["story_ranking"] == ranked["story_ranking"]
    assert set(bound["project_presentation"]["technologies"]) == {"Python", "SQLite", "Redis"}
    assert {key: value for key, value in bound["project_presentation"].items() if key != "technologies"} == {
        "project_id": "alpha", "display_name": "Authoritative Alpha",
        "url": "https://github.com/owner/alpha",
    }
    assert bound["final_bullets"] == generated["final_bullets"]
    assert (project, generated, ranked) == before


def test_server_binding_does_not_guess_repository_from_name_or_jd():
    project = {"project_id": "alpha", "project_name": "WorkAgent API"}
    generated = candidate("alpha", "WorkAgent API", "JDTechnology", "Implemented validation.")
    generated["jd_requirements"] = {"must_have_skills": ["EmployerTechnology"]}
    generated["url"] = "https://github.com/model/guess"
    bound = api_server.bind_story_ranked_project_candidate(project, generated, entry("alpha", "WorkAgent API"))
    assert bound["project_presentation"]["technologies"] == []
    assert bound["project_presentation"]["url"] == ""
