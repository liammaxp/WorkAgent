from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import api_server
from backend import project_retrieval_v2


PROJECTS = (
    ("a_hike_through_time", "liammaxp/a-hike-through-time"),
    ("course_management_database", "liammaxp/course-management-database-system"),
    ("event_lottery_system", "liammaxp/event-lottery-system-application"),
    ("furniture_search_inventory", "liammaxp/furniture-search-and-inventory-system"),
    ("workagent", "liammaxp/workagent"),
)
JD_FIXTURES = {
    "backend_engineering": {"requirements": ["Python", "API", "retrieval"]},
    "full_stack_engineering": {"requirements": ["React", "API", "validation"]},
    "data_database_engineering": {"requirements": ["database", "testing"]},
    "automation_testing": {"requirements": ["automation", "testing", "Kubernetes"]},
}
V2_MODE = api_server.RESUME_EVIDENCE_RETRIEVAL_MODE_V2


def project(project_id, repo):
    return {
        "project_id": project_id,
        "project_name": project_id.replace("_", " ").title(),
        "repo": repo,
        "technologies": ["Python"],
        "mechanisms": ["validation", "testing"],
    }


def safe_evidence(item, count=3, summary="Safe project-supported implementation evidence"):
    hits = []
    for index in range(count):
        hits.append(
            {
                "project_id": item["project_id"],
                "chunk_id": f"chk_{item['project_id']}_{index}",
                "source_id": f"raw_{item['project_id']}_{index}",
                "repo": item["repo"],
                "path": f"src/{index}.py",
                "commit_sha": "abc123",
                "source_type": "file_snapshot",
                "chunk_type": "file_section",
                "symbol": f"symbol_{index}",
                "score": 0.9 - index * 0.05,
                "keyword_score": 0.7,
                "symbol_score": 0.2,
                "vector_score": 0.8,
                "search_sources": ["keyword", "symbol", "vector"],
                "query_groups": ["project_identity", "mechanisms"],
                "match_reasons": ["keyword_exact:validation"],
                "summary": summary,
                "keywords": ["Python", "validation"],
                "technical_tags": ["testing"],
                "text_hash": "b" * 64,
                "text_chars": 128,
            }
        )
    return project_retrieval_v2.adapt_hybrid_hits_for_resume_evidence(
        hits,
        project_id=item["project_id"],
        project_name=item["project_name"],
        authorized_repositories=[item["repo"]],
        limit=count,
    )


def test_five_project_multi_jd_outputs_are_isolated_bounded_and_three_run_deterministic(monkeypatch):
    monkeypatch.setattr(
        api_server,
        "retrieve_evidence_for_project",
        lambda *_args, **_kwargs: pytest.fail("v2 deterministic runs must not fall back to legacy"),
    )

    def v2_provider(item, **_kwargs):
        return safe_evidence(item)

    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", v2_provider)
    for fixture_name, jd_targets in JD_FIXTURES.items():
        operation_outputs = []
        for _run in range(3):
            per_project = {}
            for project_id, repo in PROJECTS:
                item = project(project_id, repo)
                evidence = api_server.retrieve_evidence_for_project_for_resume(
                    item,
                    jd_targets=jd_targets,
                    retrieval_mode=V2_MODE,
                )
                compacted = api_server.compact_github_evidence_for_prompt(evidence)
                assert {entry["project_id"] for entry in compacted} == {project_id}
                assert {entry["repo"] for entry in compacted} == {repo}
                assert len(compacted) <= api_server.RESUME_PROJECT_EVIDENCE_LIMIT
                assert len(api_server.serialize_resume_evidence_for_budget(compacted)) <= api_server.MAX_PROMPT_EVIDENCE_CHARS
                assert "Kubernetes" not in json.dumps(compacted, ensure_ascii=False)
                per_project[project_id] = compacted
            operation_outputs.append(json.dumps(per_project, ensure_ascii=False, sort_keys=True))
        assert operation_outputs[0] == operation_outputs[1] == operation_outputs[2], fixture_name


def test_writer_prompt_compaction_preserves_hybrid_order_and_does_not_mutate():
    item = project("workagent", "liammaxp/workagent")
    evidence = safe_evidence(item, count=api_server.RESUME_PROJECT_EVIDENCE_LIMIT)
    before = deepcopy(evidence)
    compacted = api_server.compact_retrieval_v2_evidence_for_prompt(evidence)
    assert evidence == before
    assert [entry["chunk_id"] for entry in compacted] == [entry["chunk_id"] for entry in evidence]
    assert [entry["final_score"] for entry in compacted] == [entry["final_score"] for entry in evidence]
    assert [entry["search_sources"] for entry in compacted] == [entry["search_sources"] for entry in evidence]
    assert [entry["query_groups"] for entry in compacted] == [entry["query_groups"] for entry in evidence]


def test_character_budget_exact_boundary_and_one_character_over():
    item = project("workagent", "liammaxp/workagent")
    compacted = api_server.compact_retrieval_v2_evidence_for_prompt(safe_evidence(item, count=1))
    exact = len(api_server.serialize_resume_evidence_for_budget(compacted))
    assert api_server.truncate_resume_evidence_for_prompt_budget(compacted, exact + 1) == compacted
    assert api_server.truncate_resume_evidence_for_prompt_budget(compacted, exact) == compacted
    assert api_server.truncate_resume_evidence_for_prompt_budget(compacted, exact - 1) == []


def test_one_item_over_budget_truncates_prefix_without_secondary_ranking():
    item = project("workagent", "liammaxp/workagent")
    compacted = api_server.compact_retrieval_v2_evidence_for_prompt(safe_evidence(item, count=3))
    first_only = compacted[:1]
    first_budget = len(api_server.serialize_resume_evidence_for_budget(first_only))
    assert api_server.truncate_resume_evidence_for_prompt_budget(compacted, first_budget) == first_only


def test_long_summary_many_provenance_entries_and_duplicates_remain_bounded():
    item = project("workagent", "liammaxp/workagent")
    evidence = safe_evidence(item, count=8, summary="S" * 5000)
    evidence[0]["keywords"] = [f"keyword-{index}-" + "K" * 200 for index in range(100)]
    evidence[0]["technical_tags"] = [f"tag-{index}-" + "T" * 200 for index in range(100)]
    evidence[0]["match_reasons"] = [f"reason-{index}-" + "R" * 300 for index in range(100)]
    evidence.append(deepcopy(evidence[0]))
    compacted = api_server.compact_retrieval_v2_evidence_for_prompt(evidence)
    assert len(compacted) <= api_server.RESUME_PROJECT_EVIDENCE_LIMIT
    assert len({entry["chunk_id"] for entry in compacted}) == len(compacted)
    assert all(len(entry["summary"]) <= project_retrieval_v2.MAX_V2_SAFE_SUMMARY_CHARS for entry in compacted)
    assert len(api_server.serialize_resume_evidence_for_budget(compacted)) <= api_server.MAX_PROMPT_EVIDENCE_CHARS


@pytest.mark.parametrize("value", (None, "", "0", "false", "invalid", "automatic_fallback"))
def test_mode_contract_has_only_explicit_v2_and_legacy(value):
    assert api_server.normalize_resume_evidence_retrieval_mode(value) == api_server.RESUME_EVIDENCE_RETRIEVAL_MODE_LEGACY
    assert api_server.normalize_resume_evidence_retrieval_mode(V2_MODE) == V2_MODE
