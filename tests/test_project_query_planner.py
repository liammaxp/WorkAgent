from __future__ import annotations

import json

from backend import project_query_planner as planner


def all_queries(plan):
    return [query for group in planner.QUERY_GROUPS for query in plan[group]]


def project_memory():
    return {
        "projects": [
            {
                "project_id": "WorkAgent",
                "project_name": "WorkAgent",
                "repo": "workagent/repository",
                "positioning": "Resume tailoring evidence pipeline",
                "summary": "Deterministic local project memory for retrieval.",
                "tech_stack": ["Python", "SQLite"],
                "workflows": ["evidence cards", "retrieval reranking", "deterministic merge"],
                "validation": ["LaTeX validation", "quality gate", "fallback retry"],
                "real_metrics": ["token use", "manual review time"],
                "symbols": ["retrieve_evidence_for_project", "ResumeQualityGate"],
            },
            {
                "project_id": "OtherProject",
                "project_name": "OtherProject",
                "tech_stack": ["Kubernetes", "Rust"],
                "workflows": ["distributed consensus"],
                "symbols": ["otherProjectOnly"],
            },
        ]
    }


def test_identity_queries_are_focused_bounded_deduplicated_and_deterministic():
    first = planner.build_project_query_plan(project_id="WorkAgent", project_memory=project_memory())
    second = planner.build_project_query_plan(project_id="WorkAgent", project_memory=project_memory())
    assert first == second
    assert first["project_identity"]
    assert any("WorkAgent" in query and "evidence" in query.casefold() for query in first["project_identity"])
    assert len({query.casefold() for query in first["project_identity"]}) == len(first["project_identity"])


def test_jd_alignment_is_compact_and_excludes_boilerplate_and_complete_jd():
    jd = {
        "technologies": ["FastAPI", "PostgreSQL"],
        "requirements": ["backend retrieval reliability validation"],
        "benefits": "health insurance and salary",
        "body": "Complete employer equal opportunity location boilerplate " * 100,
    }
    plan = planner.build_project_query_plan(
        project_id="WorkAgent", project_memory=project_memory(), jd_targets=jd
    )
    output = " ".join(plan["jd_alignment"]).casefold()
    assert "fastapi" in output and "postgresql" in output
    assert "benefit" not in output and "employer" not in output and "location" not in output
    assert "fastapi" not in " ".join(plan["mechanisms"]).casefold()


def test_mechanism_and_validation_queries_only_use_supported_project_terms():
    plan = planner.build_project_query_plan(project_id="WorkAgent", project_memory=project_memory())
    mechanisms = " ".join(plan["mechanisms"]).casefold()
    validation = " ".join(plan["validation_repair"]).casefold()
    assert "retrieval" in mechanisms and "sqlite" in mechanisms and "deterministic" in mechanisms
    assert "kubernetes" not in mechanisms and "distributed" not in mechanisms
    assert "latex" in validation and "quality" in validation and "retry" in validation
    empty = planner.build_project_query_plan(
        project_id="Plain", project_memory={"projects": [{"project_id": "Plain", "summary": "simple app"}]}
    )
    assert empty["validation_repair"] == []
    assert "hallucination solved" not in " ".join(all_queries(plan)).casefold()


def test_symbols_preserve_identifier_styles_and_reject_prose_and_oversized_values():
    known = {
        "WorkAgent": [
            "snake_case", "camelCase", "PascalCase", "snake_case",
            "this is unsupported free form prose", "x" * 300,
        ],
        "OtherProject": ["otherProjectOnly"],
    }
    plan = planner.build_project_query_plan(
        project_id="WorkAgent", project_memory=project_memory(), known_symbols=known
    )
    output = " ".join(plan["symbols"])
    assert "snake_case" in output and "camelCase" in output and "PascalCase" in output
    assert output.count("snake_case") == 1
    assert "unsupported free form" not in output
    assert "otherProjectOnly" not in output


def test_metric_queries_are_search_targets_without_invented_outcomes():
    plan = planner.build_project_query_plan(project_id="WorkAgent", project_memory=project_memory())
    output = " ".join(plan["metrics_impact"]).casefold()
    assert "metric evidence search" in output
    assert "token" in output and "manual" in output
    assert "%" not in output
    assert not any(character.isdigit() for character in output)


def test_all_query_and_symbol_bounds_are_enforced():
    plan = planner.build_project_query_plan(
        project_id="WorkAgent",
        project_memory=project_memory(),
        jd_targets={"targets": [f"technology_{index}" for index in range(200)]},
        known_symbols={"WorkAgent": [f"symbol_{index}" for index in range(200)]},
    )
    assert len(all_queries(plan)) <= planner.MAX_TOTAL_QUERIES
    assert all(len(plan[group]) <= planner.MAX_QUERIES_PER_GROUP for group in planner.QUERY_GROUPS)
    assert all(len(query) <= planner.MAX_QUERY_CHARS for query in all_queries(plan))
    assert all(len(query.split()) <= planner.MAX_TERMS_PER_QUERY for query in all_queries(plan))
    assert sum(len(query.split()) for query in plan["symbols"]) <= planner.MAX_SYMBOLS


def test_missing_malformed_and_nested_raw_inputs_are_safe():
    plan = planner.build_project_query_plan(
        project_id=None,
        project_memory={"projects": "invalid", "raw_text": "private"},
        compact_facts={"raw_text": "private", "nested": {"content": "private"}},
        jd_targets=object(),
        known_symbols={"raw_text": "private"},
    )
    assert plan["project_id"] == ""
    assert all(plan[group] == [] for group in planner.QUERY_GROUPS)


def test_reordered_set_like_inputs_produce_stable_semantic_output():
    first = planner.build_project_query_plan(
        project_id="WorkAgent",
        project_memory=project_memory(),
        jd_targets={"technologies": ["SQLite", "Python", "FastAPI"]},
        known_symbols={"WorkAgent": ["z_symbol", "a_symbol"]},
    )
    second = planner.build_project_query_plan(
        project_id="WorkAgent",
        project_memory=project_memory(),
        jd_targets={"technologies": ["FastAPI", "Python", "SQLite"]},
        known_symbols={"WorkAgent": ["a_symbol", "z_symbol"]},
    )
    assert first == second

    mixed_first = planner.build_project_query_plan(
        project_id="WorkAgent", jd_targets={"technologies": ["python", "Python"]}
    )
    mixed_second = planner.build_project_query_plan(
        project_id="WorkAgent", jd_targets={"technologies": ["Python", "python"]}
    )
    assert mixed_first == mixed_second


def test_cross_project_facts_and_symbols_never_contaminate_selected_project():
    plan = planner.build_project_query_plan(
        project_id="WorkAgent",
        project_memory=project_memory(),
        compact_facts={
            "WorkAgent": {"mechanisms": ["cache validation"], "symbols": ["selectedOnly"]},
            "OtherProject": {"mechanisms": ["Kubernetes consensus"], "symbols": ["otherOnly"]},
        },
        jd_targets={"technologies": ["Kubernetes"]},
        known_symbols={"WorkAgent": ["selectedKnown"], "OtherProject": ["otherKnown"]},
    )
    mechanisms = " ".join(plan["mechanisms"]).casefold()
    symbols = " ".join(plan["symbols"])
    assert "cache" in mechanisms and "kubernetes" not in mechanisms
    assert "selectedOnly" in symbols and "selectedKnown" in symbols
    assert "otherOnly" not in symbols and "otherKnown" not in symbols
    assert "kubernetes" in " ".join(plan["jd_alignment"]).casefold()


def test_raw_diff_secrets_and_large_bodies_never_enter_queries():
    sensitive = (
        "diff --git a/backend/a.py b/backend/a.py\n"
        "SECRET_API_KEY=example-secret\n-----BEGIN PRIVATE KEY-----\n"
    )
    plan = planner.build_project_query_plan(
        project_id="WorkAgent",
        project_memory={
            "projects": [{
                "project_id": "WorkAgent",
                "summary": "safe retrieval project",
                "raw_text": sensitive,
                "content": sensitive,
                "patch": sensitive,
            }]
        },
        compact_facts={"WorkAgent": {"body": sensitive, "mechanisms": [sensitive]}},
        jd_targets={"body": sensitive, "technologies": ["Python"]},
    )
    serialized = json.dumps(plan, sort_keys=True)
    assert "diff --git" not in serialized
    assert "example-secret" not in serialized
    assert "PRIVATE KEY" not in serialized


def test_plan_has_exact_required_groups_and_no_side_effect_dependencies():
    plan = planner.build_project_query_plan(project_id="WorkAgent")
    assert list(plan) == ["project_id", *planner.QUERY_GROUPS]
    source = open(planner.__file__, encoding="utf-8").read()
    for forbidden in (
        "chromadb", "MEMORY_STORE", "openai", "ProjectCapabilityFact",
        "project_capability_reader", "github_raw_storage", "github_evidence_chunks",
    ):
        assert forbidden not in source
