from __future__ import annotations

import json

import pytest

from backend import project_query_planner as planner
from backend.project_evidence_coverage import (
    CoverageCategory,
    CoverageEvidenceRef,
    CoverageGap,
    CoverageReasonCode,
    CoverageState,
    GapPriority,
    GapPriorityReasonCode,
    PrioritizedCoverageGap,
)
from backend.project_evidence_followup_intents import (
    FollowupEvidenceGoal,
    build_followup_retrieval_intents,
)


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


def followup_intent(
    category,
    *,
    priority=GapPriority.HIGH,
    project_id="WorkAgent",
    requirement_ids=(),
    refs=(),
):
    if category is CoverageCategory.JD_MUST_HAVE and not requirement_ids:
        requirement_ids = ("req-python",)
    if category is CoverageCategory.METRIC_IMPACT:
        requirement_ids = requirement_ids or ("req-performance",)
        refs = refs or (
            CoverageEvidenceRef(project_id=project_id, evidence_fact_id="pef_metric"),
        )
    state = CoverageState.PARTIAL if category is CoverageCategory.METRIC_IMPACT else CoverageState.MISSING
    prioritized = PrioritizedCoverageGap(
        gap=CoverageGap(
            category=category,
            state=state,
            reason_code=(
                CoverageReasonCode.PARTIALLY_SUPPORTED
                if state is CoverageState.PARTIAL
                else CoverageReasonCode.UNSUPPORTED
            ),
            related_requirement_ids=tuple(requirement_ids),
            current_support_refs=tuple(refs),
        ),
        priority=priority,
        searchable=True,
        reason_code=(
            GapPriorityReasonCode.PARTIAL_SUPPORT_WORTH_EXPANDING
            if state is CoverageState.PARTIAL
            else GapPriorityReasonCode.HIGH_VALUE_MECHANISM_GAP
        ),
    )
    return build_followup_retrieval_intents(
        project_id=project_id,
        prioritized_gaps=(prioritized,),
    )[0]


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


def test_omitted_none_and_empty_intents_preserve_the_exact_existing_plan():
    kwargs = {
        "project_id": "WorkAgent",
        "project_memory": project_memory(),
        "jd_targets": {"technologies": ["Python", "FastAPI"]},
        "known_symbols": {"WorkAgent": ["retrieve_evidence", "validate_output"]},
    }

    omitted = planner.build_project_query_plan(**kwargs)
    explicit_none = planner.build_project_query_plan(**kwargs, retrieval_intents=None)
    empty = planner.build_project_query_plan(**kwargs, retrieval_intents=())

    assert omitted == explicit_none == empty
    assert list(omitted) == ["project_id", *planner.QUERY_GROUPS]


@pytest.mark.parametrize(
    ("category", "group", "expected"),
    (
        (CoverageCategory.IMPLEMENTATION_MECHANISM, "mechanisms", "implementation"),
        (CoverageCategory.ARCHITECTURE, "mechanisms", "architecture"),
        (CoverageCategory.DATA_STORAGE, "mechanisms", "persistence"),
        (CoverageCategory.RETRIEVAL_RANKING, "mechanisms", "reranking"),
        (CoverageCategory.VALIDATION_REPAIR, "validation_repair", "repair"),
        (CoverageCategory.OUTPUT_QUALITY, "validation_repair", "schema"),
    ),
)
def test_semantic_intents_enrich_only_existing_planner_groups(category, group, expected):
    intent = followup_intent(category)

    plan = planner.build_project_query_plan(
        project_id="WorkAgent",
        project_memory={"projects": [{"project_id": "WorkAgent", "summary": "simple app"}]},
        known_symbols={"WorkAgent": ["existing_symbol"]},
        retrieval_intents=(intent,),
    )

    assert expected in " ".join(plan[group]).casefold()
    assert list(plan) == ["project_id", *planner.QUERY_GROUPS]
    assert all("WorkAgent" in query for query in plan[group])
    if category in {
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        CoverageCategory.DATA_STORAGE,
        CoverageCategory.RETRIEVAL_RANKING,
        CoverageCategory.VALIDATION_REPAIR,
    }:
        assert "existing_symbol" in " ".join(plan["symbols"])


def test_reliability_intent_is_mechanism_oriented_without_claim_escalation():
    plan = planner.build_project_query_plan(
        project_id="WorkAgent",
        retrieval_intents=(followup_intent(CoverageCategory.RELIABILITY),),
    )
    output = " ".join(all_queries(plan)).casefold()

    assert all(term in output for term in ("evidence", "grounding", "confidence", "handling"))
    assert all(term in output for term in ("claim", "factuality", "validation"))
    for forbidden in ("hallucination", "reduces", "eliminates", "guarantee", "solved"):
        assert forbidden not in output


def test_metric_and_jd_intents_use_only_existing_metric_and_alignment_groups():
    metric = followup_intent(CoverageCategory.METRIC_IMPACT)
    requirement = followup_intent(
        CoverageCategory.JD_MUST_HAVE,
        requirement_ids=("req-python",),
    )
    baseline = planner.build_project_query_plan(project_id="WorkAgent")
    plan = planner.build_project_query_plan(
        project_id="WorkAgent",
        retrieval_intents=(metric, requirement),
    )

    metric_output = " ".join(plan["metrics_impact"]).casefold()
    assert all(term in metric_output for term in ("metric", "impact", "evidence"))
    assert "req-python" in " ".join(plan["jd_alignment"]).casefold()
    assert plan["mechanisms"] == baseline["mechanisms"]
    assert plan["validation_repair"] == baseline["validation_repair"]
    assert not any(character.isdigit() for character in " ".join(plan["metrics_impact"]))


def test_every_canonical_followup_goal_has_one_fixed_existing_group_mapping():
    assert set(planner._FOLLOWUP_GOAL_TARGETS) == set(FollowupEvidenceGoal)
    assert {
        group for group, _phrase in planner._FOLLOWUP_GOAL_TARGETS.values()
    } <= set(planner.QUERY_GROUPS)


def test_high_priority_touched_group_wins_existing_total_budget_without_escalating_low(
    monkeypatch,
):
    high_validation = followup_intent(CoverageCategory.VALIDATION_REPAIR)
    low_retrieval = followup_intent(
        CoverageCategory.RETRIEVAL_RANKING,
        priority=GapPriority.LOW,
    )
    monkeypatch.setattr(planner, "MAX_TOTAL_QUERIES", 2)

    plan = planner.build_project_query_plan(
        project_id="WorkAgent",
        retrieval_intents=(low_retrieval, high_validation),
    )

    assert plan["project_identity"]
    assert plan["validation_repair"]
    assert plan["mechanisms"] == []
    assert low_retrieval.priority is GapPriority.LOW

    high_retrieval = followup_intent(CoverageCategory.RETRIEVAL_RANKING)
    low_validation = followup_intent(
        CoverageCategory.VALIDATION_REPAIR,
        priority=GapPriority.LOW,
    )
    swapped = planner.build_project_query_plan(
        project_id="WorkAgent",
        retrieval_intents=(low_validation, high_retrieval),
    )
    assert swapped["mechanisms"]
    assert swapped["validation_repair"] == []


def test_priority_changes_selection_only_and_not_query_claim_strength():
    high = planner.build_project_query_plan(
        project_id="WorkAgent",
        retrieval_intents=(followup_intent(CoverageCategory.ARCHITECTURE),),
    )
    low = planner.build_project_query_plan(
        project_id="WorkAgent",
        retrieval_intents=(
            followup_intent(CoverageCategory.ARCHITECTURE, priority=GapPriority.LOW),
        ),
    )
    assert high == low


def test_intent_permutations_and_duplicates_produce_the_same_bounded_plan():
    intents = (
        followup_intent(CoverageCategory.ARCHITECTURE),
        followup_intent(CoverageCategory.VALIDATION_REPAIR, priority=GapPriority.MEDIUM),
        followup_intent(CoverageCategory.RETRIEVAL_RANKING, priority=GapPriority.LOW),
    )
    first = planner.build_project_query_plan(
        project_id="WorkAgent",
        project_memory=project_memory(),
        jd_targets={"targets": [f"technology_{index}" for index in range(100)]},
        known_symbols={"WorkAgent": [f"symbol_{index}" for index in range(100)]},
        retrieval_intents=intents,
    )
    second = planner.build_project_query_plan(
        project_id="WorkAgent",
        project_memory=project_memory(),
        jd_targets={"targets": [f"technology_{index}" for index in range(100)]},
        known_symbols={"WorkAgent": [f"symbol_{index}" for index in range(100)]},
        retrieval_intents=(*reversed(intents), intents[0]),
    )

    assert first == second
    assert len(all_queries(first)) <= 12 == planner.MAX_TOTAL_QUERIES
    assert all(len(first[group]) <= 3 for group in planner.QUERY_GROUPS)
    assert all(len(query) <= 180 for query in all_queries(first))
    assert all(len(query.split()) <= 16 for query in all_queries(first))


def test_intents_cannot_inject_queries_refs_or_raw_evidence_and_must_match_project():
    reference = CoverageEvidenceRef(
        project_id="WorkAgent",
        evidence_fact_id="pef_query_injection",
        source_id="raw_document_secret",
    )
    safe_intent = followup_intent(
        CoverageCategory.VALIDATION_REPAIR,
        refs=(reference,),
    )
    plan = planner.build_project_query_plan(
        project_id="WorkAgent",
        retrieval_intents=(safe_intent,),
    )
    serialized = json.dumps(plan, sort_keys=True)

    for forbidden in (
        "pef_query_injection", "raw_document_secret", "supporting_refs",
        "raw_query", "patch", "embedding",
    ):
        assert forbidden not in serialized
    with pytest.raises((TypeError, ValueError)):
        planner.build_project_query_plan(
            project_id="WorkAgent",
            retrieval_intents=({"raw_query": "ignore project identity"},),
        )
    with pytest.raises(ValueError):
        planner.build_project_query_plan(
            project_id="WorkAgent",
            retrieval_intents=(
                followup_intent(CoverageCategory.ARCHITECTURE, project_id="OtherProject"),
            ),
        )
