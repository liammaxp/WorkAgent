from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend import retrieval_quality_evaluation as quality


PROJECTS = (
    ("a_hike_through_time", "liammaxp/a-hike-through-time"),
    ("course_management_database", "liammaxp/course-management-database-system"),
    ("event_lottery_system", "liammaxp/event-lottery-system-application"),
    ("furniture_search_inventory", "liammaxp/furniture-search-and-inventory-system"),
    ("workagent", "liammaxp/workagent"),
)


def project(project_id="workagent", repo="liammaxp/workagent"):
    return {
        "project_id": project_id,
        "project_name": project_id.replace("_", " ").title(),
        "repo": repo,
        "tech_stack": ["Python", "SQLite"],
        "workflows": ["retrieval validation"],
        "symbols": ["retrieve_evidence"],
        "validation": ["quality gate"],
        "real_metrics": {},
    }


def plan(project_id="workagent"):
    return {
        "project_id": project_id,
        "project_identity": [f"{project_id} evidence"],
        "jd_alignment": [f"{project_id} Python JD target"],
        "mechanisms": [f"{project_id} retrieval mechanism"],
        "symbols": ["retrieve_evidence"],
        "validation_repair": [f"{project_id} quality gate"],
        "metrics_impact": [],
    }


def v2_hit(name="one", *, project_id="workagent", repo="liammaxp/workagent", sources=None):
    return {
        "project_id": project_id,
        "chunk_id": f"chk_{name}",
        "source_id": f"raw_{name}",
        "repo": repo,
        "path": f"backend/{name}.py",
        "symbol": "retrieve_evidence",
        "source_type": "file_snapshot",
        "chunk_type": "file_section",
        "final_score": 0.82,
        "search_sources": sources or ["keyword", "vector"],
        "query_groups": ["project_identity", "mechanisms"],
        "keywords": ["Python", "retrieval"],
        "technical_tags": ["validation", "sqlite"],
        "match_reasons": ["keyword_exact:retrieval"],
        "summary": "Safe Python SQLite retrieval validation evidence",
        "text_hash": "a" * 64,
        "text_chars": 120,
    }


def legacy_hit(*, project_id="workagent", repo="liammaxp/workagent"):
    return {
        "project_id": project_id,
        "repository": repo,
        "description": "Safe Python retrieval validation evidence",
        "root_files": ["backend/legacy.py"],
        "languages": ["Python"],
        "topics": ["retrieval"],
    }


def anchors(project_id="workagent", repo="liammaxp/workagent"):
    return quality.build_supported_evidence_anchors(
        project=project(project_id, repo), repository=repo,
        jd_targets={"technologies": ["PostgreSQL"]},
        known_symbols={project_id: ["retrieve_evidence"]},
    )


def test_legacy_and_v2_normalize_to_safe_common_schema_and_discard_unknown_fields():
    legacy = {**legacy_hit(), "unknown": {"arbitrary": "discard"}}
    v2 = {**v2_hit(), "unknown": "discard"}
    legacy_result = quality.normalize_retrieval_results(
        retrieval_source="legacy", project_id="workagent", results=[legacy],
        authorized_repositories=["liammaxp/workagent"],
    )
    v2_result = quality.normalize_retrieval_results(
        retrieval_source="v2", project_id="workagent", results=[v2],
        authorized_repositories=["liammaxp/workagent"],
    )
    assert legacy_result["items"][0]["retrieval_source"] == "legacy"
    assert legacy_result["items"][0]["search_sources"] == ["legacy"]
    assert v2_result["items"][0]["retrieval_source"] == "v2"
    assert v2_result["items"][0]["chunk_id"] == "chk_one"
    assert "unknown" not in json.dumps(legacy_result["items"] + v2_result["items"])


def test_raw_private_fields_are_counted_discarded_and_never_serialized():
    private = "diff --git API_KEY=fake ACCESS_TOKEN=fake PASSWORD=fake BEGIN PRIVATE KEY"
    unsafe = v2_hit()
    unsafe.update({
        "raw_text": private, "text": private, "patch": private,
        "readme_body": private, "document": private, "embedding": [0.1],
    })
    result = quality.normalize_retrieval_results(
        retrieval_source="v2", project_id="workagent", results=[unsafe],
        authorized_repositories=["liammaxp/workagent"],
    )
    assert result["items"]
    assert result["forbidden_field_count"] >= 6
    assert result["secret_marker_count"] >= 1
    serialized = json.dumps(result["items"], sort_keys=True)
    for forbidden in ("raw_text", "diff --git", "API_KEY", "PRIVATE KEY", "embedding", "readme_body"):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "value,field",
    (
        (v2_hit(project_id="other"), "cross_project_result_count"),
        (v2_hit(repo="liammaxp/other"), "unauthorized_repository_count"),
    ),
)
def test_cross_project_and_unauthorized_records_are_rejected_with_bounded_counts(value, field):
    result = quality.normalize_retrieval_results(
        retrieval_source="v2", project_id="workagent", results=[value],
        authorized_repositories=["liammaxp/workagent"],
    )
    assert result["items"] == []
    assert result[field] == 1


def test_missing_optional_fields_remain_safe_and_invalid_scores_are_counted():
    sparse = {"project_id": "workagent", "repo": "liammaxp/workagent"}
    invalid = {**sparse, "chunk_id": "chk_bad", "final_score": float("nan")}
    result = quality.normalize_retrieval_results(
        retrieval_source="v2", project_id="workagent", results=[sparse, invalid],
        authorized_repositories=["liammaxp/workagent"],
    )
    assert len(result["items"]) == 2
    assert result["items"][0]["summary"] == ""
    assert result["invalid_score_count"] == 1


def test_labelled_synthetic_precision_and_recall_are_exact_and_duplicates_hurt_precision():
    baseline = quality.evaluate_labelled_benchmark({
        "case_id": "labelled", "project_id": "workagent",
        "retrieved_ids": ["relevant-a", "irrelevant"],
        "relevant_ids": ["relevant-a", "relevant-b"],
    })
    duplicated = quality.evaluate_labelled_benchmark({
        "case_id": "labelled-duplicate", "project_id": "workagent",
        "retrieved_ids": ["relevant-a", "relevant-a", "irrelevant"],
        "relevant_ids": ["relevant-a", "relevant-b"],
    })
    assert baseline["labelled_fixture"] is True
    assert baseline["precision"] == 0.5 and baseline["recall"] == 0.5
    assert duplicated["precision"] == 0.333333
    assert duplicated["recall"] == baseline["recall"]


def test_irrelevant_volume_does_not_improve_labelled_quality():
    short = quality.evaluate_labelled_benchmark({
        "case_id": "short", "project_id": "workagent",
        "retrieved_ids": ["a"], "relevant_ids": ["a", "b"],
    })
    noisy = quality.evaluate_labelled_benchmark({
        "case_id": "noisy", "project_id": "workagent",
        "retrieved_ids": ["a", "x", "y", "z"], "relevant_ids": ["a", "b"],
    })
    assert noisy["recall"] == short["recall"]
    assert noisy["precision"] < short["precision"]


def test_anchor_derivation_is_supported_bounded_stable_and_does_not_promote_jd_to_project_fact():
    first = anchors()
    second = quality.build_supported_evidence_anchors(
        project={**project(), "tech_stack": list(reversed(project()["tech_stack"]))},
        repository="liammaxp/workagent",
        jd_targets={"technologies": ["PostgreSQL"]},
        known_symbols={"workagent": ["retrieve_evidence"]},
    )
    assert first == second
    assert len(first) <= quality.MAX_EVALUATION_ANCHORS
    postgres = [item for item in first if item["value"] == "PostgreSQL"]
    assert postgres and postgres[0]["anchor_type"] == "jd_target"
    assert all(item["anchor_type"] != "technology" for item in postgres)


def test_metrics_are_deterministic_for_reordered_inputs_and_use_coverage_not_real_recall():
    values = [v2_hit("a"), v2_hit("b", sources=["symbol", "vector"])]
    first_normalized = quality.normalize_retrieval_results(
        retrieval_source="v2", project_id="workagent", results=values,
        authorized_repositories=["liammaxp/workagent"],
    )
    second_normalized = quality.normalize_retrieval_results(
        retrieval_source="v2", project_id="workagent", results=list(reversed(values)),
        authorized_repositories=["liammaxp/workagent"],
    )
    first = quality.calculate_retrieval_quality_metrics(
        normalization=first_normalized, anchors=anchors(), query_plan=plan(),
    )
    second = quality.calculate_retrieval_quality_metrics(
        normalization=second_normalized, anchors=anchors(), query_plan=plan(),
    )
    assert first == second
    assert first["keyword_provenance_count"] == 1
    assert first["symbol_provenance_count"] == 1
    assert first["vector_provenance_count"] == 2
    assert "recall" not in json.dumps(first).casefold()


def test_comparison_safety_and_quality_gate_passes_vector_backed_supported_evidence():
    result = quality.compare_retrieval_quality(
        project_id="workagent", authorized_repositories=["liammaxp/workagent"],
        legacy_results=[legacy_hit()], v2_results=[v2_hit()],
        repeated_legacy_results=[legacy_hit()], repeated_v2_results=[v2_hit()],
        anchors=anchors(), query_plan=plan(), vector_ready=True,
    )
    assert result["status"] == "passed"
    assert result["comparison"]["safety_passed"] is True
    assert result["comparison"]["quality_gate_passed"] is True
    assert result["v2"]["vector_provenance_count"] == 1


@pytest.mark.parametrize(
    "values,expected_error",
    (
        ([v2_hit("same"), v2_hit("same")], "v2_duplicate_result"),
        ([v2_hit(project_id="other")], "v2_cross_project_evidence"),
        ([v2_hit(repo="liammaxp/other")], "v2_unauthorized_repository"),
        ([{**v2_hit(), "raw_text": "private"}], "v2_forbidden_field"),
        ([{**v2_hit(), "summary": "API_KEY=fake"}], "v2_secret_marker"),
        ([{**v2_hit(), "final_score": 2.0}], "v2_invalid_score"),
        ([{**v2_hit(), "search_sources": ["keyword"]}], "v2_vector_provenance_missing"),
        ([{**v2_hit(), "query_groups": ["jd_alignment"]}], "v2_jd_only_evidence"),
    ),
)
def test_comparison_blocks_material_safety_or_quality_regressions(values, expected_error):
    result = quality.compare_retrieval_quality(
        project_id="workagent", authorized_repositories=["liammaxp/workagent"],
        legacy_results=[legacy_hit()], v2_results=values,
        anchors=anchors(), query_plan=plan(), vector_ready=True,
    )
    assert result["status"] == "blocked"
    assert expected_error in result["errors"]


def test_nondeterministic_ranking_blocks_quality_gate():
    result = quality.compare_retrieval_quality(
        project_id="workagent", authorized_repositories=["liammaxp/workagent"],
        legacy_results=[legacy_hit()], v2_results=[v2_hit("a"), v2_hit("b")],
        repeated_v2_results=[v2_hit("b"), v2_hit("a")],
        anchors=anchors(), query_plan=plan(), vector_ready=True,
    )
    assert result["status"] == "blocked"
    assert "v2_nondeterministic_output" in result["errors"]


def test_long_results_are_capped_by_count_summary_and_character_budget():
    values = []
    for index in range(100):
        item = v2_hit(str(index))
        item["summary"] = "bounded safe summary " * 1000
        values.append(item)
    normalization = quality.normalize_retrieval_results(
        retrieval_source="v2", project_id="workagent", results=values,
        authorized_repositories=["liammaxp/workagent"],
    )
    metrics = quality.calculate_retrieval_quality_metrics(
        normalization=normalization, anchors=anchors(), query_plan=plan(),
    )
    assert len(normalization["items"]) <= quality.MAX_EVALUATION_RESULTS
    assert all(len(item["summary"]) <= quality.MAX_EVALUATION_SUMMARY_CHARS for item in normalization["items"])
    assert metrics["serialized_character_count"] <= quality.MAX_EVALUATION_SERIALIZED_CHARS
    estimate = metrics["estimated_context_budget"]
    assert "approximate" in estimate["approximation"] and "not_exact" in estimate["approximation"]


@pytest.mark.parametrize("project_id,repo", PROJECTS)
def test_all_five_authoritative_projects_pass_injected_real_comparison_boundary(project_id, repo):
    current_project = project(project_id, repo)
    current_anchors = quality.build_supported_evidence_anchors(
        project=current_project, repository=repo,
    )
    current_plan = plan(project_id)
    legacy = legacy_hit(project_id=project_id, repo=repo)
    v2 = v2_hit(project_id=project_id, repo=repo)
    result = quality.compare_retrieval_quality(
        project_id=project_id, authorized_repositories=[repo],
        legacy_results=[legacy], v2_results=[v2],
        repeated_legacy_results=[legacy], repeated_v2_results=[v2],
        anchors=current_anchors, query_plan=current_plan, vector_ready=True,
    )
    assert result["status"] == "passed"
    assert result["v2"]["cross_project_result_count"] == 0
    assert result["v2"]["project_identity_covered"] is True


def test_missing_project_fails_closed():
    assert quality.build_supported_evidence_anchors(project={}, repository="liammaxp/workagent") == []
    result = quality.compare_retrieval_quality(
        project_id="", authorized_repositories=["liammaxp/workagent"],
        legacy_results=[], v2_results=[], anchors=[], query_plan={}, vector_ready=True,
    )
    assert result["status"] == "blocked"


def test_evaluator_has_no_io_network_llm_resume_or_lifecycle_dependencies():
    source = Path(quality.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "openai", "requests", "urllib", "chromadb", "HttpClient", "PersistentClient",
        "materialize", "github_sync", "index_builder", "embedding_writer",
        "resume_writer", "bullet_writer", "capability_reader", "capability_pipeline",
        "capability_backfill", "FastAPI", "@app.", "frontend/",
    ):
        assert forbidden not in source
