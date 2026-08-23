from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend import project_retrieval_v2 as retrieval_v2
from backend.github_evidence_chunks import build_github_evidence_chunk_record
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
from backend.project_repository_identity import build_project_repository_identity_authority


PROJECT_A = {
    "project_id": "ProjectA",
    "project_name": "Project A",
    "repo": "owner/repo-a",
    "tech_stack": ["Python"],
    "workflows": ["retrieval validation"],
    "symbols": ["retrieve_evidence"],
}
PROJECT_B = {
    "project_id": "ProjectB",
    "project_name": "Project B",
    "repo": "owner/repo-b",
    "tech_stack": ["Python", "React"],
    "workflows": ["retrieval validation API database"],
}


def followup_intent(project_id=PROJECT_A["project_id"], *, requirement_ids=()):
    prioritized = PrioritizedCoverageGap(
        gap=CoverageGap(
            category=CoverageCategory.JD_MUST_HAVE if requirement_ids else CoverageCategory.VALIDATION_REPAIR,
            state=CoverageState.MISSING,
            reason_code=CoverageReasonCode.UNSUPPORTED,
            related_requirement_ids=tuple(requirement_ids),
        ),
        priority=GapPriority.HIGH,
        searchable=True,
        reason_code=GapPriorityReasonCode.JD_MUST_HAVE_GAP,
    )
    return build_followup_retrieval_intents(
        project_id=project_id,
        prioritized_gaps=(prioritized,),
    )[0]


def authority(*projects):
    return build_project_repository_identity_authority(
        project_memory={"projects": list(projects)}
    )


def chunk(name, *, project=PROJECT_A, repo=None, text="retrieval validation API evidence"):
    return build_github_evidence_chunk_record(
        source_id=f"raw_{name}",
        project_id=project["project_id"],
        repo=repo or project["repo"],
        source_type="file_snapshot",
        chunk_type="file_section",
        path=f"backend/{name}.py",
        commit_sha="abc123",
        symbol="retrieve_evidence" if project is PROJECT_A else "other_symbol",
        text=text,
        summary=f"Safe {name} retrieval validation summary",
        raw_hash="a" * 64,
    )


def ready(project_id=PROJECT_A["project_id"], **changes):
    result = {
        "status": "ready",
        "vector_ready": True,
        "chunk_mapping_ready": True,
        "ready_for_hybrid_retrieval": True,
        "records_unresolved": 0,
        "identity_authority_status": "ready",
        "identity_conflict_count": 0,
        "identity_unresolved_count": 0,
        "projects_with_chunks": [project_id],
    }
    result.update(changes)
    return result


def vector_record(project=PROJECT_A, *, record_id="vec_a", distance=0.15):
    return {
        "vector_record_id": record_id,
        "distance": distance,
        "metadata": {
            "project_id": project["project_id"],
            "repository": project["repo"],
        },
        "rank": 1,
    }


def base_dependencies(*, projects=(PROJECT_A,), chunks=None, readiness=None, vector_search=None):
    identity = authority(*projects)
    chunk_values = list(chunks if chunks is not None else [chunk("a")])
    return {
        "vector_backend_enabled": lambda: True,
        "authority_loader": lambda _path: identity,
        "vector_metadata_reader": lambda: [
            {"vector_record_id": f"vec_{index}", "metadata": {"repository": item["repo"]}}
            for index, item in enumerate(projects)
        ],
        "readiness_inspector": lambda **_kwargs: readiness or ready(),
        "chunk_loader": lambda _path: chunk_values,
        "vector_search": vector_search or (lambda **_kwargs: [vector_record()]),
    }


def safe_hybrid_hit(name="a", *, project=PROJECT_A, repo=None, score=0.9, sources=None):
    return {
        "project_id": project["project_id"],
        "chunk_id": f"chk_{name}",
        "source_id": f"raw_{name}",
        "repo": repo or project["repo"],
        "path": f"backend/{name}.py",
        "commit_sha": "abc123",
        "source_type": "file_snapshot",
        "chunk_type": "file_section",
        "symbol": "retrieve_evidence",
        "score": score,
        "keyword_score": 0.6,
        "symbol_score": 0.7,
        "vector_score": 0.8,
        "search_sources": sources or ["keyword", "vector"],
        "query_groups": ["mechanisms"],
        "match_reasons": ["keyword_exact:retrieval"],
        "summary": f"Safe {name} summary",
        "keywords": ["retrieval", "Python"],
        "technical_tags": ["validation"],
        "text_hash": "b" * 64,
        "text_chars": 123,
    }


def test_ready_chain_reuses_planner_hybrid_and_returns_vector_backed_safe_evidence():
    vector_queries = []

    def search(**kwargs):
        vector_queries.append(kwargs["query"])
        assert kwargs["project_id"] == PROJECT_A["project_id"]
        assert kwargs["embedder"] is not None and kwargs["authority"]["status"] == "ready"
        return [vector_record()]

    result = retrieval_v2.retrieve_evidence_for_project_v2(
        PROJECT_A,
        jd_targets={"technologies": ["PostgreSQL"]},
        **base_dependencies(vector_search=search),
    )

    assert result
    assert all(item["project_id"] == PROJECT_A["project_id"] for item in result)
    assert all(item["repo"] == PROJECT_A["repo"] for item in result)
    assert all(0 <= item["final_score"] <= 1 for item in result)
    assert any("vector" in item["search_sources"] for item in result)
    assert "retrieve_evidence" not in vector_queries
    assert len({item["chunk_id"] for item in result}) == len(result)


@pytest.mark.parametrize(
    "readiness_value",
    (
        None,
        {},
        ready(status="partial"),
        ready(vector_ready=False),
        ready(chunk_mapping_ready=False),
        ready(ready_for_hybrid_retrieval=False),
        ready(records_unresolved=1),
        ready(identity_authority_status="partial"),
        ready(identity_conflict_count=1),
        ready(identity_unresolved_count=1),
        ready(project_id=PROJECT_B["project_id"]),
    ),
)
def test_incomplete_or_malformed_readiness_blocks_before_plan_or_vector(readiness_value):
    calls = []
    result = retrieval_v2.retrieve_evidence_for_project_v2(
        PROJECT_A,
        readiness_inspector=lambda **_kwargs: readiness_value,
        query_plan_builder=lambda **_kwargs: pytest.fail("planner must not run"),
        vector_search=lambda **_kwargs: calls.append(_kwargs),
        **{
            key: value
            for key, value in base_dependencies().items()
            if key not in {"readiness_inspector", "vector_search"}
        },
    )
    assert result == [] and calls == []


def test_disabled_vector_backend_blocks_before_all_v2_io():
    calls = []
    result = retrieval_v2.retrieve_evidence_for_project_v2(
        PROJECT_A,
        vector_backend_enabled=lambda: False,
        authority_loader=lambda _path: calls.append("authority"),
        vector_metadata_reader=lambda: calls.append("metadata"),
        readiness_inspector=lambda **_kwargs: calls.append("readiness"),
        chunk_loader=lambda _path: calls.append("chunks"),
        query_plan_builder=lambda **_kwargs: calls.append("planner"),
        hybrid_retriever=lambda **_kwargs: calls.append("hybrid"),
        vector_search=lambda **_kwargs: calls.append("vector"),
    )
    assert result == [] and calls == []


def test_vector_metadata_unavailable_blocks_without_similarity_or_lexical_search():
    calls = []

    def inspector(**kwargs):
        assert kwargs["vector_records"] == []
        return ready(status="blocked", vector_ready=False)

    dependencies = base_dependencies()
    dependencies.update({
        "vector_metadata_reader": lambda: [],
        "readiness_inspector": inspector,
        "query_plan_builder": lambda **_kwargs: calls.append("planner"),
        "hybrid_retriever": lambda **_kwargs: calls.append("hybrid"),
        "vector_search": lambda **_kwargs: calls.append("vector"),
    })
    assert retrieval_v2.retrieve_evidence_for_project_v2(PROJECT_A, **dependencies) == []
    assert calls == []


def test_vector_failure_after_readiness_never_returns_lexical_only_results():
    def unavailable(**_kwargs):
        raise TimeoutError("127.0.0.1:8100 ACCESS_TOKEN=fake")

    result = retrieval_v2.retrieve_evidence_for_project_v2(
        PROJECT_A,
        **base_dependencies(vector_search=unavailable),
    )
    assert result == []
    assert "127.0.0.1" not in json.dumps(result)


def test_query_plan_inputs_are_structured_and_project_identity_is_preserved():
    captured = {}

    def planner(**kwargs):
        captured.update(kwargs)
        return {
            "project_id": PROJECT_A["project_id"],
            "project_identity": ["ProjectA evidence"],
            "jd_alignment": [],
            "mechanisms": [],
            "symbols": [],
            "validation_repair": [],
            "metrics_impact": [],
        }

    result = retrieval_v2.retrieve_evidence_for_project_v2(
        PROJECT_A,
        jd_targets={"technologies": ["PostgreSQL"]},
        known_symbols={PROJECT_A["project_id"]: ["retrieve_evidence"]},
        query_plan_builder=planner,
        hybrid_retriever=lambda **_kwargs: {
            "status": "ready", "hits": [safe_hybrid_hit()], "warnings": [], "errors": []
        },
        **{key: value for key, value in base_dependencies().items() if key != "vector_search"},
    )
    assert result
    assert captured["project_id"] == PROJECT_A["project_id"]
    assert captured["project_memory"] == {"projects": [PROJECT_A]}
    assert captured["jd_targets"] == {"technologies": ["PostgreSQL"]}
    assert "Kubernetes" not in json.dumps(captured)


@pytest.mark.parametrize("argument", ("omitted", None, ()))
def test_none_and_empty_intents_do_not_change_the_strict_old_planner_contract(argument):
    calls = []

    def strict_planner(*, project_id, project_memory, compact_facts, jd_targets, known_symbols):
        calls.append({
            "project_id": project_id,
            "project_memory": project_memory,
            "compact_facts": compact_facts,
            "jd_targets": jd_targets,
            "known_symbols": known_symbols,
        })
        return {
            "project_id": project_id,
            "project_identity": [f"{project_id} evidence"],
            "jd_alignment": [],
            "mechanisms": [],
            "symbols": [],
            "validation_repair": [],
            "metrics_impact": [],
        }

    kwargs = {
        "query_plan_builder": strict_planner,
        "hybrid_retriever": lambda **_kwargs: {
            "status": "ready", "hits": [safe_hybrid_hit()], "warnings": [], "errors": []
        },
        **base_dependencies(),
    }
    if argument != "omitted":
        kwargs["retrieval_intents"] = argument

    result = retrieval_v2.retrieve_evidence_for_project_v2(PROJECT_A, **kwargs)

    assert result
    assert len(calls) == 1
    assert set(calls[0]) == {
        "project_id", "project_memory", "compact_facts", "jd_targets", "known_symbols",
    }


def test_valid_intents_are_forwarded_only_to_the_existing_planner_and_never_become_evidence():
    supplied = (followup_intent(requirement_ids=("req-python",)),)
    captured = {}

    def planner(**kwargs):
        captured.update(kwargs)
        return {
            "project_id": PROJECT_A["project_id"],
            "project_identity": ["ProjectA evidence"],
            "jd_alignment": ["ProjectA req-python evidence"],
            "mechanisms": [],
            "symbols": [],
            "validation_repair": [],
            "metrics_impact": [],
        }

    result = retrieval_v2.retrieve_evidence_for_project_v2(
        PROJECT_A,
        retrieval_intents=supplied,
        query_plan_builder=planner,
        hybrid_retriever=lambda **_kwargs: {
            "status": "ready", "hits": [safe_hybrid_hit()], "warnings": [], "errors": []
        },
        **base_dependencies(),
    )

    assert result
    assert captured["retrieval_intents"] == supplied
    serialized = json.dumps(result, sort_keys=True)
    assert "req-python" not in serialized
    assert "jd_requirement_evidence" not in serialized


def test_valid_intent_reaches_the_actual_planner_and_existing_hybrid_boundary_once():
    captured = {}

    def hybrid(**kwargs):
        captured.update(kwargs)
        return {
            "status": "ready", "hits": [safe_hybrid_hit()], "warnings": [], "errors": []
        }

    result = retrieval_v2.retrieve_evidence_for_project_v2(
        PROJECT_A,
        retrieval_intents=(followup_intent(),),
        hybrid_retriever=hybrid,
        **base_dependencies(),
    )

    assert result
    assert captured["project_id"] == PROJECT_A["project_id"]
    assert "validation" in " ".join(
        captured["query_plan"]["validation_repair"]
    ).casefold()
    assert list(captured["query_plan"]) == [
        "project_id", *retrieval_v2.QUERY_GROUPS,
    ]


def test_direct_v2_foreign_intent_fails_before_backend_or_readiness_io():
    calls = []

    result = retrieval_v2.retrieve_evidence_for_project_v2(
        PROJECT_A,
        retrieval_intents=(followup_intent(PROJECT_B["project_id"]),),
        vector_backend_enabled=lambda: calls.append("backend"),
        authority_loader=lambda _path: calls.append("authority"),
        vector_metadata_reader=lambda: calls.append("metadata"),
        readiness_inspector=lambda **_kwargs: calls.append("readiness"),
        chunk_loader=lambda _path: calls.append("chunks"),
        query_plan_builder=lambda **_kwargs: calls.append("planner"),
    )

    assert result == []
    assert calls == []


def test_project_isolation_discards_cross_project_chunks_vectors_and_repository_guessing():
    calls = []

    def search(**kwargs):
        calls.append(kwargs)
        return [vector_record(PROJECT_B, record_id="vec_b", distance=0.01), vector_record()]

    result = retrieval_v2.retrieve_evidence_for_project_v2(
        PROJECT_A,
        **base_dependencies(
            projects=(PROJECT_A, PROJECT_B),
            chunks=[chunk("a"), chunk("b", project=PROJECT_B)],
            vector_search=search,
        ),
    )
    assert result
    assert {item["project_id"] for item in result} == {PROJECT_A["project_id"]}
    assert {item["repo"] for item in result} == {PROJECT_A["repo"]}
    assert calls

    guessed = {**PROJECT_A, "project_id": "MissingProject", "repo": PROJECT_A["repo"]}
    assert retrieval_v2.retrieve_evidence_for_project_v2(
        guessed, **base_dependencies(projects=(PROJECT_A, PROJECT_B))
    ) == []


def test_adapter_preserves_order_clamps_limits_deduplicates_and_redacts_private_values():
    private = "diff --git BEGIN PRIVATE KEY API_KEY=fake ACCESS_TOKEN=fake PASSWORD=fake full patch full source body vector document"
    first = safe_hybrid_hit("first", score=0.95)
    first.update({
        "raw_text": private,
        "document": private,
        "embeddings": [0.1],
        "summary": private,
        "match_reasons": [private, "keyword_exact:retrieval"],
    })
    duplicate = {**safe_hybrid_hit("duplicate", score=0.99), "chunk_id": first["chunk_id"]}
    cross_project = safe_hybrid_hit("cross", project=PROJECT_B)
    wrong_repo = safe_hybrid_hit("wrong", repo="owner/not-authorized")
    values = [first, duplicate, cross_project, wrong_repo] + [
        safe_hybrid_hit(str(index), score=0.8) for index in range(70)
    ]

    result = retrieval_v2.adapt_hybrid_hits_for_resume_evidence(
        values,
        project_id=PROJECT_A["project_id"],
        authorized_repositories=[PROJECT_A["repo"]],
        project_name=PROJECT_A["project_name"],
        limit=10_000,
    )
    assert result[0]["chunk_id"] == first["chunk_id"]
    assert len(result) == retrieval_v2.MAX_V2_EVIDENCE_LIMIT
    assert len({item["chunk_id"] for item in result}) == len(result)
    assert result[0]["summary"] == ""
    serialized = json.dumps(result, sort_keys=True)
    for forbidden in (
        "raw_text", "document", "embedding", "diff --git", "PRIVATE KEY",
        "API_KEY", "ACCESS_TOKEN", "PASSWORD", "full patch", "full source body",
        "vector document", "owner/not-authorized", PROJECT_B["project_id"],
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize("limit", (0, -1, True, "20", None))
def test_invalid_limits_fail_closed(limit):
    calls = []
    assert retrieval_v2.retrieve_evidence_for_project_v2(
        PROJECT_A,
        limit=limit,
        vector_backend_enabled=lambda: calls.append("backend"),
    ) == []
    assert calls == []


def test_controlled_dependency_errors_are_safe_empty_without_legacy_fallback():
    for dependency in (
        {"authority_loader": lambda _path: (_ for _ in ()).throw(RuntimeError("PASSWORD=secret"))},
        {"vector_metadata_reader": lambda: (_ for _ in ()).throw(TimeoutError("127.0.0.1:8100"))},
        {"chunk_loader": lambda _path: (_ for _ in ()).throw(ValueError("full source body"))},
        {"query_plan_builder": lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("API_KEY=fake"))},
        {"hybrid_retriever": lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("diff --git"))},
    ):
        values = base_dependencies()
        values.update(dependency)
        result = retrieval_v2.retrieve_evidence_for_project_v2(PROJECT_A, **values)
        assert result == [] and json.dumps(result) == "[]"


def test_retrieval_module_has_no_write_lifecycle_or_frontend_dependencies():
    source = Path(retrieval_v2.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "materialize_saved_github_evidence",
        "github_evidence_preparation_service",
        "github_sync",
        "index_builder",
        "PersistentClient",
        "get_or_create_collection",
        "project_capability_reader",
        "project_capability_pipeline",
        "project_capability_backfill",
        "resume_writer",
        "bullet_writer",
        "frontend/",
        "@app.",
    ):
        assert forbidden not in source
