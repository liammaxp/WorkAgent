from __future__ import annotations

import json
import math
from pathlib import Path

from backend import evidence_vector_search as vector
from backend import github_evidence_chunks as chunks
from backend import project_retrieval_v2
from backend.memory_store import MemoryVectorStore


PRIVATE = "diff --git a/private.py b/private.py API_KEY=fake-key -----BEGIN PRIVATE KEY-----"


def chunk(name, *, project_id="ProjectA", repo="owner/repo-a", path="backend/retrieval.py", **changes):
    values = {
        "source_id": f"raw_{name}", "project_id": project_id, "repo": repo,
        "source_type": "file_snapshot", "chunk_type": "file_section", "path": path,
        "commit_sha": "abc123", "symbol": "", "text": "retrieval validation evidence",
        "summary": "Safe retrieval validation summary", "keywords": ["retrieval", "evidence"],
        "technical_tags": ["validation"], "raw_hash": "a" * 64,
    }
    values.update(changes)
    return chunks.build_github_evidence_chunk_record(**values)


def raw_result(name, *, project_id="ProjectA", repo="owner/repo-a", distance=0.2, **metadata):
    return {
        "id": f"vec_{name}", "distance": distance,
        "document": PRIVATE, "embedding": [0.1, 0.2],
        "metadata": {"project_id": project_id, "repository": repo, **metadata},
    }


def test_score_normalization_direction_bounds_and_malformed_values():
    assert vector.normalize_vector_score(distance=0.1) > vector.normalize_vector_score(distance=0.9)
    assert vector.normalize_vector_score(similarity=0.8) == 0.8
    assert vector.normalize_vector_score(similarity=2.0) == 1.0
    for value in (float("nan"), float("inf"), -1, "0.2", None):
        assert vector.normalize_vector_score(distance=value) is None
    assert vector.normalize_vector_score(distance=0.2) == vector.normalize_vector_score(distance=0.2)


def test_repository_result_is_sanitized_and_documents_embeddings_are_discarded():
    result = vector.normalize_repository_vector_result(
        raw_result("safe", path="backend/retrieval.py", raw_text=PRIVATE, embeddings=[1, 2]),
        project_id="ProjectA",
    )
    assert result is not None
    serialized = json.dumps(result, sort_keys=True)
    for forbidden in ("document", "embedding", "raw_text", "diff --git", "fake-key"):
        assert forbidden not in serialized
    assert result["path"] == "backend/retrieval.py"


def test_malformed_results_scores_and_project_identity_fail_closed():
    assert vector.normalize_repository_vector_result(None, project_id="ProjectA") is None
    assert vector.normalize_repository_vector_result({"id": "x", "distance": 0.1}, project_id="ProjectA") is None
    assert vector.normalize_repository_vector_result(raw_result("other", project_id="ProjectB"), project_id="ProjectA") is None
    conflict = raw_result("conflict", repository_project_id="ProjectB")
    assert vector.normalize_repository_vector_result(conflict, project_id="ProjectA") is None
    assert vector.normalize_repository_vector_result(raw_result("nan", distance=float("nan")), project_id="ProjectA") is None


def test_existing_backend_adapter_uses_safe_vector_method_without_documents():
    class Store:
        def __init__(self):
            self.calls = []
        def search_github_vector_records(self, query, n_results, *, project_id):
            self.calls.append((query, n_results, project_id))
            return [raw_result("one")]
    store = Store()
    result = vector.search_existing_github_evidence_vectors(
        query="retrieval evidence", n_results=100, project_id="ProjectA", memory_store=store
    )
    assert len(result) == 1
    assert store.calls == [(
        "evidence retrieval", vector.MAX_VECTOR_TOP_K_PER_QUERY, "ProjectA"
    )]
    assert "document" not in result[0] and "embedding" not in result[0]
    assert vector.search_existing_github_evidence_vectors(
        query=PRIVATE, project_id="ProjectA", memory_store=store
    ) == []


def test_memory_vector_store_reader_fails_closed_without_initializing_chroma(tmp_path):
    class Collection:
        def __getattr__(self, name):
            raise AssertionError(f"Chroma collection method must not run: {name}")

    store = MemoryVectorStore(tmp_path / "chroma", tmp_path / "memory.json", tmp_path / "github")
    store._github = Collection()
    store._ensure_client = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("PersistentClient must not be initialized by the read path")
    )
    assert store.search_github_vector_records("retrieval", n_results=5) == []


def test_repository_result_maps_to_project_repo_path_and_lexical_chunks():
    exact = chunk("exact", path="backend/retrieval.py", keywords=["retrieval"])
    unrelated = chunk("unrelated", path="backend/other.py", text="renderer", keywords=["renderer"])
    other_project = chunk("other", project_id="ProjectB", repo="owner/repo-a", keywords=["retrieval"])
    normalized = vector.normalize_repository_vector_result(
        raw_result("repo", path="backend/retrieval.py"), project_id="ProjectA"
    )
    hits = vector.map_vector_result_to_evidence_chunks(
        vector_result=normalized, query="retrieval", query_group="mechanisms",
        project_id="ProjectA", chunks=[unrelated, other_project, exact], top_k=3,
    )
    assert len(hits) == 1 and hits[0]["chunk_id"] == exact["chunk_id"]
    assert hits[0]["search_source"] == "vector"
    assert "path_exact" in hits[0]["match_reasons"]
    assert "text" not in hits[0]


def test_no_safe_repo_or_lexical_mapping_returns_no_hit():
    normalized = vector.normalize_repository_vector_result(raw_result("repo"), project_id="ProjectA")
    assert vector.map_vector_result_to_evidence_chunks(
        vector_result=normalized, query="missing", query_group="mechanisms",
        project_id="ProjectA", chunks=[chunk("one")],
    ) == []
    wrong_repo = chunk("wrong", repo="owner/different", keywords=["retrieval"])
    assert vector.map_vector_result_to_evidence_chunks(
        vector_result=normalized, query="retrieval", query_group="mechanisms",
        project_id="ProjectA", chunks=[wrong_repo],
    ) == []


def test_multi_query_executes_eligible_groups_independently_and_skips_symbols():
    calls = []
    def fake_search(*, query, n_results, project_id):
        calls.append((query, n_results, project_id))
        return [raw_result(query.replace(" ", "_"))]
    plan = {
        "project_id": "ProjectA", "project_identity": ["ProjectA evidence"],
        "jd_alignment": ["backend retrieval"], "mechanisms": ["retrieval"],
        "symbols": ["retrieve_evidence"], "validation_repair": ["validation"],
        "metrics_impact": ["latency"],
    }
    result = vector.vector_search_project_query_plan(
        query_plan=plan, project_id="ProjectA", chunks=[chunk("one")], vector_search=fake_search,
    )
    assert [call[0] for call in calls] == [
        "evidence projecta", "backend retrieval", "retrieval", "validation", "latency"
    ]
    assert all(call[0] != "retrieve_evidence" for call in calls)
    assert {hit["query_group"] for hit in result} <= set(vector.ELIGIBLE_VECTOR_QUERY_GROUPS)


def test_cross_project_vector_results_and_jd_terms_cannot_cross_boundary():
    def fake_search(**_kwargs):
        return [raw_result("b", project_id="ProjectB", repo="owner/repo-b"), raw_result("a")]
    plan = {group: [] for group in vector.QUERY_GROUPS}
    plan.update(project_id="ProjectA", jd_alignment=["redis websocket retrieval"])
    result = vector.vector_search_project_query_plan(
        query_plan=plan, project_id="ProjectA",
        chunks=[chunk("a"), chunk("b", project_id="ProjectB", repo="owner/repo-b")],
        vector_search=fake_search,
    )
    assert result and {hit["project_id"] for hit in result} == {"ProjectA"}
    assert all(hit["vector_record_id"] != "vec_b" for hit in result)


def test_vector_and_mapping_scores_are_distinct_finite_bounded_and_ordered():
    strong = chunk("strong", keywords=["retrieval"], technical_tags=["retrieval"], symbol="retrieval")
    weak = chunk("weak", text="retrieval", keywords=[], technical_tags=[], path="backend/weak.py")
    high = vector.normalize_repository_vector_result(raw_result("high", distance=0.1), project_id="ProjectA")
    lower = vector.normalize_repository_vector_result(raw_result("lower", distance=0.3), project_id="ProjectA")
    strong_hit = vector.map_vector_result_to_evidence_chunks(
        vector_result=high, query="retrieval", query_group="mechanisms",
        project_id="ProjectA", chunks=[strong], top_k=1,
    )[0]
    weak_hit = vector.map_vector_result_to_evidence_chunks(
        vector_result=high, query="retrieval", query_group="mechanisms",
        project_id="ProjectA", chunks=[weak], top_k=1,
    )[0]
    lower_strong = vector.map_vector_result_to_evidence_chunks(
        vector_result=lower, query="retrieval", query_group="mechanisms",
        project_id="ProjectA", chunks=[strong], top_k=1,
    )[0]
    assert strong_hit["mapping_score"] > weak_hit["mapping_score"]
    assert strong_hit["score"] > weak_hit["score"]
    assert strong_hit["score"] > lower_strong["score"]
    assert all(math.isfinite(hit["score"]) and 0 <= hit["score"] <= 1 for hit in (strong_hit, weak_hit, lower_strong))


def test_limits_invalid_values_and_empty_inputs_fail_closed():
    plan = {group: [] for group in vector.QUERY_GROUPS}
    plan.update(project_id="ProjectA", mechanisms=[f"retrieval term_{index}" for index in range(30)])
    calls = []
    def fake_search(**kwargs):
        calls.append(kwargs)
        return [raw_result(str(index), distance=index / 100) for index in range(50)]
    hits = vector.vector_search_project_query_plan(
        query_plan=plan, project_id="ProjectA", chunks=[chunk(str(index)) for index in range(20)],
        vector_search=fake_search, top_k_per_query=100, chunks_per_vector_result=100,
    )
    assert len(calls) <= vector.MAX_VECTOR_QUERIES
    assert all(call["n_results"] == vector.MAX_VECTOR_TOP_K_PER_QUERY for call in calls)
    assert len(hits) <= vector.MAX_VECTOR_HITS_TOTAL
    assert vector.vector_search_project_query_plan(
        query_plan=plan, project_id="ProjectA", chunks=[chunk("x")], vector_search=fake_search,
        top_k_per_query=0,
    ) == []
    assert vector.vector_search_project_query_plan(
        query_plan={}, project_id="ProjectA", chunks=[], vector_search=fake_search,
    ) == []


def test_determinism_for_reordered_chunks_results_queries_and_metadata():
    records = [raw_result("b", distance=0.2), raw_result("a", distance=0.2)]
    def fake_forward(**_kwargs): return records
    def fake_reverse(**_kwargs): return list(reversed(records))
    plan_a = {group: [] for group in vector.QUERY_GROUPS}
    plan_a.update(project_id="ProjectA", mechanisms=["validation retrieval", "evidence"])
    plan_b = dict(plan_a, mechanisms=["evidence", "retrieval validation"])
    chunk_values = [chunk("b"), chunk("a")]
    first = vector.vector_search_project_query_plan(
        query_plan=plan_a, project_id="ProjectA", chunks=chunk_values, vector_search=fake_forward,
    )
    second = vector.vector_search_project_query_plan(
        query_plan=plan_b, project_id="ProjectA", chunks=list(reversed(chunk_values)), vector_search=fake_reverse,
    )
    assert first == second


def test_returned_hits_exclude_documents_embeddings_raw_and_secrets():
    private_chunk = chunk("private", text=PRIVATE, summary=PRIVATE, keywords=["retrieval"])
    def fake_search(**_kwargs): return [raw_result("private", raw_text=PRIVATE, embeddings=[1, 2])]
    plan = {group: [] for group in vector.QUERY_GROUPS}
    plan.update(project_id="ProjectA", mechanisms=["retrieval"])
    hits = vector.vector_search_project_query_plan(
        query_plan=plan, project_id="ProjectA", chunks=[private_chunk], vector_search=fake_search,
    )
    serialized = json.dumps(hits, sort_keys=True)
    assert hits
    for forbidden in ("document", "embedding", "raw_text", "diff --git", "fake-key", "PRIVATE KEY"):
        assert forbidden not in serialized
    assert all("text" not in hit for hit in hits)


def test_unsafe_query_never_reaches_vector_backend_and_scaffold_stays_empty(monkeypatch):
    calls = []
    def fake_search(**kwargs): calls.append(kwargs); return []
    plan = {group: [] for group in vector.QUERY_GROUPS}
    plan.update(project_id="ProjectA", mechanisms=[PRIVATE])
    assert vector.vector_search_project_query_plan(
        query_plan=plan, project_id="ProjectA", chunks=[chunk("one")], vector_search=fake_search,
    ) == []
    assert calls == []
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "1")
    assert project_retrieval_v2.retrieve_evidence_for_project_v2({"project_id": "ProjectA"}) == []


def test_vector_module_has_no_resume_api_llm_or_capability_dependencies():
    source = Path(vector.__file__).read_text(encoding="utf-8").casefold()
    for forbidden in (
        "chromadb", "openai", "api_server", "tailor_resume", "projectcapabilityfact",
        "project_capability_reader", "project_capability_pipeline", "project_capability_backfill",
    ):
        assert forbidden not in source
