from __future__ import annotations

import json
import math
from pathlib import Path

from backend import evidence_hybrid_retrieval as hybrid
from backend.github_evidence_chunks import build_github_evidence_chunk_record


PROJECT_A = "ProjectA"
PROJECT_B = "ProjectB"
REPO_A = "owner/repo-a"
PRIVATE = "diff --git a/private b/private API_KEY=fake BEGIN PRIVATE KEY"


def chunk(name: str, *, project_id: str = PROJECT_A, repo: str = REPO_A, path: str = "backend/retrieval.py", text: str = "retrieval validation API evidence", symbol: str = "", summary: str = "Safe retrieval validation summary", source_id: str | None = None):
    return build_github_evidence_chunk_record(
        source_id=source_id or f"raw_{name}", project_id=project_id, repo=repo,
        source_type="file_snapshot", chunk_type="file_section", path=path,
        commit_sha="abc123", symbol=symbol, text=text, summary=summary,
        raw_hash="a" * 64,
    )


def plan(**groups):
    values = {group: [] for group in (
        "project_identity", "jd_alignment", "mechanisms", "symbols",
        "validation_repair", "metrics_impact",
    )}
    values.update(groups)
    return {"project_id": PROJECT_A, **values}


def readiness(*, project_id: str = PROJECT_A, **changes):
    value = {
        "status": "ready", "vector_ready": True, "chunk_mapping_ready": True,
        "ready_for_hybrid_retrieval": True, "projects_with_chunks": [project_id],
    }
    value.update(changes)
    return value


def source_hit(name: str, source: str = "keyword", *, project_id: str = PROJECT_A, score: float = 21.0, query_group: str = "mechanisms", query: str = "retrieval validation", path: str = "backend/retrieval.py", text_hash: str | None = None, summary: str = "Safe summary", reasons=None, chunk_id: str | None = None, source_id: str | None = None, vector_record_id: str = "vec_a", vector_rank: int = 1):
    return {
        "hit_id": f"hit_{name}_{source}", "chunk_id": chunk_id or f"chk_{name}",
        "source_id": source_id or f"raw_{name}", "project_id": project_id,
        "repo": REPO_A if project_id == PROJECT_A else "owner/repo-b", "path": path,
        "commit_sha": "abc123", "source_type": "file_snapshot", "chunk_type": "file_section",
        "symbol": "retrieve_evidence", "search_source": source,
        "query_group": query_group, "query": query, "score": score,
        "matched_terms": ["retrieval", "validation"],
        "match_reasons": reasons or (["symbol_exact:retrieve_evidence"] if source == "symbol" else ["keyword_exact:retrieval"]),
        "summary": summary, "keywords": ["retrieval"], "technical_tags": ["validation"],
        "text_hash": text_hash or ("a" * 64), "text_chars": 120,
        **({"vector_record_id": vector_record_id, "vector_rank": vector_rank} if source == "vector" else {}),
    }


def fake_vector_search(**kwargs):
    return [{
        "id": "vec_a", "similarity": 0.9,
        "metadata": {"project_id": kwargs["project_id"], "repository": REPO_A},
    }]


def test_fixed_score_calibration_is_bounded_directional_and_fail_closed():
    assert hybrid.normalize_lexical_retrieval_score(0) == 0
    assert 0 < hybrid.normalize_lexical_retrieval_score(6) < hybrid.normalize_lexical_retrieval_score(42) == 1
    assert 0 < hybrid.normalize_symbol_retrieval_score(12) < hybrid.normalize_symbol_retrieval_score(31) == 1
    for invalid in (-1, float("nan"), float("inf"), "10", True, None):
        assert hybrid.normalize_lexical_retrieval_score(invalid) is None
        assert hybrid.normalize_symbol_retrieval_score(invalid) is None
    assert hybrid.normalize_lexical_retrieval_score(10) == hybrid.normalize_lexical_retrieval_score(10)


def test_source_normalization_is_project_scoped_bounded_and_private_fields_never_survive():
    value = source_hit("one", summary=PRIVATE)
    value.update({"raw_text": PRIVATE, "document": PRIVATE, "embeddings": [1, 2]})
    normalized = hybrid.normalize_hybrid_source_hit(value, project_id=PROJECT_A)
    assert normalized and normalized["summary"] == ""
    serialized = json.dumps(normalized, sort_keys=True)
    for forbidden in ("raw_text", "document", "embedding", "diff --git", "fake", "PRIVATE KEY"):
        assert forbidden not in serialized
    assert hybrid.normalize_hybrid_source_hit(value, project_id=PROJECT_B) is None
    assert hybrid.normalize_hybrid_source_hit({**value, "search_source": "unknown"}, project_id=PROJECT_A) is None
    assert hybrid.normalize_hybrid_source_hit({**value, "chunk_id": ""}, project_id=PROJECT_A) is None
    assert hybrid.normalize_hybrid_source_hit({**value, "score": math.inf}, project_id=PROJECT_A) is None


def test_exact_chunk_merges_all_sources_and_repeated_queries_count_once():
    values = [
        source_hit("same", "keyword", query="retrieval"),
        source_hit("same", "keyword", query="validation"),
        source_hit("same", "symbol", score=20, query_group="symbols", query="retrieve_evidence"),
        source_hit("same", "vector", score=0.8, query="retrieval"),
    ]
    result = hybrid.merge_hybrid_evidence_candidates(source_hits=values, project_id=PROJECT_A)
    assert len(result) == 1
    hit = result[0]
    assert hit["source_agreement_count"] == 3
    assert hit["search_sources"] == ["keyword", "symbol", "vector"]
    assert len(hit["query_matches"]) == 4
    assert hit["hit_id"].startswith("hyb_")
    keyword_only = hybrid.merge_hybrid_evidence_candidates(source_hits=values[:2], project_id=PROJECT_A)[0]
    assert keyword_only["source_agreement_count"] == 1
    assert hit["score"] > keyword_only["score"]


def test_exact_content_dedupe_is_conservative_and_never_crosses_path_or_project():
    exact = [
        source_hit("a", text_hash="b" * 64, path="backend/a.py"),
        source_hit("b", "vector", score=0.7, text_hash="b" * 64, path="backend/a.py"),
    ]
    merged = hybrid.merge_hybrid_evidence_candidates(source_hits=exact, project_id=PROJECT_A)
    assert len(merged) == 1 and merged[0]["duplicate_chunk_ids"] == ["chk_b"]
    assert len(hybrid.merge_hybrid_evidence_candidates(
        source_hits=[exact[0], {**exact[1], "path": "backend/b.py"}], project_id=PROJECT_A,
    )) == 2
    assert len(hybrid.merge_hybrid_evidence_candidates(
        source_hits=[exact[0], {**exact[1], "summary": "Almost the same wording", "text_hash": "c" * 64}],
        project_id=PROJECT_A,
    )) == 2
    cross_project = {**exact[1], "project_id": PROJECT_B, "repo": "owner/repo-b"}
    isolated = hybrid.merge_hybrid_evidence_candidates(source_hits=[exact[0], cross_project], project_id=PROJECT_A)
    assert len(isolated) == 1 and isolated[0]["project_id"] == PROJECT_A


def test_score_ordering_rewards_exact_strength_without_overrewarding_weak_agreement():
    strong_symbol = hybrid.merge_hybrid_evidence_candidates(
        source_hits=[source_hit("symbol", "symbol", score=31, query_group="symbols")], project_id=PROJECT_A,
    )[0]
    weak_keyword = hybrid.merge_hybrid_evidence_candidates(
        source_hits=[source_hit("weak", score=1, reasons=["text_token:retrieval"])], project_id=PROJECT_A,
    )[0]
    strong_vector = hybrid.merge_hybrid_evidence_candidates(
        source_hits=[source_hit("vector", "vector", score=0.9)], project_id=PROJECT_A,
    )[0]
    weak_multi = hybrid.merge_hybrid_evidence_candidates(source_hits=[
        source_hit("multi", "keyword", score=1),
        source_hit("multi", "symbol", score=1, query_group="symbols"),
        source_hit("multi", "vector", score=0.1),
    ], project_id=PROJECT_A)[0]
    strong_keyword = hybrid.merge_hybrid_evidence_candidates(
        source_hits=[source_hit("keyword", score=30)], project_id=PROJECT_A,
    )[0]
    assert strong_symbol["score"] > weak_keyword["score"]
    assert strong_vector["score"] > weak_keyword["score"]
    assert strong_keyword["score"] > weak_multi["score"]
    assert all(math.isfinite(item["score"]) and 0 <= item["score"] <= 1 for item in [strong_symbol, weak_keyword, strong_vector, weak_multi])


def test_jd_only_is_discounted_and_metrics_are_not_marked_verified():
    mechanism = hybrid.merge_hybrid_evidence_candidates(
        source_hits=[source_hit("m", score=20, query_group="mechanisms")], project_id=PROJECT_A,
    )[0]
    jd = hybrid.merge_hybrid_evidence_candidates(
        source_hits=[source_hit("j", score=20, query_group="jd_alignment")], project_id=PROJECT_A,
    )[0]
    metric = hybrid.merge_hybrid_evidence_candidates(
        source_hits=[source_hit("x", score=20, query_group="metrics_impact")], project_id=PROJECT_A,
    )[0]
    assert jd["score"] < mechanism["score"]
    assert "verified" not in json.dumps(metric).casefold()


def test_runner_reuses_all_sources_excludes_symbols_from_vector_and_isolates_projects():
    chunks = [
        chunk("a", symbol="retrieve_evidence"),
        chunk("b", project_id=PROJECT_B, repo="owner/repo-b", text="retrieval validation API database React Python"),
    ]
    calls = []
    def vector_search(**kwargs):
        calls.append(kwargs)
        return [
            {"id": "vec_a", "similarity": 0.9, "metadata": {"project_id": PROJECT_A, "repository": REPO_A}},
            {"id": "vec_b", "similarity": 1.0, "metadata": {"project_id": PROJECT_B, "repository": "owner/repo-b"}},
        ]
    result = hybrid.run_project_hybrid_retrieval(
        project_id=PROJECT_A,
        query_plan=plan(project_identity=["ProjectA evidence"], mechanisms=["retrieval validation"], symbols=["retrieve_evidence"]),
        chunks=chunks, vector_search=vector_search, readiness=readiness(), top_k=10,
    )
    assert result["status"] == "ready" and result["hits"]
    assert {hit["project_id"] for hit in result["hits"]} == {PROJECT_A}
    assert all(call["query"] != "retrieve_evidence" for call in calls)
    assert {"keyword", "symbol", "vector"} <= set().union(*(hit["search_sources"] for hit in result["hits"]))


def test_readiness_and_adapter_fail_closed_without_lexical_fallback():
    calls = []
    def vector_search(**kwargs): calls.append(kwargs); return []
    for blocked in (
        readiness(vector_ready=False), readiness(chunk_mapping_ready=False),
        readiness(ready_for_hybrid_retrieval=False), {"status": "ready"}, None,
    ):
        result = hybrid.run_project_hybrid_retrieval(
            project_id=PROJECT_A, query_plan=plan(mechanisms=["retrieval"]), chunks=[chunk("a")],
            vector_search=vector_search, readiness=blocked,
        )
        assert result["status"] == "blocked" and result["hits"] == []
    assert calls == []
    assert hybrid.run_project_hybrid_retrieval(
        project_id=PROJECT_A, query_plan=plan(mechanisms=["retrieval"]), chunks=[chunk("a")],
        vector_search=None, readiness=readiness(),
    )["status"] == "blocked"


def test_empty_invalid_and_oversized_limits_are_bounded():
    empty_plan = hybrid.run_project_hybrid_retrieval(
        project_id=PROJECT_A, query_plan=plan(), chunks=[chunk("a")],
        vector_search=fake_vector_search, readiness=readiness(),
    )
    assert empty_plan["status"] == "empty"
    assert hybrid.run_project_hybrid_retrieval(
        project_id=PROJECT_A, query_plan=plan(mechanisms=["retrieval"]), chunks=[],
        vector_search=fake_vector_search, readiness=readiness(),
    )["status"] == "empty"
    assert hybrid.run_project_hybrid_retrieval(
        project_id=PROJECT_A, query_plan=plan(mechanisms=["retrieval"]), chunks=[chunk("a")],
        vector_search=fake_vector_search, readiness=readiness(), top_k=0,
    )["status"] == "blocked"
    many = [source_hit(str(index), score=20 + index / 100) for index in range(80)]
    merged = hybrid.merge_hybrid_evidence_candidates(source_hits=many, project_id=PROJECT_A)
    selected = hybrid.rerank_hybrid_evidence_candidates(candidates=merged, top_k=1000)
    assert len(selected) <= hybrid.MAX_HYBRID_TOP_K
    assert hybrid.rerank_hybrid_evidence_candidates(candidates=[{"bad": True}], top_k=5) == []


def test_group_and_path_diversity_caps_apply_when_alternatives_exist():
    hits = []
    for index in range(5):
        hits.extend([
            source_hit(f"jd{index}", score=30-index, query_group="jd_alignment", path="same.py"),
            source_hit(f"m{index}", score=29-index, query_group="mechanisms", path=f"m{index}.py"),
            source_hit(f"v{index}", score=28-index, query_group="validation_repair", path=f"v{index}.py"),
        ])
    candidates = hybrid.merge_hybrid_evidence_candidates(source_hits=hits, project_id=PROJECT_A)
    selected = hybrid.rerank_hybrid_evidence_candidates(candidates=candidates, top_k=8)
    assert sum(hit["query_groups"] == ["jd_alignment"] for hit in selected) <= 2
    assert sum(hit["path"] == "same.py" for hit in selected) <= math.ceil(8 * hybrid.MAX_PATH_FRACTION)
    assert selected[0]["score"] == max(hit["score"] for hit in candidates)


def test_coverage_is_final_hit_derived_deterministic_and_informational():
    query_plan = plan(project_identity=["identity"], mechanisms=["retrieval"], symbols=["retrieve_evidence"], metrics_impact=["latency"])
    hits = hybrid.merge_hybrid_evidence_candidates(source_hits=[
        source_hit("a", query_group="mechanisms"),
        source_hit("a", "vector", score=0.8, query_group="validation_repair"),
    ], project_id=PROJECT_A)
    first = hybrid.build_hybrid_retrieval_coverage(query_plan=query_plan, hits=hits, source_hit_counts={"keyword": 1, "vector": 1})
    second = hybrid.build_hybrid_retrieval_coverage(query_plan={**query_plan}, hits=list(reversed(hits)), source_hit_counts={"vector": 1, "keyword": 1})
    assert first == second
    assert first["groups_with_hits"] == ["mechanisms", "validation_repair"]
    assert first["missing_query_groups"] == ["project_identity", "symbols", "metrics_impact"]
    assert first["multi_source_hit_count"] == 1
    assert "follow" not in json.dumps(first).casefold()


def test_execution_is_deterministic_across_reordered_chunks_queries_and_vector_results():
    chunks = [chunk("a", path="a.py"), chunk("b", path="b.py")]
    query_plan_a = plan(mechanisms=["validation retrieval", "retrieval API"])
    query_plan_b = plan(mechanisms=list(reversed(query_plan_a["mechanisms"])))
    records = [
        {"id": "vec_b", "similarity": 0.8, "metadata": {"project_id": PROJECT_A, "repository": REPO_A}},
        {"id": "vec_a", "similarity": 0.8, "metadata": {"project_id": PROJECT_A, "repository": REPO_A}},
    ]
    first = hybrid.run_project_hybrid_retrieval(
        project_id=PROJECT_A, query_plan=query_plan_a, chunks=chunks,
        vector_search=lambda **_: records, readiness=readiness(),
    )
    second = hybrid.run_project_hybrid_retrieval(
        project_id=PROJECT_A, query_plan=query_plan_b, chunks=list(reversed(chunks)),
        vector_search=lambda **_: list(reversed(records)), readiness=readiness(),
    )
    assert first == second


def test_provenance_and_output_limits_are_deterministic():
    hits = []
    for index in range(80):
        hits.append(source_hit(
            "same", "keyword", query=f"retrieval q{index}",
            reasons=[f"keyword_exact:term{index}"],
        ))
    result = hybrid.merge_hybrid_evidence_candidates(source_hits=list(reversed(hits)), project_id=PROJECT_A)[0]
    assert len(result["query_matches"]) <= hybrid.MAX_HYBRID_QUERY_MATCHES
    assert len(result["match_reasons"]) <= hybrid.MAX_HYBRID_MATCH_REASONS
    assert result == hybrid.merge_hybrid_evidence_candidates(source_hits=hits, project_id=PROJECT_A)[0]


def test_boundary_has_no_resume_capability_materialization_api_or_frontend_integration():
    source = Path("backend/evidence_hybrid_retrieval.py").read_text(encoding="utf-8")
    for forbidden in (
        "retrieve_evidence_for_project_v2", "retrieve_evidence_for_project_for_resume",
        "tailor_resume", "capability_reader", "capability_pipeline", "capability_backfill",
        "materialize_saved_github_evidence", "PersistentClient", "FastAPI", "@app.",
    ):
        assert forbidden not in source


def test_final_result_serialization_never_contains_raw_or_secret_content():
    private = chunk("private", text=PRIVATE, summary=PRIVATE)
    result = hybrid.run_project_hybrid_retrieval(
        project_id=PROJECT_A, query_plan=plan(mechanisms=["retrieval validation"]),
        chunks=[private], vector_search=fake_vector_search, readiness=readiness(),
    )
    serialized = json.dumps(result, sort_keys=True)
    for forbidden in ("text\"", "raw_text", "snippet", "document", "embedding", "diff --git", "API_KEY", "PRIVATE KEY"):
        assert forbidden not in serialized
