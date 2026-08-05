from __future__ import annotations

import json
import math
from pathlib import Path

from backend import evidence_chunk_search as search
from backend import github_evidence_chunks as chunks
from backend import github_raw_storage
from backend import project_query_planner
from backend import project_retrieval_v2


PRIVATE_BODY = "diff --git a/private.py b/private.py SECRET_API_KEY=example-secret"


def chunk(
    chunk_id,
    *,
    project_id="ProjectA",
    path="backend/retrieval.py",
    symbol="",
    text="",
    summary="",
    keywords=(),
    tags=(),
):
    return chunks.build_github_evidence_chunk_record(
        source_id=f"raw_{chunk_id}",
        project_id=project_id,
        repo=project_id,
        source_type="file_snapshot",
        chunk_type="file_section",
        path=path,
        commit_sha="abc123",
        symbol=symbol,
        text=text,
        summary=summary,
        keywords=list(keywords),
        technical_tags=list(tags),
        raw_hash="a" * 64,
    )


def test_keyword_fields_match_with_expected_weight_order_and_safe_hits():
    records = [
        chunk("keyword", keywords=["retrieval"]),
        chunk("tag", tags=["retrieval"]),
        chunk("summary", summary="retrieval behavior"),
        chunk("text", text="internal retrieval implementation"),
        chunk("none", text="unrelated renderer"),
    ]
    hits = search.keyword_search_evidence_chunks(
        chunks=records, query="retrieval", project_id="ProjectA", top_k=10
    )
    assert [hit["score"] for hit in hits] == [6.0, 6.0, 4.0, 2.0]
    assert {hit["chunk_id"] for hit in hits} == {record["chunk_id"] for record in records[:4]}
    assert all("text" not in hit and "raw_text" not in hit for hit in hits)


def test_exact_technical_and_multi_field_matches_outrank_generic_prose():
    technical = chunk(
        "technical", symbol="retrieve_evidence", text="retrieval evidence", keywords=["retrieval"], tags=["retrieval"]
    )
    prose = chunk("prose", text="This project resume system uses retrieval")
    hits = search.keyword_search_evidence_chunks(
        chunks=[prose, technical], query="retrieval", project_id="ProjectA"
    )
    assert hits[0]["chunk_id"] == technical["chunk_id"]
    generic = search.keyword_search_evidence_chunks(
        chunks=[chunk("generic", keywords=["resume"]), chunk("specific", keywords=["validation"])],
        query="resume validation",
        project_id="ProjectA",
    )
    assert generic[0]["chunk_id"].endswith(generic[0]["chunk_id"].split("_")[-1])
    assert generic[0]["score"] > generic[-1]["score"]


def test_keyword_no_match_empty_unsafe_and_bounds_fail_closed():
    records = [chunk(str(index), keywords=["retrieval"]) for index in range(80)]
    assert search.keyword_search_evidence_chunks(
        chunks=records, query="missing", project_id="ProjectA"
    ) == []
    assert search.keyword_search_evidence_chunks(chunks=records, query="", project_id="ProjectA") == []
    assert search.keyword_search_evidence_chunks(
        chunks=records, query="SECRET_API_KEY=example-secret", project_id="ProjectA"
    ) == []
    assert search.keyword_search_evidence_chunks(
        chunks=records, query="retrieval", project_id="ProjectA", top_k=-1
    ) == []
    assert len(search.keyword_search_evidence_chunks(
        chunks=records, query="retrieval", project_id="ProjectA", top_k=1000
    )) == search.MAX_TOP_K


def test_symbol_styles_exact_path_and_dotted_module_matches():
    records = [
        chunk("snake", symbol="retrieve_evidence_for_project"),
        chunk("camel", symbol="resumeQualityGate"),
        chunk("pascal", symbol="ProjectQueryPlan"),
        chunk("path", path="backend/api_server.py"),
        chunk("token", text="backend.project_query_planner validates output"),
    ]
    for symbol, expected in (
        ("retrieve_evidence_for_project", records[0]),
        ("resumeQualityGate", records[1]),
        ("ProjectQueryPlan", records[2]),
        ("api_server.py", records[3]),
        ("backend/api_server.py", records[3]),
        ("backend.project_query_planner", records[4]),
    ):
        hits = search.symbol_search_evidence_chunks(
            chunks=list(reversed(records)), symbols=[symbol], project_id="ProjectA"
        )
        assert hits and hits[0]["chunk_id"] == expected["chunk_id"]
        assert "text" not in hits[0]


def test_symbol_search_rejects_short_ambiguous_and_deduplicates_symbols():
    record = chunk("gate", symbol="resume_quality_gate", text="api gate")
    assert search.symbol_search_evidence_chunks(
        chunks=[record], symbols=["api", "api"], project_id="ProjectA"
    ) == []
    hits = search.symbol_search_evidence_chunks(
        chunks=[record], symbols=["resume_quality_gate", "resume_quality_gate"], project_id="ProjectA"
    )
    assert len(hits) == 1
    assert hits[0]["matched_terms"] == ["resume_quality_gate"]


def test_project_isolation_is_strict_even_with_overlapping_query_terms():
    project_a = chunk("a", project_id="ProjectA", tags=["sqlite", "retrieval"])
    project_b = chunk("b", project_id="ProjectB", tags=["sqlite", "redis", "websocket"])
    hits = search.keyword_search_evidence_chunks(
        chunks=[project_b, project_a], query="sqlite redis websocket", project_id="ProjectA"
    )
    assert {hit["project_id"] for hit in hits} == {"ProjectA"}
    assert search.keyword_search_evidence_chunks(
        chunks=[project_a], query="sqlite", project_id=""
    ) == []
    missing_identity = dict(project_a, project_id="")
    assert search.keyword_search_evidence_chunks(
        chunks=[missing_identity], query="sqlite", project_id="ProjectA"
    ) == []


def test_query_plan_groups_search_independently_without_final_merge():
    records = [
        chunk("mechanism", keywords=["retrieval"]),
        chunk("validation", tags=["validation"]),
        chunk("metric", keywords=["latency"]),
        chunk("symbol", symbol="retrieve_evidence_for_project"),
    ]
    plan = {
        "project_id": "ProjectA",
        "project_identity": [],
        "jd_alignment": [],
        "mechanisms": ["retrieval"],
        "symbols": ["retrieve_evidence_for_project"],
        "validation_repair": ["validation"],
        "metrics_impact": ["latency metric evidence search"],
    }
    result = search.search_project_query_plan(chunks=records, query_plan=plan, top_k_per_query=5)
    assert result["mechanisms"][0]["query_group"] == "mechanisms"
    assert result["validation_repair"][0]["query_group"] == "validation_repair"
    assert result["metrics_impact"][0]["query_group"] == "metrics_impact"
    assert result["symbols"][0]["search_source"] == "symbol"
    assert result["mechanisms"][0]["query"] == "retrieval"


def test_scores_and_order_are_finite_deterministic_and_input_order_independent():
    records = [
        chunk("b", path="b.py", keywords=["retrieval"]),
        chunk("a", path="a.py", keywords=["retrieval"]),
    ]
    first = search.keyword_search_evidence_chunks(
        chunks=records, query="retrieval", project_id="ProjectA"
    )
    second = search.keyword_search_evidence_chunks(
        chunks=list(reversed(records)), query="retrieval", project_id="ProjectA"
    )
    assert first == second
    assert [hit["path"] for hit in first] == ["a.py", "b.py"]
    assert all(math.isfinite(hit["score"]) and hit["score"] > 0 for hit in first)


def test_reordered_keywords_tags_query_terms_and_symbols_are_stable():
    first_chunk = chunk("stable", keywords=["evidence", "retrieval"], tags=["cache", "validation"])
    second_chunk = chunk("stable", keywords=["retrieval", "evidence"], tags=["validation", "cache"])
    first = search.keyword_search_evidence_chunks(
        chunks=[first_chunk], query="retrieval evidence", project_id="ProjectA"
    )
    second = search.keyword_search_evidence_chunks(
        chunks=[second_chunk], query="evidence retrieval", project_id="ProjectA"
    )
    assert first == second
    symbols_first = search.symbol_search_evidence_chunks(
        chunks=[chunk("symbols", symbol="ProjectQueryPlan", text="retrieve_evidence")],
        symbols=["ProjectQueryPlan", "retrieve_evidence"], project_id="ProjectA",
    )
    symbols_second = search.symbol_search_evidence_chunks(
        chunks=[chunk("symbols", symbol="ProjectQueryPlan", text="retrieve_evidence")],
        symbols=["retrieve_evidence", "ProjectQueryPlan"], project_id="ProjectA",
    )
    assert symbols_first == symbols_second


def test_malformed_chunks_are_skipped_or_safely_normalized():
    valid = chunk("valid", text="retrieval")
    malformed = [None, [], {"project_id": "ProjectA"}, dict(valid, text={"nested": "retrieval"})]
    hits = search.keyword_search_evidence_chunks(
        chunks=[*malformed, valid], query="retrieval", project_id="ProjectA"
    )
    assert hits
    assert all(isinstance(hit["score"], float) for hit in hits)
    assert {hit["chunk_id"] for hit in hits} == {valid["chunk_id"]}


def test_malformed_large_text_is_bounded_and_sensitive_labels_are_not_returned():
    record = dict(
        chunk("malformed"),
        text="retrieval " * 10000,
        keywords=["retrieval", "SECRET_API_KEY"],
        technical_tags=["validation", "access_token"],
    )
    hit = search.keyword_search_evidence_chunks(
        chunks=[record], query="retrieval", project_id="ProjectA"
    )[0]
    serialized = json.dumps(hit, sort_keys=True).casefold()
    assert "secret_api_key" not in serialized
    assert "access_token" not in serialized


def test_hit_redaction_excludes_source_bodies_secrets_and_unsafe_summary():
    record = chunk(
        "private",
        text=PRIVATE_BODY,
        summary=PRIVATE_BODY,
        keywords=["retrieval"],
    )
    hit = search.keyword_search_evidence_chunks(
        chunks=[record], query="retrieval", project_id="ProjectA"
    )[0]
    serialized = json.dumps(hit, sort_keys=True)
    assert "text" not in hit and "raw_text" not in hit
    for forbidden in ("diff --git", "example-secret", "PRIVATE KEY"):
        assert forbidden not in serialized
    assert hit["summary"] == ""


def test_bounds_on_hit_fields_and_oversized_queries():
    record = chunk(
        "bounded",
        summary="retrieval " * 200,
        keywords=[f"term_{index}" for index in range(100)],
        tags=[f"tag_{index}" for index in range(100)],
        text=" ".join(f"query_{index}" for index in range(100)),
    )
    hits = search.keyword_search_evidence_chunks(
        chunks=[record], query=" ".join(f"query_{index}" for index in range(100)),
        project_id="ProjectA",
    )
    assert hits
    hit = hits[0]
    assert len(hit["query"]) <= search.MAX_SEARCH_QUERY_CHARS
    assert len(hit["matched_terms"]) <= search.MAX_MATCHED_TERMS
    assert len(hit["match_reasons"]) <= search.MAX_MATCH_REASONS
    assert len(hit["summary"]) <= search.MAX_HIT_SUMMARY_CHARS


def test_existing_redaction_planner_and_scaffold_boundaries_remain_safe(monkeypatch):
    raw = github_raw_storage.build_github_raw_source_record(raw_text=PRIVATE_BODY)
    raw_safe = github_raw_storage.redact_github_raw_source_record(raw)
    evidence_chunk = chunk("safe", text=PRIVATE_BODY, keywords=["retrieval"])
    chunk_safe = chunks.redact_github_evidence_chunk_record(evidence_chunk)
    plan = project_query_planner.build_project_query_plan(
        project_id="ProjectA", project_memory={"projects": [{"project_id": "ProjectA", "raw_text": PRIVATE_BODY}]}
    )
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "1")
    assert "raw_text" not in raw_safe and "text" not in chunk_safe
    assert PRIVATE_BODY not in json.dumps(plan)
    assert project_retrieval_v2.retrieve_evidence_for_project_v2({"project_id": "ProjectA"}) == []


def test_search_module_has_no_vector_api_llm_or_capability_dependencies():
    source = Path(search.__file__).read_text(encoding="utf-8").casefold()
    for forbidden in (
        "chromadb", "memory_store", "openai", "projectcapabilityfact",
        "project_capability_reader", "project_capability_pipeline", "project_capability_backfill",
    ):
        assert forbidden not in source
