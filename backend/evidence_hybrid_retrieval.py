"""Deterministic project-scoped orchestration across existing evidence search sources."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable, Mapping, Sequence, TypedDict

from backend import evidence_memory
from backend.evidence_chunk_search import EvidenceSearchHit, search_project_query_plan
from backend.evidence_vector_search import VectorEvidenceHit, vector_search_project_query_plan
from backend.project_query_planner import MAX_TOTAL_QUERIES, ProjectQueryPlan, QUERY_GROUPS


DEFAULT_HYBRID_TOP_K = 20
MAX_HYBRID_TOP_K = 50
MAX_SOURCE_HITS_TOTAL = 500
MAX_NORMALIZED_CANDIDATES = 300
MAX_HYBRID_QUERY_MATCHES = 24
MAX_HYBRID_MATCHED_TERMS = 24
MAX_HYBRID_MATCH_REASONS = 24
MAX_DUPLICATE_CHUNK_IDS = 16
MAX_SOURCE_IDS_PER_HIT = 16
MAX_VECTOR_RECORD_IDS_PER_HIT = 16
MAX_HYBRID_SUMMARY_CHARS = 500
MAX_SAFE_LABELS = 32
MAX_WARNINGS = 16
MAX_ERRORS = 8

# A single lexical target can match symbol, path, keyword, tag, summary, text, and phrase.
KEYWORD_SCORE_CAP = 42.0
# A single symbol target can match symbol, path, identifier, and three components.
SYMBOL_SCORE_CAP = 31.0
KEYWORD_WEIGHT = 0.90
SYMBOL_WEIGHT = 1.00
VECTOR_WEIGHT = 0.95
AGREEMENT_BONUS_PER_ADDITIONAL_SOURCE = 0.05
MAX_AGREEMENT_BONUS_SOURCES = 2
QUERY_DIVERSITY_BONUS = 0.02
MAX_QUERY_DIVERSITY_BONUS = 3
JD_ALIGNMENT_ONLY_FACTOR = 0.72
MIN_GROUP_COVERAGE_SCORE = 0.15

MAX_JD_ONLY_FRACTION = 0.25
MAX_PRIMARY_GROUP_FRACTION = 0.50
MAX_PATH_FRACTION = 0.40

SEMANTIC_GROUP_ORDER = (
    "project_identity",
    "mechanisms",
    "symbols",
    "validation_repair",
    "metrics_impact",
    "jd_alignment",
)
_GROUP_PRIORITY = {group: index for index, group in enumerate(SEMANTIC_GROUP_ORDER)}
_SOURCE_ORDER = {"keyword": 0, "symbol": 1, "vector": 2}
_ALLOWED_SOURCES = frozenset(_SOURCE_ORDER)
_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")
_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9_+.#/:-]{1,160}$")
_HASH_RE = re.compile(r"^[a-fA-F0-9]{32,128}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_UNSAFE_RE = re.compile(
    r"(?i)(?:diff\s+--git|begin\s+(?:rsa\s+)?private\s+key|api[_-]?key\s*=|"
    r"access[_-]?token\s*=|password\s*=|secret\s*=|credential\s*=|"
    r"private[_-]?key|authorization\s*:|bearer\s+[a-z0-9._-]+)"
)


class EvidenceSourceContribution(TypedDict):
    search_source: str
    query_group: str
    query: str
    source_score: float
    raw_score: float
    matched_terms: list[str]
    match_reasons: list[str]
    vector_record_id: str | None
    vector_rank: int | None


class NormalizedHybridSourceHit(TypedDict):
    chunk_id: str
    source_id: str
    project_id: str
    repo: str
    path: str
    commit_sha: str
    source_type: str
    chunk_type: str
    symbol: str
    summary: str
    keywords: list[str]
    technical_tags: list[str]
    text_hash: str
    text_chars: int
    contribution: EvidenceSourceContribution
    exact_match_quality: int


class HybridEvidenceHit(TypedDict):
    hit_id: str
    project_id: str
    chunk_id: str
    duplicate_chunk_ids: list[str]
    source_id: str
    source_ids: list[str]
    repo: str
    path: str
    commit_sha: str
    source_type: str
    chunk_type: str
    symbol: str
    score: float
    keyword_score: float
    symbol_score: float
    vector_score: float
    source_agreement_count: int
    search_sources: list[str]
    query_groups: list[str]
    query_matches: list[EvidenceSourceContribution]
    matched_terms: list[str]
    match_reasons: list[str]
    vector_record_ids: list[str]
    summary: str
    keywords: list[str]
    technical_tags: list[str]
    text_hash: str
    text_chars: int
    exact_match_quality: int


class HybridRetrievalCoverage(TypedDict):
    requested_query_groups: list[str]
    groups_with_hits: list[str]
    missing_query_groups: list[str]
    keyword_hit_count: int
    symbol_hit_count: int
    vector_hit_count: int
    hybrid_hit_count: int
    multi_source_hit_count: int
    project_identity_covered: bool
    mechanisms_covered: bool
    symbols_covered: bool
    validation_repair_covered: bool
    metrics_impact_covered: bool
    jd_alignment_covered: bool


class HybridRetrievalResult(TypedDict):
    status: str
    project_id: str
    hits: list[HybridEvidenceHit]
    coverage: HybridRetrievalCoverage
    warnings: list[str]
    errors: list[str]


def _safe_string(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = " ".join(_CONTROL_RE.sub(" ", value).split())
    if not cleaned or _UNSAFE_RE.search(cleaned):
        return ""
    return cleaned[:limit]


def _project_id(value: Any) -> str:
    safe = _safe_string(value, 160)
    return safe if _PROJECT_ID_RE.fullmatch(safe) else ""


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _safe_labels(value: Any, limit: int = MAX_SAFE_LABELS) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    labels: dict[str, str] = {}
    candidates = sorted(value, key=lambda item: (str(item).casefold(), str(item)))[: limit * 4]
    for item in candidates:
        safe = _safe_string(item, 160)
        if not safe or not _SAFE_LABEL_RE.fullmatch(safe):
            continue
        key = safe.casefold()
        current = labels.get(key)
        if current is None or safe < current:
            labels[key] = safe
    return [labels[key] for key in sorted(labels)][:limit]


def _safe_query(value: Any) -> str:
    return _safe_string(value, 180)


def normalize_lexical_retrieval_score(raw_score: Any) -> float | None:
    """Calibrate a non-negative lexical score against the fixed single-target cap."""

    value = _finite_nonnegative(raw_score)
    if value is None:
        return None
    return round(min(value / KEYWORD_SCORE_CAP, 1.0), 6)


def normalize_symbol_retrieval_score(raw_score: Any) -> float | None:
    """Calibrate a non-negative symbol score against the fixed single-target cap."""

    value = _finite_nonnegative(raw_score)
    if value is None:
        return None
    return round(min(value / SYMBOL_SCORE_CAP, 1.0), 6)


def _exact_quality(reasons: Sequence[str]) -> int:
    quality = 0
    for reason in reasons:
        code = reason.split(":", 1)[0].casefold()
        quality = max(quality, {
            "symbol_exact": 5,
            "path_exact": 4,
            "identifier_exact": 4,
            "keyword_exact": 3,
            "technical_tag": 3,
            "repository_exact": 2,
            "summary_phrase": 2,
            "summary_match": 1,
        }.get(code, 0))
    return quality


def normalize_hybrid_source_hit(
    hit: Any, *, project_id: Any,
) -> NormalizedHybridSourceHit | None:
    """Convert an existing lexical, symbol, or vector hit into one safe source shape."""

    requested_project = _project_id(project_id)
    if not requested_project or not isinstance(hit, Mapping):
        return None
    hit_project = _project_id(hit.get("project_id"))
    if not hit_project or hit_project.casefold() != requested_project.casefold():
        return None
    search_source = _safe_string(hit.get("search_source"), 16).casefold()
    query_group = _safe_string(hit.get("query_group"), 64)
    if search_source not in _ALLOWED_SOURCES or query_group not in QUERY_GROUPS:
        return None
    chunk_id = _safe_string(hit.get("chunk_id"), 160)
    source_id = _safe_string(hit.get("source_id"), 160)
    query = _safe_query(hit.get("query"))
    if not chunk_id or not source_id or not query:
        return None
    raw_score = _finite_nonnegative(hit.get("score"))
    if raw_score is None or raw_score <= 0:
        return None
    if search_source == "keyword":
        source_score = normalize_lexical_retrieval_score(raw_score)
        raw_score = min(raw_score, 612.0)
    elif search_source == "symbol":
        source_score = normalize_symbol_retrieval_score(raw_score)
        raw_score = min(raw_score, 496.0)
    else:
        if raw_score > 1:
            return None
        source_score = round(raw_score, 6)
    if source_score is None or source_score <= 0:
        return None
    matched_terms = _safe_labels(hit.get("matched_terms"), MAX_HYBRID_MATCHED_TERMS)
    match_reasons = _safe_labels(hit.get("match_reasons"), MAX_HYBRID_MATCH_REASONS)
    vector_record_id: str | None = None
    vector_rank: int | None = None
    if search_source == "vector":
        vector_record_id = _safe_string(hit.get("vector_record_id"), 180)
        rank = hit.get("vector_rank")
        if not vector_record_id or isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            return None
        vector_rank = rank
    text_hash = _safe_string(hit.get("text_hash"), 128)
    if text_hash and not _HASH_RE.fullmatch(text_hash):
        text_hash = ""
    text_chars = hit.get("text_chars")
    if isinstance(text_chars, bool) or not isinstance(text_chars, int) or text_chars < 0:
        text_chars = 0
    contribution: EvidenceSourceContribution = {
        "search_source": search_source,
        "query_group": query_group,
        "query": query,
        "source_score": source_score,
        "raw_score": round(raw_score, 6),
        "matched_terms": matched_terms,
        "match_reasons": match_reasons,
        "vector_record_id": vector_record_id,
        "vector_rank": vector_rank,
    }
    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "project_id": hit_project,
        "repo": _safe_string(hit.get("repo"), 160),
        "path": _safe_string(hit.get("path"), 300),
        "commit_sha": _safe_string(hit.get("commit_sha"), 160),
        "source_type": _safe_string(hit.get("source_type"), 64),
        "chunk_type": _safe_string(hit.get("chunk_type"), 64),
        "symbol": _safe_string(hit.get("symbol"), 128),
        "summary": _safe_string(hit.get("summary"), MAX_HYBRID_SUMMARY_CHARS),
        "keywords": _safe_labels(hit.get("keywords")),
        "technical_tags": _safe_labels(hit.get("technical_tags")),
        "text_hash": text_hash.casefold(),
        "text_chars": text_chars,
        "contribution": contribution,
        "exact_match_quality": _exact_quality(match_reasons),
    }


def _source_identity(hit: NormalizedHybridSourceHit) -> tuple[str, ...]:
    return (
        hit["project_id"].casefold(), hit["repo"].replace("\\", "/").strip("/").casefold(),
        hit["path"].replace("\\", "/").strip("/").casefold(),
        hit["source_type"].casefold(), hit["chunk_type"].casefold(), hit["symbol"].casefold(),
        hit["text_hash"], str(hit["text_chars"]),
    )


def _contribution_key(value: EvidenceSourceContribution) -> tuple[Any, ...]:
    return (
        _SOURCE_ORDER[value["search_source"]],
        _GROUP_PRIORITY.get(value["query_group"], len(_GROUP_PRIORITY)),
        value["query"].casefold(), value["vector_record_id"] or "", value["vector_rank"] or 0,
        -value["source_score"],
    )


def _final_score(contributions: Sequence[EvidenceSourceContribution]) -> tuple[float, float, float, float]:
    keyword = max((item["source_score"] for item in contributions if item["search_source"] == "keyword"), default=0.0)
    symbol = max((item["source_score"] for item in contributions if item["search_source"] == "symbol"), default=0.0)
    vector = max((item["source_score"] for item in contributions if item["search_source"] == "vector"), default=0.0)
    sources = {item["search_source"] for item in contributions}
    queries = {(item["query_group"], item["query"].casefold()) for item in contributions}
    groups = {item["query_group"] for item in contributions}
    base = max(keyword * KEYWORD_WEIGHT, symbol * SYMBOL_WEIGHT, vector * VECTOR_WEIGHT)
    agreement = AGREEMENT_BONUS_PER_ADDITIONAL_SOURCE * min(max(len(sources) - 1, 0), MAX_AGREEMENT_BONUS_SOURCES)
    diversity = QUERY_DIVERSITY_BONUS * min(max(len(queries) - 1, 0), MAX_QUERY_DIVERSITY_BONUS)
    factor = JD_ALIGNMENT_ONLY_FACTOR if groups == {"jd_alignment"} else 1.0
    score = min(max((base + agreement + diversity) * factor, 0.0), 1.0)
    return round(score, 6), round(keyword, 6), round(symbol, 6), round(vector, 6)


def _build_merged_hit(values: Sequence[NormalizedHybridSourceHit]) -> HybridEvidenceHit:
    ordered = sorted(values, key=lambda item: (item["chunk_id"], item["source_id"], _source_identity(item), _contribution_key(item["contribution"])))
    primary = ordered[0]
    contributions_by_key: dict[tuple[Any, ...], EvidenceSourceContribution] = {}
    for item in ordered:
        contribution = item["contribution"]
        key = _contribution_key(contribution)[:-1]
        current = contributions_by_key.get(key)
        if current is None or contribution["source_score"] > current["source_score"]:
            contributions_by_key[key] = contribution
    contributions = sorted(contributions_by_key.values(), key=_contribution_key)[:MAX_HYBRID_QUERY_MATCHES]
    score, keyword_score, symbol_score, vector_score = _final_score(contributions)
    chunk_ids = sorted({item["chunk_id"] for item in ordered})
    source_ids = sorted({item["source_id"] for item in ordered})[:MAX_SOURCE_IDS_PER_HIT]
    search_sources = sorted({item["search_source"] for item in contributions}, key=lambda item: _SOURCE_ORDER[item])
    query_groups = sorted({item["query_group"] for item in contributions}, key=lambda item: _GROUP_PRIORITY[item])
    vector_record_ids = sorted({item["vector_record_id"] for item in contributions if item["vector_record_id"]})[:MAX_VECTOR_RECORD_IDS_PER_HIT]
    terms = _safe_labels([term for item in contributions for term in item["matched_terms"]], MAX_HYBRID_MATCHED_TERMS)
    reasons = _safe_labels([reason for item in contributions for reason in item["match_reasons"]], MAX_HYBRID_MATCH_REASONS)
    keywords = _safe_labels([label for item in ordered for label in item["keywords"]])
    tags = _safe_labels([label for item in ordered for label in item["technical_tags"]])
    summaries = sorted({item["summary"] for item in ordered if item["summary"]}, key=lambda item: (-len(item), item.casefold(), item))
    hit_id = evidence_memory.stable_record_id("hyb", [primary["project_id"], chunk_ids[0], chunk_ids[1:]])
    return {
        "hit_id": hit_id,
        "project_id": primary["project_id"],
        "chunk_id": chunk_ids[0],
        "duplicate_chunk_ids": chunk_ids[1:1 + MAX_DUPLICATE_CHUNK_IDS],
        "source_id": source_ids[0],
        "source_ids": source_ids,
        "repo": primary["repo"],
        "path": primary["path"],
        "commit_sha": primary["commit_sha"],
        "source_type": primary["source_type"],
        "chunk_type": primary["chunk_type"],
        "symbol": primary["symbol"],
        "score": score,
        "keyword_score": keyword_score,
        "symbol_score": symbol_score,
        "vector_score": vector_score,
        "source_agreement_count": len(search_sources),
        "search_sources": search_sources,
        "query_groups": query_groups,
        "query_matches": contributions,
        "matched_terms": terms,
        "match_reasons": reasons,
        "vector_record_ids": vector_record_ids,
        "summary": summaries[0] if summaries else "",
        "keywords": keywords,
        "technical_tags": tags,
        "text_hash": primary["text_hash"],
        "text_chars": primary["text_chars"],
        "exact_match_quality": max(item["exact_match_quality"] for item in ordered),
    }


def _merge_normalized(values: Sequence[NormalizedHybridSourceHit], maximum: int) -> list[HybridEvidenceHit]:
    identities_by_chunk: dict[str, set[tuple[str, ...]]] = {}
    for item in values:
        identities_by_chunk.setdefault(item["chunk_id"], set()).add(_source_identity(item))
    safe_values = [item for item in values if len(identities_by_chunk[item["chunk_id"]]) == 1]
    grouped: dict[tuple[str, ...], list[NormalizedHybridSourceHit]] = {}
    for item in safe_values:
        normalized_repo = item["repo"].replace("\\", "/").strip("/").casefold()
        normalized_path = item["path"].replace("\\", "/").strip("/").casefold()
        if item["text_hash"] and normalized_repo and normalized_path:
            key = (
                "content", item["project_id"].casefold(), item["text_hash"], normalized_repo,
                normalized_path, item["source_type"].casefold(), item["chunk_type"].casefold(),
            )
        else:
            key = ("chunk", item["project_id"].casefold(), item["chunk_id"])
        grouped.setdefault(key, []).append(item)
    hits = [_build_merged_hit(grouped[key]) for key in sorted(grouped)]
    return sorted(hits, key=_hybrid_order)[:maximum]


def merge_hybrid_evidence_candidates(
    *, source_hits: Any, project_id: Any, max_candidates: int = MAX_NORMALIZED_CANDIDATES,
) -> list[HybridEvidenceHit]:
    """Normalize and conservatively merge exact chunk or exact permitted-content duplicates."""

    requested_project = _project_id(project_id)
    if not requested_project or not isinstance(source_hits, (list, tuple)):
        return []
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates <= 0:
        return []
    limit = min(max_candidates, MAX_NORMALIZED_CANDIDATES)
    normalized = [
        value for item in source_hits[:MAX_SOURCE_HITS_TOTAL]
        if (value := normalize_hybrid_source_hit(item, project_id=requested_project)) is not None
    ]
    normalized.sort(key=lambda item: (item["chunk_id"], _source_identity(item), _contribution_key(item["contribution"])))
    return _merge_normalized(normalized, limit)


def _hybrid_order(hit: HybridEvidenceHit) -> tuple[Any, ...]:
    best_group = min((_GROUP_PRIORITY[group] for group in hit["query_groups"]), default=len(_GROUP_PRIORITY))
    return (
        -hit["score"], -hit["source_agreement_count"], -hit["exact_match_quality"], best_group,
        -hit["symbol_score"], -hit["vector_score"], -hit["keyword_score"],
        hit["repo"].casefold(), hit["path"].casefold(), hit["symbol"].casefold(),
        hit["chunk_id"], hit["hit_id"],
    )


def _primary_group(hit: HybridEvidenceHit) -> str:
    return min(hit["query_groups"], key=lambda group: _GROUP_PRIORITY[group]) if hit["query_groups"] else ""


def rerank_hybrid_evidence_candidates(
    *, candidates: Any, top_k: Any = DEFAULT_HYBRID_TOP_K,
) -> list[HybridEvidenceHit]:
    """Select deterministic group coverage, then fill with bounded group/path diversity."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        return []
    limit = min(top_k, MAX_HYBRID_TOP_K)
    if not isinstance(candidates, (list, tuple)):
        return []
    valid: list[HybridEvidenceHit] = []
    for item in candidates:
        if not isinstance(item, Mapping) or not _project_id(item.get("project_id")):
            continue
        try:
            _hybrid_order(item)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError):
            continue
        valid.append(item)  # type: ignore[arg-type]
    ranked = sorted(valid, key=_hybrid_order)
    if not ranked:
        return []
    selected: list[HybridEvidenceHit] = []
    selected_ids: set[str] = set()

    def add(hit: HybridEvidenceHit) -> None:
        if hit["hit_id"] not in selected_ids and len(selected) < limit:
            selected.append(hit)
            selected_ids.add(hit["hit_id"])

    add(ranked[0])
    for group in SEMANTIC_GROUP_ORDER[:-1]:
        if any(group in item["query_groups"] for item in selected):
            continue
        candidate = next((item for item in ranked if group in item["query_groups"] and item["score"] >= MIN_GROUP_COVERAGE_SCORE), None)
        if candidate is not None:
            add(candidate)

    jd_cap = max(1, math.floor(limit * MAX_JD_ONLY_FRACTION))
    group_cap = max(1, math.ceil(limit * MAX_PRIMARY_GROUP_FRACTION))
    path_cap = max(1, math.ceil(limit * MAX_PATH_FRACTION))
    for relaxed in (False, True):
        for hit in ranked:
            if hit["hit_id"] in selected_ids or len(selected) >= limit:
                continue
            if not relaxed:
                jd_only = hit["query_groups"] == ["jd_alignment"]
                primary_group = _primary_group(hit)
                path_key = hit["path"].replace("\\", "/").casefold()
                if jd_only and sum(item["query_groups"] == ["jd_alignment"] for item in selected) >= jd_cap:
                    continue
                if primary_group and sum(_primary_group(item) == primary_group for item in selected) >= group_cap:
                    continue
                if path_key and sum(item["path"].replace("\\", "/").casefold() == path_key for item in selected) >= path_cap:
                    continue
            add(hit)
        if len(selected) >= limit:
            break
    return sorted(selected, key=_hybrid_order)[:limit]


def _validated_plan(query_plan: Any, project_id: str) -> ProjectQueryPlan | None:
    if not isinstance(query_plan, Mapping):
        return None
    plan_project = _project_id(query_plan.get("project_id"))
    if not plan_project or plan_project.casefold() != project_id.casefold():
        return None
    remaining = MAX_TOTAL_QUERIES
    groups: dict[str, list[str]] = {}
    for group in QUERY_GROUPS:
        values = query_plan.get(group)
        if not isinstance(values, (list, tuple)):
            groups[group] = []
            continue
        safe = sorted({_safe_query(value) for value in values if _safe_query(value)})[:3]
        groups[group] = safe[:remaining]
        remaining -= len(groups[group])
    return ProjectQueryPlan(project_id=plan_project, **groups)


def build_hybrid_retrieval_coverage(
    *, query_plan: Any, hits: Any, source_hit_counts: Mapping[str, int] | None = None,
) -> HybridRetrievalCoverage:
    requested = [
        group for group in SEMANTIC_GROUP_ORDER
        if isinstance(query_plan, Mapping) and isinstance(query_plan.get(group), (list, tuple)) and bool(query_plan.get(group))
    ]
    selected = [item for item in hits if isinstance(item, Mapping)] if isinstance(hits, (list, tuple)) else []
    covered_set = {group for item in selected for group in item.get("query_groups", []) if group in _GROUP_PRIORITY}
    covered = [group for group in SEMANTIC_GROUP_ORDER if group in covered_set]
    missing = [group for group in requested if group not in covered_set]
    counts = source_hit_counts if isinstance(source_hit_counts, Mapping) else {}
    def safe_count(source: str) -> int:
        value = counts.get(source, 0)
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0

    result: HybridRetrievalCoverage = {
        "requested_query_groups": requested,
        "groups_with_hits": covered,
        "missing_query_groups": missing,
        "keyword_hit_count": safe_count("keyword"),
        "symbol_hit_count": safe_count("symbol"),
        "vector_hit_count": safe_count("vector"),
        "hybrid_hit_count": len(selected),
        "multi_source_hit_count": sum(int(item.get("source_agreement_count", 0)) > 1 for item in selected),
        "project_identity_covered": "project_identity" in covered_set,
        "mechanisms_covered": "mechanisms" in covered_set,
        "symbols_covered": "symbols" in covered_set,
        "validation_repair_covered": "validation_repair" in covered_set,
        "metrics_impact_covered": "metrics_impact" in covered_set,
        "jd_alignment_covered": "jd_alignment" in covered_set,
    }
    return result


def _empty_coverage() -> HybridRetrievalCoverage:
    return build_hybrid_retrieval_coverage(query_plan={}, hits=[], source_hit_counts={})


def _result(
    status: str, project_id: str, *, hits: Sequence[HybridEvidenceHit] = (),
    coverage: HybridRetrievalCoverage | None = None,
    warnings: Sequence[str] = (), errors: Sequence[str] = (),
) -> HybridRetrievalResult:
    return {
        "status": status,
        "project_id": project_id,
        "hits": list(hits)[:MAX_HYBRID_TOP_K],
        "coverage": coverage or _empty_coverage(),
        "warnings": sorted(set(warnings))[:MAX_WARNINGS],
        "errors": sorted(set(errors))[:MAX_ERRORS],
    }


def _readiness_ready(readiness: Any, project_id: str) -> bool:
    if not isinstance(readiness, Mapping):
        return False
    if not (
        readiness.get("status") == "ready"
        and readiness.get("vector_ready") is True
        and readiness.get("chunk_mapping_ready") is True
        and readiness.get("ready_for_hybrid_retrieval") is True
    ):
        return False
    projects = readiness.get("projects_with_chunks")
    return isinstance(projects, (list, tuple)) and any(
        isinstance(item, str) and item.casefold() == project_id.casefold() for item in projects
    )


def run_project_hybrid_retrieval(
    *,
    project_id: Any,
    query_plan: Any,
    chunks: Any,
    vector_search: Callable[..., Any],
    readiness: Any,
    top_k: Any = DEFAULT_HYBRID_TOP_K,
    keyword_top_k_per_query: Any = 10,
    symbol_top_k_per_query: Any = 10,
    vector_top_k_per_query: Any = 5,
) -> HybridRetrievalResult:
    """Run bounded hybrid retrieval without persistence, fallback, or resume integration."""

    requested_project = _project_id(project_id)
    if not requested_project:
        return _result("blocked", "", errors=["invalid_project_identity"])
    if not _readiness_ready(readiness, requested_project):
        return _result("blocked", requested_project, errors=["hybrid_readiness_required"])
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        return _result("blocked", requested_project, errors=["invalid_top_k"])
    limit = min(top_k, MAX_HYBRID_TOP_K)
    plan = _validated_plan(query_plan, requested_project)
    if plan is None:
        return _result("blocked", requested_project, errors=["invalid_query_plan"])
    if not any(plan[group] for group in QUERY_GROUPS):
        return _result("empty", requested_project, coverage=build_hybrid_retrieval_coverage(query_plan=plan, hits=[], source_hit_counts={}))
    if not isinstance(chunks, (list, tuple)) or not chunks:
        return _result("empty", requested_project, coverage=build_hybrid_retrieval_coverage(query_plan=plan, hits=[], source_hit_counts={}))
    if not callable(vector_search):
        return _result("blocked", requested_project, errors=["vector_adapter_required"])
    for value, code in (
        (keyword_top_k_per_query, "invalid_keyword_limit"),
        (symbol_top_k_per_query, "invalid_symbol_limit"),
        (vector_top_k_per_query, "invalid_vector_limit"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return _result("blocked", requested_project, errors=[code])
    try:
        lexical = search_project_query_plan(
            chunks=chunks, query_plan=plan, project_id=requested_project,
            top_k_per_query=max(keyword_top_k_per_query, symbol_top_k_per_query),
        )
        raw_hits: list[EvidenceSearchHit | VectorEvidenceHit] = []
        for group in QUERY_GROUPS:
            group_hits = lexical.get(group, []) if isinstance(lexical, Mapping) else []
            if isinstance(group_hits, list):
                source_limit = symbol_top_k_per_query if group == "symbols" else keyword_top_k_per_query
                raw_hits.extend(group_hits[: min(source_limit * max(len(plan[group]), 1), 150)])
        raw_hits.extend(vector_search_project_query_plan(
            query_plan=plan, project_id=requested_project, chunks=chunks,
            vector_search=vector_search, top_k_per_query=vector_top_k_per_query,
        ))
        normalized = [
            value for item in raw_hits
            if (value := normalize_hybrid_source_hit(item, project_id=requested_project)) is not None
        ]
        normalized.sort(key=lambda item: (item["chunk_id"], _source_identity(item), _contribution_key(item["contribution"])))
        warnings: list[str] = []
        if len(normalized) > MAX_SOURCE_HITS_TOTAL:
            warnings.append("source_hit_limit_reached")
            normalized = normalized[:MAX_SOURCE_HITS_TOTAL]
        source_counts = Counter(item["contribution"]["search_source"] for item in normalized)
        candidates = _merge_normalized(normalized, MAX_NORMALIZED_CANDIDATES)
        if len({item["chunk_id"] for item in normalized}) > MAX_NORMALIZED_CANDIDATES:
            warnings.append("candidate_limit_reached")
        selected = rerank_hybrid_evidence_candidates(candidates=candidates, top_k=limit)
        coverage = build_hybrid_retrieval_coverage(
            query_plan=plan, hits=selected, source_hit_counts=source_counts,
        )
        if not selected:
            return _result("empty", requested_project, coverage=coverage, warnings=warnings)
        return _result("ready", requested_project, hits=selected, coverage=coverage, warnings=warnings)
    except Exception:
        return _result("error", requested_project, errors=["hybrid_retrieval_failed"])
