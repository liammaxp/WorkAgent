"""Deterministic project-scoped lexical search over evidence chunks."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence, TypedDict

from backend import evidence_memory
from backend.project_query_planner import QUERY_GROUPS


MAX_SEARCH_QUERY_CHARS = 180
MAX_SEARCH_TERMS = 16
MAX_SYMBOL_SEARCH_ITEMS = 16
MAX_MATCHED_TERMS = 16
MAX_MATCH_REASONS = 16
MAX_HIT_SUMMARY_CHARS = 500
MAX_INTERNAL_TEXT_CHARS = 3600
MAX_SAFE_LABELS = 32
DEFAULT_TOP_K = 10
MAX_TOP_K = 50
MIN_SYMBOL_CHARS = 4

SYMBOL_EXACT_WEIGHT = 12.0
PATH_EXACT_WEIGHT = 8.0
KEYWORD_EXACT_WEIGHT = 6.0
TECHNICAL_TAG_WEIGHT = 6.0
SUMMARY_TOKEN_WEIGHT = 4.0
TEXT_TOKEN_WEIGHT = 2.0
PHRASE_MATCH_WEIGHT = 4.0
IDENTIFIER_TOKEN_WEIGHT = 5.0
SYMBOL_COMPONENT_WEIGHT = 2.0
GENERIC_TERM_MULTIPLIER = 0.25

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.#/-]{1,127}")
_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9_+.#/-]{1,128}$")
_SYMBOL_RE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]{2,127}|[A-Za-z_$][A-Za-z0-9_$]{2,127}|"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|"
    r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.(?:py|js|jsx|ts|tsx|json|md|sql|yaml|yml)|"
    r"[A-Za-z0-9_.-]{1,120}\.(?:py|js|jsx|ts|tsx|json|md|sql|yaml|yml)|"
    r"/[A-Za-z0-9_./{}:-]{2,127})$"
)
_UNSAFE_RE = re.compile(
    r"(?i)(?:diff\s+--git|begin\s+(?:rsa\s+)?private\s+key|api[_-]?key\s*=|"
    r"access[_-]?token\s*=|password\s*=|secret\s*=|credential\s*=)"
)
_SENSITIVE_LABEL_RE = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|password|secret|credential|private[_-]?key)"
)
_GENERIC_TERMS = frozenset({
    "ai", "analysis", "application", "generate", "improve", "project", "resume", "system",
})
_STOPWORDS = frozenset({
    "and", "are", "for", "from", "into", "of", "or", "that", "the", "this", "to", "with",
})


class EvidenceSearchHit(TypedDict):
    hit_id: str
    chunk_id: str
    source_id: str
    project_id: str
    repo: str
    path: str
    commit_sha: str
    source_type: str
    chunk_type: str
    symbol: str
    search_source: str
    query_group: str
    query: str
    score: float
    matched_terms: list[str]
    match_reasons: list[str]
    summary: str
    keywords: list[str]
    technical_tags: list[str]
    text_hash: str
    text_chars: int


def _safe_string(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = " ".join(_CONTROL_RE.sub(" ", value).split())
    if not cleaned or _UNSAFE_RE.search(cleaned):
        return ""
    return cleaned[:limit]


def _safe_labels(value: Any, limit: int = MAX_SAFE_LABELS) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    labels: dict[str, str] = {}
    for item in value[: limit * 4]:
        if not isinstance(item, str):
            continue
        label = item.strip()
        if (
            not _SAFE_LABEL_RE.fullmatch(label)
            or _UNSAFE_RE.search(label)
            or _SENSITIVE_LABEL_RE.search(label)
        ):
            continue
        key = label.casefold()
        current = labels.get(key)
        if current is None or label < current:
            labels[key] = label
    return [labels[key] for key in sorted(labels)][:limit]


def _query(value: Any) -> tuple[str, list[str]]:
    safe = _safe_string(value, MAX_SEARCH_QUERY_CHARS)
    if not safe:
        return "", []
    terms: dict[str, str] = {}
    for token in _TOKEN_RE.findall(safe):
        normalized = token.strip("/.-").casefold()
        if not normalized or normalized in _STOPWORDS or _UNSAFE_RE.search(normalized):
            continue
        terms.setdefault(normalized, normalized)
    normalized_terms = [terms[key] for key in sorted(terms)][:MAX_SEARCH_TERMS]
    if not normalized_terms:
        return "", []
    return " ".join(normalized_terms)[:MAX_SEARCH_QUERY_CHARS], normalized_terms


def _top_k(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return 0
    return min(value, MAX_TOP_K)


def _chunk(value: Any, project_id: str) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    chunk_id = _safe_string(value.get("chunk_id"), 160)
    source_id = _safe_string(value.get("source_id"), 160)
    chunk_project_id = _safe_string(value.get("project_id"), 160)
    if not chunk_id or not source_id or not chunk_project_id:
        return None
    if chunk_project_id.casefold() != project_id.casefold():
        return None
    text = value.get("text", "")
    if not isinstance(text, str):
        text = ""
    text = text[:MAX_INTERNAL_TEXT_CHARS]
    summary = _safe_string(value.get("summary"), MAX_HIT_SUMMARY_CHARS)
    symbol = _safe_string(value.get("symbol"), 128)
    if symbol and not _SAFE_LABEL_RE.fullmatch(symbol):
        symbol = ""
    text_chars = value.get("text_chars", len(text))
    if isinstance(text_chars, bool) or not isinstance(text_chars, int) or text_chars < 0:
        text_chars = len(text)
    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "project_id": chunk_project_id,
        "repo": _safe_string(value.get("repo"), 160),
        "path": _safe_string(value.get("path"), 300),
        "commit_sha": _safe_string(value.get("commit_sha"), 160),
        "source_type": _safe_string(value.get("source_type"), 64),
        "chunk_type": _safe_string(value.get("chunk_type"), 64),
        "symbol": symbol,
        "text": text,
        "summary": summary,
        "keywords": _safe_labels(value.get("keywords")),
        "technical_tags": _safe_labels(value.get("technical_tags")),
        "text_hash": _safe_string(value.get("text_hash"), 160),
        "text_chars": text_chars,
    }


def _field_tokens(value: str) -> set[str]:
    return {token.strip("/.-").casefold() for token in _TOKEN_RE.findall(value) if token.strip("/.-")}


def _path_values(path: str) -> set[str]:
    normalized = path.replace("\\", "/").casefold()
    values = {normalized}
    values.update(part for part in normalized.split("/") if part)
    return values


def _reason(code: str, term: str) -> str:
    return f"{code}:{term}"[:160]


def _build_hit(
    chunk: Mapping[str, Any],
    *,
    search_source: str,
    query_group: str,
    query: str,
    score: float,
    matched_terms: Sequence[str],
    match_reasons: Sequence[str],
) -> EvidenceSearchHit:
    bounded_score = round(float(score), 4)
    if not math.isfinite(bounded_score) or bounded_score <= 0:
        raise ValueError("search hit score must be finite and positive")
    terms = sorted(set(matched_terms), key=lambda item: (item.casefold(), item))[:MAX_MATCHED_TERMS]
    reasons = sorted(set(match_reasons))[:MAX_MATCH_REASONS]
    return {
        "hit_id": evidence_memory.stable_record_id(
            "hit", [chunk["chunk_id"], search_source, query_group, query]
        ),
        "chunk_id": chunk["chunk_id"],
        "source_id": chunk["source_id"],
        "project_id": chunk["project_id"],
        "repo": chunk["repo"],
        "path": chunk["path"],
        "commit_sha": chunk["commit_sha"],
        "source_type": chunk["source_type"],
        "chunk_type": chunk["chunk_type"],
        "symbol": chunk["symbol"],
        "search_source": search_source,
        "query_group": _safe_string(query_group, 64),
        "query": query,
        "score": bounded_score,
        "matched_terms": terms,
        "match_reasons": reasons,
        "summary": chunk["summary"],
        "keywords": chunk["keywords"],
        "technical_tags": chunk["technical_tags"],
        "text_hash": chunk["text_hash"],
        "text_chars": chunk["text_chars"],
    }


def _sort_hits(scored: Sequence[tuple[EvidenceSearchHit, int]]) -> list[EvidenceSearchHit]:
    return [
        hit
        for hit, _quality in sorted(
            scored,
            key=lambda item: (
                -item[0]["score"],
                -item[1],
                item[0]["path"].casefold(),
                item[0]["symbol"].casefold(),
                item[0]["chunk_id"],
            ),
        )
    ]


def keyword_search_evidence_chunks(
    *,
    chunks: Any,
    query: Any,
    project_id: Any,
    query_group: Any = "mechanisms",
    top_k: Any = DEFAULT_TOP_K,
) -> list[EvidenceSearchHit]:
    requested_project = _safe_string(project_id, 160)
    safe_query, terms = _query(query)
    limit = _top_k(top_k)
    if not requested_project or not safe_query or not terms or not limit or not isinstance(chunks, (list, tuple)):
        return []
    scored: list[tuple[EvidenceSearchHit, int]] = []
    for value in chunks:
        item = _chunk(value, requested_project)
        if item is None:
            continue
        symbol = item["symbol"].casefold()
        path_values = _path_values(item["path"])
        keywords = {label.casefold() for label in item["keywords"]}
        tags = {label.casefold() for label in item["technical_tags"]}
        summary_tokens = _field_tokens(item["summary"])
        text_tokens = _field_tokens(item["text"])
        summary_lower = item["summary"].casefold()
        text_lower = item["text"].casefold()
        score = 0.0
        quality = 0
        matched: list[str] = []
        reasons: list[str] = []
        for term in terms:
            multiplier = GENERIC_TERM_MULTIPLIER if term in _GENERIC_TERMS else 1.0
            term_score = 0.0
            if term == symbol:
                term_score += SYMBOL_EXACT_WEIGHT
                quality += 4
                reasons.append(_reason("symbol_exact", term))
            if term in path_values:
                term_score += PATH_EXACT_WEIGHT
                quality += 3
                reasons.append(_reason("path_exact", term))
            if term in keywords:
                term_score += KEYWORD_EXACT_WEIGHT
                quality += 2
                reasons.append(_reason("keyword_exact", term))
            if term in tags:
                term_score += TECHNICAL_TAG_WEIGHT
                quality += 2
                reasons.append(_reason("technical_tag", term))
            if term in summary_tokens:
                term_score += SUMMARY_TOKEN_WEIGHT
                quality += 1
                reasons.append(_reason("summary_match", term))
            if term in text_tokens:
                term_score += TEXT_TOKEN_WEIGHT
                reasons.append(_reason("text_token", term))
            if term_score:
                matched.append(term)
                score += term_score * multiplier
        if len(terms) > 1 and safe_query in summary_lower:
            score += PHRASE_MATCH_WEIGHT
            quality += 1
            reasons.append("summary_phrase")
        elif len(terms) > 1 and safe_query in text_lower:
            score += PHRASE_MATCH_WEIGHT
            reasons.append("text_phrase")
        if score <= 0 or not matched:
            continue
        hit = _build_hit(
            item,
            search_source="keyword",
            query_group=_safe_string(query_group, 64),
            query=safe_query,
            score=score,
            matched_terms=matched,
            match_reasons=reasons,
        )
        scored.append((hit, quality))
    return _sort_hits(scored)[:limit]


def _safe_symbols(value: Any) -> list[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple)):
        candidates = list(value[: MAX_SYMBOL_SEARCH_ITEMS * 4])
    else:
        return []
    symbols: dict[str, str] = {}
    for value in candidates:
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        if (
            len(candidate) < MIN_SYMBOL_CHARS
            or not _SYMBOL_RE.fullmatch(candidate)
            or _SENSITIVE_LABEL_RE.search(candidate)
        ):
            continue
        key = candidate.casefold()
        current = symbols.get(key)
        if current is None or candidate < current:
            symbols[key] = candidate
    return [symbols[key] for key in sorted(symbols)][:MAX_SYMBOL_SEARCH_ITEMS]


def symbol_search_evidence_chunks(
    *,
    chunks: Any,
    symbols: Any,
    project_id: Any,
    query_group: Any = "symbols",
    top_k: Any = DEFAULT_TOP_K,
) -> list[EvidenceSearchHit]:
    requested_project = _safe_string(project_id, 160)
    requested_symbols = _safe_symbols(symbols)
    limit = _top_k(top_k)
    if not requested_project or not requested_symbols or not limit or not isinstance(chunks, (list, tuple)):
        return []
    query = " ".join(requested_symbols)[:MAX_SEARCH_QUERY_CHARS]
    scored: list[tuple[EvidenceSearchHit, int]] = []
    for value in chunks:
        item = _chunk(value, requested_project)
        if item is None:
            continue
        chunk_symbol = item["symbol"].casefold()
        path_values = _path_values(item["path"])
        identifiers = _field_tokens(item["text"])
        identifiers.update(label.casefold() for label in item["keywords"])
        score = 0.0
        quality = 0
        matched: list[str] = []
        reasons: list[str] = []
        for symbol in requested_symbols:
            normalized = symbol.casefold()
            components = {
                part for part in re.split(r"[_./-]+", normalized) if len(part) >= MIN_SYMBOL_CHARS
            }
            symbol_score = 0.0
            if normalized == chunk_symbol:
                symbol_score += SYMBOL_EXACT_WEIGHT
                quality += 5
                reasons.append(_reason("symbol_exact", normalized))
            if normalized in path_values:
                symbol_score += PATH_EXACT_WEIGHT
                quality += 4
                reasons.append(_reason("path_exact", normalized))
            if normalized in identifiers:
                symbol_score += IDENTIFIER_TOKEN_WEIGHT
                quality += 2
                reasons.append(_reason("identifier_exact", normalized))
            component_matches = sorted(components & identifiers)
            if len(normalized) >= 6 and component_matches:
                symbol_score += SYMBOL_COMPONENT_WEIGHT * min(len(component_matches), 3)
                reasons.extend(_reason("symbol_component", term) for term in component_matches[:3])
            if symbol_score:
                score += symbol_score
                matched.append(normalized)
        if score <= 0 or not matched:
            continue
        hit = _build_hit(
            item,
            search_source="symbol",
            query_group=_safe_string(query_group, 64),
            query=query,
            score=score,
            matched_terms=matched,
            match_reasons=reasons,
        )
        scored.append((hit, quality))
    return _sort_hits(scored)[:limit]


def search_project_query_plan(
    *,
    chunks: Any,
    query_plan: Any,
    project_id: Any = None,
    top_k_per_query: Any = DEFAULT_TOP_K,
) -> dict[str, list[EvidenceSearchHit]]:
    """Search each plan query independently without cross-query merging."""

    if not isinstance(query_plan, Mapping):
        return {group: [] for group in QUERY_GROUPS}
    requested_project = project_id if project_id is not None else query_plan.get("project_id")
    results: dict[str, list[EvidenceSearchHit]] = {group: [] for group in QUERY_GROUPS}
    for group in QUERY_GROUPS:
        queries = query_plan.get(group)
        if not isinstance(queries, (list, tuple)):
            continue
        for query in queries[:3]:
            if group == "symbols":
                hits = symbol_search_evidence_chunks(
                    chunks=chunks,
                    symbols=_safe_symbols(str(query).split()) if isinstance(query, str) else [],
                    project_id=requested_project,
                    query_group=group,
                    top_k=top_k_per_query,
                )
            else:
                hits = keyword_search_evidence_chunks(
                    chunks=chunks,
                    query=query,
                    project_id=requested_project,
                    query_group=group,
                    top_k=top_k_per_query,
                )
            results[group].extend(hits)
    return results
