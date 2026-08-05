"""Deterministic, read-only quality evaluation for legacy and v2 retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from backend.project_query_planner import QUERY_GROUPS
from backend.project_repository_identity import normalize_project_id, normalize_repository_identity


MAX_EVALUATION_RESULTS = 50
MAX_EVALUATION_ANCHORS = 64
MAX_EVALUATION_LABELS = 32
MAX_EVALUATION_SUMMARY_CHARS = 600
MAX_EVALUATION_SERIALIZED_CHARS = 50_000
MAX_EVALUATION_WARNINGS = 16
MAX_EVALUATION_ERRORS = 8
APPROXIMATE_CHARACTERS_PER_TOKEN = 4
MATERIAL_ANCHOR_COVERAGE_DROP = 0.20

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.-]+")
_SECRET_RE = re.compile(
    r"(?i)(?:diff\s+--git|begin\s+(?:rsa\s+)?private\s+key|api[_-]?key\s*[:=]|"
    r"access[_-]?token\s*[:=]|password\s*[:=]|credential\s*[:=]|"
    r"full\s+patch|full\s+source\s+body|vector\s+document)"
)
_FORBIDDEN_FIELDS = frozenset({
    "body", "content", "diff", "diff_body", "document", "documents", "embedding",
    "embeddings", "patch", "patch_body", "raw", "raw_text", "readme", "readme_body",
    "snippet", "text", "token", "tokens",
})
_SAFE_SOURCES = frozenset({"keyword", "legacy", "symbol", "vector"})
_PROJECT_FIELDS = {
    "mechanism": ("confirmed_features", "key_modules", "mechanisms", "workflows"),
    "metric": ("real_metrics",),
    "symbol": ("known_symbols", "symbols"),
    "technology": ("tech_stack", "technologies", "tools"),
    "validation": ("testing", "validation", "validation_repair"),
}
_JD_FIELDS = frozenset({"keywords", "requirements", "skills", "technologies", "validation"})


class RetrievalEvidenceAnchor(TypedDict):
    anchor_id: str
    anchor_type: str
    value: str
    evidence_required: bool


class RetrievalEvaluationItem(TypedDict):
    retrieval_source: str
    project_id: str
    chunk_id: str | None
    source_id: str | None
    repo: str
    path: str
    symbol: str
    source_type: str
    chunk_type: str
    score: float
    search_sources: list[str]
    query_groups: list[str]
    keywords: list[str]
    technical_tags: list[str]
    match_reasons: list[str]
    summary: str
    text_hash: str
    text_chars: int


class RetrievalNormalizationResult(TypedDict):
    retrieval_source: str
    project_id: str
    items: list[RetrievalEvaluationItem]
    input_result_count: int
    cross_project_result_count: int
    unauthorized_repository_count: int
    forbidden_field_count: int
    secret_marker_count: int
    invalid_score_count: int
    warnings: list[str]
    errors: list[str]


class RetrievalQualityMetrics(TypedDict):
    status: str
    result_count: int
    unique_result_count: int
    duplicate_count: int
    duplicate_rate: float
    unique_chunk_count: int
    unique_source_count: int
    unique_path_count: int
    unique_symbol_count: int
    repository_count: int
    cross_project_result_count: int
    unauthorized_repository_count: int
    forbidden_field_count: int
    secret_marker_count: int
    invalid_score_count: int
    minimum_score: float
    maximum_score: float
    mean_score: float
    keyword_provenance_count: int
    symbol_provenance_count: int
    vector_provenance_count: int
    multi_source_count: int
    requested_query_groups: list[str]
    covered_query_groups: list[str]
    missing_query_groups: list[str]
    path_diversity: float
    source_type_diversity: int
    chunk_type_diversity: int
    technical_tag_coverage: int
    known_symbol_coverage: float
    supported_anchor_coverage: float
    project_identity_covered: bool
    matched_anchor_ids: list[str]
    serialized_character_count: int
    estimated_context_budget: dict[str, Any]
    deterministic_output: bool


class RetrievalComparisonResult(TypedDict):
    status: str
    project_id: str
    legacy: RetrievalQualityMetrics
    v2: RetrievalQualityMetrics
    comparison: dict[str, Any]
    warnings: list[str]
    errors: list[str]


class RetrievalBenchmarkCase(TypedDict):
    case_id: str
    project_id: str
    retrieved_ids: list[str]
    relevant_ids: list[str]


def _safe_string(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = " ".join(_CONTROL_RE.sub(" ", value).split())
    if not cleaned or _SECRET_RE.search(cleaned):
        return ""
    return cleaned[:limit]


def _safe_labels(value: Any, *, allowed: frozenset[str] | None = None) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    labels: dict[str, str] = {}
    candidates = sorted(value, key=lambda item: (str(item).casefold(), str(item)))
    for item in candidates[: MAX_EVALUATION_LABELS * 4]:
        safe = _safe_string(item, 160)
        normalized = safe.casefold()
        if not safe or (allowed is not None and normalized not in allowed):
            continue
        labels[normalized] = safe
    return [labels[key] for key in sorted(labels)][:MAX_EVALUATION_LABELS]


def _structured_strings(value: Any, *, depth: int = 0) -> list[str]:
    if depth > 2:
        return []
    if isinstance(value, str):
        safe = _safe_string(value, 160)
        return [safe] if safe else []
    if isinstance(value, Mapping):
        output: list[str] = []
        for key in sorted(value, key=lambda item: str(item).casefold())[:MAX_EVALUATION_LABELS]:
            safe_key = _safe_string(str(key), 80)
            if safe_key:
                output.append(safe_key)
            output.extend(_structured_strings(value[key], depth=depth + 1))
        return output[: MAX_EVALUATION_LABELS * 2]
    if isinstance(value, (list, tuple, set, frozenset)):
        output = []
        for item in sorted(value, key=lambda candidate: (str(candidate).casefold(), str(candidate)))[
            :MAX_EVALUATION_LABELS
        ]:
            output.extend(_structured_strings(item, depth=depth + 1))
        return output[: MAX_EVALUATION_LABELS * 2]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return [str(value)] if math.isfinite(number) else []
    return []


def build_supported_evidence_anchors(
    *,
    project: Any,
    repository: Any,
    jd_targets: Any = None,
    known_symbols: Any = None,
) -> list[RetrievalEvidenceAnchor]:
    """Derive stable anchors only from structured project, identity, and JD inputs."""

    if not isinstance(project, Mapping):
        return []
    project_id = normalize_project_id(project.get("project_id"))
    repo = normalize_repository_identity(repository)
    if not project_id or not repo:
        return []
    candidates: list[tuple[str, str]] = [
        ("project_identity", project_id),
        ("project_identity", repo),
    ]
    title = _safe_string(project.get("project_name") or project.get("name"), 160)
    if title:
        candidates.append(("project_identity", title))
    for anchor_type, fields in _PROJECT_FIELDS.items():
        values: list[str] = []
        for field in fields:
            values.extend(_structured_strings(project.get(field)))
        if anchor_type == "symbol":
            scoped = known_symbols.get(project_id) if isinstance(known_symbols, Mapping) else known_symbols
            values.extend(_structured_strings(scoped))
        candidates.extend((anchor_type, value) for value in values)
    if isinstance(jd_targets, Mapping):
        for field in sorted(_JD_FIELDS):
            candidates.extend(
                ("jd_target", value) for value in _structured_strings(jd_targets.get(field))
            )
    anchors: dict[tuple[str, str], RetrievalEvidenceAnchor] = {}
    for anchor_type, value in candidates:
        safe = _safe_string(value, 160)
        normalized = safe.casefold()
        if not safe or len(normalized) < 2:
            continue
        key = (anchor_type, normalized)
        digest = hashlib.sha256(f"{anchor_type}\0{normalized}".encode("utf-8")).hexdigest()[:24]
        anchors[key] = {
            "anchor_id": f"anc_{digest}",
            "anchor_type": anchor_type,
            "value": safe,
            "evidence_required": True,
        }
    return [anchors[key] for key in sorted(anchors)][:MAX_EVALUATION_ANCHORS]


def _privacy_counts(value: Any, *, budget: list[int], depth: int = 0) -> tuple[int, int]:
    if budget[0] <= 0 or depth > 5:
        return 0, 0
    budget[0] -= 1
    forbidden = 0
    secrets = 0
    if isinstance(value, Mapping):
        for key, item in list(value.items())[:MAX_EVALUATION_LABELS * 4]:
            if str(key).strip().casefold() in _FORBIDDEN_FIELDS:
                forbidden += 1
            nested_forbidden, nested_secrets = _privacy_counts(item, budget=budget, depth=depth + 1)
            forbidden += nested_forbidden
            secrets += nested_secrets
    elif isinstance(value, (list, tuple)):
        for item in value[:MAX_EVALUATION_RESULTS * 4]:
            nested_forbidden, nested_secrets = _privacy_counts(item, budget=budget, depth=depth + 1)
            forbidden += nested_forbidden
            secrets += nested_secrets
    elif isinstance(value, str) and _SECRET_RE.search(value[:4000]):
        secrets += 1
    return forbidden, secrets


def _identity_repository(value: Mapping[str, Any]) -> str:
    for key in ("repository", "repo", "github_repository", "repository_url", "url"):
        if repository := normalize_repository_identity(value.get(key)):
            return repository
    return ""


def _path(value: Mapping[str, Any]) -> str:
    direct = _safe_string(value.get("path"), 400)
    if direct:
        return direct
    for key in ("changed_file_paths", "root_files"):
        values = value.get(key)
        if isinstance(values, (list, tuple)):
            for item in values[:MAX_EVALUATION_LABELS]:
                if safe := _safe_string(item, 400):
                    return safe
    return ""


def _score(value: Mapping[str, Any]) -> tuple[float, int]:
    present = "final_score" in value or "score" in value
    raw = value.get("final_score", value.get("score"))
    if not present:
        return 0.0, 0
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0.0, 1
    number = float(raw)
    if not math.isfinite(number) or number < 0 or number > 1:
        return 0.0, 1
    return round(number, 6), 0


def normalize_retrieval_results(
    *,
    retrieval_source: Any,
    project_id: Any,
    results: Any,
    authorized_repositories: Sequence[str],
) -> RetrievalNormalizationResult:
    """Normalize one provider into a safe common model and reject identity leakage."""

    source = str(retrieval_source).strip().casefold()
    requested_project = normalize_project_id(project_id)
    authorized = {
        repository
        for value in authorized_repositories
        if (repository := normalize_repository_identity(value))
    }
    empty: RetrievalNormalizationResult = {
        "retrieval_source": source if source in {"legacy", "v2"} else "",
        "project_id": requested_project,
        "items": [],
        "input_result_count": 0,
        "cross_project_result_count": 0,
        "unauthorized_repository_count": 0,
        "forbidden_field_count": 0,
        "secret_marker_count": 0,
        "invalid_score_count": 0,
        "warnings": [],
        "errors": [],
    }
    if source not in {"legacy", "v2"} or not requested_project or not authorized:
        empty["errors"] = ["invalid_evaluation_identity"]
        return empty
    if not isinstance(results, (list, tuple)):
        empty["errors"] = ["invalid_retrieval_results"]
        return empty
    empty["input_result_count"] = min(len(results), MAX_EVALUATION_RESULTS * 4)
    items: list[RetrievalEvaluationItem] = []
    for raw in results[: MAX_EVALUATION_RESULTS * 4]:
        if not isinstance(raw, Mapping):
            continue
        forbidden, secrets = _privacy_counts(raw, budget=[1000])
        empty["forbidden_field_count"] += forbidden
        empty["secret_marker_count"] += secrets
        repository = _identity_repository(raw)
        explicit_project = normalize_project_id(
            raw.get("project_id") or raw.get("repository_project_id")
        )
        if explicit_project and explicit_project != requested_project:
            empty["cross_project_result_count"] += 1
            continue
        if repository not in authorized:
            empty["unauthorized_repository_count"] += 1
            continue
        score, invalid_score = _score(raw)
        empty["invalid_score_count"] += invalid_score
        summary = _safe_string(raw.get("summary") or raw.get("description"), MAX_EVALUATION_SUMMARY_CHARS)
        text_chars = raw.get("text_chars")
        if isinstance(text_chars, bool) or not isinstance(text_chars, int) or text_chars < 0:
            text_chars = len(summary)
        chunk_id = _safe_string(raw.get("chunk_id"), 180) or None
        source_id = _safe_string(raw.get("source_id"), 180) or None
        item: RetrievalEvaluationItem = {
            "retrieval_source": source,
            "project_id": requested_project,
            "chunk_id": chunk_id,
            "source_id": source_id,
            "repo": repository,
            "path": _path(raw),
            "symbol": _safe_string(raw.get("symbol"), 180),
            "source_type": _safe_string(raw.get("source_type") or raw.get("source"), 64),
            "chunk_type": _safe_string(raw.get("chunk_type"), 64),
            "score": score,
            "search_sources": (
                _safe_labels(raw.get("search_sources"), allowed=_SAFE_SOURCES)
                if source == "v2" else ["legacy"]
            ),
            "query_groups": _safe_labels(raw.get("query_groups"), allowed=frozenset(QUERY_GROUPS)),
            "keywords": _safe_labels(
                raw.get("keywords")
                or raw.get("resume_relevant_keywords")
                or raw.get("topics")
                or raw.get("languages")
            ),
            "technical_tags": _safe_labels(raw.get("technical_tags") or raw.get("languages")),
            "match_reasons": _safe_labels(raw.get("match_reasons")),
            "summary": summary,
            "text_hash": _safe_string(raw.get("text_hash"), 64).casefold(),
            "text_chars": min(text_chars, 2_000_000),
        }
        candidate = [*items, item]
        if len(json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"))) > MAX_EVALUATION_SERIALIZED_CHARS:
            empty["warnings"].append("evaluation_context_limit_reached")
            break
        items.append(item)
        if len(items) >= MAX_EVALUATION_RESULTS:
            if len(results) > MAX_EVALUATION_RESULTS:
                empty["warnings"].append("evaluation_result_limit_reached")
            break
    empty["items"] = items
    empty["warnings"] = sorted(set(empty["warnings"]))[:MAX_EVALUATION_WARNINGS]
    return empty


def _normalized_text(value: str) -> tuple[str, set[str]]:
    normalized = " ".join(value.casefold().replace("\\", "/").split())
    return normalized, set(_TOKEN_RE.findall(normalized))


def _anchor_matches(anchor: RetrievalEvidenceAnchor, item: RetrievalEvaluationItem) -> bool:
    values = [
        item["project_id"], item["repo"], item["path"], item["symbol"],
        item["source_type"], item["chunk_type"], item["summary"],
        *item["keywords"], *item["technical_tags"], *item["query_groups"], *item["match_reasons"],
    ]
    anchor_text, anchor_tokens = _normalized_text(anchor["value"])
    if not anchor_text:
        return False
    for value in values:
        text, tokens = _normalized_text(value)
        if anchor_text == text or (len(anchor_tokens) > 1 and anchor_text in text):
            return True
        if len(anchor_tokens) == 1 and anchor_tokens <= tokens:
            return True
    return False


def _item_identity(item: RetrievalEvaluationItem) -> str:
    if item["chunk_id"]:
        return f"chunk:{item['chunk_id']}"
    if item["source_id"]:
        return f"source:{item['source_id']}:{item['path']}"
    safe = {
        "repo": item["repo"], "path": item["path"], "symbol": item["symbol"],
        "summary": item["summary"], "source_type": item["source_type"],
    }
    return "eval:" + hashlib.sha256(
        json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def calculate_retrieval_quality_metrics(
    *,
    normalization: RetrievalNormalizationResult,
    anchors: Sequence[RetrievalEvidenceAnchor],
    query_plan: Any = None,
    deterministic_output: bool = True,
) -> RetrievalQualityMetrics:
    items = normalization.get("items", []) if isinstance(normalization, Mapping) else []
    safe_items = [item for item in items if isinstance(item, Mapping)][:MAX_EVALUATION_RESULTS]
    identities = [_item_identity(item) for item in safe_items]
    unique_identities = set(identities)
    duplicate_count = len(identities) - len(unique_identities)
    scores = [item["score"] for item in safe_items]
    requested_groups = [
        group for group in QUERY_GROUPS
        if isinstance(query_plan, Mapping) and isinstance(query_plan.get(group), (list, tuple)) and query_plan[group]
    ]
    covered_values = {group for item in safe_items for group in item["query_groups"]}
    covered_groups = [group for group in QUERY_GROUPS if group in covered_values and group in requested_groups]
    safe_anchors = [anchor for anchor in anchors if isinstance(anchor, Mapping)][:MAX_EVALUATION_ANCHORS]
    matched = [
        anchor for anchor in safe_anchors
        if any(_anchor_matches(anchor, item) for item in safe_items)
    ]
    symbol_anchors = [anchor for anchor in safe_anchors if anchor.get("anchor_type") == "symbol"]
    matched_symbol_ids = {
        anchor["anchor_id"] for anchor in matched if anchor.get("anchor_type") == "symbol"
    }
    serialized = json.dumps(safe_items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    character_count = len(serialized)
    result_count = len(safe_items)
    return {
        "status": "ready" if safe_items else "empty",
        "result_count": result_count,
        "unique_result_count": len(unique_identities),
        "duplicate_count": duplicate_count,
        "duplicate_rate": round(duplicate_count / result_count, 6) if result_count else 0.0,
        "unique_chunk_count": len({item["chunk_id"] for item in safe_items if item["chunk_id"]}),
        "unique_source_count": len({item["source_id"] for item in safe_items if item["source_id"]}),
        "unique_path_count": len({item["path"] for item in safe_items if item["path"]}),
        "unique_symbol_count": len({item["symbol"] for item in safe_items if item["symbol"]}),
        "repository_count": len({item["repo"] for item in safe_items}),
        "cross_project_result_count": int(normalization.get("cross_project_result_count", 0)),
        "unauthorized_repository_count": int(normalization.get("unauthorized_repository_count", 0)),
        "forbidden_field_count": int(normalization.get("forbidden_field_count", 0)),
        "secret_marker_count": int(normalization.get("secret_marker_count", 0)),
        "invalid_score_count": int(normalization.get("invalid_score_count", 0)),
        "minimum_score": round(min(scores), 6) if scores else 0.0,
        "maximum_score": round(max(scores), 6) if scores else 0.0,
        "mean_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "keyword_provenance_count": sum("keyword" in item["search_sources"] for item in safe_items),
        "symbol_provenance_count": sum("symbol" in item["search_sources"] for item in safe_items),
        "vector_provenance_count": sum("vector" in item["search_sources"] for item in safe_items),
        "multi_source_count": sum(len(set(item["search_sources"])) > 1 for item in safe_items),
        "requested_query_groups": requested_groups,
        "covered_query_groups": covered_groups,
        "missing_query_groups": [group for group in requested_groups if group not in covered_groups],
        "path_diversity": round(len({item["path"] for item in safe_items if item["path"]}) / result_count, 6) if result_count else 0.0,
        "source_type_diversity": len({item["source_type"] for item in safe_items if item["source_type"]}),
        "chunk_type_diversity": len({item["chunk_type"] for item in safe_items if item["chunk_type"]}),
        "technical_tag_coverage": len({tag.casefold() for item in safe_items for tag in item["technical_tags"]}),
        "known_symbol_coverage": round(len(matched_symbol_ids) / len(symbol_anchors), 6) if symbol_anchors else 1.0,
        "supported_anchor_coverage": round(len(matched) / len(safe_anchors), 6) if safe_anchors else 0.0,
        "project_identity_covered": any(anchor.get("anchor_type") == "project_identity" for anchor in matched),
        "matched_anchor_ids": sorted(anchor["anchor_id"] for anchor in matched),
        "serialized_character_count": character_count,
        "estimated_context_budget": {
            "character_count": character_count,
            "approximate_token_count": math.ceil(character_count / APPROXIMATE_CHARACTERS_PER_TOKEN),
            "approximation": "approximate_characters_divided_by_4_not_exact_token_usage",
        },
        "deterministic_output": deterministic_output is True,
    }


def evaluate_labelled_benchmark(case: RetrievalBenchmarkCase) -> dict[str, Any]:
    """Calculate precision/recall only for an explicitly labelled synthetic case."""

    if not isinstance(case, Mapping):
        return {"status": "error", "precision": 0.0, "recall": 0.0}
    retrieved = [
        safe for value in case.get("retrieved_ids", [])[:MAX_EVALUATION_RESULTS]
        if (safe := _safe_string(value, 180))
    ] if isinstance(case.get("retrieved_ids"), list) else []
    relevant = {
        safe for value in case.get("relevant_ids", [])[:MAX_EVALUATION_RESULTS]
        if (safe := _safe_string(value, 180))
    } if isinstance(case.get("relevant_ids"), list) else set()
    recovered = {value for value in retrieved if value in relevant}
    return {
        "status": "ready" if relevant else "blocked",
        "case_id": _safe_string(case.get("case_id"), 120),
        "labelled_fixture": True,
        "retrieved_count": len(retrieved),
        "relevant_count": len(relevant),
        "relevant_retrieved_count": len(recovered),
        "precision": round(len(recovered) / len(retrieved), 6) if retrieved else 0.0,
        "recall": round(len(recovered) / len(relevant), 6) if relevant else 0.0,
    }


def compare_normalized_retrieval_quality(
    *,
    project_id: Any,
    legacy_normalization: RetrievalNormalizationResult,
    v2_normalization: RetrievalNormalizationResult,
    anchors: Sequence[RetrievalEvidenceAnchor],
    query_plan: Any,
    vector_ready: bool,
    legacy_deterministic: bool = True,
    v2_deterministic: bool = True,
) -> RetrievalComparisonResult:
    """Compare already-safe normalizations without provider or filesystem access."""

    requested_project = normalize_project_id(project_id)
    legacy = legacy_normalization
    v2 = v2_normalization
    legacy_metrics = calculate_retrieval_quality_metrics(
        normalization=legacy, anchors=anchors, query_plan=query_plan,
        deterministic_output=legacy_deterministic,
    )
    v2_metrics = calculate_retrieval_quality_metrics(
        normalization=v2, anchors=anchors, query_plan=query_plan,
        deterministic_output=v2_deterministic,
    )
    errors: list[str] = []
    for condition, code in (
        (v2_metrics["cross_project_result_count"] == 0, "v2_cross_project_evidence"),
        (v2_metrics["unauthorized_repository_count"] == 0, "v2_unauthorized_repository"),
        (v2_metrics["forbidden_field_count"] == 0, "v2_forbidden_field"),
        (v2_metrics["secret_marker_count"] == 0, "v2_secret_marker"),
        (v2_metrics["invalid_score_count"] == 0, "v2_invalid_score"),
        (v2_metrics["duplicate_count"] == 0, "v2_duplicate_result"),
        (v2_metrics["deterministic_output"], "v2_nondeterministic_output"),
        (v2_metrics["result_count"] <= MAX_EVALUATION_RESULTS, "v2_result_limit_exceeded"),
        (v2_metrics["serialized_character_count"] <= MAX_EVALUATION_SERIALIZED_CHARS, "v2_context_budget_exceeded"),
        (v2_metrics["result_count"] > 0, "v2_no_supported_evidence"),
        (v2_metrics["project_identity_covered"], "v2_project_identity_not_covered"),
        (not vector_ready or v2_metrics["vector_provenance_count"] > 0, "v2_vector_provenance_missing"),
        (
            not v2_metrics["covered_query_groups"]
            or set(v2_metrics["covered_query_groups"]) != {"jd_alignment"},
            "v2_jd_only_evidence",
        ),
        (
            v2_metrics["supported_anchor_coverage"] + MATERIAL_ANCHOR_COVERAGE_DROP
            >= legacy_metrics["supported_anchor_coverage"],
            "v2_material_anchor_coverage_regression",
        ),
    ):
        if not condition:
            errors.append(code)
    safety_codes = {
        "v2_cross_project_evidence", "v2_unauthorized_repository", "v2_forbidden_field",
        "v2_secret_marker", "v2_invalid_score", "v2_duplicate_result",
        "v2_nondeterministic_output", "v2_result_limit_exceeded", "v2_context_budget_exceeded",
    }
    safety_passed = not any(code in safety_codes for code in errors)
    quality_passed = not errors
    comparison = {
        "anchor_coverage_delta": round(
            v2_metrics["supported_anchor_coverage"] - legacy_metrics["supported_anchor_coverage"], 6
        ),
        "query_group_coverage_delta": len(v2_metrics["covered_query_groups"]) - len(legacy_metrics["covered_query_groups"]),
        "duplicate_rate_delta": round(v2_metrics["duplicate_rate"] - legacy_metrics["duplicate_rate"], 6),
        "context_character_delta": v2_metrics["serialized_character_count"] - legacy_metrics["serialized_character_count"],
        "material_anchor_coverage_drop_threshold": MATERIAL_ANCHOR_COVERAGE_DROP,
        "maximum_result_count": MAX_EVALUATION_RESULTS,
        "maximum_serialized_characters": MAX_EVALUATION_SERIALIZED_CHARS,
        "vector_ready": vector_ready is True,
        "safety_passed": safety_passed,
        "quality_gate_passed": quality_passed,
    }
    legacy_warnings = legacy.get("warnings", []) if isinstance(legacy, Mapping) else []
    v2_warnings = v2.get("warnings", []) if isinstance(v2, Mapping) else []
    legacy_errors = legacy.get("errors", []) if isinstance(legacy, Mapping) else []
    v2_errors = v2.get("errors", []) if isinstance(v2, Mapping) else []
    return {
        "status": "passed" if quality_passed else "blocked",
        "project_id": requested_project,
        "legacy": legacy_metrics,
        "v2": v2_metrics,
        "comparison": comparison,
        "warnings": sorted(set(legacy_warnings + v2_warnings))[:MAX_EVALUATION_WARNINGS],
        "errors": sorted(set(errors + legacy_errors + v2_errors))[:MAX_EVALUATION_ERRORS],
    }


def compare_retrieval_quality(
    *,
    project_id: Any,
    authorized_repositories: Sequence[str],
    legacy_results: Any,
    v2_results: Any,
    anchors: Sequence[RetrievalEvidenceAnchor],
    query_plan: Any,
    vector_ready: bool,
    repeated_legacy_results: Any = None,
    repeated_v2_results: Any = None,
) -> RetrievalComparisonResult:
    requested_project = normalize_project_id(project_id)
    legacy = normalize_retrieval_results(
        retrieval_source="legacy", project_id=requested_project, results=legacy_results,
        authorized_repositories=authorized_repositories,
    )
    v2 = normalize_retrieval_results(
        retrieval_source="v2", project_id=requested_project, results=v2_results,
        authorized_repositories=authorized_repositories,
    )
    legacy_deterministic = True
    v2_deterministic = True
    if repeated_legacy_results is not None:
        repeated = normalize_retrieval_results(
            retrieval_source="legacy", project_id=requested_project, results=repeated_legacy_results,
            authorized_repositories=authorized_repositories,
        )
        legacy_deterministic = repeated["items"] == legacy["items"]
    if repeated_v2_results is not None:
        repeated = normalize_retrieval_results(
            retrieval_source="v2", project_id=requested_project, results=repeated_v2_results,
            authorized_repositories=authorized_repositories,
        )
        v2_deterministic = repeated["items"] == v2["items"]
    return compare_normalized_retrieval_quality(
        project_id=requested_project,
        legacy_normalization=legacy,
        v2_normalization=v2,
        anchors=anchors,
        query_plan=query_plan,
        vector_ready=vector_ready,
        legacy_deterministic=legacy_deterministic,
        v2_deterministic=v2_deterministic,
    )
