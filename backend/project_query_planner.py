"""Deterministic bounded query planning for project evidence retrieval."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence, TypedDict

from backend.project_evidence_coverage import CoverageCategory, GapPriority
from backend.project_evidence_followup_intents import (
    FollowupEvidenceGoal,
    FollowupRetrievalIntent,
    validate_followup_retrieval_intents,
)
from backend.project_evidence_models import EvidenceType


MAX_QUERY_CHARS = 180
MAX_QUERIES_PER_GROUP = 3
MAX_TOTAL_QUERIES = 12
MAX_TERMS_PER_QUERY = 16
MAX_SYMBOLS = 16

QUERY_GROUPS = (
    "project_identity",
    "jd_alignment",
    "mechanisms",
    "symbols",
    "validation_repair",
    "metrics_impact",
)
_RAW_KEYS = frozenset({
    "body", "content", "diff", "file_content", "hunk", "patch", "raw_text", "text",
})
_IDENTITY_KEYS = frozenset({
    "core_problem", "core_value", "identity", "name", "positioning", "project_id",
    "project_name", "repo", "repository", "short_summary", "summary",
})
_PROJECT_FACT_KEYS = frozenset({
    "confirmed_features", "features", "mechanisms", "technical_tags", "tech_stack",
    "tools", "workflows", "validation", "quality", "reliability", "metrics",
    "real_metrics", "impact", "symbols", "functions", "classes", "routes", "modules",
    "files", "path",
})
_JD_KEYS = frozenset({
    "technologies", "technology", "technical_skills", "skills", "requirements",
    "responsibilities", "capabilities", "backend", "data", "retrieval", "reliability",
    "validation", "systems", "targets",
})
_SYMBOL_KEYS = frozenset({"symbols", "functions", "classes", "routes", "modules", "files", "path"})
_SECRET_RE = re.compile(
    r"(?i)(?:begin\s+(?:rsa\s+)?private\s+key|api[_-]?key\s*=|access[_-]?token\s*=|"
    r"secret\s*=|password\s*=|credential\s*=|diff\s+--git)"
)
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.#/-]{1,63}")
_SYMBOL_RE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]{2,127}|[A-Za-z_$][A-Za-z0-9_$]{2,127}|"
    r"[A-Za-z0-9_.-]{1,120}\.(?:py|js|jsx|ts|tsx|json|md|sql|yaml|yml))$"
)
_BOILERPLATE = frozenset({
    "accommodation", "applicant", "benefit", "benefits", "candidate", "disability",
    "employer", "employment", "equal", "insurance", "location", "opportunity", "salary",
    "veteran", "workplace",
})
_GENERIC_STOP = frozenset({
    "about", "also", "and", "are", "complete", "description", "for", "from", "have",
    "into", "our", "that", "the", "their", "this", "through", "with", "your",
})
_MECHANISM_TERMS = (
    ("evidence card", "evidence cards"),
    ("local memory", "local memory"),
    ("quality gate", "quality gate"),
    ("structured output", "structured output"),
    ("deterministic merge", "deterministic merge"),
    ("pipeline orchestration", "pipeline orchestration"),
    ("rerank", "reranking"),
    ("retrieval", "retrieval"),
    ("validation", "validation"),
    ("sqlite", "SQLite"),
    ("cache", "cache"),
    ("fallback", "fallback"),
)
_VALIDATION_TERMS = (
    ("unsupported claim", "unsupported claim blocking"),
    ("template pollution", "template pollution prevention"),
    ("quality gate", "quality gate"),
    ("error handling", "error handling"),
    ("latex", "LaTeX repair"),
    ("validation", "validation"),
    ("fallback", "fallback"),
    ("retry", "retry"),
    ("repair", "repair"),
)
_METRIC_TERMS = (
    ("manual review", "manual review time"),
    ("success rate", "success rate"),
    ("failure rate", "failure rate"),
    ("token", "token use"),
    ("latency", "latency"),
    ("throughput", "throughput"),
    ("precision", "precision"),
    ("recall", "recall"),
    ("cost", "cost"),
)
_FOLLOWUP_PRIORITY = {
    GapPriority.HIGH: 0,
    GapPriority.MEDIUM: 1,
    GapPriority.LOW: 2,
}
_FOLLOWUP_GOAL_TARGETS: dict[FollowupEvidenceGoal, tuple[str, str]] = {
    FollowupEvidenceGoal.CONCRETE_IMPLEMENTATION_MECHANISM: (
        "mechanisms", "concrete implementation mechanism",
    ),
    FollowupEvidenceGoal.TECHNICAL_WORKFLOW: ("mechanisms", "technical workflow"),
    FollowupEvidenceGoal.ALGORITHM_OR_PROCESSING_STEP: (
        "mechanisms", "algorithm processing step",
    ),
    FollowupEvidenceGoal.SYSTEM_COMPONENTS: ("mechanisms", "system architecture components"),
    FollowupEvidenceGoal.COMPONENT_RELATIONSHIPS: (
        "mechanisms", "component relationships",
    ),
    FollowupEvidenceGoal.DATA_OR_CONTROL_FLOW: ("mechanisms", "data control flow"),
    FollowupEvidenceGoal.SERVICE_BOUNDARIES: ("mechanisms", "service boundaries"),
    FollowupEvidenceGoal.PERSISTENCE_MECHANISM: ("mechanisms", "persistence mechanism"),
    FollowupEvidenceGoal.DATABASE_OR_CACHE_MECHANISM: (
        "mechanisms", "database cache mechanism",
    ),
    FollowupEvidenceGoal.STORAGE_LIFECYCLE: ("mechanisms", "storage lifecycle"),
    FollowupEvidenceGoal.RETRIEVAL_MECHANISM: ("mechanisms", "retrieval mechanism"),
    FollowupEvidenceGoal.INDEXING_MECHANISM: ("mechanisms", "indexing mechanism"),
    FollowupEvidenceGoal.RANKING_OR_RERANKING_MECHANISM: (
        "mechanisms", "ranking reranking mechanism",
    ),
    FollowupEvidenceGoal.EVIDENCE_SELECTION_MECHANISM: (
        "mechanisms", "evidence selection mechanism",
    ),
    FollowupEvidenceGoal.VALIDATION_MECHANISM: (
        "validation_repair", "validation mechanism",
    ),
    FollowupEvidenceGoal.REPAIR_MECHANISM: ("validation_repair", "repair mechanism"),
    FollowupEvidenceGoal.RETRY_OR_FALLBACK_BEHAVIOR: (
        "validation_repair", "retry fallback behavior",
    ),
    FollowupEvidenceGoal.FAILURE_HANDLING: ("validation_repair", "failure handling"),
    FollowupEvidenceGoal.DETERMINISTIC_VERIFICATION: (
        "validation_repair", "deterministic verification",
    ),
    FollowupEvidenceGoal.OUTPUT_VALIDATION: ("validation_repair", "output validation"),
    FollowupEvidenceGoal.QUALITY_GATE: ("validation_repair", "quality gate"),
    FollowupEvidenceGoal.SCHEMA_VALIDATION: ("validation_repair", "schema validation"),
    FollowupEvidenceGoal.CONSISTENCY_ENFORCEMENT: (
        "validation_repair", "consistency enforcement",
    ),
    FollowupEvidenceGoal.EVIDENCE_GROUNDING: ("mechanisms", "evidence grounding"),
    FollowupEvidenceGoal.CONFIDENCE_HANDLING: ("mechanisms", "confidence handling"),
    FollowupEvidenceGoal.CLAIM_VALIDATION: ("validation_repair", "claim validation"),
    FollowupEvidenceGoal.SOURCE_ENFORCEMENT: ("mechanisms", "source citation enforcement"),
    FollowupEvidenceGoal.FACTUALITY_EVALUATION: (
        "validation_repair", "factuality validation",
    ),
    FollowupEvidenceGoal.METRIC_OR_IMPACT_EVIDENCE: (
        "metrics_impact", "metric impact evidence",
    ),
    FollowupEvidenceGoal.JD_REQUIREMENT_EVIDENCE: (
        "jd_alignment", "requirement evidence",
    ),
}
_FOLLOWUP_EVIDENCE_TYPE_TERMS = {
    EvidenceType.FEATURE: "feature implementation",
    EvidenceType.BUG_FIX: "bug fix",
    EvidenceType.ARCHITECTURE: "architecture",
    EvidenceType.WORKFLOW: "workflow",
    EvidenceType.VALIDATION: "validation",
    EvidenceType.FAILURE_RECOVERY: "failure recovery",
    EvidenceType.DATA_PERSISTENCE: "data persistence",
    EvidenceType.RETRIEVAL: "retrieval",
    EvidenceType.OPTIMIZATION: "optimization",
    EvidenceType.INTEGRATION: "integration",
    EvidenceType.TESTING: "testing",
    EvidenceType.CONFIGURATION: "configuration",
    EvidenceType.DOCUMENTATION: "documentation",
}
_FOLLOWUP_GROUP_PURPOSE = {
    "jd_alignment": "JD target",
    "mechanisms": "mechanism",
    "validation_repair": "validation repair",
    "metrics_impact": "metric evidence search",
}
_SYMBOL_RELEVANT_CATEGORIES = frozenset({
    CoverageCategory.IMPLEMENTATION_MECHANISM,
    CoverageCategory.DATA_STORAGE,
    CoverageCategory.RETRIEVAL_RANKING,
    CoverageCategory.VALIDATION_REPAIR,
})


class ProjectQueryPlan(TypedDict):
    project_id: str
    project_identity: list[str]
    jd_alignment: list[str]
    mechanisms: list[str]
    symbols: list[str]
    validation_repair: list[str]
    metrics_impact: list[str]


def _safe_string(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = " ".join(_CONTROL_RE.sub(" ", value).split())
    if not cleaned or _SECRET_RE.search(cleaned):
        return ""
    return cleaned[: MAX_QUERY_CHARS * 4]


def _tokens(value: Any) -> list[str]:
    cleaned = _safe_string(value)
    if not cleaned:
        return []
    result: dict[str, str] = {}
    for token in _TOKEN_RE.findall(cleaned):
        normalized = token.strip("/.-").casefold()
        if not normalized or normalized in _BOILERPLATE or normalized in _GENERIC_STOP:
            continue
        result.setdefault(normalized, token.strip("/.-"))
    return [result[key] for key in sorted(result)]


def _safe_values(value: Any, allowed_keys: frozenset[str], *, depth: int = 0) -> list[str]:
    if depth > 2:
        return []
    if isinstance(value, str):
        return [value] if _safe_string(value) else []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value[:64]:
            if isinstance(item, str):
                if _safe_string(item):
                    result.append(item)
            elif isinstance(item, Mapping):
                result.extend(_safe_values(item, allowed_keys, depth=depth + 1))
        return result
    if isinstance(value, Mapping):
        result = []
        for key in sorted(value, key=lambda item: str(item).casefold()):
            normalized_key = str(key).casefold()
            if normalized_key in _RAW_KEYS or normalized_key not in allowed_keys:
                continue
            result.extend(_safe_values(value[key], allowed_keys, depth=depth + 1))
        return result
    return []


def _select_project(project_id: str, project_memory: Any) -> Mapping[str, Any]:
    if not isinstance(project_memory, Mapping):
        return {}
    projects = project_memory.get("projects", project_memory)
    if isinstance(projects, Mapping):
        candidates: Sequence[Any] = [projects]
    elif isinstance(projects, (list, tuple)):
        candidates = projects
    else:
        return {}
    requested = project_id.casefold()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        identifiers = (
            candidate.get("project_id"), candidate.get("project_name"), candidate.get("name")
        )
        if any(isinstance(item, str) and item.strip().casefold() == requested for item in identifiers):
            return candidate
    return {}


def _select_project_scoped(value: Any, project_id: str) -> Any:
    if not isinstance(value, Mapping):
        return value
    projects = value.get("projects")
    if projects is not None:
        return _select_project(project_id, value)
    for key, item in value.items():
        if str(key).casefold() == project_id.casefold():
            return item
    item_id = value.get("project_id")
    if isinstance(item_id, str) and item_id.strip().casefold() != project_id.casefold():
        return {}
    return value


def _terms_from_values(values: Sequence[str]) -> list[str]:
    terms: dict[str, str] = {}
    for value in values:
        for token in _tokens(value):
            key = token.casefold()
            current = terms.get(key)
            if current is None or token < current:
                terms[key] = token
    return [terms[key] for key in sorted(terms)]


def _supported_phrases(values: Sequence[str], vocabulary: Sequence[tuple[str, str]]) -> list[str]:
    haystack = " ".join(_safe_string(value).casefold() for value in values)
    return [output for trigger, output in vocabulary if trigger in haystack]


def _identity_terms(project_id: str, project: Mapping[str, Any]) -> list[str]:
    values = [project_id]
    for key in sorted(_IDENTITY_KEYS):
        if key in project and key not in _RAW_KEYS:
            values.extend(_safe_values(project[key], _IDENTITY_KEYS))
    return _terms_from_values(values)


def _symbols(value: Any, project_id: str) -> list[str]:
    scoped = _select_project_scoped(value, project_id)
    values = _safe_values(scoped, _SYMBOL_KEYS)
    found: dict[str, str] = {}
    for raw in values:
        for candidate in re.split(r"[\s,;]+", _safe_string(raw)):
            candidate = candidate.strip("()[]{}:'\"")
            if _SYMBOL_RE.fullmatch(candidate):
                key = candidate.casefold()
                current = found.get(key)
                if current is None or candidate < current:
                    found[key] = candidate
    return [found[key] for key in sorted(found)][:MAX_SYMBOLS]


def _make_queries(identity: Sequence[str], terms: Sequence[str], purpose: str) -> list[str]:
    identity_prefix = list(identity[:2])
    unique: dict[str, str] = {}
    for term in terms:
        for token in _tokens(term):
            unique.setdefault(token.casefold(), token)
    remaining = [unique[key] for key in sorted(unique) if key not in {x.casefold() for x in identity_prefix}]
    queries: list[str] = []
    batch_size = max(1, MAX_TERMS_PER_QUERY - len(identity_prefix) - 2)
    for index in range(0, len(remaining), batch_size):
        words = [*identity_prefix, purpose, *remaining[index:index + batch_size]][:MAX_TERMS_PER_QUERY]
        query = " ".join(words)[:MAX_QUERY_CHARS].strip()
        if query and any(character.isalnum() for character in query):
            queries.append(query)
        if len(queries) >= MAX_QUERIES_PER_GROUP:
            break
    return queries


def _symbol_queries(symbols: Sequence[str]) -> list[str]:
    queries = []
    for index in range(0, len(symbols), 6):
        query = " ".join(symbols[index:index + 6])[:MAX_QUERY_CHARS]
        if query:
            queries.append(query)
        if len(queries) >= MAX_QUERIES_PER_GROUP:
            break
    return queries


def _dedupe_queries(values: Sequence[str]) -> list[str]:
    result: dict[str, str] = {}
    for value in values:
        key = value.casefold()
        if key not in result:
            result[key] = value
    return list(result.values())


def _followup_query_candidates(
    *,
    identity: Sequence[str],
    retrieval_intents: Sequence[FollowupRetrievalIntent],
    symbols_available: bool,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    candidates = {group: [] for group in QUERY_GROUPS}
    group_priorities: dict[str, int] = {}
    for intent in retrieval_intents:
        priority = _FOLLOWUP_PRIORITY[intent.priority]
        terms_by_group: dict[str, list[str]] = {}
        for goal in intent.evidence_goals:
            group, phrase = _FOLLOWUP_GOAL_TARGETS[goal]
            terms_by_group.setdefault(group, []).append(phrase)
        if "jd_alignment" in terms_by_group:
            terms_by_group["jd_alignment"].extend(intent.requirement_ids)
        type_terms = [
            _FOLLOWUP_EVIDENCE_TYPE_TERMS[evidence_type]
            for evidence_type in intent.preferred_evidence_types
            if evidence_type in _FOLLOWUP_EVIDENCE_TYPE_TERMS
        ]
        for group, terms in terms_by_group.items():
            queries = _make_queries(
                identity,
                [*terms, *type_terms],
                _FOLLOWUP_GROUP_PURPOSE[group],
            )
            candidates[group] = _dedupe_queries([*candidates[group], *queries])
            group_priorities[group] = min(priority, group_priorities.get(group, priority))
        if symbols_available and any(
            category in _SYMBOL_RELEVANT_CATEGORIES
            for category in intent.target_categories
        ):
            group_priorities["symbols"] = min(
                priority,
                group_priorities.get("symbols", priority),
            )
    return candidates, group_priorities


def build_project_query_plan(
    *,
    project_id: Any,
    project_memory: Any = None,
    compact_facts: Any = None,
    jd_targets: Any = None,
    known_symbols: Any = None,
    retrieval_intents: Any = None,
) -> ProjectQueryPlan:
    """Build a side-effect-free query plan without treating targets as evidence."""

    validated_intents = validate_followup_retrieval_intents(
        project_id=project_id,
        retrieval_intents=retrieval_intents,
    )
    project_id_value = _safe_string(project_id)[:120]
    project = _select_project(project_id_value, project_memory) if project_id_value else {}
    scoped_facts = _select_project_scoped(compact_facts, project_id_value)
    identity = _identity_terms(project_id_value, project)
    project_values = [
        *_safe_values(project, _PROJECT_FACT_KEYS),
        *_safe_values(scoped_facts, _PROJECT_FACT_KEYS),
    ]
    jd_values = _safe_values(jd_targets, _JD_KEYS)
    mechanism_terms = _supported_phrases(project_values, _MECHANISM_TERMS)
    validation_terms = _supported_phrases(project_values, _VALIDATION_TERMS)
    metric_terms = _supported_phrases([*project_values, *jd_values], _METRIC_TERMS)
    jd_terms = _terms_from_values(jd_values)
    symbol_values = _symbols(known_symbols, project_id_value)
    symbol_values.extend(_symbols(project, project_id_value))
    symbol_values.extend(_symbols(scoped_facts, project_id_value))
    deduped_symbols = {
        symbol.casefold(): symbol for symbol in sorted(symbol_values, key=lambda item: (item.casefold(), item))
    }
    identity_queries = _make_queries([], identity, "project identity")
    groups: dict[str, list[str]] = {
        "project_identity": identity_queries,
        "jd_alignment": _make_queries(identity, jd_terms, "JD target"),
        "mechanisms": _make_queries(identity, mechanism_terms, "mechanism"),
        "symbols": _symbol_queries(
            [deduped_symbols[key] for key in sorted(deduped_symbols)][:MAX_SYMBOLS]
        ),
        "validation_repair": _make_queries(identity, validation_terms, "validation repair"),
        "metrics_impact": _make_queries(identity, metric_terms, "metric evidence search"),
    }
    if validated_intents:
        followup, followup_priorities = _followup_query_candidates(
            identity=[
                project_id_value,
                *(item for item in identity if item.casefold() != project_id_value.casefold()),
            ],
            retrieval_intents=validated_intents,
            symbols_available=bool(groups["symbols"]),
        )
        for group in QUERY_GROUPS:
            if not followup[group]:
                continue
            if group == "jd_alignment" and groups[group]:
                groups[group] = _dedupe_queries([
                    groups[group][0],
                    *followup[group],
                    *groups[group][1:],
                ])[:MAX_QUERIES_PER_GROUP]
            else:
                groups[group] = _dedupe_queries([
                    *followup[group],
                    *groups[group],
                ])[:MAX_QUERIES_PER_GROUP]
        touched_groups = sorted(
            (
                group
                for group in followup_priorities
                if group not in {"project_identity", "jd_alignment"}
            ),
            key=lambda group: (followup_priorities[group], QUERY_GROUPS.index(group)),
        )
        allocation_order = [
            "project_identity",
            "jd_alignment",
            *touched_groups,
            *(
                group
                for group in QUERY_GROUPS
                if group not in {"project_identity", "jd_alignment", *touched_groups}
            ),
        ]
        remaining = MAX_TOTAL_QUERIES
        allocated = {group: [] for group in QUERY_GROUPS}
        for group in allocation_order:
            allocated[group] = groups[group][:remaining]
            remaining -= len(allocated[group])
        return ProjectQueryPlan(
            project_id=project_id_value,
            **{group: allocated[group] for group in QUERY_GROUPS},
        )
    remaining = MAX_TOTAL_QUERIES
    bounded: dict[str, list[str]] = {}
    for group in QUERY_GROUPS:
        bounded[group] = groups[group][:remaining]
        remaining -= len(bounded[group])
    return ProjectQueryPlan(project_id=project_id_value, **bounded)
