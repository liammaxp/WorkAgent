"""Pure, deterministic runtime coverage assessment for project evidence.

Coverage is a derived decision aid.  It references the authoritative project
evidence models without replacing them, performs no retrieval or persistence,
and never carries source bodies or descriptive evidence text in its output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import re
from typing import Any

from backend.project_evidence_models import (
    ClaimSubjectType,
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    ProjectCapabilityFact,
    ProjectClaimBoundary,
    ProjectEvidenceFact,
)
from backend.project_claim_boundaries import evaluate_project_numeric_claim
from backend.project_repository_identity import normalize_project_id


MAX_SUPPORTING_REFS_PER_DIMENSION = 8
MAX_REQUIREMENTS = 32
MAX_REQUIREMENT_TERMS = 16
MAX_IDENTIFIER_LENGTH = 300
MAX_PRIORITIZED_FOLLOWUP_GAPS = 4

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_REQUIREMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_WORD_RE = re.compile(r"[a-z0-9+#.]+")


class CoverageState(str, Enum):
    COVERED = "covered"
    PARTIAL = "partial"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    BLOCKED = "blocked"


class CoverageCategory(str, Enum):
    PROJECT_IDENTITY = "project_identity"
    ARCHITECTURE = "architecture"
    IMPLEMENTATION_MECHANISM = "implementation_mechanism"
    DATA_STORAGE = "data_storage"
    RETRIEVAL_RANKING = "retrieval_ranking"
    VALIDATION_REPAIR = "validation_repair"
    OUTPUT_QUALITY = "output_quality"
    RELIABILITY = "reliability"
    METRIC_IMPACT = "metric_impact"
    JD_MUST_HAVE = "jd_must_have"


class CoverageReasonCode(str, Enum):
    SUPPORTED_BY_EVIDENCE = "supported_by_evidence"
    SUPPORTED_BY_CAPABILITY = "supported_by_capability"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    NO_RELEVANT_REQUIREMENT = "no_relevant_requirement"
    NO_METRIC_SUPPORT = "no_metric_support"
    WRONG_PROJECT = "wrong_project"
    INSUFFICIENT_SPECIFICITY = "insufficient_specificity"
    CLAIM_BOUNDARY_RESTRICTED = "claim_boundary_restricted"


class GapPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    IGNORE = "ignore"


class GapPriorityReasonCode(str, Enum):
    JD_MUST_HAVE_GAP = "jd_must_have_gap"
    JD_RELEVANT_PARTIAL_SUPPORT = "jd_relevant_partial_support"
    HIGH_VALUE_MECHANISM_GAP = "high_value_mechanism_gap"
    PARTIAL_SUPPORT_WORTH_EXPANDING = "partial_support_worth_expanding"
    NO_METRIC_SIGNAL = "no_metric_signal"
    CLAIM_BOUNDARY_RESTRICTED = "claim_boundary_restricted"
    BLOCKED_UPSTREAM = "blocked_upstream"
    INSUFFICIENT_PROJECT_RELEVANCE = "insufficient_project_relevance"


_CATEGORY_ORDER = tuple(CoverageCategory)
_CATEGORY_INDEX = {category: index for index, category in enumerate(_CATEGORY_ORDER)}
_POSITIVE_STATUSES = frozenset({EvidenceStatus.ACCEPTED})
_METRIC_PRIORITY = {
    MetricSupport.NONE: 0,
    MetricSupport.APPROXIMATE: 1,
    MetricSupport.EXPLICIT: 2,
}
_GAP_PRIORITY_INDEX = {
    GapPriority.HIGH: 0,
    GapPriority.MEDIUM: 1,
    GapPriority.LOW: 2,
    GapPriority.IGNORE: 3,
}

_CONTEXT_DEPENDENT_CATEGORIES = frozenset({
    CoverageCategory.ARCHITECTURE,
    CoverageCategory.DATA_STORAGE,
    CoverageCategory.RETRIEVAL_RANKING,
    CoverageCategory.VALIDATION_REPAIR,
    CoverageCategory.OUTPUT_QUALITY,
    CoverageCategory.RELIABILITY,
    CoverageCategory.METRIC_IMPACT,
})

_CATEGORY_SIGNALS: dict[CoverageCategory, frozenset[str]] = {
    CoverageCategory.ARCHITECTURE: frozenset({
        "architecture", "component", "layer", "orchestration", "pipeline",
        "service", "system", "workflow",
    }),
    CoverageCategory.DATA_STORAGE: frozenset({
        "atomic_persistence", "cache", "database", "database_storage",
        "datastore", "persistence", "persistent_storage", "postgres",
        "postgresql", "redis", "sqlite", "storage",
    }),
    CoverageCategory.RETRIEVAL_RANKING: frozenset({
        "bm25", "index", "indexing", "keyword_search",
        "rank", "ranking", "rerank", "reranker", "reranking", "retrieval",
        "retriever", "search", "semantic_search", "vector_search",
    }),
    CoverageCategory.VALIDATION_REPAIR: frozenset({
        "claim_validation", "error_handling", "fallback", "failure_recovery",
        "repair", "retry", "schema_validation", "structured_validation",
        "validation", "validation_gate",
    }),
    CoverageCategory.OUTPUT_QUALITY: frozenset({
        "claim_validation", "factuality_evaluation", "output_evidence_validation",
        "output_quality", "quality_control", "quality_dimensions",
        "structured_output_validation", "unsupported_claim_blocking",
    }),
    CoverageCategory.RELIABILITY: frozenset({
        "claim_validation", "confidence_threshold", "evidence_grounding",
        "factuality_evaluation", "fallback", "low_evidence_refusal",
        "source_attribution", "source_enforcement", "source_grounding",
        "structured_validation", "unsupported_claim_blocking",
    }),
}

_CONCRETE_IMPLEMENTATION_SIGNALS = frozenset({
    "algorithm", "api", "cache", "client", "component", "database",
    "endpoint", "fallback", "index", "indexing", "orchestration", "parser",
    "persistence", "pipeline", "queue", "ranking", "repair", "repository",
    "retrieval", "retry", "route", "schema", "search", "server", "storage",
    "transaction", "validation", "vector", "worker", "workflow",
})

_FOLLOWUP_VALUE_CATEGORIES = frozenset({
    CoverageCategory.ARCHITECTURE,
    CoverageCategory.IMPLEMENTATION_MECHANISM,
    CoverageCategory.DATA_STORAGE,
    CoverageCategory.RETRIEVAL_RANKING,
    CoverageCategory.VALIDATION_REPAIR,
    CoverageCategory.OUTPUT_QUALITY,
})

_STRICT_METRIC_REQUIREMENT_SIGNALS = frozenset({
    "benchmark", "kpi", "latency", "measurable", "measurement", "metric",
    "metrics", "percentile", "quantitative", "throughput",
})

_CAPABILITY_CATEGORIES: dict[str, frozenset[CoverageCategory]] = {
    "claim_validation": frozenset({
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        CoverageCategory.VALIDATION_REPAIR,
        CoverageCategory.OUTPUT_QUALITY,
        CoverageCategory.RELIABILITY,
    }),
    "data_persistence": frozenset({
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        CoverageCategory.DATA_STORAGE,
    }),
    "deterministic_document_generation": frozenset({
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        CoverageCategory.OUTPUT_QUALITY,
    }),
    "evidence_grounded_generation": frozenset({
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        CoverageCategory.RELIABILITY,
    }),
    "failure_recovery": frozenset({
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        CoverageCategory.VALIDATION_REPAIR,
        CoverageCategory.RELIABILITY,
    }),
    "frontend_backend_integration": frozenset({
        CoverageCategory.ARCHITECTURE,
        CoverageCategory.IMPLEMENTATION_MECHANISM,
    }),
    "latex_validation_and_repair": frozenset({
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        CoverageCategory.VALIDATION_REPAIR,
        CoverageCategory.OUTPUT_QUALITY,
    }),
    "llm_reliability": frozenset({
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        CoverageCategory.RELIABILITY,
    }),
    "output_quality_control": frozenset({
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        CoverageCategory.VALIDATION_REPAIR,
        CoverageCategory.OUTPUT_QUALITY,
    }),
    "project_memory_management": frozenset({
        CoverageCategory.ARCHITECTURE,
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        CoverageCategory.DATA_STORAGE,
    }),
    "retrieval_and_reranking": frozenset({
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        CoverageCategory.RETRIEVAL_RANKING,
    }),
    "structured_evidence_extraction": frozenset({
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        CoverageCategory.VALIDATION_REPAIR,
    }),
    "template_pollution_blocking": frozenset({
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        CoverageCategory.VALIDATION_REPAIR,
        CoverageCategory.OUTPUT_QUALITY,
    }),
    "test_and_regression_hardening": frozenset({
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        CoverageCategory.VALIDATION_REPAIR,
    }),
    "workflow_orchestration": frozenset({
        CoverageCategory.ARCHITECTURE,
        CoverageCategory.IMPLEMENTATION_MECHANISM,
    }),
}

_ALLOWED_CLAIM_PREFIXES: dict[CoverageCategory, frozenset[str]] = {
    CoverageCategory.ARCHITECTURE: frozenset({
        "architecture", "capability", "implementation", "mechanism", "workflow",
    }),
    CoverageCategory.IMPLEMENTATION_MECHANISM: frozenset({"capability", "implementation", "mechanism"}),
    CoverageCategory.DATA_STORAGE: frozenset({
        "capability", "implementation", "mechanism", "persistence", "technology",
    }),
    CoverageCategory.RETRIEVAL_RANKING: frozenset({
        "capability", "implementation", "mechanism", "retrieval", "technology",
    }),
    CoverageCategory.VALIDATION_REPAIR: frozenset({
        "capability", "implementation", "mechanism", "testing", "validation",
    }),
    CoverageCategory.OUTPUT_QUALITY: frozenset({
        "capability", "implementation", "mechanism", "testing", "validation",
    }),
    CoverageCategory.RELIABILITY: frozenset({
        "capability", "implementation", "mechanism", "validation",
    }),
    CoverageCategory.METRIC_IMPACT: frozenset({"metric"}),
    CoverageCategory.JD_MUST_HAVE: frozenset({
        "architecture", "capability", "implementation", "mechanism", "persistence",
        "retrieval", "technology", "testing", "validation", "workflow",
    }),
}


def _bounded_identifier(value: Any, name: str, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        if required:
            raise ValueError(f"{name} must not be blank")
        return None
    if len(normalized) > MAX_IDENTIFIER_LENGTH or _CONTROL_RE.search(normalized):
        raise ValueError(f"{name} is not a bounded identifier")
    return normalized


def _normalized_project_id(value: Any) -> str:
    normalized = normalize_project_id(value)
    if not normalized:
        raise ValueError("project_id must be a normalized project identifier")
    return normalized


def _stable_strings(values: Sequence[str], name: str, *, maximum: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    normalized: dict[str, str] = {}
    for value in values:
        item = _bounded_identifier(value, name, required=True)
        assert item is not None
        key = item.casefold()
        if key not in normalized or item < normalized[key]:
            normalized[key] = item
    return tuple(normalized[key] for key in sorted(normalized)[:maximum])


@dataclass(frozen=True, slots=True)
class CoverageRequirement:
    """A bounded, already-normalized JD requirement used only at runtime."""

    requirement_id: str
    target_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        requirement_id = _bounded_identifier(self.requirement_id, "requirement_id", required=True)
        assert requirement_id is not None
        if not _REQUIREMENT_ID_RE.fullmatch(requirement_id):
            raise ValueError("requirement_id must be a normalized identifier")
        object.__setattr__(self, "requirement_id", requirement_id)
        terms = _stable_strings(self.target_terms, "target_terms", maximum=MAX_REQUIREMENT_TERMS)
        if not terms:
            raise ValueError("target_terms must contain at least one term")
        object.__setattr__(self, "target_terms", terms)


@dataclass(frozen=True, slots=True)
class CoverageEvidenceRef:
    project_id: str
    evidence_fact_id: str | None = None
    capability_fact_id: str | None = None
    claim_boundary_id: str | None = None
    source_id: str | None = None
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_id", _normalized_project_id(self.project_id))
        for name in (
            "evidence_fact_id", "capability_fact_id", "claim_boundary_id", "source_id", "chunk_id",
        ):
            object.__setattr__(self, name, _bounded_identifier(getattr(self, name), name))
        if not any((
            self.evidence_fact_id,
            self.capability_fact_id,
            self.claim_boundary_id,
            self.source_id,
            self.chunk_id,
        )):
            raise ValueError("coverage evidence reference requires at least one authority identifier")

    def to_dict(self) -> dict[str, str]:
        return {
            name: value
            for name in (
                "project_id", "evidence_fact_id", "capability_fact_id",
                "claim_boundary_id", "source_id", "chunk_id",
            )
            if (value := getattr(self, name)) is not None
        }


@dataclass(frozen=True, slots=True)
class CoverageDimensionResult:
    category: CoverageCategory
    state: CoverageState
    reason_code: CoverageReasonCode
    supporting_refs: tuple[CoverageEvidenceRef, ...] = ()
    requirement_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", CoverageCategory(self.category))
        object.__setattr__(self, "state", CoverageState(self.state))
        object.__setattr__(self, "reason_code", CoverageReasonCode(self.reason_code))
        refs = tuple(sorted(set(self.supporting_refs), key=_ref_sort_key))[:MAX_SUPPORTING_REFS_PER_DIMENSION]
        if any(ref.project_id == "" for ref in refs):
            raise ValueError("supporting reference project_id must not be blank")
        object.__setattr__(self, "supporting_refs", refs)
        object.__setattr__(
            self,
            "requirement_ids",
            _stable_strings(self.requirement_ids, "requirement_ids", maximum=MAX_REQUIREMENTS),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "state": self.state.value,
            "reason_code": self.reason_code.value,
            "supporting_refs": [ref.to_dict() for ref in self.supporting_refs],
            "requirement_ids": list(self.requirement_ids),
        }


@dataclass(frozen=True, slots=True)
class CoverageGap:
    category: CoverageCategory
    state: CoverageState
    reason_code: CoverageReasonCode
    related_requirement_ids: tuple[str, ...] = ()
    current_support_refs: tuple[CoverageEvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        category = CoverageCategory(self.category)
        state = CoverageState(self.state)
        if state not in {CoverageState.PARTIAL, CoverageState.MISSING, CoverageState.BLOCKED}:
            raise ValueError("a coverage gap must be partial, missing, or blocked")
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "reason_code", CoverageReasonCode(self.reason_code))
        object.__setattr__(
            self,
            "related_requirement_ids",
            _stable_strings(
                self.related_requirement_ids,
                "related_requirement_ids",
                maximum=MAX_REQUIREMENTS,
            ),
        )
        object.__setattr__(
            self,
            "current_support_refs",
            tuple(sorted(set(self.current_support_refs), key=_ref_sort_key))[
                :MAX_SUPPORTING_REFS_PER_DIMENSION
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "state": self.state.value,
            "reason_code": self.reason_code.value,
            "related_requirement_ids": list(self.related_requirement_ids),
            "current_support_refs": [ref.to_dict() for ref in self.current_support_refs],
        }


@dataclass(frozen=True, slots=True)
class CoverageReport:
    project_id: str
    dimensions: tuple[CoverageDimensionResult, ...]
    gaps: tuple[CoverageGap, ...]
    overall_state: CoverageState

    def __post_init__(self) -> None:
        project_id = _normalized_project_id(self.project_id)
        dimensions = tuple(sorted(self.dimensions, key=lambda item: _CATEGORY_INDEX[item.category]))
        if tuple(item.category for item in dimensions) != _CATEGORY_ORDER:
            raise ValueError("coverage report must contain each category exactly once")
        if any(ref.project_id != project_id for item in dimensions for ref in item.supporting_refs):
            raise ValueError("coverage references must match report project_id")
        gaps = tuple(sorted(self.gaps, key=lambda item: _CATEGORY_INDEX[item.category]))
        if len({item.category for item in gaps}) != len(gaps):
            raise ValueError("coverage report contains duplicate gaps")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "gaps", gaps)
        object.__setattr__(self, "overall_state", CoverageState(self.overall_state))

    def for_category(self, category: CoverageCategory | str) -> CoverageDimensionResult:
        selected = CoverageCategory(category)
        return next(item for item in self.dimensions if item.category is selected)

    def to_dict(self) -> dict[str, Any]:
        counts = {state.value: 0 for state in CoverageState}
        for item in self.dimensions:
            counts[item.state.value] += 1
        return {
            "project_id": self.project_id,
            "dimensions": [item.to_dict() for item in self.dimensions],
            "gaps": [item.to_dict() for item in self.gaps],
            "overall_state": self.overall_state.value,
            "summary_counts": counts,
        }


@dataclass(frozen=True, slots=True)
class PrioritizedCoverageGap:
    """One bounded, actionable gap selected for possible later retrieval."""

    gap: CoverageGap
    priority: GapPriority
    searchable: bool
    reason_code: GapPriorityReasonCode

    def __post_init__(self) -> None:
        if not isinstance(self.gap, CoverageGap):
            raise TypeError("gap must be a CoverageGap")
        priority = GapPriority(self.priority)
        reason = GapPriorityReasonCode(self.reason_code)
        if not isinstance(self.searchable, bool):
            raise TypeError("searchable must be a boolean")
        if priority is GapPriority.IGNORE and self.searchable:
            raise ValueError("ignored coverage gaps cannot be searchable")
        if priority is not GapPriority.IGNORE and not self.searchable:
            raise ValueError("actionable coverage gaps must be searchable")
        if self.gap.state is CoverageState.BLOCKED and self.searchable:
            raise ValueError("blocked coverage gaps cannot be searchable")
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "reason_code", reason)

    @property
    def category(self) -> CoverageCategory:
        return self.gap.category

    @property
    def coverage_state(self) -> CoverageState:
        return self.gap.state

    @property
    def related_requirement_ids(self) -> tuple[str, ...]:
        return self.gap.related_requirement_ids

    @property
    def supporting_refs(self) -> tuple[CoverageEvidenceRef, ...]:
        return self.gap.current_support_refs

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "coverage_state": self.coverage_state.value,
            "priority": self.priority.value,
            "searchable": self.searchable,
            "reason_code": self.reason_code.value,
            "coverage_reason_code": self.gap.reason_code.value,
            "related_requirement_ids": list(self.related_requirement_ids),
            "supporting_refs": [ref.to_dict() for ref in self.supporting_refs],
        }


def _ref_sort_key(ref: CoverageEvidenceRef) -> tuple[str, ...]:
    return (
        ref.project_id,
        ref.evidence_fact_id or "",
        ref.capability_fact_id or "",
        ref.claim_boundary_id or "",
        ref.source_id or "",
        ref.chunk_id or "",
    )


def _words(values: Sequence[str]) -> frozenset[str]:
    words: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = "_".join(_WORD_RE.findall(value.casefold()))
        if not normalized:
            continue
        words.add(normalized)
        words.update(part for part in normalized.split("_") if part)
    return frozenset(words)


def _fact_words(fact: ProjectEvidenceFact) -> frozenset[str]:
    return _words((
        fact.mechanism,
        *fact.implementation,
        *fact.technical_tags,
        fact.evidence_type.value,
    ))


def _capability_words(fact: ProjectCapabilityFact) -> frozenset[str]:
    return _words((
        fact.capability_type,
        *fact.mechanisms,
        *fact.technical_tags,
    ))


def _has_signal(words: frozenset[str], signals: frozenset[str]) -> bool:
    return bool(words & signals)


def _has_concrete_implementation(fact: ProjectEvidenceFact) -> bool:
    if not fact.mechanism or not fact.implementation:
        return False
    return _has_signal(_fact_words(fact), _CONCRETE_IMPLEMENTATION_SIGNALS)


def _fact_metric_support(fact: ProjectEvidenceFact) -> MetricSupport:
    if fact.metric_support is MetricSupport.NONE:
        return MetricSupport.NONE
    return (
        fact.metric_support
        if any(evaluate_project_numeric_claim(value, fact.metric_support)[0] for value in fact.safe_impact)
        else MetricSupport.NONE
    )


def _fact_supports_category(
    fact: ProjectEvidenceFact,
    category: CoverageCategory,
) -> bool:
    words = _fact_words(fact)
    if category is CoverageCategory.PROJECT_IDENTITY:
        return any(ref.project_id == fact.project_id for ref in fact.source_refs)
    if category is CoverageCategory.IMPLEMENTATION_MECHANISM:
        return _has_concrete_implementation(fact)
    if category is CoverageCategory.ARCHITECTURE:
        return fact.evidence_type in {EvidenceType.ARCHITECTURE, EvidenceType.WORKFLOW} or _has_signal(
            words, _CATEGORY_SIGNALS[category]
        )
    if category is CoverageCategory.DATA_STORAGE:
        return fact.evidence_type is EvidenceType.DATA_PERSISTENCE or _has_signal(
            words, _CATEGORY_SIGNALS[category]
        )
    if category is CoverageCategory.RETRIEVAL_RANKING:
        return fact.evidence_type is EvidenceType.RETRIEVAL or _has_signal(
            words, _CATEGORY_SIGNALS[category]
        )
    if category is CoverageCategory.VALIDATION_REPAIR:
        return fact.evidence_type in {EvidenceType.VALIDATION, EvidenceType.FAILURE_RECOVERY} or _has_signal(
            words, _CATEGORY_SIGNALS[category]
        )
    if category in {CoverageCategory.OUTPUT_QUALITY, CoverageCategory.RELIABILITY}:
        return _has_signal(words, _CATEGORY_SIGNALS[category])
    if category is CoverageCategory.METRIC_IMPACT:
        return _fact_metric_support(fact) is not MetricSupport.NONE
    return False


def _capability_supports_category(
    fact: ProjectCapabilityFact,
    category: CoverageCategory,
) -> bool:
    if not fact.present:
        return False
    return category in _CAPABILITY_CATEGORIES.get(fact.capability_type, ())


def _claim_prefixes(values: Sequence[str]) -> frozenset[str]:
    return frozenset(
        claim.partition(":")[0].strip().casefold()
        for claim in values
        if claim.strip()
    )


def _boundary_restricts(
    boundaries: tuple[ProjectClaimBoundary, ...],
    category: CoverageCategory,
) -> bool:
    if not boundaries or category is CoverageCategory.PROJECT_IDENTITY:
        return False
    allowed_prefixes = _ALLOWED_CLAIM_PREFIXES.get(category, frozenset())
    for boundary in boundaries:
        allowed = _claim_prefixes(boundary.allowed_claims)
        forbidden = _claim_prefixes(boundary.forbidden_claims)
        if category is CoverageCategory.METRIC_IMPACT and boundary.metric_support is MetricSupport.NONE:
            return True
        if not (allowed & allowed_prefixes) or forbidden & allowed_prefixes:
            return True
    return False


def _matching_boundaries(
    *,
    project_boundaries: tuple[ProjectClaimBoundary, ...],
    subject_type: ClaimSubjectType,
    subject_id: str,
) -> tuple[ProjectClaimBoundary, ...]:
    return tuple(
        boundary
        for boundary in project_boundaries
        if (
            (boundary.subject_type is subject_type and boundary.subject_id == subject_id)
            or boundary.subject_type is ClaimSubjectType.PROJECT
        )
    )


def _safe_ref_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > MAX_IDENTIFIER_LENGTH or _CONTROL_RE.search(normalized):
        return None
    return normalized


def _fact_ref(
    fact: ProjectEvidenceFact,
    boundaries: tuple[ProjectClaimBoundary, ...],
) -> CoverageEvidenceRef:
    source = min(fact.source_refs, key=lambda item: item.source_id)
    metadata_chunk_id = source.metadata.get("chunk_id") if isinstance(source.metadata, Mapping) else None
    return CoverageEvidenceRef(
        project_id=fact.project_id,
        evidence_fact_id=fact.evidence_fact_id,
        claim_boundary_id=boundaries[0].boundary_id if boundaries else None,
        source_id=_safe_ref_id(source.source_id),
        chunk_id=_safe_ref_id(metadata_chunk_id),
    )


def _capability_ref(
    capability: ProjectCapabilityFact,
    evidence_by_id: Mapping[str, ProjectEvidenceFact],
    boundaries: tuple[ProjectClaimBoundary, ...],
) -> CoverageEvidenceRef:
    evidence_fact_id = min(capability.source_evidence_fact_ids)
    evidence = evidence_by_id[evidence_fact_id]
    source = min(evidence.source_refs, key=lambda item: item.source_id)
    metadata_chunk_id = source.metadata.get("chunk_id") if isinstance(source.metadata, Mapping) else None
    return CoverageEvidenceRef(
        project_id=capability.project_id,
        evidence_fact_id=evidence_fact_id,
        capability_fact_id=capability.capability_id,
        claim_boundary_id=boundaries[0].boundary_id if boundaries else None,
        source_id=_safe_ref_id(source.source_id),
        chunk_id=_safe_ref_id(metadata_chunk_id),
    )


def _requirement_from_value(value: Any) -> CoverageRequirement | None:
    if isinstance(value, CoverageRequirement):
        return value
    if isinstance(value, str):
        term = " ".join(value.split())
        identifier = "-".join(_WORD_RE.findall(term.casefold()))[:160]
        if not term or not identifier or not _REQUIREMENT_ID_RE.fullmatch(identifier):
            return None
        return CoverageRequirement(identifier, (term,))
    if not isinstance(value, Mapping):
        return None
    required = value.get("required", value.get("must_have", True))
    if required is not True:
        return None
    requirement_id = value.get("requirement_id", value.get("id"))
    targets = value.get("target_terms", value.get("targets", value.get("terms", value.get("keywords"))))
    if isinstance(targets, str):
        targets = (targets,)
    if not isinstance(requirement_id, str) or not isinstance(targets, Sequence):
        return None
    try:
        return CoverageRequirement(requirement_id, tuple(targets))
    except (TypeError, ValueError):
        return None


def _normalize_requirements(values: Any) -> tuple[CoverageRequirement, ...]:
    if values is None:
        return ()
    if isinstance(values, Mapping):
        if "requirement_id" in values or "id" in values:
            candidates: Sequence[Any] = (values,)
        else:
            candidates = tuple(
                {"requirement_id": key, "target_terms": target}
                for key, target in values.items()
            )
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        candidates = values
    else:
        candidates = (values,)
    definitions: dict[str, dict[tuple[str, ...], CoverageRequirement]] = {}
    for value in candidates:
        requirement = _requirement_from_value(value)
        if requirement is not None:
            definitions.setdefault(requirement.requirement_id, {})[
                requirement.target_terms
            ] = requirement
    return tuple(
        next(iter(definitions[key].values()))
        for key in sorted(definitions)[:MAX_REQUIREMENTS]
        if len(definitions[key]) == 1
    )


def _requirement_supported(
    requirement: CoverageRequirement,
    words: frozenset[str],
) -> bool:
    target_words = _words(requirement.target_terms)
    return bool(target_words) and target_words.issubset(words)


def _overall_state(dimensions: Sequence[CoverageDimensionResult]) -> CoverageState:
    relevant = [item.state for item in dimensions if item.state is not CoverageState.NOT_APPLICABLE]
    if relevant and all(state is CoverageState.COVERED for state in relevant):
        return CoverageState.COVERED
    if any(state in {CoverageState.COVERED, CoverageState.PARTIAL} for state in relevant):
        return CoverageState.PARTIAL
    if any(state is CoverageState.BLOCKED for state in relevant):
        return CoverageState.BLOCKED
    return CoverageState.MISSING


def build_project_evidence_coverage_report(
    *,
    project_id: str,
    jd_requirements: Any = (),
    evidence_facts: Sequence[ProjectEvidenceFact] = (),
    capability_facts: Sequence[ProjectCapabilityFact] = (),
    claim_boundaries: Sequence[ProjectClaimBoundary] = (),
    compact_facts: Any = None,
) -> CoverageReport:
    """Build a bounded coverage report from existing structured authority.

    ``compact_facts`` is accepted for forward-compatible call sites but is not
    treated as evidence authority in this first evaluator.  Retrieval hits and
    compact prose therefore cannot create positive coverage.
    """

    requested_project = _normalized_project_id(project_id)
    del compact_facts

    all_evidence = tuple(item for item in evidence_facts if isinstance(item, ProjectEvidenceFact))
    all_capabilities = tuple(item for item in capability_facts if isinstance(item, ProjectCapabilityFact))
    all_boundaries = tuple(item for item in claim_boundaries if isinstance(item, ProjectClaimBoundary))
    saw_wrong_project = any(
        item.project_id != requested_project
        for item in (*all_evidence, *all_capabilities, *all_boundaries)
    )

    evidence = tuple(sorted(
        (
            item for item in all_evidence
             if normalize_project_id(item.project_id) == item.project_id == requested_project
             and item.status in _POSITIVE_STATUSES
             and item.source_refs
             and _safe_ref_id(item.evidence_fact_id) is not None
             and all(
                 normalize_project_id(ref.project_id) == ref.project_id == requested_project
                 and _safe_ref_id(ref.source_id) is not None
                 for ref in item.source_refs
             )
        ),
        key=lambda item: item.evidence_fact_id,
    ))
    evidence_by_id = {item.evidence_fact_id: item for item in evidence}
    capabilities = tuple(sorted(
        (
            item for item in all_capabilities
             if normalize_project_id(item.project_id) == item.project_id == requested_project
             and item.present
             and item.source_evidence_fact_ids
             and _safe_ref_id(item.capability_id) is not None
             and all(source_id in evidence_by_id for source_id in item.source_evidence_fact_ids)
        ),
        key=lambda item: item.capability_id,
    ))
    boundaries = tuple(sorted(
        (
            item for item in all_boundaries
            if normalize_project_id(item.project_id) == item.project_id == requested_project
            and _safe_ref_id(item.boundary_id) is not None
        ),
        key=lambda item: item.boundary_id,
    ))
    requirements = _normalize_requirements(jd_requirements)

    dimensions: list[CoverageDimensionResult] = []
    gap_requirement_ids: dict[CoverageCategory, tuple[str, ...]] = {}
    for category in _CATEGORY_ORDER:
        if category is CoverageCategory.JD_MUST_HAVE:
            requirement_refs: dict[str, list[CoverageEvidenceRef]] = {
                item.requirement_id: [] for item in requirements
            }
            restricted_refs: list[CoverageEvidenceRef] = []
            for requirement in requirements:
                for fact in evidence:
                    if not _requirement_supported(requirement, _fact_words(fact)):
                        continue
                    matched = _matching_boundaries(
                        project_boundaries=boundaries,
                        subject_type=ClaimSubjectType.EVIDENCE_FACT,
                        subject_id=fact.evidence_fact_id,
                    )
                    ref = _fact_ref(fact, matched)
                    if _boundary_restricts(matched, category):
                        restricted_refs.append(ref)
                    else:
                        requirement_refs[requirement.requirement_id].append(ref)
                for capability in capabilities:
                    if not _requirement_supported(requirement, _capability_words(capability)):
                        continue
                    matched = _matching_boundaries(
                        project_boundaries=boundaries,
                        subject_type=ClaimSubjectType.CAPABILITY_FACT,
                        subject_id=capability.capability_id,
                    )
                    ref = _capability_ref(capability, evidence_by_id, matched)
                    if _boundary_restricts(matched, category):
                        restricted_refs.append(ref)
                    else:
                        requirement_refs[requirement.requirement_id].append(ref)
            covered_requirements = tuple(
                key for key in sorted(requirement_refs) if requirement_refs[key]
            )
            uncovered_requirements = tuple(
                item.requirement_id
                for item in requirements
                if item.requirement_id not in covered_requirements
            )
            gap_requirement_ids[category] = uncovered_requirements
            refs = tuple(ref for key in covered_requirements for ref in requirement_refs[key])
            if not requirements:
                state = CoverageState.NOT_APPLICABLE
                reason = CoverageReasonCode.NO_RELEVANT_REQUIREMENT
            elif len(covered_requirements) == len(requirements):
                state = CoverageState.COVERED
                reason = CoverageReasonCode.SUPPORTED_BY_EVIDENCE
            elif covered_requirements:
                state = CoverageState.PARTIAL
                reason = CoverageReasonCode.PARTIALLY_SUPPORTED
            elif restricted_refs:
                state = CoverageState.BLOCKED
                reason = CoverageReasonCode.CLAIM_BOUNDARY_RESTRICTED
                refs = tuple(restricted_refs)
            else:
                state = CoverageState.MISSING
                reason = CoverageReasonCode.UNSUPPORTED
            dimensions.append(CoverageDimensionResult(
                category=category,
                state=state,
                reason_code=reason,
                supporting_refs=refs,
                requirement_ids=tuple(item.requirement_id for item in requirements),
            ))
            continue

        fact_refs: list[CoverageEvidenceRef] = []
        capability_refs: list[CoverageEvidenceRef] = []
        restricted_refs: list[CoverageEvidenceRef] = []
        metric_supports: list[MetricSupport] = []
        category_relevant = False
        for fact in evidence:
            if category is CoverageCategory.METRIC_IMPACT and fact.safe_impact:
                category_relevant = True
            if not _fact_supports_category(fact, category):
                continue
            category_relevant = True
            matched = _matching_boundaries(
                project_boundaries=boundaries,
                subject_type=ClaimSubjectType.EVIDENCE_FACT,
                subject_id=fact.evidence_fact_id,
            )
            ref = _fact_ref(fact, matched)
            if _boundary_restricts(matched, category):
                restricted_refs.append(ref)
                continue
            fact_refs.append(ref)
            if category is CoverageCategory.METRIC_IMPACT:
                support = _fact_metric_support(fact)
                if matched:
                    support = min(
                        (support, *(boundary.metric_support for boundary in matched)),
                        key=lambda item: _METRIC_PRIORITY[item],
                    )
                metric_supports.append(support)
        for capability in capabilities:
            if not _capability_supports_category(capability, category):
                continue
            category_relevant = True
            matched = _matching_boundaries(
                project_boundaries=boundaries,
                subject_type=ClaimSubjectType.CAPABILITY_FACT,
                subject_id=capability.capability_id,
            )
            ref = _capability_ref(capability, evidence_by_id, matched)
            if _boundary_restricts(matched, category):
                restricted_refs.append(ref)
                continue
            capability_refs.append(ref)
            if category is CoverageCategory.METRIC_IMPACT:
                support = capability.metric_support
                if matched:
                    support = min(
                        (support, *(boundary.metric_support for boundary in matched)),
                        key=lambda item: _METRIC_PRIORITY[item],
                    )
                metric_supports.append(support)

        refs = tuple(fact_refs + capability_refs)
        if category is CoverageCategory.METRIC_IMPACT and refs:
            strongest = max(metric_supports, key=lambda item: _METRIC_PRIORITY[item])
            if strongest is MetricSupport.EXPLICIT:
                state = CoverageState.COVERED
                reason = (
                    CoverageReasonCode.SUPPORTED_BY_EVIDENCE
                    if fact_refs else CoverageReasonCode.SUPPORTED_BY_CAPABILITY
                )
            else:
                state = CoverageState.PARTIAL
                reason = CoverageReasonCode.PARTIALLY_SUPPORTED
        elif refs:
            state = CoverageState.COVERED
            reason = (
                CoverageReasonCode.SUPPORTED_BY_EVIDENCE
                if fact_refs else CoverageReasonCode.SUPPORTED_BY_CAPABILITY
            )
        elif restricted_refs:
            state = CoverageState.BLOCKED
            reason = CoverageReasonCode.CLAIM_BOUNDARY_RESTRICTED
            refs = tuple(restricted_refs)
        elif category in _CONTEXT_DEPENDENT_CATEGORIES and not category_relevant:
            state = CoverageState.NOT_APPLICABLE
            reason = (
                CoverageReasonCode.NO_METRIC_SUPPORT
                if category is CoverageCategory.METRIC_IMPACT
                else CoverageReasonCode.NO_RELEVANT_REQUIREMENT
            )
        else:
            state = CoverageState.MISSING
            reason = (
                CoverageReasonCode.WRONG_PROJECT
                if saw_wrong_project and not evidence
                else CoverageReasonCode.NO_METRIC_SUPPORT
                if category is CoverageCategory.METRIC_IMPACT
                else CoverageReasonCode.UNSUPPORTED
            )
        dimensions.append(CoverageDimensionResult(
            category=category,
            state=state,
            reason_code=reason,
            supporting_refs=refs,
        ))

    gaps = tuple(
        CoverageGap(
            category=item.category,
            state=item.state,
            reason_code=item.reason_code,
            related_requirement_ids=gap_requirement_ids.get(item.category, item.requirement_ids),
            current_support_refs=item.supporting_refs,
        )
        for item in dimensions
        if item.state in {CoverageState.PARTIAL, CoverageState.MISSING, CoverageState.BLOCKED}
    )
    return CoverageReport(
        project_id=requested_project,
        dimensions=tuple(dimensions),
        gaps=gaps,
        overall_state=_overall_state(dimensions),
    )


def _requirement_targets_category(
    requirement: CoverageRequirement,
    category: CoverageCategory,
) -> bool:
    words = _words(requirement.target_terms)
    if category is CoverageCategory.JD_MUST_HAVE:
        return True
    if category is CoverageCategory.METRIC_IMPACT:
        return bool(words & _STRICT_METRIC_REQUIREMENT_SIGNALS)
    if category is CoverageCategory.IMPLEMENTATION_MECHANISM:
        signals = _CONCRETE_IMPLEMENTATION_SIGNALS | _words((category.value,))
        return bool(words & signals)
    signals = _CATEGORY_SIGNALS.get(category, frozenset()) | _words((category.value,))
    return bool(words & signals)


def _safe_gap_for_prioritization(
    *,
    coverage_report: CoverageReport,
    gap: CoverageGap,
    related_requirement_ids: Sequence[str],
) -> CoverageGap | None:
    dimension = coverage_report.for_category(gap.category)
    if (
        dimension.state not in {CoverageState.PARTIAL, CoverageState.MISSING, CoverageState.BLOCKED}
        or dimension.state is not gap.state
        or dimension.reason_code is not gap.reason_code
    ):
        return None
    dimension_refs = set(dimension.supporting_refs)
    safe_refs = tuple(
        ref
        for ref in gap.current_support_refs
        if ref.project_id == coverage_report.project_id and ref in dimension_refs
    )
    return CoverageGap(
        category=gap.category,
        state=gap.state,
        reason_code=gap.reason_code,
        related_requirement_ids=tuple(related_requirement_ids),
        current_support_refs=safe_refs,
    )


def _ignored_gap(
    gap: CoverageGap,
    reason_code: GapPriorityReasonCode,
) -> PrioritizedCoverageGap:
    return PrioritizedCoverageGap(
        gap=gap,
        priority=GapPriority.IGNORE,
        searchable=False,
        reason_code=reason_code,
    )


def _prioritize_one_gap(
    *,
    coverage_report: CoverageReport,
    gap: CoverageGap,
    requirements: tuple[CoverageRequirement, ...],
    project_has_evidence: bool,
) -> PrioritizedCoverageGap | None:
    matched_requirement_ids = tuple(
        requirement.requirement_id
        for requirement in requirements
        if _requirement_targets_category(requirement, gap.category)
    )
    related_requirement_ids = _stable_strings(
        (*gap.related_requirement_ids, *matched_requirement_ids),
        "related_requirement_ids",
        maximum=MAX_REQUIREMENTS,
    )
    safe_gap = _safe_gap_for_prioritization(
        coverage_report=coverage_report,
        gap=gap,
        related_requirement_ids=related_requirement_ids,
    )
    if safe_gap is None:
        return None
    if safe_gap.reason_code is CoverageReasonCode.CLAIM_BOUNDARY_RESTRICTED:
        return _ignored_gap(safe_gap, GapPriorityReasonCode.CLAIM_BOUNDARY_RESTRICTED)
    if safe_gap.reason_code is CoverageReasonCode.WRONG_PROJECT:
        return _ignored_gap(safe_gap, GapPriorityReasonCode.INSUFFICIENT_PROJECT_RELEVANCE)
    if safe_gap.state is CoverageState.BLOCKED:
        return _ignored_gap(safe_gap, GapPriorityReasonCode.BLOCKED_UPSTREAM)
    if safe_gap.category is CoverageCategory.PROJECT_IDENTITY:
        return _ignored_gap(safe_gap, GapPriorityReasonCode.INSUFFICIENT_PROJECT_RELEVANCE)

    valid_partial_support = (
        safe_gap.state is CoverageState.PARTIAL and bool(safe_gap.current_support_refs)
    )
    if safe_gap.category is CoverageCategory.METRIC_IMPACT:
        metric_requirement_ids = tuple(
            requirement.requirement_id
            for requirement in requirements
            if _requirement_targets_category(requirement, CoverageCategory.METRIC_IMPACT)
        )
        if valid_partial_support and metric_requirement_ids:
            metric_gap = CoverageGap(
                category=safe_gap.category,
                state=safe_gap.state,
                reason_code=safe_gap.reason_code,
                related_requirement_ids=metric_requirement_ids,
                current_support_refs=safe_gap.current_support_refs,
            )
            return PrioritizedCoverageGap(
                gap=metric_gap,
                priority=GapPriority.MEDIUM,
                searchable=True,
                reason_code=GapPriorityReasonCode.JD_RELEVANT_PARTIAL_SUPPORT,
            )
        return _ignored_gap(safe_gap, GapPriorityReasonCode.NO_METRIC_SIGNAL)

    jd_relevant = bool(related_requirement_ids)
    if safe_gap.category is CoverageCategory.JD_MUST_HAVE:
        if not jd_relevant:
            return _ignored_gap(safe_gap, GapPriorityReasonCode.INSUFFICIENT_PROJECT_RELEVANCE)
        return PrioritizedCoverageGap(
            gap=safe_gap,
            priority=GapPriority.HIGH,
            searchable=True,
            reason_code=(
                GapPriorityReasonCode.JD_RELEVANT_PARTIAL_SUPPORT
                if valid_partial_support
                else GapPriorityReasonCode.JD_MUST_HAVE_GAP
            ),
        )

    if safe_gap.category in _FOLLOWUP_VALUE_CATEGORIES:
        if jd_relevant:
            return PrioritizedCoverageGap(
                gap=safe_gap,
                priority=GapPriority.HIGH,
                searchable=True,
                reason_code=(
                    GapPriorityReasonCode.JD_RELEVANT_PARTIAL_SUPPORT
                    if valid_partial_support
                    else GapPriorityReasonCode.JD_MUST_HAVE_GAP
                ),
            )
        if valid_partial_support:
            return PrioritizedCoverageGap(
                gap=safe_gap,
                priority=GapPriority.MEDIUM,
                searchable=True,
                reason_code=GapPriorityReasonCode.PARTIAL_SUPPORT_WORTH_EXPANDING,
            )
        if safe_gap.state is CoverageState.MISSING and project_has_evidence:
            return PrioritizedCoverageGap(
                gap=safe_gap,
                priority=GapPriority.LOW,
                searchable=True,
                reason_code=GapPriorityReasonCode.HIGH_VALUE_MECHANISM_GAP,
            )
        return _ignored_gap(safe_gap, GapPriorityReasonCode.INSUFFICIENT_PROJECT_RELEVANCE)

    if safe_gap.category is CoverageCategory.RELIABILITY:
        if jd_relevant:
            return PrioritizedCoverageGap(
                gap=safe_gap,
                priority=GapPriority.HIGH,
                searchable=True,
                reason_code=(
                    GapPriorityReasonCode.JD_RELEVANT_PARTIAL_SUPPORT
                    if valid_partial_support
                    else GapPriorityReasonCode.JD_MUST_HAVE_GAP
                ),
            )
        if valid_partial_support:
            return PrioritizedCoverageGap(
                gap=safe_gap,
                priority=GapPriority.MEDIUM,
                searchable=True,
                reason_code=GapPriorityReasonCode.PARTIAL_SUPPORT_WORTH_EXPANDING,
            )
    return _ignored_gap(safe_gap, GapPriorityReasonCode.INSUFFICIENT_PROJECT_RELEVANCE)


def prioritize_project_evidence_gaps(
    *,
    coverage_report: CoverageReport,
    jd_requirements: Any = (),
    max_followup_gaps: int = MAX_PRIORITIZED_FOLLOWUP_GAPS,
) -> tuple[PrioritizedCoverageGap, ...]:
    """Select bounded semantic gaps for a later retrieval-intent step.

    This function classifies only existing coverage gaps.  It does not build
    queries, execute retrieval, inspect source content, or relax blockers.
    """

    if not isinstance(coverage_report, CoverageReport):
        raise TypeError("coverage_report must be a CoverageReport")
    if isinstance(max_followup_gaps, bool) or not isinstance(max_followup_gaps, int):
        raise TypeError("max_followup_gaps must be an integer")
    if not 0 <= max_followup_gaps <= MAX_PRIORITIZED_FOLLOWUP_GAPS:
        raise ValueError(
            f"max_followup_gaps must be between 0 and {MAX_PRIORITIZED_FOLLOWUP_GAPS}"
        )

    requirements = _normalize_requirements(jd_requirements)
    project_has_evidence = any(
        ref.project_id == coverage_report.project_id
        for dimension in coverage_report.dimensions
        for ref in dimension.supporting_refs
    )
    decisions = tuple(
        decision
        for gap in coverage_report.gaps
        if (
            decision := _prioritize_one_gap(
                coverage_report=coverage_report,
                gap=gap,
                requirements=requirements,
                project_has_evidence=project_has_evidence,
            )
        ) is not None
    )
    actionable = tuple(
        item
        for item in sorted(
            decisions,
            key=lambda item: (
                _GAP_PRIORITY_INDEX[item.priority],
                _CATEGORY_INDEX[item.category],
            ),
        )
        if item.priority is not GapPriority.IGNORE and item.searchable
    )
    return actionable[:max_followup_gaps]


__all__ = [
    "CoverageCategory",
    "CoverageDimensionResult",
    "CoverageEvidenceRef",
    "CoverageGap",
    "CoverageReasonCode",
    "CoverageReport",
    "CoverageRequirement",
    "CoverageState",
    "GapPriority",
    "GapPriorityReasonCode",
    "MAX_SUPPORTING_REFS_PER_DIMENSION",
    "MAX_PRIORITIZED_FOLLOWUP_GAPS",
    "PrioritizedCoverageGap",
    "build_project_evidence_coverage_report",
    "prioritize_project_evidence_gaps",
]
