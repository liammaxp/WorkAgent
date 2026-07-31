"""Deterministic, conservative project evidence claim-boundary generation.

The module consumes already-scored Evidence Facts and already-accepted
Capability Facts.  It does not persist data, inspect job descriptions, call a
model, or generate resume prose.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from types import MappingProxyType
from typing import Iterable, Mapping

from backend.project_capability_extractor import (
    SOURCE_CATEGORY_CAPABILITY,
    SOURCE_CATEGORY_DIRECT,
    SOURCE_CATEGORY_PROJECT,
    classify_project_evidence_source_category,
)
from backend.project_capability_taxonomy import (
    get_project_capability_definition,
    validate_project_capability_type,
)
from backend.project_evidence_models import (
    MAX_CLAIM_LENGTH,
    MAX_LIST_ITEMS,
    MAX_NOTES_LENGTH,
    ClaimSubjectType,
    Confidence,
    EvidenceStatus,
    MetricSupport,
    ProjectCapabilityFact,
    ProjectClaimBoundary,
    ProjectEvidenceFact,
    build_project_evidence_stable_id,
)


MIN_DIRECT_CLAIM_SCORE = 60
MIN_CONTEXT_CLAIM_SCORE = 40
MAX_DECISION_SAMPLES = 100
MAX_FORBIDDEN_CLAIMS = 40
MAX_ALLOWED_CLAIMS = MAX_LIST_ITEMS

CLAIM_TYPES = (
    "problem",
    "mechanism",
    "implementation",
    "bounded_impact",
    "technology",
    "capability",
    "metric",
    "architecture",
    "workflow",
    "validation",
    "testing",
    "persistence",
    "retrieval",
)
_CLAIM_TYPE_SET = frozenset(CLAIM_TYPES)
CLAIM_TYPE_PRIORITY = MappingProxyType({
    "problem": 0,
    "mechanism": 1,
    "implementation": 2,
    "technology": 3,
    "capability": 4,
    "bounded_impact": 5,
    "metric": 6,
    "architecture": 7,
    "workflow": 8,
    "validation": 9,
    "testing": 10,
    "persistence": 11,
    "retrieval": 12,
})
CLAIM_LIMITS = MappingProxyType({
    "problem": 10,
    "mechanism": 30,
    "implementation": 40,
    "bounded_impact": 20,
    "technology": 30,
    "capability": 18,
    "metric": 10,
    "architecture": 10,
    "workflow": 10,
    "validation": 10,
    "testing": 10,
    "persistence": 10,
    "retrieval": 10,
})
_SERIALIZED_PREFIX = MappingProxyType({
    **{claim_type: claim_type for claim_type in CLAIM_TYPES},
    "bounded_impact": "impact",
})
_PREFIX_TO_TYPE = MappingProxyType({value: key for key, value in _SERIALIZED_PREFIX.items()})

GLOBAL_FORBIDDEN_POLICIES = tuple(sorted({
    "absolute_guarantee",
    "cross_project_technology",
    "hallucination_elimination",
    "invented_numeric_impact",
    "legacy_capability_label_promotion",
    "mechanism_to_impact_inference",
    "unsupported_architecture",
    "unsupported_ats_improvement",
    "unsupported_capability",
    "unsupported_cost_reduction",
    "unsupported_deployment_claim",
    "unsupported_enterprise_grade",
    "unsupported_factuality_claim",
    "unsupported_high_availability",
    "unsupported_latency_improvement",
    "unsupported_metric",
    "unsupported_performance_improvement",
    "unsupported_production_scale",
    "unsupported_reliability_improvement",
    "unsupported_token_reduction",
    "unsupported_user_adoption",
}))

_GENERIC_MECHANISMS = frozenset({
    "added intelligent features",
    "added validation",
    "built ai-powered system",
    "built an ai-powered system",
    "developed backend functionality",
    "enhanced retrieval",
    "improved pipeline",
    "improved project workflow",
    "improved reliability",
    "optimized performance",
    "unknown",
    "updated backend logic",
    "used advanced technologies",
    "worked on resume generation",
})
_GENERIC_IMPLEMENTATION = _GENERIC_MECHANISMS | frozenset({
    "implementation",
    "implementation details",
    "updated code",
})
_PATH_ANCHORS = frozenset({
    "app", "backend", "client", "frontend", "lib", "server", "src", "tests",
})
_IDENTIFIER_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[/\\]")
_EMBEDDED_WINDOWS_ABSOLUTE_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[/\\]")
_NUMERIC_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:~\s*)?\d+(?:\.\d+)?(?:\s*(?:%|percent|x|times))?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_APPROXIMATE_RE = re.compile(
    r"(?:\b(?:about|approximately|around|roughly|up[\s_-]+to)\b|~\s*\d|\d\s*[-\u2013]\s*\d)",
    re.IGNORECASE,
)
_NON_RESUME_METRIC_RE = re.compile(
    r"(?:\b(?:quality[\s_-]+score|tests?|test[\s_-]+cases?|lines?|files?|commits?|"
    r"versions?|http[\s_-]+status|status[\s_-]+codes?|model[\s_-]+versions?|issues?|"
    r"sha|dates?|phases?|steps?|stages?)\b|\b[0-9a-f]{7,64}\b|"
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b)",
    re.IGNORECASE,
)
_UNSAFE_CONTENT_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\b(?:raw_text|raw_patch|full_patch|complete_diff|repository_dump|full_file_content)\s*[:=]",
    r"(?:^|\s)diff\s+--git\s+",
    r"(?:^|\s)@@\s+-\d+",
    r"(?:^|\s)(?:---|\+\+\+)\s+[ab]/",
    r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
    r"\b(?:api[_ -]?key|access[_ -]?token|password|authorization)\s*[:=]\s*\S+",
    r"\b(?:def|class|function)\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*[:{]",
    r"\b(?:import|from)\s+[A-Za-z_][A-Za-z0-9_.]*\s*(?:import\s+|;)",
))
_IMPACT_POLICY_PATTERNS = MappingProxyType({
    "absolute_guarantee": re.compile(
        r"\b(?:guarantee(?:d|s)?|eliminat(?:e|ed|es|ing)|fully[\s_-]+prevent(?:ed|s|ing)?|"
        r"zero[\s_-]+(?:errors?|failures?|hallucinations?))\b",
        re.IGNORECASE,
    ),
    "hallucination_elimination": re.compile(
        r"\b(?:eliminat(?:e|ed|es|ing)|zero)[\s_-]+hallucinations?\b",
        re.IGNORECASE,
    ),
    "unsupported_factuality_claim": re.compile(
        r"\b(?:guaranteed?[\s_-]+factual(?:ity)?|factually[\s_-]+correct)\b",
        re.IGNORECASE,
    ),
    "unsupported_token_reduction": re.compile(
        r"\b(?:reduc(?:e|ed|es|ing)|sav(?:e|ed|es|ing))[\s_-]+(?:token|context)s?\b",
        re.IGNORECASE,
    ),
    "unsupported_cost_reduction": re.compile(
        r"\b(?:reduc(?:e|ed|es|ing)|sav(?:e|ed|es|ing))[\s_-]+(?:cost|money|dollars?)\b",
        re.IGNORECASE,
    ),
    "unsupported_latency_improvement": re.compile(
        r"\b(?:improv(?:e|ed|es|ing)|reduc(?:e|ed|es|ing))[\s_-]+(?:latency|response[\s_-]+time)\b",
        re.IGNORECASE,
    ),
    "unsupported_ats_improvement": re.compile(
        r"\b(?:improv(?:e|ed|es|ing)|increas(?:e|ed|es|ing))[\s_-]+ats(?:[\s_-]+(?:success|score))?\b",
        re.IGNORECASE,
    ),
    "unsupported_reliability_improvement": re.compile(
        r"\bimprov(?:e|ed|es|ing)[\s_-]+reliability\b",
        re.IGNORECASE,
    ),
    "unsupported_performance_improvement": re.compile(
        r"\b(?:improv(?:e|ed|es|ing)|optimiz(?:e|ed|es|ing))[\s_-]+performance\b",
        re.IGNORECASE,
    ),
    "unsupported_production_scale": re.compile(r"\bproduction[\s_-]+scale\b", re.IGNORECASE),
    "unsupported_enterprise_grade": re.compile(r"\benterprise[\s_-]+grade\b", re.IGNORECASE),
    "unsupported_high_availability": re.compile(r"\bhigh[\s_-]+availability\b", re.IGNORECASE),
    "unsupported_deployment_claim": re.compile(
        r"\b(?:deployed?|deployment|production[\s_-]+deployment)\b",
        re.IGNORECASE,
    ),
    "unsupported_user_adoption": re.compile(
        r"\b(?:users?|customers?)[\s_-]+(?:adopted?|used|served)\b",
        re.IGNORECASE,
    ),
})


@dataclass(frozen=True)
class ProjectStructuredClaim:
    project_id: str
    claim_type: str
    value: str
    evidence_fact_ids: tuple[str, ...]
    capability_fact_ids: tuple[str, ...]
    metric_support: str
    confidence: str
    rule_id: str
    quality_score: int = 0
    source_category: str = "other"

    def __post_init__(self) -> None:
        normalized = _normalize(self.value)
        if self.claim_type not in _CLAIM_TYPE_SET:
            raise ValueError("unknown claim type")
        if not _normalize(self.project_id):
            raise ValueError("missing project ID")
        if not normalized:
            raise ValueError("blank claim value")
        if self.metric_support not in {item.value for item in MetricSupport}:
            raise ValueError("invalid metric support")
        if self.confidence not in {item.value for item in Confidence}:
            raise ValueError("invalid confidence")
        if not _IDENTIFIER_RE.fullmatch(self.rule_id):
            raise ValueError("invalid claim rule ID")
        object.__setattr__(self, "value", normalized)
        object.__setattr__(self, "evidence_fact_ids", tuple(sorted(set(self.evidence_fact_ids))))
        object.__setattr__(self, "capability_fact_ids", tuple(sorted(set(self.capability_fact_ids))))
        object.__setattr__(self, "quality_score", max(0, min(100, int(self.quality_score))))

    @property
    def serialized(self) -> str:
        return f"{_SERIALIZED_PREFIX[self.claim_type]}:{self.value}"


@dataclass(frozen=True)
class ProjectClaimBoundaryDecision:
    project_id: str
    decision_code: str
    evidence_fact_id: str = ""
    capability_fact_id: str = ""


@dataclass(frozen=True)
class ProjectClaimBoundaryConflict:
    project_id: str
    claim_type: str
    claim_hash: str
    reason_code: str
    evidence_fact_ids: tuple[str, ...] = ()
    capability_fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectClaimBoundaryGroupCount:
    project_id: str
    claim_type: str
    confidence: str
    metric_support: str
    evidence_source_category: str
    count: int


@dataclass(frozen=True)
class ProjectForbiddenPolicyCount:
    policy_code: str
    count: int


@dataclass(frozen=True)
class ProjectClaimBoundaryValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectClaimBoundaryReport:
    project_count: int
    evidence_fact_count: int
    capability_fact_count: int
    qualifying_evidence_fact_count: int
    evidence_boundaries_created: int
    capability_boundaries_created: int
    project_boundaries_created: int
    allowed_claim_count: int
    forbidden_claim_count: int
    mechanism_claim_count: int
    implementation_claim_count: int
    problem_claim_count: int
    impact_claim_count: int
    technology_claim_count: int
    capability_claim_count: int
    metric_claim_count: int
    weak_fact_blocked_count: int
    rejected_fact_blocked_count: int
    low_quality_blocked_count: int
    contextual_only_restricted_count: int
    unsupported_metric_blocked_count: int
    unsupported_impact_blocked_count: int
    unsupported_capability_blocked_count: int
    project_mismatch_count: int
    conflict_count: int
    truncated_claim_count: int
    grouped_counts: tuple[ProjectClaimBoundaryGroupCount, ...] = ()
    forbidden_policy_counts: tuple[ProjectForbiddenPolicyCount, ...] = ()
    decisions: tuple[ProjectClaimBoundaryDecision, ...] = ()
    conflicts: tuple[ProjectClaimBoundaryConflict, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _ClaimSourceResult:
    claims: tuple[ProjectStructuredClaim, ...] = ()
    forbidden: tuple[str, ...] = ()
    blocked_claims: tuple[str, ...] = ()
    decision_code: str = "no_allowed_claim"
    unsupported_metric_count: int = 0
    unsupported_impact_count: int = 0
    project_mismatch_count: int = 0


@dataclass(frozen=True)
class _ProjectBoundaryResult:
    boundary: ProjectClaimBoundary | None
    claims: tuple[ProjectStructuredClaim, ...]
    forbidden: tuple[str, ...]
    evidence_boundary_count: int
    capability_boundary_count: int
    qualifying_evidence_count: int
    unsupported_metric_count: int
    unsupported_impact_count: int
    unsupported_capability_count: int
    project_mismatch_count: int
    truncated_count: int
    conflicts: tuple[ProjectClaimBoundaryConflict, ...]
    decisions: tuple[ProjectClaimBoundaryDecision, ...]


def _normalize(value: str) -> str:
    return " ".join(str(value or "").split())


def normalize_project_claim(value: str) -> str:
    """Return the authoritative whitespace-normalized claim representation."""

    if not isinstance(value, str):
        raise TypeError("claim value must be a string")
    return _normalize(value)


def normalize_project_serialized_claim(value: str) -> str:
    """Validate and normalize one serialized project claim without widening it."""

    normalized = normalize_project_claim(value)
    prefix, separator, claim_value = normalized.partition(":")
    if not separator or prefix not in _PREFIX_TO_TYPE:
        raise ValueError("unsupported serialized claim type")
    safe_value = _safe_claim_value(_PREFIX_TO_TYPE[prefix], claim_value)
    if not safe_value:
        raise ValueError("unsafe or blank serialized claim")
    return f"{prefix}:{safe_value}"


def _quality_score(fact: ProjectEvidenceFact) -> int:
    value = fact.quality_score
    if value is None or isinstance(value, bool):
        return 0
    return max(0, min(100, int(value)))


def _quality_blockers(fact: ProjectEvidenceFact) -> tuple[str, ...]:
    values = fact.quality_breakdown.get("blocker_codes", ())
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(sorted({_normalize(value) for value in values if _normalize(value)}))


def _valid_provenance(fact: ProjectEvidenceFact) -> bool:
    return bool(fact.source_refs) and all(ref.project_id == fact.project_id for ref in fact.source_refs)


def _generic_mechanism(value: str) -> bool:
    parts = [_normalize(part).casefold() for part in value.split(";") if _normalize(part)]
    return not parts or all(part in _GENERIC_MECHANISMS for part in parts)


def _concrete_implementation(values: Iterable[str]) -> bool:
    normalized = [_normalize(value).casefold() for value in values if _normalize(value)]
    return bool(normalized) and any(value not in _GENERIC_IMPLEMENTATION for value in normalized)


def _contains_unsafe_content(value: str) -> bool:
    return any(pattern.search(value) for pattern in _UNSAFE_CONTENT_PATTERNS)


def _safe_implementation(value: str) -> str:
    normalized = _normalize(value.replace("\\", "/"))
    if not normalized or _contains_unsafe_content(normalized):
        return ""
    if re.match(r"^(?:GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\s+/", normalized, re.IGNORECASE):
        return normalized
    if normalized.startswith("/api/"):
        return normalized
    path_part, separator, symbol = normalized.partition("::")
    is_absolute = bool(_WINDOWS_ABSOLUTE_RE.match(path_part)) or path_part.startswith(("/", "~/", "//"))
    if is_absolute:
        parts = [part for part in path_part.replace("\\", "/").split("/") if part and part not in {".", ".."}]
        anchor_index = next(
            (index for index, part in enumerate(parts) if part.casefold() in _PATH_ANCHORS),
            None,
        )
        path_part = "/".join(parts[anchor_index:] if anchor_index is not None else parts[-1:])
        normalized = path_part + (f"::{symbol}" if separator and symbol else "")
    elif _EMBEDDED_WINDOWS_ABSOLUTE_RE.search(normalized):
        return ""
    if normalized.startswith(("../", "..\\")) or not normalized:
        return ""
    return normalized


def _safe_claim_value(claim_type: str, value: str) -> str:
    normalized = _safe_implementation(value) if claim_type == "implementation" else _normalize(value)
    if not normalized or _contains_unsafe_content(normalized):
        return ""
    if claim_type != "implementation" and (
        _EMBEDDED_WINDOWS_ABSOLUTE_RE.search(normalized)
        or re.search(r"(?:^|\s)/(?:Users|home|var|tmp|etc)/", normalized, re.IGNORECASE)
    ):
        return ""
    prefix = _SERIALIZED_PREFIX[claim_type]
    if len(prefix) + 1 + len(normalized) > MAX_CLAIM_LENGTH:
        return ""
    if claim_type == "implementation" and (
        _WINDOWS_ABSOLUTE_RE.match(normalized) or normalized.startswith(("~/", "//"))
    ):
        return ""
    return normalized


def _fact_confidence(fact: ProjectEvidenceFact, source_category: str) -> str:
    if source_category == SOURCE_CATEGORY_PROJECT:
        return Confidence.LOW.value
    return Confidence.HIGH.value if _quality_score(fact) >= 80 else Confidence.MEDIUM.value


def _metric_policies(value: str, support: MetricSupport) -> tuple[bool, tuple[str, ...]]:
    if not _NUMERIC_RE.search(value):
        return False, ()
    if _NON_RESUME_METRIC_RE.search(value):
        return False, ()
    if support is MetricSupport.NONE:
        return False, ("invented_numeric_impact", "unsupported_metric")
    if support is MetricSupport.APPROXIMATE and not _APPROXIMATE_RE.search(value):
        return False, ("unsupported_metric",)
    return True, ()


def is_project_resume_metric_claim(value: str) -> bool:
    """Return whether a numeric value is a resume metric rather than an identifier/count."""

    normalized = normalize_project_claim(value)
    return bool(_NUMERIC_RE.search(normalized) and not _NON_RESUME_METRIC_RE.search(normalized))


def evaluate_project_numeric_claim(
    value: str,
    metric_support: MetricSupport | str,
) -> tuple[bool, tuple[str, ...]]:
    """Apply the existing exact/approximate numeric-claim policy."""

    support = metric_support if isinstance(metric_support, MetricSupport) else MetricSupport(metric_support)
    return _metric_policies(normalize_project_claim(value), support)


def _impact_blockers(value: str) -> tuple[str, ...]:
    return tuple(sorted(code for code, pattern in _IMPACT_POLICY_PATTERNS.items() if pattern.search(value)))


def get_project_claim_safety_blockers(value: str) -> tuple[str, ...]:
    """Return existing absolute/unsupported impact policy codes for one claim."""

    return _impact_blockers(normalize_project_claim(value))


def _new_claim(
    fact: ProjectEvidenceFact,
    claim_type: str,
    value: str,
    rule_id: str,
    *,
    source_category: str,
    metric_support: MetricSupport = MetricSupport.NONE,
) -> ProjectStructuredClaim | None:
    safe_value = _safe_claim_value(claim_type, value)
    if not safe_value:
        return None
    return ProjectStructuredClaim(
        project_id=fact.project_id,
        claim_type=claim_type,
        value=safe_value,
        evidence_fact_ids=(fact.evidence_fact_id,),
        capability_fact_ids=(),
        metric_support=metric_support.value,
        confidence=_fact_confidence(fact, source_category),
        rule_id=rule_id,
        quality_score=_quality_score(fact),
        source_category=source_category,
    )


def _legacy_capability_policies(fact: ProjectEvidenceFact) -> tuple[str, ...]:
    if classify_project_evidence_source_category(fact) != SOURCE_CATEGORY_CAPABILITY:
        return ()
    policies: set[str] = {"legacy_capability_label_promotion"}
    for value in fact.technical_tags:
        try:
            canonical = validate_project_capability_type(value)
        except (TypeError, ValueError):
            continue
        policies.add(f"unsupported_capability:{canonical}")
    return tuple(sorted(policies))


def _claims_from_fact(fact: ProjectEvidenceFact) -> _ClaimSourceResult:
    if not isinstance(fact, ProjectEvidenceFact):
        raise TypeError("claim-boundary generation expects ProjectEvidenceFact")
    category = classify_project_evidence_source_category(fact)
    quality = _quality_score(fact)
    forbidden = set(GLOBAL_FORBIDDEN_POLICIES)
    forbidden.update(_legacy_capability_policies(fact))
    if not _valid_provenance(fact):
        forbidden.add("cross_project_technology")
        return _ClaimSourceResult(
            forbidden=tuple(sorted(forbidden)),
            decision_code="project_provenance_mismatch" if fact.source_refs else "missing_provenance",
            project_mismatch_count=1 if fact.source_refs else 0,
        )
    if fact.status is EvidenceStatus.WEAK:
        return _ClaimSourceResult(forbidden=tuple(sorted(forbidden)), decision_code="weak_fact_blocked")
    if fact.status is EvidenceStatus.REJECTED:
        return _ClaimSourceResult(forbidden=tuple(sorted(forbidden)), decision_code="rejected_fact_blocked")
    if category == SOURCE_CATEGORY_CAPABILITY:
        return _ClaimSourceResult(forbidden=tuple(sorted(forbidden)), decision_code="legacy_capability_context_blocked")
    if category == SOURCE_CATEGORY_PROJECT:
        if fact.status is not EvidenceStatus.SUPPORTING or quality < MIN_CONTEXT_CLAIM_SCORE or _quality_blockers(fact):
            return _ClaimSourceResult(forbidden=tuple(sorted(forbidden)), decision_code="context_fact_blocked")
        claim = _new_claim(
            fact,
            "problem",
            fact.problem,
            "explicit_project_context_problem",
            source_category=category,
        ) if fact.problem else None
        return _ClaimSourceResult(
            claims=(claim,) if claim is not None else (),
            forbidden=tuple(sorted(forbidden)),
            decision_code="context_problem_allowed" if claim is not None else "contextual_only_restricted",
        )
    if category != SOURCE_CATEGORY_DIRECT or fact.status is not EvidenceStatus.ACCEPTED:
        return _ClaimSourceResult(forbidden=tuple(sorted(forbidden)), decision_code="unsupported_source_category")
    if quality < MIN_DIRECT_CLAIM_SCORE:
        return _ClaimSourceResult(forbidden=tuple(sorted(forbidden)), decision_code="low_quality_blocked")
    if _quality_blockers(fact):
        return _ClaimSourceResult(forbidden=tuple(sorted(forbidden)), decision_code="claim_safety_blocked")
    if _generic_mechanism(fact.mechanism):
        return _ClaimSourceResult(forbidden=tuple(sorted(forbidden)), decision_code="generic_mechanism_blocked")

    claims: list[ProjectStructuredClaim] = []
    blocked_claims: set[str] = set()
    unsupported_metric_count = 0
    unsupported_impact_count = 0
    if fact.problem:
        claim = _new_claim(fact, "problem", fact.problem, "explicit_problem", source_category=category)
        if claim is not None:
            claims.append(claim)
    mechanism = _new_claim(fact, "mechanism", fact.mechanism, "explicit_mechanism", source_category=category)
    if mechanism is not None:
        claims.append(mechanism)
    elif fact.mechanism:
        forbidden.add("oversized_or_unsafe_claim")
    concrete_implementation = _concrete_implementation(fact.implementation)
    if concrete_implementation:
        for value in fact.implementation:
            claim = _new_claim(fact, "implementation", value, "explicit_implementation", source_category=category)
            if claim is not None:
                claims.append(claim)
            elif _normalize(value):
                forbidden.add("oversized_or_unsafe_claim")
    if concrete_implementation:
        for tag in fact.technical_tags:
            claim = _new_claim(fact, "technology", tag, "implementation_linked_technical_tag", source_category=category)
            if claim is not None:
                claims.append(claim)

    for impact in fact.safe_impact:
        normalized = _normalize(impact)
        impact_policies = _impact_blockers(normalized)
        metric_allowed, metric_policies = _metric_policies(normalized, fact.metric_support)
        if impact_policies:
            forbidden.update(impact_policies)
            blocked_claims.add(f"impact:{normalized}")
            unsupported_impact_count += 1
            continue
        if _NUMERIC_RE.search(normalized) and not metric_allowed and metric_policies:
            forbidden.update(metric_policies)
            blocked_claims.update({f"impact:{normalized}", f"metric:{normalized}"})
            unsupported_metric_count += 1
            continue
        impact_claim = _new_claim(
            fact,
            "bounded_impact",
            normalized,
            "explicit_safe_impact",
            source_category=category,
        )
        if impact_claim is not None:
            claims.append(impact_claim)
        if metric_allowed:
            metric_claim = _new_claim(
                fact,
                "metric",
                normalized,
                "explicit_supported_metric",
                source_category=category,
                metric_support=fact.metric_support,
            )
            if metric_claim is not None:
                claims.append(metric_claim)

    for value in fact.forbidden_claims:
        normalized = _normalize(value)
        if not normalized:
            continue
        prefix = normalized.partition(":")[0]
        if prefix in _PREFIX_TO_TYPE and normalized.partition(":")[2]:
            blocked_claims.add(normalized)
            forbidden.add(normalized)
        elif _contains_unsafe_content(normalized) or len(normalized) + 17 > MAX_CLAIM_LENGTH:
            forbidden.add("unsafe_source_forbidden_content")
        else:
            forbidden.add(f"source_forbidden:{normalized}")
    return _ClaimSourceResult(
        claims=tuple(claims),
        forbidden=tuple(sorted(forbidden)),
        blocked_claims=tuple(sorted(blocked_claims)),
        decision_code="evidence_claims_allowed" if claims else "no_safe_claim_value",
        unsupported_metric_count=unsupported_metric_count,
        unsupported_impact_count=unsupported_impact_count,
    )


def _claims_from_capability(
    capability: ProjectCapabilityFact,
    evidence_facts_by_id: Mapping[str, ProjectEvidenceFact],
) -> _ClaimSourceResult:
    if not isinstance(capability, ProjectCapabilityFact):
        raise TypeError("claim-boundary generation expects ProjectCapabilityFact")
    forbidden = set(GLOBAL_FORBIDDEN_POLICIES)
    if not capability.present:
        forbidden.add("unsupported_capability")
        return _ClaimSourceResult(forbidden=tuple(sorted(forbidden)), decision_code="capability_not_present")
    try:
        canonical = validate_project_capability_type(capability.capability_type)
        definition = get_project_capability_definition(canonical)
    except (TypeError, ValueError):
        forbidden.add("unsupported_capability")
        return _ClaimSourceResult(forbidden=tuple(sorted(forbidden)), decision_code="unknown_capability_type")
    evidence: list[ProjectEvidenceFact] = []
    for evidence_id in capability.source_evidence_fact_ids:
        fact = evidence_facts_by_id.get(evidence_id)
        if fact is None:
            forbidden.add(f"unsupported_capability:{canonical}")
            return _ClaimSourceResult(forbidden=tuple(sorted(forbidden)), decision_code="unknown_capability_evidence")
        if fact.project_id != capability.project_id:
            forbidden.update({"cross_project_technology", f"unsupported_capability:{canonical}"})
            return _ClaimSourceResult(
                forbidden=tuple(sorted(forbidden)),
                decision_code="capability_project_mismatch",
                project_mismatch_count=1,
            )
        evidence.append(fact)
    forbidden.update(f"taxonomy:{code}" for code in definition.forbidden_inferences)
    for value in capability.forbidden_claims:
        normalized = _normalize(value)
        if normalized and not _contains_unsafe_content(normalized) and len(normalized) + 17 <= MAX_CLAIM_LENGTH:
            forbidden.add(f"source_forbidden:{normalized}")
    quality = max((_quality_score(fact) for fact in evidence), default=0)
    source_category = (
        SOURCE_CATEGORY_DIRECT
        if any(classify_project_evidence_source_category(fact) == SOURCE_CATEGORY_DIRECT for fact in evidence)
        else SOURCE_CATEGORY_PROJECT
    )
    confidence = capability.confidence.value
    claim = ProjectStructuredClaim(
        project_id=capability.project_id,
        claim_type="capability",
        value=canonical,
        evidence_fact_ids=tuple(capability.source_evidence_fact_ids),
        capability_fact_ids=(capability.capability_id,),
        metric_support=MetricSupport.NONE.value,
        confidence=confidence,
        rule_id="present_capability_fact",
        quality_score=quality,
        source_category=source_category,
    )
    return _ClaimSourceResult(
        claims=(claim,),
        forbidden=tuple(sorted(forbidden)),
        decision_code="capability_claim_allowed",
    )


def _claim_key(claim: ProjectStructuredClaim) -> tuple[str, str]:
    return (claim.claim_type, claim.value.casefold())


def _claim_hash(serialized: str) -> str:
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _merge_metric_support(values: Iterable[str], claim_type: str) -> str:
    supports = set(values)
    if claim_type != "metric" or not supports:
        return MetricSupport.NONE.value
    if MetricSupport.NONE.value in supports:
        return MetricSupport.NONE.value
    if MetricSupport.APPROXIMATE.value in supports:
        return MetricSupport.APPROXIMATE.value
    return MetricSupport.EXPLICIT.value


def _strongest_confidence(values: Iterable[str]) -> str:
    priority = {Confidence.LOW.value: 0, Confidence.MEDIUM.value: 1, Confidence.HIGH.value: 2}
    return max(values, key=lambda value: priority[value], default=Confidence.LOW.value)


def _strongest_source_category(values: Iterable[str]) -> str:
    priority = {SOURCE_CATEGORY_DIRECT: 0, SOURCE_CATEGORY_PROJECT: 1, SOURCE_CATEGORY_CAPABILITY: 2, "other": 3}
    return min(values, key=lambda value: priority.get(value, 4), default="other")


def _merge_claims(claims: Iterable[ProjectStructuredClaim]) -> list[ProjectStructuredClaim]:
    grouped: dict[tuple[str, str], list[ProjectStructuredClaim]] = {}
    for claim in claims:
        grouped.setdefault(_claim_key(claim), []).append(claim)
    merged: list[ProjectStructuredClaim] = []
    for key in sorted(grouped, key=lambda item: (CLAIM_TYPE_PRIORITY[item[0]], item[1])):
        items = grouped[key]
        exemplar = min(items, key=lambda item: (item.value.casefold(), item.value, item.rule_id))
        merged.append(ProjectStructuredClaim(
            project_id=exemplar.project_id,
            claim_type=exemplar.claim_type,
            value=exemplar.value,
            evidence_fact_ids=tuple(sorted({value for item in items for value in item.evidence_fact_ids})),
            capability_fact_ids=tuple(sorted({value for item in items for value in item.capability_fact_ids})),
            metric_support=_merge_metric_support((item.metric_support for item in items), exemplar.claim_type),
            confidence=_strongest_confidence(item.confidence for item in items),
            rule_id=min(item.rule_id for item in items),
            quality_score=max(item.quality_score for item in items),
            source_category=_strongest_source_category(item.source_category for item in items),
        ))
    return merged


def _claim_rank(claim: ProjectStructuredClaim) -> tuple[int, int, int, int, str]:
    source_priority = 0 if claim.source_category == SOURCE_CATEGORY_DIRECT else 1
    return (
        -claim.quality_score,
        source_priority,
        CLAIM_TYPE_PRIORITY[claim.claim_type],
        -(len(claim.evidence_fact_ids) + len(claim.capability_fact_ids)),
        _claim_hash(claim.serialized),
    )


def _limit_claims(claims: Iterable[ProjectStructuredClaim]) -> tuple[list[ProjectStructuredClaim], int]:
    grouped: dict[str, list[ProjectStructuredClaim]] = {}
    for claim in claims:
        grouped.setdefault(claim.claim_type, []).append(claim)
    selected: list[ProjectStructuredClaim] = []
    truncated = 0
    for claim_type in sorted(grouped, key=lambda value: CLAIM_TYPE_PRIORITY[value]):
        ordered = sorted(grouped[claim_type], key=_claim_rank)
        limit = CLAIM_LIMITS[claim_type]
        selected.extend(ordered[:limit])
        truncated += max(0, len(ordered) - limit)
    if len(selected) > MAX_ALLOWED_CLAIMS:
        selected = sorted(selected, key=_claim_rank)
        truncated += len(selected) - MAX_ALLOWED_CLAIMS
        selected = selected[:MAX_ALLOWED_CLAIMS]
    selected.sort(key=lambda item: (CLAIM_TYPE_PRIORITY[item.claim_type], item.value.casefold(), item.value))
    return selected, truncated


def _claim_note(claim: ProjectStructuredClaim) -> str:
    evidence = ",".join(claim.evidence_fact_ids)
    capabilities = ",".join(claim.capability_fact_ids)
    return (
        f"claim_meta|{_claim_hash(claim.serialized)}|{claim.confidence}|{claim.metric_support}|"
        f"{claim.source_category}|{claim.rule_id}|e={evidence}|c={capabilities}"
    )


def _boundary_metric_support(claims: Iterable[ProjectStructuredClaim]) -> MetricSupport:
    values = [claim.metric_support for claim in claims if claim.claim_type == "metric"]
    if not values:
        return MetricSupport.NONE
    if MetricSupport.NONE.value in values:
        return MetricSupport.NONE
    if MetricSupport.APPROXIMATE.value in values:
        return MetricSupport.APPROXIMATE
    return MetricSupport.EXPLICIT


def _build_boundary(
    project_id: str,
    subject_type: ClaimSubjectType,
    subject_id: str,
    claims: Iterable[ProjectStructuredClaim],
    forbidden_values: Iterable[str],
) -> tuple[ProjectClaimBoundary | None, tuple[ProjectStructuredClaim, ...], tuple[str, ...], int]:
    limited, truncated = _limit_claims(_merge_claims(claims))
    supported: list[ProjectStructuredClaim] = []
    for claim in limited:
        if len(_claim_note(claim)) > MAX_NOTES_LENGTH:
            truncated += 1
            continue
        supported.append(claim)
    if not supported:
        return None, (), (), truncated
    forbidden = sorted(set(forbidden_values))
    if len(forbidden) > MAX_FORBIDDEN_CLAIMS:
        truncated += len(forbidden) - MAX_FORBIDDEN_CLAIMS
        forbidden = forbidden[:MAX_FORBIDDEN_CLAIMS]
    allowed = [claim.serialized for claim in supported]
    blocked = {value.casefold() for value in forbidden}
    supported = [claim for claim in supported if claim.serialized.casefold() not in blocked]
    allowed = [claim.serialized for claim in supported]
    if not supported:
        return None, (), tuple(forbidden), truncated
    notes = sorted(_claim_note(claim) for claim in supported)
    metric_support = _boundary_metric_support(supported)
    boundary_id = build_project_evidence_stable_id("pcb_", project_id, {
        "subject_type": subject_type.value,
        "subject_id": subject_id,
        "allowed_claims": allowed,
        "forbidden_claims": forbidden,
        "metric_support": metric_support.value,
        "notes": notes,
    })
    boundary = ProjectClaimBoundary(
        project_id=project_id,
        subject_type=subject_type,
        subject_id=subject_id,
        allowed_claims=allowed,
        forbidden_claims=forbidden,
        metric_support=metric_support,
        notes=notes,
        boundary_id=boundary_id,
    )
    return boundary, tuple(supported), tuple(forbidden), truncated


def build_project_evidence_claim_boundary(fact: ProjectEvidenceFact) -> ProjectClaimBoundary | None:
    """Build one evidence-subject boundary without mutating the fact."""

    result = _claims_from_fact(fact)
    boundary, _claims, _forbidden, _truncated = _build_boundary(
        fact.project_id,
        ClaimSubjectType.EVIDENCE_FACT,
        fact.evidence_fact_id,
        result.claims,
        (*result.forbidden, *result.blocked_claims),
    )
    return boundary


def build_project_capability_claim_boundary(
    capability: ProjectCapabilityFact,
    *,
    evidence_facts_by_id: Mapping[str, ProjectEvidenceFact],
) -> ProjectClaimBoundary | None:
    """Build one boundary only for a present, canonical, fully-resolved capability."""

    result = _claims_from_capability(capability, evidence_facts_by_id)
    boundary, _claims, _forbidden, _truncated = _build_boundary(
        capability.project_id,
        ClaimSubjectType.CAPABILITY_FACT,
        capability.capability_id,
        result.claims,
        result.forbidden,
    )
    return boundary


def _project_subject_id(project_id: str) -> str:
    if len(project_id) <= 100:
        return project_id
    return build_project_evidence_stable_id("pcb_", project_id, {"subject_type": "project"})


def _build_project_result(
    project_id: str,
    evidence_facts: Iterable[ProjectEvidenceFact],
    capability_facts: Iterable[ProjectCapabilityFact],
    *,
    evidence_facts_by_id: Mapping[str, ProjectEvidenceFact] | None = None,
) -> _ProjectBoundaryResult:
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("project_id must not be blank")
    facts = list(evidence_facts)
    capabilities = list(capability_facts)
    if any(not isinstance(fact, ProjectEvidenceFact) for fact in facts):
        raise TypeError("evidence_facts must contain ProjectEvidenceFact values")
    if any(not isinstance(capability, ProjectCapabilityFact) for capability in capabilities):
        raise TypeError("capability_facts must contain ProjectCapabilityFact values")
    mismatched = sum(fact.project_id != project_id for fact in facts) + sum(
        capability.project_id != project_id for capability in capabilities
    )
    facts = [fact for fact in facts if fact.project_id == project_id]
    capabilities = [capability for capability in capabilities if capability.project_id == project_id]
    by_id = dict(evidence_facts_by_id) if evidence_facts_by_id is not None else {
        fact.evidence_fact_id: fact for fact in facts
    }
    claims: list[ProjectStructuredClaim] = []
    forbidden: set[str] = set(GLOBAL_FORBIDDEN_POLICIES)
    blocked: set[str] = set()
    decisions: list[ProjectClaimBoundaryDecision] = []
    evidence_boundary_count = 0
    capability_boundary_count = 0
    qualifying_evidence = 0
    unsupported_metric = 0
    unsupported_impact = 0
    unsupported_capability = 0
    project_mismatch = mismatched
    for fact in sorted(facts, key=lambda item: item.evidence_fact_id):
        result = _claims_from_fact(fact)
        claims.extend(result.claims)
        forbidden.update(result.forbidden)
        blocked.update(result.blocked_claims)
        unsupported_metric += result.unsupported_metric_count
        unsupported_impact += result.unsupported_impact_count
        project_mismatch += result.project_mismatch_count
        blocked_for_fact = {value.casefold() for value in result.blocked_claims}
        if any(claim.serialized.casefold() not in blocked_for_fact for claim in result.claims):
            evidence_boundary_count += 1
            qualifying_evidence += 1
        decisions.append(ProjectClaimBoundaryDecision(
            project_id=project_id,
            evidence_fact_id=fact.evidence_fact_id,
            decision_code=result.decision_code,
        ))
    for capability in sorted(capabilities, key=lambda item: item.capability_id):
        result = _claims_from_capability(capability, by_id)
        claims.extend(result.claims)
        forbidden.update(result.forbidden)
        project_mismatch += result.project_mismatch_count
        if result.claims:
            capability_boundary_count += 1
        else:
            unsupported_capability += 1
        decisions.append(ProjectClaimBoundaryDecision(
            project_id=project_id,
            capability_fact_id=capability.capability_id,
            decision_code=result.decision_code,
        ))
    if not capabilities:
        unsupported_capability += 1

    merged = _merge_claims(claims)
    conflicts: list[ProjectClaimBoundaryConflict] = []
    safe_claims: list[ProjectStructuredClaim] = []
    blocked_folded = {value.casefold() for value in blocked}
    for claim in merged:
        if claim.serialized.casefold() in blocked_folded or (
            claim.claim_type == "metric" and claim.metric_support == MetricSupport.NONE.value
        ):
            conflicts.append(ProjectClaimBoundaryConflict(
                project_id=project_id,
                claim_type=claim.claim_type,
                claim_hash=_claim_hash(claim.serialized),
                reason_code="forbidden_wins",
                evidence_fact_ids=claim.evidence_fact_ids,
                capability_fact_ids=claim.capability_fact_ids,
            ))
            forbidden.add(claim.serialized)
            continue
        safe_claims.append(claim)
    boundary, final_claims, final_forbidden, truncated = _build_boundary(
        project_id,
        ClaimSubjectType.PROJECT,
        _project_subject_id(project_id),
        safe_claims,
        forbidden,
    )
    return _ProjectBoundaryResult(
        boundary=boundary,
        claims=final_claims,
        forbidden=final_forbidden,
        evidence_boundary_count=evidence_boundary_count,
        capability_boundary_count=capability_boundary_count,
        qualifying_evidence_count=qualifying_evidence,
        unsupported_metric_count=unsupported_metric,
        unsupported_impact_count=unsupported_impact,
        unsupported_capability_count=unsupported_capability,
        project_mismatch_count=project_mismatch,
        truncated_count=truncated,
        conflicts=tuple(sorted(conflicts, key=lambda item: (item.project_id, item.claim_type, item.claim_hash, item.reason_code))),
        decisions=tuple(sorted(decisions, key=lambda item: (
            item.project_id, item.evidence_fact_id, item.capability_fact_id, item.decision_code,
        ))),
    )


def build_project_claim_boundary(
    project_id: str,
    evidence_facts: Iterable[ProjectEvidenceFact],
    capability_facts: Iterable[ProjectCapabilityFact] = (),
) -> ProjectClaimBoundary | None:
    """Aggregate exact-project claims; project aliases are never inferred."""

    return _build_project_result(project_id, evidence_facts, capability_facts).boundary


def _payload_hash(value: ProjectEvidenceFact | ProjectCapabilityFact) -> str:
    return hashlib.sha256(value.to_json().encode("utf-8")).hexdigest()


def _dedupe_records(
    records: Iterable[ProjectEvidenceFact | ProjectCapabilityFact],
    id_name: str,
) -> tuple[list[ProjectEvidenceFact | ProjectCapabilityFact], set[str], list[ProjectClaimBoundaryConflict]]:
    grouped: dict[str, list[ProjectEvidenceFact | ProjectCapabilityFact]] = {}
    for record in records:
        grouped.setdefault(getattr(record, id_name), []).append(record)
    output: list[ProjectEvidenceFact | ProjectCapabilityFact] = []
    invalid_ids: set[str] = set()
    conflicts: list[ProjectClaimBoundaryConflict] = []
    for record_id in sorted(grouped):
        items = grouped[record_id]
        payloads = {_payload_hash(item) for item in items}
        projects = {item.project_id for item in items}
        if len(payloads) > 1 or len(projects) > 1:
            invalid_ids.add(record_id)
            for project_id in sorted(projects):
                conflicts.append(ProjectClaimBoundaryConflict(
                    project_id=project_id,
                    claim_type="evidence" if id_name == "evidence_fact_id" else "capability",
                    claim_hash=hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:16],
                    reason_code="same_id_different_payload" if len(payloads) > 1 else "same_id_cross_project",
                ))
            continue
        output.append(min(items, key=lambda item: item.to_json()))
    return output, invalid_ids, conflicts


def build_project_claim_boundaries_by_project(
    evidence_facts: Iterable[ProjectEvidenceFact],
    capability_facts: Iterable[ProjectCapabilityFact] = (),
) -> tuple[dict[str, ProjectClaimBoundary], ProjectClaimBoundaryReport]:
    """Build deterministic project boundaries and a content-free audit report."""

    evidence_records = list(evidence_facts)
    capability_records = list(capability_facts)
    if any(not isinstance(fact, ProjectEvidenceFact) for fact in evidence_records):
        raise TypeError("evidence_facts must contain ProjectEvidenceFact values")
    if any(not isinstance(capability, ProjectCapabilityFact) for capability in capability_records):
        raise TypeError("capability_facts must contain ProjectCapabilityFact values")
    projects = sorted({item.project_id for item in [*evidence_records, *capability_records]})
    unique_evidence_raw, invalid_evidence_ids, global_conflicts = _dedupe_records(
        evidence_records, "evidence_fact_id"
    )
    unique_capability_raw, invalid_capability_ids, capability_conflicts = _dedupe_records(
        capability_records, "capability_id"
    )
    unique_evidence = [item for item in unique_evidence_raw if isinstance(item, ProjectEvidenceFact)]
    unique_capabilities = [item for item in unique_capability_raw if isinstance(item, ProjectCapabilityFact)]
    global_evidence_by_id = {fact.evidence_fact_id: fact for fact in unique_evidence}
    global_conflicts.extend(capability_conflicts)
    boundaries: dict[str, ProjectClaimBoundary] = {}
    all_claims: list[ProjectStructuredClaim] = []
    all_forbidden: list[str] = []
    all_decisions: list[ProjectClaimBoundaryDecision] = []
    all_conflicts: list[ProjectClaimBoundaryConflict] = list(global_conflicts)
    qualifying = evidence_boundaries = capability_boundaries = 0
    unsupported_metric = unsupported_impact = unsupported_capability = 0
    project_mismatch = truncated = 0
    for project_id in projects:
        result = _build_project_result(
            project_id,
            [fact for fact in unique_evidence if fact.project_id == project_id],
            [capability for capability in unique_capabilities if capability.project_id == project_id],
            evidence_facts_by_id=global_evidence_by_id,
        )
        if result.boundary is not None:
            boundaries[project_id] = result.boundary
        all_claims.extend(result.claims)
        all_forbidden.extend(result.forbidden)
        all_decisions.extend(result.decisions)
        all_conflicts.extend(result.conflicts)
        qualifying += result.qualifying_evidence_count
        evidence_boundaries += result.evidence_boundary_count
        capability_boundaries += result.capability_boundary_count
        unsupported_metric += result.unsupported_metric_count
        unsupported_impact += result.unsupported_impact_count
        unsupported_capability += result.unsupported_capability_count
        project_mismatch += result.project_mismatch_count
        truncated += result.truncated_count
    project_mismatch += len(invalid_evidence_ids) + len(invalid_capability_ids)

    type_counts = {claim_type: 0 for claim_type in CLAIM_TYPES}
    group_counts: dict[tuple[str, str, str, str, str], int] = {}
    for claim in all_claims:
        type_counts[claim.claim_type] += 1
        key = (
            claim.project_id,
            claim.claim_type,
            claim.confidence,
            claim.metric_support,
            claim.source_category,
        )
        group_counts[key] = group_counts.get(key, 0) + 1
    forbidden_counts: dict[str, int] = {}
    for value in all_forbidden:
        policy = value.partition(":")[0]
        forbidden_counts[policy] = forbidden_counts.get(policy, 0) + 1
    report = ProjectClaimBoundaryReport(
        project_count=len(projects),
        evidence_fact_count=len(evidence_records),
        capability_fact_count=len(capability_records),
        qualifying_evidence_fact_count=qualifying,
        evidence_boundaries_created=evidence_boundaries,
        capability_boundaries_created=capability_boundaries,
        project_boundaries_created=len(boundaries),
        allowed_claim_count=len(all_claims),
        forbidden_claim_count=len(all_forbidden),
        mechanism_claim_count=type_counts["mechanism"],
        implementation_claim_count=type_counts["implementation"],
        problem_claim_count=type_counts["problem"],
        impact_claim_count=type_counts["bounded_impact"],
        technology_claim_count=type_counts["technology"],
        capability_claim_count=type_counts["capability"],
        metric_claim_count=type_counts["metric"],
        weak_fact_blocked_count=sum(fact.status is EvidenceStatus.WEAK for fact in evidence_records),
        rejected_fact_blocked_count=sum(fact.status is EvidenceStatus.REJECTED for fact in evidence_records),
        low_quality_blocked_count=sum(_quality_score(fact) < MIN_DIRECT_CLAIM_SCORE for fact in evidence_records),
        contextual_only_restricted_count=sum(
            classify_project_evidence_source_category(fact) == SOURCE_CATEGORY_PROJECT for fact in evidence_records
        ),
        unsupported_metric_blocked_count=unsupported_metric,
        unsupported_impact_blocked_count=unsupported_impact,
        unsupported_capability_blocked_count=unsupported_capability,
        project_mismatch_count=project_mismatch,
        conflict_count=len(all_conflicts),
        truncated_claim_count=truncated,
        grouped_counts=tuple(
            ProjectClaimBoundaryGroupCount(
                project_id=key[0],
                claim_type=key[1],
                confidence=key[2],
                metric_support=key[3],
                evidence_source_category=key[4],
                count=count,
            )
            for key, count in sorted(group_counts.items())
        ),
        forbidden_policy_counts=tuple(
            ProjectForbiddenPolicyCount(policy_code=code, count=count)
            for code, count in sorted(forbidden_counts.items())
        ),
        decisions=tuple(sorted(all_decisions, key=lambda item: (
            item.project_id, item.evidence_fact_id, item.capability_fact_id, item.decision_code,
        ))[:MAX_DECISION_SAMPLES]),
        conflicts=tuple(sorted(all_conflicts, key=lambda item: (
            item.project_id, item.claim_type, item.claim_hash, item.reason_code,
        ))[:MAX_DECISION_SAMPLES]),
    )
    return dict(sorted(boundaries.items())), report


def _parse_claim_meta(note: str) -> tuple[str, str, str, str, str, tuple[str, ...], tuple[str, ...]] | None:
    parts = note.split("|")
    if len(parts) != 8 or parts[0] != "claim_meta" or not parts[6].startswith("e=") or not parts[7].startswith("c="):
        return None
    evidence = tuple(value for value in parts[6][2:].split(",") if value)
    capabilities = tuple(value for value in parts[7][2:].split(",") if value)
    return parts[1], parts[2], parts[3], parts[4], parts[5], evidence, capabilities


def get_project_claim_boundary_evidence_ids(
    boundary: ProjectClaimBoundary,
) -> tuple[str, ...]:
    """Return exact Evidence Fact IDs from validated claim metadata notes."""

    if not isinstance(boundary, ProjectClaimBoundary):
        raise TypeError("boundary must be a ProjectClaimBoundary")
    identifiers: set[str] = set()
    for note in boundary.notes:
        parsed = _parse_claim_meta(note)
        if parsed is not None:
            identifiers.update(parsed[5])
    return tuple(sorted(identifiers))


def validate_project_claim_boundary(
    boundary: ProjectClaimBoundary,
    *,
    evidence_facts_by_id: Mapping[str, ProjectEvidenceFact] | None = None,
    capability_facts_by_id: Mapping[str, ProjectCapabilityFact] | None = None,
) -> ProjectClaimBoundaryValidationResult:
    """Validate structure, safety, ordering, and optional support bindings."""

    if not isinstance(boundary, ProjectClaimBoundary):
        raise TypeError("validate_project_claim_boundary expects ProjectClaimBoundary")
    errors: set[str] = set()
    if not _normalize(boundary.project_id):
        errors.add("missing_project_id")
    allowed_folded = [value.casefold() for value in boundary.allowed_claims]
    forbidden_folded = [value.casefold() for value in boundary.forbidden_claims]
    if len(allowed_folded) != len(set(allowed_folded)):
        errors.add("duplicate_allowed_claim")
    if len(forbidden_folded) != len(set(forbidden_folded)):
        errors.add("duplicate_forbidden_claim")
    if set(allowed_folded) & set(forbidden_folded):
        errors.add("allowed_forbidden_conflict")
    if boundary.allowed_claims != sorted(boundary.allowed_claims, key=lambda item: (item.casefold(), item)):
        errors.add("non_deterministic_allowed_order")
    if boundary.forbidden_claims != sorted(boundary.forbidden_claims, key=lambda item: (item.casefold(), item)):
        errors.add("non_deterministic_forbidden_order")
    if boundary.notes != sorted(boundary.notes):
        errors.add("non_deterministic_note_order")

    claims_by_hash: dict[str, tuple[str, str]] = {}
    for serialized in boundary.allowed_claims:
        prefix, separator, value = serialized.partition(":")
        if not separator:
            errors.add("invalid_claim_prefix")
            continue
        if prefix not in _PREFIX_TO_TYPE:
            errors.add("unknown_claim_type" if _IDENTIFIER_RE.fullmatch(prefix) else "invalid_claim_prefix")
            continue
        if not _normalize(value):
            errors.add("blank_claim_value")
            continue
        claim_type = _PREFIX_TO_TYPE[prefix]
        if claim_type not in _CLAIM_TYPE_SET:
            errors.add("unknown_claim_type")
        if _contains_unsafe_content(value):
            errors.add("unsafe_claim_content")
        if claim_type != "implementation" and (
            _EMBEDDED_WINDOWS_ABSOLUTE_RE.search(value)
            or re.search(r"(?:^|\s)/(?:Users|home|var|tmp|etc)/", value, re.IGNORECASE)
        ):
            errors.add("unsafe_absolute_path")
        if claim_type == "implementation" and (
            _WINDOWS_ABSOLUTE_RE.match(value) or value.startswith(("~/", "//"))
        ):
            errors.add("unsafe_absolute_path")
        if claim_type == "metric" and (
            boundary.metric_support is MetricSupport.NONE
            or not _NUMERIC_RE.search(value)
            or _NON_RESUME_METRIC_RE.search(value)
        ):
            errors.add("unsupported_metric_claim")
        claims_by_hash[_claim_hash(serialized)] = (claim_type, value)

    supported_hashes: set[str] = set()
    for note in boundary.notes:
        parsed = _parse_claim_meta(note)
        if parsed is None:
            errors.add("invalid_claim_metadata")
            continue
        claim_hash, confidence, metric, source_category, rule_id, evidence_ids, capability_ids = parsed
        if claim_hash not in claims_by_hash:
            errors.add("unknown_claim_metadata_hash")
            continue
        if claim_hash in supported_hashes:
            errors.add("duplicate_claim_metadata")
        supported_hashes.add(claim_hash)
        if confidence not in {item.value for item in Confidence}:
            errors.add("invalid_claim_confidence")
        if metric not in {item.value for item in MetricSupport}:
            errors.add("invalid_claim_metric_support")
        if source_category not in {SOURCE_CATEGORY_DIRECT, SOURCE_CATEGORY_PROJECT, SOURCE_CATEGORY_CAPABILITY, "other"}:
            errors.add("invalid_claim_source_category")
        if not _IDENTIFIER_RE.fullmatch(rule_id):
            errors.add("invalid_claim_rule_id")
        claim_type, value = claims_by_hash[claim_hash]
        if evidence_facts_by_id is not None:
            resolved: list[ProjectEvidenceFact] = []
            for evidence_id in evidence_ids:
                fact = evidence_facts_by_id.get(evidence_id)
                if fact is None:
                    errors.add("unknown_evidence_fact_id")
                    continue
                resolved.append(fact)
                if fact.project_id != boundary.project_id:
                    errors.add("cross_project_evidence_binding")
            if claim_type == "technology" and not any(
                fact.status is EvidenceStatus.ACCEPTED
                and classify_project_evidence_source_category(fact) == SOURCE_CATEGORY_DIRECT
                and _quality_score(fact) >= MIN_DIRECT_CLAIM_SCORE
                and any(tag.casefold() == value.casefold() for tag in fact.technical_tags)
                and _concrete_implementation(fact.implementation)
                for fact in resolved
            ):
                errors.add("technology_without_direct_support")
            if claim_type == "metric" and not any(
                fact.metric_support is not MetricSupport.NONE for fact in resolved
            ):
                errors.add("metric_without_supported_evidence")
        if capability_facts_by_id is not None:
            resolved_capabilities: list[ProjectCapabilityFact] = []
            for capability_id in capability_ids:
                capability = capability_facts_by_id.get(capability_id)
                if capability is None:
                    errors.add("unknown_capability_fact_id")
                    continue
                resolved_capabilities.append(capability)
                if capability.project_id != boundary.project_id:
                    errors.add("capability_project_mismatch")
                if not capability.present:
                    errors.add("capability_not_present")
            if claim_type == "capability":
                matches = False
                for capability in resolved_capabilities:
                    try:
                        canonical = validate_project_capability_type(capability.capability_type)
                    except (TypeError, ValueError):
                        errors.add("unknown_capability_type")
                        continue
                    if capability.present and capability.project_id == boundary.project_id and canonical == value:
                        matches = True
                if not matches:
                    errors.add("capability_claim_mismatch")
        if claim_type == "capability" and not capability_ids:
            errors.add("capability_claim_without_fact")
    if set(claims_by_hash) - supported_hashes:
        errors.add("missing_claim_support_metadata")
    return ProjectClaimBoundaryValidationResult(valid=not errors, errors=tuple(sorted(errors)))


def list_project_claim_types() -> tuple[str, ...]:
    return CLAIM_TYPES


__all__ = [
    "CLAIM_LIMITS",
    "CLAIM_TYPES",
    "GLOBAL_FORBIDDEN_POLICIES",
    "MAX_ALLOWED_CLAIMS",
    "MAX_DECISION_SAMPLES",
    "MIN_CONTEXT_CLAIM_SCORE",
    "MIN_DIRECT_CLAIM_SCORE",
    "ProjectClaimBoundaryConflict",
    "ProjectClaimBoundaryDecision",
    "ProjectClaimBoundaryGroupCount",
    "ProjectClaimBoundaryReport",
    "ProjectClaimBoundaryValidationResult",
    "ProjectForbiddenPolicyCount",
    "ProjectStructuredClaim",
    "build_project_capability_claim_boundary",
    "build_project_claim_boundaries_by_project",
    "build_project_evidence_claim_boundary",
    "build_project_claim_boundary",
    "evaluate_project_numeric_claim",
    "get_project_claim_boundary_evidence_ids",
    "get_project_claim_safety_blockers",
    "is_project_resume_metric_claim",
    "list_project_claim_types",
    "normalize_project_claim",
    "normalize_project_serialized_claim",
    "validate_project_claim_boundary",
]
