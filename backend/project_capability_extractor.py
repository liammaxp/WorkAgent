"""Deterministic project evidence signal and capability extraction.

This module maps already-scored :class:`ProjectEvidenceFact` objects to the
strict signal vocabulary defined by the project evidence capability taxonomy and then
evaluates each project independently.  It is deliberately pure: it performs
no persistence, retrieval, network access, model calls, or claim generation.

Text matching is limited to explicit canonical phrases and conservative named
variants.  Generic technology tags and loose words such as ``cache``,
``validate``, ``memory``, ``sort``, or ``diff`` never imply a capability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from backend.project_capability_taxonomy import (
    CAPABILITY_ALIASES,
    CAPABILITY_TAXONOMY,
    ProjectCapabilityDefinition,
    list_project_capability_definitions,
    list_project_capability_overlap_rules,
    list_project_evidence_signal_identifiers,
)
from backend.project_evidence_models import (
    Confidence,
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    ProjectCapabilityFact,
    ProjectEvidenceFact,
    EvidenceSourceRef,
)


MAX_DECISION_SAMPLES = 100
MAX_MECHANISMS = 12
HIGH_CONFIDENCE_SCORE_MARGIN = 10

SOURCE_CATEGORY_DIRECT = "direct_evidence"
SOURCE_CATEGORY_PROJECT = "project_context"
SOURCE_CATEGORY_CAPABILITY = "capability_context"
SOURCE_CATEGORY_OTHER = "other"

DIRECT_SOURCE_TYPES = frozenset({
    "github_evidence_chunk",
    "github_evidence_card",
    "github_evidence_raw_change_summary",
    "project_change_evidence_card",
    "project_change_raw_change_summary",
})
PROJECT_CONTEXT_SOURCE_TYPES = frozenset({
    "project_memory",
    "project_compact_facts",
    "compact_facts",
})
CAPABILITY_CONTEXT_SOURCE_TYPES = frozenset({
    "github_evidence_capability_fact",
    "project_change_capability_fact",
})

_ALLOWED_RULE_FIELDS = frozenset({
    "evidence_type",
    "technical_tags",
    "mechanism",
    "implementation",
    "safe_impact",
    "source_metadata",
    "quality_metadata",
})
_VALID_EVIDENCE_TYPES = frozenset(item.value for item in EvidenceType)
_FIELD_PRIORITY = {
    "evidence_type": 0,
    "technical_tags": 1,
    "source_metadata": 2,
    "mechanism": 3,
    "implementation": 4,
    "safe_impact": 5,
    "quality_metadata": 6,
}
_RULE_IDENTIFIER_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:/")
_METRIC_CLAIM_RE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*%|"
    r"\b\d+(?:\.\d+)?\s*x\s+faster\b|"
    r"\b(?:reduced|decreased|increased|improved|saved)\b.{0,80}\b\d+(?:\.\d+)?\b)",
    re.IGNORECASE,
)
_SIGNAL_METADATA_KEYS = frozenset({
    "capability_signals",
    "operation",
    "operations",
    "signal",
    "signals",
    "technical_signals",
})


@dataclass(frozen=True)
class ProjectSignalExtractionRule:
    rule_id: str
    signal: str
    allowed_fields: tuple[str, ...]
    required_all_patterns: tuple[str, ...] = ()
    required_any_patterns: tuple[str, ...] = ()
    forbidden_patterns: tuple[str, ...] = ()
    exact_values: tuple[str, ...] = ()
    accepted_evidence_types: tuple[str, ...] = ()
    minimum_quality_score: int = 0
    direct_source_only: bool = False


@dataclass(frozen=True)
class ProjectSignalEvidenceBinding:
    signal: str
    evidence_fact_id: str
    project_id: str
    source_category: str
    quality_score: int
    structural_status: str
    confidence: str
    evidence_type: str
    matched_field: str
    rule_id: str


@dataclass(frozen=True)
class ProjectFactSignalExtraction:
    evidence_fact_id: str
    project_id: str
    signals: tuple[str, ...]
    bindings: tuple[ProjectSignalEvidenceBinding, ...]
    rejected_candidates: tuple[str, ...]


@dataclass(frozen=True)
class ProjectSignalExtractionDecision:
    evidence_fact_id: str
    project_id: str
    decision_code: str
    signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectSignalExtractionReport:
    fact_count: int
    facts_with_signals: int
    facts_without_signals: int
    signal_binding_count: int
    unique_signal_count: int
    ambiguous_match_rejected_count: int
    unsupported_signal_candidate_count: int
    contextual_signal_count: int
    direct_signal_count: int
    decisions: tuple[ProjectSignalExtractionDecision, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectCapabilityExtractionDecision:
    project_id: str
    capability_type: str
    decision_code: str
    supporting_evidence_fact_ids: tuple[str, ...] = ()
    missing_required_group_indexes: tuple[int, ...] = ()


@dataclass(frozen=True)
class ProjectCapabilityExtractionConflict:
    project_id: str
    conflict_code: str
    evidence_fact_ids: tuple[str, ...]
    capability_type: str = ""
    rule_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectCapabilityOverlapDecision:
    project_id: str
    left_capability: str
    right_capability: str
    shared_evidence_fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class ProjectCapabilityDecisionGroupCount:
    capability_type: str
    decision_code: str
    count: int


@dataclass(frozen=True)
class ProjectCapabilityExtractionReport:
    project_count: int
    fact_count: int
    signal_binding_count: int
    capability_candidates_evaluated: int
    capabilities_emitted: int
    missing_required_group_count: int
    insufficient_quality_count: int
    insufficient_direct_fact_count: int
    insufficient_total_fact_count: int
    direct_provenance_required_count: int
    contextual_only_rejected_count: int
    high_risk_blocked_count: int
    project_mismatch_count: int
    duplicate_fact_binding_count: int
    conflict_count: int
    overlap_count: int = 0
    decisions: tuple[ProjectCapabilityExtractionDecision, ...] = ()
    conflicts: tuple[ProjectCapabilityExtractionConflict, ...] = ()
    overlaps: tuple[ProjectCapabilityOverlapDecision, ...] = ()
    grouped_decision_counts: tuple[ProjectCapabilityDecisionGroupCount, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _phrase_pattern(value: str) -> str:
    tokens = [token for token in re.split(r"[_\s-]+", value.strip()) if token]
    body = r"[\s_-]+".join(re.escape(token) for token in tokens)
    return rf"(?<![a-z0-9]){body}(?![a-z0-9])"


# These are explicit linguistic variants, not fuzzy synonyms.  Every signal
# also receives its canonical snake/space/hyphen form below.
_TEXT_PATTERN_OVERRIDES: Mapping[str, tuple[str, ...]] = {
    "allowed_forbidden_claim_handling": (
        r"(?<![a-z0-9])allowed(?:[\s_-]+and[\s_-]+|\s*/\s*)forbidden[\s_-]+claim[\s_-]+handling(?![a-z0-9])",
    ),
    "atomic_persistence": (r"(?<![a-z0-9])atomic[\s_-]+write(?:s|n|ing)?(?![a-z0-9])",),
    "backend_route": (
        r"(?<![a-z0-9])api[\s_-]+routes?(?:[\s_-]+updates?)?(?![a-z0-9])",
    ),
    "cache_reuse": (
        r"(?<![a-z0-9])reus(?:e|ed|ing)[\s_-]+(?:the[\s_-]+)?cache(?:d)?(?![a-z0-9])",
        r"(?<![a-z0-9])cache(?:d)?[\s_-]+(?:result|record|analysis|context|value)s?[\s_-]+(?:is[\s_-]+|are[\s_-]+|was[\s_-]+|were[\s_-]+)?reus(?:e|ed|ing)(?![a-z0-9])",
    ),
    "candidate_filtering": (r"(?<![a-z0-9])filter(?:ed|ing)?[\s_-]+candidates?(?![a-z0-9])",),
    "canonical_serialization": (r"(?<![a-z0-9])canonical[\s_-]+json(?![a-z0-9])",),
    "changed_file_detection": (r"(?<![a-z0-9])detect(?:s|ed|ing)?[\s_-]+changed[\s_-]+files?(?![a-z0-9])",),
    "claim_validation": (r"(?<![a-z0-9])validat(?:e|es|ed|ing)[\s_-]+claims?(?![a-z0-9])",),
    "diff_processing": (r"(?<![a-z0-9])process(?:es|ed|ing)?[\s_-]+(?:git[\s_-]+)?diffs?(?![a-z0-9])",),
    "exact_deduplication": (
        r"(?<![a-z0-9])exact[\s_-]+duplicates?(?![a-z0-9])",
        r"(?<![a-z0-9])deterministic[\s_-]+deduplication(?![a-z0-9])",
    ),
    "fallback": (r"(?<![a-z0-9])fallback[\s_-]+(?:path|handling|behavior|strategy)(?![a-z0-9])",),
    "field_normalization": (r"(?<![a-z0-9])normalized?[\s_-]+fields?(?![a-z0-9])",),
    "integrity_conflict_detection": (r"(?<![a-z0-9])integrity[\s_-]+conflicts?(?![a-z0-9])",),
    "invariant_check": (r"(?<![a-z0-9])check(?:s|ed|ing)?[\s_-]+invariants?(?![a-z0-9])",),
    "latex_compile_check": (
        r"(?<![a-z0-9])(?:latex|tex)[\s_-]+compil(?:e|es|ed|ing|ation)[\s_-]+check(?:s|ed|ing)?(?![a-z0-9])",
        r"(?<![a-z0-9])check(?:s|ed|ing)?[\s_-]+(?:latex|tex)[\s_-]+compil(?:e|es|ed|ing|ation)(?![a-z0-9])",
    ),
    "latex_repair": (r"(?<![a-z0-9])repair(?:s|ed|ing)?[\s_-]+latex(?![a-z0-9])",),
    "low_evidence_refusal": (r"(?<![a-z0-9])refus(?:e|es|ed|ing)[\s_-]+(?:output[\s_-]+)?(?:when[\s_-]+)?(?:evidence[\s_-]+is[\s_-]+)?(?:low|insufficient)(?![a-z0-9])",),
    "load_validate_write_lifecycle": (
        r"(?<![a-z0-9])load\s*/\s*validate\s*/\s*write[\s_-]+lifecycle(?![a-z0-9])",
    ),
    "output_evidence_validation": (r"(?<![a-z0-9])validat(?:e|es|ed|ing)[\s_-]+output[\s_-]+against[\s_-]+evidence(?![a-z0-9])",),
    "prior_state_comparison": (r"(?<![a-z0-9])compar(?:e|es|ed|ing)[\s_-]+(?:against[\s_-]+)?prior[\s_-]+state(?![a-z0-9])",),
    "rag_terminology": (r"(?<![a-z0-9])rag(?![a-z0-9])",),
    "regression_testing": (r"(?<![a-z0-9])regression[\s_-]+tests?(?![a-z0-9])",),
    "reranking": (r"(?<![a-z0-9])rerank(?:s|ed|er|ers|ing)?(?![a-z0-9])",),
    "retry": (r"(?<![a-z0-9])retr(?:y|ies|ied|ying)[\s_-]+(?:path|handling|behavior|strategy|operation)s?(?![a-z0-9])",),
    "schema_versioning": (r"(?<![a-z0-9])schema[\s_-]+versions?(?![a-z0-9])",),
    "source_grounding": (r"(?<![a-z0-9])ground(?:s|ed|ing)?[\s_-]+(?:the[\s_-]+)?(?:generated[\s_-]+)?output[\s_-]+(?:in|to|with)[\s_-]+(?:source[\s_-]+)?evidence(?![a-z0-9])",),
    "stable_identity": (
        r"(?<![a-z0-9])(?:stable|deterministic)[\s_-]+(?:id|ids|identity|identities|identifier|identifiers)(?![a-z0-9])",
    ),
    "stage_io_contract": (r"(?<![a-z0-9])stage[\s_-]+i\s*/\s*o[\s_-]+contract(?![a-z0-9])",),
    "structured_output_validation": (r"(?<![a-z0-9])validat(?:e|es|ed|ing)[\s_-]+structured[\s_-]+output(?![a-z0-9])",),
    "unsupported_claim_blocking": (
        r"(?<![a-z0-9])(?:block|blocks|blocked|blocking|reject|rejects|rejected|rejecting|filter|filters|filtered|filtering)[\s_-]+unsupported[\s_-]+claims?(?![a-z0-9])",
        r"(?<![a-z0-9])unsupported[\s_-]+claims?[\s_-]+(?:blocker|blocking|rejection|filtering)(?![a-z0-9])",
    ),
}


def _build_default_rules() -> tuple[ProjectSignalExtractionRule, ...]:
    rules: list[ProjectSignalExtractionRule] = []
    for signal in list_project_evidence_signal_identifiers():
        patterns = (_phrase_pattern(signal), *_TEXT_PATTERN_OVERRIDES.get(signal, ()))
        rules.append(ProjectSignalExtractionRule(
            rule_id=f"text_{signal}",
            signal=signal,
            allowed_fields=("mechanism", "implementation", "safe_impact"),
            required_any_patterns=tuple(dict.fromkeys(patterns)),
        ))
        rules.append(ProjectSignalExtractionRule(
            rule_id=f"tag_{signal}",
            signal=signal,
            allowed_fields=("technical_tags", "source_metadata"),
            exact_values=(signal,),
        ))
    rules.append(ProjectSignalExtractionRule(
        rule_id="metadata_schema_version",
        signal="schema_versioning",
        allowed_fields=("source_metadata",),
        exact_values=("schema version",),
    ))
    rules.append(ProjectSignalExtractionRule(
        rule_id="structured_retrieval_evidence_type",
        signal="retrieval",
        allowed_fields=("evidence_type",),
        exact_values=("retrieval",),
        accepted_evidence_types=("retrieval",),
    ))
    return tuple(sorted(rules, key=lambda item: item.rule_id))


SIGNAL_EXTRACTION_RULES = _build_default_rules()


def validate_project_evidence_signal_extraction_rules(
    rules: Iterable[ProjectSignalExtractionRule] | None = None,
) -> tuple[str, ...]:
    items = tuple(SIGNAL_EXTRACTION_RULES if rules is None else rules)
    registry = set(list_project_evidence_signal_identifiers())
    errors: list[str] = []
    ids = [item.rule_id for item in items]
    if len(ids) != len(set(ids)):
        errors.append("rules:duplicate_rule_id")
    if ids != sorted(ids):
        errors.append("rules:unstable_order")
    for index, item in enumerate(items):
        prefix = f"rule[{index}]"
        if not isinstance(item, ProjectSignalExtractionRule):
            errors.append(f"{prefix}:invalid_type")
            continue
        if not _RULE_IDENTIFIER_RE.fullmatch(item.rule_id):
            errors.append(f"{prefix}:invalid_rule_id")
        if item.signal not in registry:
            errors.append(f"{prefix}:unknown_signal")
        if not item.allowed_fields or any(field not in _ALLOWED_RULE_FIELDS for field in item.allowed_fields):
            errors.append(f"{prefix}:invalid_field")
        if len(item.allowed_fields) != len(set(item.allowed_fields)):
            errors.append(f"{prefix}:duplicate_field")
        if not (item.exact_values or item.required_all_patterns or item.required_any_patterns):
            errors.append(f"{prefix}:empty_matcher")
        if isinstance(item.minimum_quality_score, bool) or not 0 <= item.minimum_quality_score <= 100:
            errors.append(f"{prefix}:invalid_quality_threshold")
        if any(value not in _VALID_EVIDENCE_TYPES for value in item.accepted_evidence_types):
            errors.append(f"{prefix}:invalid_evidence_type")
        if len(item.accepted_evidence_types) != len(set(item.accepted_evidence_types)):
            errors.append(f"{prefix}:duplicate_evidence_type")
        for pattern in (*item.required_all_patterns, *item.required_any_patterns, *item.forbidden_patterns):
            try:
                re.compile(pattern, re.IGNORECASE)
            except (TypeError, re.error):
                errors.append(f"{prefix}:invalid_pattern")
                break
    return tuple(sorted(set(errors)))


_DEFAULT_RULE_ERRORS = validate_project_evidence_signal_extraction_rules(SIGNAL_EXTRACTION_RULES)
if _DEFAULT_RULE_ERRORS:  # pragma: no cover - import-time invariant
    raise RuntimeError(f"Invalid project evidence signal rules: {_DEFAULT_RULE_ERRORS!r}")


def list_project_evidence_signal_extraction_rules() -> tuple[ProjectSignalExtractionRule, ...]:
    return SIGNAL_EXTRACTION_RULES


def classify_project_evidence_source_category(fact: ProjectEvidenceFact) -> str:
    if not isinstance(fact, ProjectEvidenceFact):
        raise TypeError("classify_project_evidence_source_category expects ProjectEvidenceFact")
    source_types = {ref.source_type for ref in fact.source_refs}
    if source_types & DIRECT_SOURCE_TYPES:
        return SOURCE_CATEGORY_DIRECT
    if source_types & CAPABILITY_CONTEXT_SOURCE_TYPES:
        return SOURCE_CATEGORY_CAPABILITY
    if source_types & PROJECT_CONTEXT_SOURCE_TYPES:
        return SOURCE_CATEGORY_PROJECT
    return SOURCE_CATEGORY_OTHER


def _safe_quality_score(fact: ProjectEvidenceFact) -> int:
    value = fact.quality_score
    if value is None or isinstance(value, bool):
        return 0
    return max(0, min(100, int(value)))


def get_project_evidence_quality_score(fact: ProjectEvidenceFact) -> int:
    """Return the extractor's authoritative bounded Evidence Fact quality."""

    if not isinstance(fact, ProjectEvidenceFact):
        raise TypeError("get_project_evidence_quality_score expects ProjectEvidenceFact")
    return _safe_quality_score(fact)


def _flatten_metadata_values(value: Any) -> list[str]:
    if isinstance(value, str):
        normalized = " ".join(value.split())
        return [normalized] if normalized else []
    if isinstance(value, (list, tuple)):
        output: list[str] = []
        for item in value:
            output.extend(_flatten_metadata_values(item))
        return output
    return []


def _field_values(fact: ProjectEvidenceFact, field: str) -> tuple[str, ...]:
    if field == "evidence_type":
        return (fact.evidence_type.value,)
    if field == "technical_tags":
        return tuple(fact.technical_tags)
    if field == "mechanism":
        return (fact.mechanism,)
    if field == "implementation":
        return tuple(fact.implementation)
    if field == "safe_impact":
        return tuple(fact.safe_impact)
    if field == "source_metadata":
        output: list[str] = []
        for ref in sorted(fact.source_refs, key=lambda item: (item.source_type, item.source_id, item.content_hash)):
            for key in sorted(ref.metadata):
                if key in _SIGNAL_METADATA_KEYS:
                    output.extend(_flatten_metadata_values(ref.metadata[key]))
                elif key == "schema_version" and str(ref.metadata[key]).strip():
                    output.append("schema version")
        return tuple(output)
    if field == "quality_metadata":
        output = []
        for key in ("reason_codes", "blocker_codes"):
            output.extend(_flatten_metadata_values(fact.quality_breakdown.get(key)))
        return tuple(output)
    return ()


def _rule_matches_value(rule: ProjectSignalExtractionRule, value: str) -> bool:
    normalized = " ".join(value.split())
    if not normalized:
        return False
    folded = normalized.casefold()
    if rule.exact_values and folded not in {item.casefold() for item in rule.exact_values}:
        return False
    if rule.required_all_patterns and not all(re.search(pattern, normalized, re.IGNORECASE) for pattern in rule.required_all_patterns):
        return False
    if rule.required_any_patterns and not any(re.search(pattern, normalized, re.IGNORECASE) for pattern in rule.required_any_patterns):
        return False
    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in rule.forbidden_patterns):
        return False
    return True


def _rejected_candidate_codes(
    fact: ProjectEvidenceFact,
    emitted_signals: set[str],
) -> tuple[str, ...]:
    text = "\n".join((fact.mechanism, *fact.implementation, *fact.safe_impact))
    candidates: list[str] = []
    ambiguous = {
        "ambiguous_cache": (r"(?<![a-z0-9])cache(?:d|s|ing)?(?![a-z0-9])", {"cache_reuse"}),
        "ambiguous_diff": (r"(?<![a-z0-9])diffs?(?![a-z0-9])", {"diff_processing", "diff_only_analysis"}),
        "ambiguous_memory": (r"(?<![a-z0-9])memory(?![a-z0-9])", {"project_memory_read", "project_memory_write"}),
        "ambiguous_sort": (r"(?<![a-z0-9])sort(?:s|ed|ing)?(?![a-z0-9])", {"deterministic_ordering"}),
        "ambiguous_validate": (
            r"(?<![a-z0-9])validat(?:e|es|ed|ing|ion)(?![a-z0-9])",
            {
                "claim_validation", "latex_validation", "metric_support_validation",
                "output_evidence_validation", "schema_validation",
                "structured_output_validation", "validation_gate",
            },
        ),
    }
    for code, (pattern, expected) in sorted(ambiguous.items()):
        if re.search(pattern, text, re.IGNORECASE) and not (emitted_signals & expected):
            candidates.append(code)
    aliases = set(CAPABILITY_ALIASES)
    if any(tag.casefold() in aliases for tag in fact.technical_tags):
        candidates.append("legacy_capability_alias")
    metadata_values = set(_field_values(fact, "source_metadata"))
    if any(
        _RULE_IDENTIFIER_RE.fullmatch(value.casefold())
        and value.casefold() not in set(list_project_evidence_signal_identifiers())
        for value in metadata_values
    ):
        candidates.append("unsupported_signal_candidate")
    return tuple(sorted(set(candidates)))


def extract_project_evidence_fact_signals(
    fact: ProjectEvidenceFact,
    *,
    rules: Iterable[ProjectSignalExtractionRule] | None = None,
) -> ProjectFactSignalExtraction:
    if not isinstance(fact, ProjectEvidenceFact):
        raise TypeError("extract_project_evidence_fact_signals expects ProjectEvidenceFact")
    selected_rules = tuple(SIGNAL_EXTRACTION_RULES if rules is None else rules)
    errors = () if selected_rules is SIGNAL_EXTRACTION_RULES else validate_project_evidence_signal_extraction_rules(selected_rules)
    if errors:
        raise ValueError(f"invalid project evidence signal extraction rules: {errors[0]}")
    category = classify_project_evidence_source_category(fact)
    quality_score = _safe_quality_score(fact)
    candidates: list[ProjectSignalEvidenceBinding] = []
    for rule in selected_rules:
        if rule.accepted_evidence_types and fact.evidence_type.value not in rule.accepted_evidence_types:
            continue
        if quality_score < rule.minimum_quality_score:
            continue
        if rule.direct_source_only and category != SOURCE_CATEGORY_DIRECT:
            continue
        matched_field = ""
        for field in rule.allowed_fields:
            if any(_rule_matches_value(rule, value) for value in _field_values(fact, field)):
                matched_field = field
                break
        if not matched_field:
            continue
        candidates.append(ProjectSignalEvidenceBinding(
            signal=rule.signal,
            evidence_fact_id=fact.evidence_fact_id,
            project_id=fact.project_id,
            source_category=category,
            quality_score=quality_score,
            structural_status=fact.status.value,
            confidence=fact.confidence.value,
            evidence_type=fact.evidence_type.value,
            matched_field=matched_field,
            rule_id=rule.rule_id,
        ))
    candidates.sort(key=lambda item: (
        item.signal,
        _FIELD_PRIORITY[item.matched_field],
        item.rule_id,
    ))
    bindings: list[ProjectSignalEvidenceBinding] = []
    seen_signals: set[str] = set()
    for binding in candidates:
        if binding.signal in seen_signals:
            continue
        seen_signals.add(binding.signal)
        bindings.append(binding)
    signals = tuple(sorted(seen_signals))
    rejected = _rejected_candidate_codes(fact, seen_signals)
    return ProjectFactSignalExtraction(
        evidence_fact_id=fact.evidence_fact_id,
        project_id=fact.project_id,
        signals=signals,
        bindings=tuple(bindings),
        rejected_candidates=rejected,
    )


def extract_project_evidence_fact_signals_many(
    facts: Iterable[ProjectEvidenceFact],
) -> tuple[list[ProjectFactSignalExtraction], ProjectSignalExtractionReport]:
    records = list(facts)
    if any(not isinstance(fact, ProjectEvidenceFact) for fact in records):
        raise TypeError("extract_project_evidence_fact_signals_many expects ProjectEvidenceFact values")
    extractions = [extract_project_evidence_fact_signals(fact) for fact in records]
    extractions.sort(key=lambda item: (item.project_id, item.evidence_fact_id, item.signals))
    bindings = [binding for item in extractions for binding in item.bindings]
    decisions = tuple(sorted((
        ProjectSignalExtractionDecision(
            evidence_fact_id=item.evidence_fact_id,
            project_id=item.project_id,
            decision_code="signals_extracted" if item.signals else "no_explicit_signal",
            signals=item.signals,
        )
        for item in extractions
    ), key=lambda item: (item.project_id, item.evidence_fact_id, item.decision_code))[:MAX_DECISION_SAMPLES])
    report = ProjectSignalExtractionReport(
        fact_count=len(records),
        facts_with_signals=sum(bool(item.signals) for item in extractions),
        facts_without_signals=sum(not item.signals for item in extractions),
        signal_binding_count=len(bindings),
        unique_signal_count=len({binding.signal for binding in bindings}),
        ambiguous_match_rejected_count=sum(
            code.startswith("ambiguous_")
            for item in extractions
            for code in item.rejected_candidates
        ),
        unsupported_signal_candidate_count=sum(
            code == "unsupported_signal_candidate"
            for item in extractions
            for code in item.rejected_candidates
        ),
        contextual_signal_count=sum(
            binding.source_category in {SOURCE_CATEGORY_PROJECT, SOURCE_CATEGORY_CAPABILITY}
            for binding in bindings
        ),
        direct_signal_count=sum(binding.source_category == SOURCE_CATEGORY_DIRECT for binding in bindings),
        decisions=decisions,
    )
    return extractions, report


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalized_path(value: str | None) -> str:
    path = (value or "").replace("\\", "/")
    return "" if path.startswith("/") or _WINDOWS_ABSOLUTE_PATH_RE.match(path) else path


def _ref_identity(ref: EvidenceSourceRef, *, include_hash: bool = True) -> tuple[Any, ...]:
    return (
        ref.project_id,
        ref.source_type,
        ref.source_id,
        ref.content_hash if include_hash else "",
        ref.commit_sha or "",
        _normalized_path(ref.file_path),
        ref.symbol or "",
        ref.start_line,
        ref.end_line,
    )


def _independent_evidence_identity(fact: ProjectEvidenceFact) -> str:
    payload = sorted({_ref_identity(ref) for ref in fact.source_refs})
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def get_independent_project_evidence_identity(fact: ProjectEvidenceFact) -> str:
    """Return the privacy-safe identity used for independent direct evidence."""

    if not isinstance(fact, ProjectEvidenceFact):
        raise TypeError("get_independent_project_evidence_identity expects ProjectEvidenceFact")
    return _independent_evidence_identity(fact)


def _base_lineage_identity(fact: ProjectEvidenceFact) -> str:
    payload = sorted({_ref_identity(ref, include_hash=False) for ref in fact.source_refs})
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _fact_payload_hash(fact: ProjectEvidenceFact) -> str:
    return hashlib.sha256(fact.to_json().encode("utf-8")).hexdigest()


def _fact_priority(
    fact: ProjectEvidenceFact,
    category: str,
) -> tuple[int, int, str]:
    return (
        0 if category == SOURCE_CATEGORY_DIRECT else 1,
        -_safe_quality_score(fact),
        fact.evidence_fact_id,
    )


def _eligible_kind(
    fact: ProjectEvidenceFact,
    category: str,
    definition: ProjectCapabilityDefinition,
) -> str | None:
    evidence_type = fact.evidence_type.value
    if (
        fact.status is EvidenceStatus.ACCEPTED
        and category in definition.accepted_source_categories
        and evidence_type in definition.accepted_evidence_types
    ):
        return "direct"
    if (
        definition.allows_contextual_support
        and fact.status is EvidenceStatus.SUPPORTING
        and category in definition.contextual_source_categories
        and evidence_type in definition.contextual_evidence_types
    ):
        return "contextual"
    return None


def classify_project_capability_evidence_kind(
    fact: ProjectEvidenceFact,
    definition: ProjectCapabilityDefinition,
) -> str | None:
    """Expose the extractor's existing direct/contextual proof classification."""

    if not isinstance(fact, ProjectEvidenceFact):
        raise TypeError("classify_project_capability_evidence_kind expects ProjectEvidenceFact")
    if not isinstance(definition, ProjectCapabilityDefinition):
        raise TypeError("definition must be a ProjectCapabilityDefinition")
    return _eligible_kind(fact, classify_project_evidence_source_category(fact), definition)


def _metric_support_for_facts(facts: Sequence[ProjectEvidenceFact]) -> MetricSupport:
    metric_facts = [
        fact
        for fact in facts
        if any(_METRIC_CLAIM_RE.search(value) for value in fact.safe_impact)
    ]
    supports = [fact.metric_support for fact in metric_facts]
    if not supports or any(value is MetricSupport.NONE for value in supports):
        return MetricSupport.NONE
    if all(value is MetricSupport.EXPLICIT for value in supports):
        return MetricSupport.EXPLICIT
    if all(value in {MetricSupport.EXPLICIT, MetricSupport.APPROXIMATE} for value in supports):
        return MetricSupport.APPROXIMATE
    return MetricSupport.NONE


def _selected_mechanisms(facts: Sequence[ProjectEvidenceFact]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        value = " ".join(fact.mechanism.split())
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
        if len(output) >= MAX_MECHANISMS:
            break
    return output


def select_project_evidence_mechanisms(
    facts: Sequence[ProjectEvidenceFact],
) -> tuple[str, ...]:
    """Return the extractor's bounded, normalized, deduplicated mechanisms."""

    if any(not isinstance(fact, ProjectEvidenceFact) for fact in facts):
        raise TypeError("select_project_evidence_mechanisms expects ProjectEvidenceFact values")
    return tuple(_selected_mechanisms(facts))


def _selected_tags(facts: Sequence[ProjectEvidenceFact]) -> list[str]:
    excluded = set(list_project_evidence_signal_identifiers()) | set(CAPABILITY_ALIASES) | set(CAPABILITY_TAXONOMY)
    values = {
        " ".join(tag.split())
        for fact in facts
        for tag in fact.technical_tags
        if " ".join(tag.split()) and " ".join(tag.split()).casefold() not in excluded
    }
    return sorted(values, key=lambda value: (value.casefold(), value))


def select_project_evidence_technical_tags(
    facts: Sequence[ProjectEvidenceFact],
) -> tuple[str, ...]:
    """Return technical tags selected by the existing extractor rules."""

    if any(not isinstance(fact, ProjectEvidenceFact) for fact in facts):
        raise TypeError("select_project_evidence_technical_tags expects ProjectEvidenceFact values")
    return tuple(_selected_tags(facts))


def _confidence_for_proof(
    definition: ProjectCapabilityDefinition,
    selected_facts: Sequence[ProjectEvidenceFact],
    selected_kinds: Mapping[str, str],
) -> Confidence:
    direct = [fact for fact in selected_facts if selected_kinds.get(fact.evidence_fact_id) == "direct"]
    independent = {_independent_evidence_identity(fact) for fact in direct}
    all_direct = len(direct) == len(selected_facts)
    strong = all(_safe_quality_score(fact) >= definition.minimum_quality_score + HIGH_CONFIDENCE_SCORE_MARGIN for fact in direct)
    if all_direct and len(independent) >= 2 and strong:
        return Confidence.HIGH
    return Confidence.MEDIUM


def derive_project_capability_confidence(
    definition: ProjectCapabilityDefinition,
    facts: Sequence[ProjectEvidenceFact],
) -> Confidence:
    """Apply the extractor's existing confidence rule to deterministic proof inputs."""

    if not isinstance(definition, ProjectCapabilityDefinition):
        raise TypeError("definition must be a ProjectCapabilityDefinition")
    if any(not isinstance(fact, ProjectEvidenceFact) for fact in facts):
        raise TypeError("facts must contain ProjectEvidenceFact values")
    selected_kinds = {
        fact.evidence_fact_id: kind
        for fact in facts
        if (kind := classify_project_capability_evidence_kind(fact, definition)) is not None
    }
    return _confidence_for_proof(definition, facts, selected_kinds)


def _empty_capability_report(*, project_count: int = 0, fact_count: int = 0) -> ProjectCapabilityExtractionReport:
    return ProjectCapabilityExtractionReport(
        project_count=project_count,
        fact_count=fact_count,
        signal_binding_count=0,
        capability_candidates_evaluated=0,
        capabilities_emitted=0,
        missing_required_group_count=0,
        insufficient_quality_count=0,
        insufficient_direct_fact_count=0,
        insufficient_total_fact_count=0,
        direct_provenance_required_count=0,
        contextual_only_rejected_count=0,
        high_risk_blocked_count=0,
        project_mismatch_count=0,
        duplicate_fact_binding_count=0,
        conflict_count=0,
    )


def _extract_one_project(
    project_id: str,
    facts: Sequence[ProjectEvidenceFact],
    *,
    external_conflicts: Sequence[ProjectCapabilityExtractionConflict] = (),
) -> tuple[list[ProjectCapabilityFact], ProjectCapabilityExtractionReport]:
    matched = [fact for fact in facts if fact.project_id == project_id]
    project_mismatch_count = len(facts) - len(matched)
    ordered = sorted(matched, key=lambda fact: (fact.evidence_fact_id, _fact_payload_hash(fact)))
    unique: list[ProjectEvidenceFact] = []
    duplicate_count = 0
    conflicts: list[ProjectCapabilityExtractionConflict] = list(external_conflicts)
    hard_conflict_ids: set[str] = {
        fact_id for conflict in external_conflicts for fact_id in conflict.evidence_fact_ids
    }
    by_id: dict[str, tuple[str, ProjectEvidenceFact]] = {}
    for fact in ordered:
        payload_hash = _fact_payload_hash(fact)
        previous = by_id.get(fact.evidence_fact_id)
        if previous is None:
            by_id[fact.evidence_fact_id] = (payload_hash, fact)
            unique.append(fact)
        elif previous[0] == payload_hash:
            duplicate_count += 1
        else:
            hard_conflict_ids.add(fact.evidence_fact_id)
            conflicts.append(ProjectCapabilityExtractionConflict(
                project_id=project_id,
                conflict_code="same_fact_id_different_payload",
                evidence_fact_ids=(fact.evidence_fact_id,),
            ))

    extractions, signal_report = extract_project_evidence_fact_signals_many(unique)
    fact_by_id = {fact.evidence_fact_id: fact for fact in unique}
    extraction_by_id = {item.evidence_fact_id: item for item in extractions}
    categories = {fact.evidence_fact_id: classify_project_evidence_source_category(fact) for fact in unique}

    by_lineage: dict[str, list[ProjectEvidenceFact]] = {}
    for fact in unique:
        if categories[fact.evidence_fact_id] == SOURCE_CATEGORY_DIRECT:
            by_lineage.setdefault(_base_lineage_identity(fact), []).append(fact)
    for lineage_facts in by_lineage.values():
        hashes = {ref.content_hash for fact in lineage_facts for ref in fact.source_refs}
        signal_sets = {extraction_by_id[fact.evidence_fact_id].signals for fact in lineage_facts}
        if len(hashes) > 1 and len(signal_sets) > 1:
            ids = tuple(sorted({fact.evidence_fact_id for fact in lineage_facts}))
            hard_conflict_ids.update(ids)
            conflicts.append(ProjectCapabilityExtractionConflict(
                project_id=project_id,
                conflict_code="same_lineage_conflicting_signals",
                evidence_fact_ids=ids,
                rule_ids=tuple(sorted({
                    binding.rule_id
                    for fact in lineage_facts
                    for binding in extraction_by_id[fact.evidence_fact_id].bindings
                })),
            ))

    emitted: list[ProjectCapabilityFact] = []
    decisions: list[ProjectCapabilityExtractionDecision] = []
    missing_required_group_count = 0
    insufficient_quality_count = 0
    insufficient_direct_fact_count = 0
    insufficient_total_fact_count = 0
    direct_provenance_required_count = 0
    contextual_only_rejected_count = 0
    high_risk_blocked_count = 0

    for definition in list_project_capability_definitions():
        required_signals = set().union(*map(set, definition.required_signal_groups))
        relevant_signals = required_signals | set(definition.supporting_signals)
        relevant_bindings = [
            binding
            for item in extractions
            for binding in item.bindings
            if binding.signal in relevant_signals
        ]
        candidate_ids = {binding.evidence_fact_id for binding in relevant_bindings}
        conflict_ids = candidate_ids & hard_conflict_ids
        eligible: list[tuple[ProjectSignalEvidenceBinding, str]] = []
        eligible_before_quality: list[tuple[ProjectSignalEvidenceBinding, str]] = []
        for binding in relevant_bindings:
            fact = fact_by_id[binding.evidence_fact_id]
            kind = _eligible_kind(fact, categories[fact.evidence_fact_id], definition)
            if kind is None or fact.evidence_fact_id in hard_conflict_ids:
                continue
            eligible_before_quality.append((binding, kind))
            if _safe_quality_score(fact) >= definition.minimum_quality_score:
                eligible.append((binding, kind))

        group_candidates: list[list[tuple[ProjectSignalEvidenceBinding, str]]] = []
        missing_indexes: list[int] = []
        low_quality_group = False
        for index, group in enumerate(definition.required_signal_groups):
            values = [(binding, kind) for binding, kind in eligible if binding.signal in group]
            group_candidates.append(values)
            if not values:
                missing_indexes.append(index)
                if any(binding.signal in group for binding, _kind in eligible_before_quality):
                    low_quality_group = True

        selected_ids: list[str] = []
        selected_kinds: dict[str, str] = {}
        for values in group_candidates:
            if not values:
                continue
            binding, kind = min(values, key=lambda item: (
                0 if item[1] == "direct" else 1,
                *_fact_priority(fact_by_id[item[0].evidence_fact_id], categories[item[0].evidence_fact_id]),
                item[0].signal,
                item[0].rule_id,
            ))
            if binding.evidence_fact_id not in selected_ids:
                selected_ids.append(binding.evidence_fact_id)
            selected_kinds[binding.evidence_fact_id] = kind

        additional = sorted({
            binding.evidence_fact_id
            for binding, _kind in eligible
            if binding.evidence_fact_id not in selected_ids
        }, key=lambda fact_id: _fact_priority(fact_by_id[fact_id], categories[fact_id]))

        def direct_identity_count() -> int:
            return len({
                _independent_evidence_identity(fact_by_id[fact_id])
                for fact_id in selected_ids
                if selected_kinds.get(fact_id) == "direct"
            })

        for fact_id in list(additional):
            if direct_identity_count() >= definition.minimum_direct_fact_count:
                break
            kind = next(kind for binding, kind in eligible if binding.evidence_fact_id == fact_id)
            if kind != "direct":
                continue
            selected_ids.append(fact_id)
            selected_kinds[fact_id] = kind
            additional.remove(fact_id)
        for fact_id in list(additional):
            if len(set(selected_ids)) >= definition.minimum_total_fact_count:
                break
            kind = next(kind for binding, kind in eligible if binding.evidence_fact_id == fact_id)
            selected_ids.append(fact_id)
            selected_kinds[fact_id] = kind
            additional.remove(fact_id)

        # One extra independent strong direct fact may be part of the proof for
        # high confidence.  Repetition from the same lineage cannot promote it.
        if not missing_indexes and direct_identity_count() < 2:
            for fact_id in additional:
                kind = next(kind for binding, kind in eligible if binding.evidence_fact_id == fact_id)
                fact = fact_by_id[fact_id]
                if (
                    kind == "direct"
                    and _safe_quality_score(fact) >= definition.minimum_quality_score + HIGH_CONFIDENCE_SCORE_MARGIN
                    and _independent_evidence_identity(fact) not in {
                        _independent_evidence_identity(fact_by_id[item])
                        for item in selected_ids
                        if selected_kinds.get(item) == "direct"
                    }
                ):
                    selected_ids.append(fact_id)
                    selected_kinds[fact_id] = kind
                    break

        direct_count = direct_identity_count()
        total_count = len(set(selected_ids))
        insufficient_direct = direct_count < definition.minimum_direct_fact_count
        insufficient_total = total_count < definition.minimum_total_fact_count
        direct_required = definition.requires_direct_provenance and direct_count == 0
        has_contextual = any(kind == "contextual" for _binding, kind in eligible)
        has_direct = any(kind == "direct" for _binding, kind in eligible)
        contextual_only = bool(relevant_bindings) and has_contextual and not has_direct
        group_uses_context = any(
            selected_kinds.get(fact_id) == "contextual" for fact_id in selected_ids
        )
        high_risk_failure = False
        if definition.high_risk and relevant_bindings:
            if definition.capability_type == "llm_reliability":
                high_risk_failure = bool(missing_indexes or direct_count < 2 or group_uses_context)
            elif definition.capability_type == "token_or_context_efficiency":
                high_risk_failure = bool(missing_indexes or direct_required or group_uses_context)

        if missing_indexes:
            missing_required_group_count += len(missing_indexes)
        insufficient_quality_count += int(low_quality_group)
        insufficient_direct_fact_count += int(insufficient_direct)
        insufficient_total_fact_count += int(insufficient_total)
        direct_provenance_required_count += int(direct_required)
        contextual_only_rejected_count += int(contextual_only)
        high_risk_blocked_count += int(high_risk_failure)

        can_emit = not any((
            missing_indexes,
            low_quality_group,
            insufficient_direct,
            insufficient_total,
            direct_required,
            contextual_only,
            conflict_ids,
            high_risk_failure,
        ))
        selected_ids = list(dict.fromkeys(selected_ids))
        selected_facts = [fact_by_id[fact_id] for fact_id in selected_ids]
        if can_emit:
            metric_support = _metric_support_for_facts(selected_facts)
            mechanisms = _selected_mechanisms(selected_facts)
            confidence = _confidence_for_proof(definition, selected_facts, selected_kinds)
            emitted.append(ProjectCapabilityFact(
                project_id=project_id,
                capability_type=definition.capability_type,
                present=True,
                source_evidence_fact_ids=selected_ids,
                confidence=confidence,
                mechanisms=mechanisms,
                allowed_resume_claims=[],
                forbidden_claims=[],
                metric_support=metric_support,
                technical_tags=_selected_tags(selected_facts),
            ))
            decision_code = "emitted"
        elif conflict_ids:
            decision_code = "conflict"
        elif high_risk_failure:
            decision_code = "high_risk_blocked"
        elif contextual_only:
            decision_code = "contextual_only_rejected"
        elif low_quality_group:
            decision_code = "insufficient_quality"
        elif missing_indexes:
            decision_code = "missing_required_groups"
        elif direct_required:
            decision_code = "direct_provenance_required"
        elif insufficient_direct:
            decision_code = "insufficient_direct_fact_count"
        else:
            decision_code = "insufficient_total_fact_count"
        decisions.append(ProjectCapabilityExtractionDecision(
            project_id=project_id,
            capability_type=definition.capability_type,
            decision_code=decision_code,
            supporting_evidence_fact_ids=tuple(sorted(selected_ids)) if can_emit else (),
            missing_required_group_indexes=tuple(missing_indexes),
        ))

    emitted.sort(key=lambda item: (item.project_id, item.capability_type, item.capability_id))
    overlaps: list[ProjectCapabilityOverlapDecision] = []
    emitted_by_type = {item.capability_type: item for item in emitted}
    for rule in list_project_capability_overlap_rules():
        left = emitted_by_type.get(rule.left_capability)
        right = emitted_by_type.get(rule.right_capability)
        if left is None or right is None:
            continue
        shared = tuple(sorted(set(left.source_evidence_fact_ids) & set(right.source_evidence_fact_ids)))
        if shared:
            overlaps.append(ProjectCapabilityOverlapDecision(
                project_id=project_id,
                left_capability=rule.left_capability,
                right_capability=rule.right_capability,
                shared_evidence_fact_ids=shared,
            ))

    conflicts = sorted(set(conflicts), key=lambda item: (
        item.project_id,
        item.conflict_code,
        item.capability_type,
        item.evidence_fact_ids,
        item.rule_ids,
    ))
    decisions.sort(key=lambda item: (
        item.project_id,
        item.capability_type,
        item.decision_code,
        item.supporting_evidence_fact_ids,
    ))
    report = ProjectCapabilityExtractionReport(
        project_count=1,
        fact_count=len(facts),
        signal_binding_count=signal_report.signal_binding_count,
        capability_candidates_evaluated=len(list_project_capability_definitions()),
        capabilities_emitted=len(emitted),
        missing_required_group_count=missing_required_group_count,
        insufficient_quality_count=insufficient_quality_count,
        insufficient_direct_fact_count=insufficient_direct_fact_count,
        insufficient_total_fact_count=insufficient_total_fact_count,
        direct_provenance_required_count=direct_provenance_required_count,
        contextual_only_rejected_count=contextual_only_rejected_count,
        high_risk_blocked_count=high_risk_blocked_count,
        project_mismatch_count=project_mismatch_count,
        duplicate_fact_binding_count=duplicate_count,
        conflict_count=len(conflicts),
        overlap_count=len(overlaps),
        decisions=tuple(decisions[:MAX_DECISION_SAMPLES]),
        conflicts=tuple(conflicts[:MAX_DECISION_SAMPLES]),
        overlaps=tuple(overlaps[:MAX_DECISION_SAMPLES]),
        grouped_decision_counts=tuple(
            ProjectCapabilityDecisionGroupCount(
                capability_type=capability_type,
                decision_code=decision_code,
                count=sum(
                    item.capability_type == capability_type and item.decision_code == decision_code
                    for item in decisions
                ),
            )
            for capability_type, decision_code in sorted({
                (item.capability_type, item.decision_code) for item in decisions
            })
        ),
    )
    return emitted, report


def extract_project_capabilities(
    project_id: str,
    facts: Iterable[ProjectEvidenceFact],
) -> tuple[list[ProjectCapabilityFact], ProjectCapabilityExtractionReport]:
    if not isinstance(project_id, str):
        raise TypeError("project_id must be a string")
    normalized_project_id = " ".join(project_id.split())
    if not normalized_project_id:
        raise ValueError("project_id must not be blank")
    records = list(facts)
    if any(not isinstance(fact, ProjectEvidenceFact) for fact in records):
        raise TypeError("extract_project_capabilities expects ProjectEvidenceFact values")
    return _extract_one_project(normalized_project_id, records)


def _aggregate_reports(
    reports: Sequence[ProjectCapabilityExtractionReport],
    *,
    fact_count: int,
) -> ProjectCapabilityExtractionReport:
    decisions = sorted(
        (item for report in reports for item in report.decisions),
        key=lambda item: (item.project_id, item.capability_type, item.decision_code),
    )
    conflicts = sorted(set(
        item for report in reports for item in report.conflicts
    ), key=lambda item: (item.project_id, item.conflict_code, item.evidence_fact_ids))
    overlaps = sorted(set(
        item for report in reports for item in report.overlaps
    ), key=lambda item: (item.project_id, item.left_capability, item.right_capability))
    grouped_keys = sorted({
        (item.capability_type, item.decision_code)
        for report in reports
        for item in report.grouped_decision_counts
    })
    return ProjectCapabilityExtractionReport(
        project_count=sum(report.project_count for report in reports),
        fact_count=fact_count,
        signal_binding_count=sum(report.signal_binding_count for report in reports),
        capability_candidates_evaluated=sum(report.capability_candidates_evaluated for report in reports),
        capabilities_emitted=sum(report.capabilities_emitted for report in reports),
        missing_required_group_count=sum(report.missing_required_group_count for report in reports),
        insufficient_quality_count=sum(report.insufficient_quality_count for report in reports),
        insufficient_direct_fact_count=sum(report.insufficient_direct_fact_count for report in reports),
        insufficient_total_fact_count=sum(report.insufficient_total_fact_count for report in reports),
        direct_provenance_required_count=sum(report.direct_provenance_required_count for report in reports),
        contextual_only_rejected_count=sum(report.contextual_only_rejected_count for report in reports),
        high_risk_blocked_count=sum(report.high_risk_blocked_count for report in reports),
        project_mismatch_count=sum(report.project_mismatch_count for report in reports),
        duplicate_fact_binding_count=sum(report.duplicate_fact_binding_count for report in reports),
        conflict_count=len(conflicts),
        overlap_count=len(overlaps),
        decisions=tuple(decisions[:MAX_DECISION_SAMPLES]),
        conflicts=tuple(conflicts[:MAX_DECISION_SAMPLES]),
        overlaps=tuple(overlaps[:MAX_DECISION_SAMPLES]),
        grouped_decision_counts=tuple(
            ProjectCapabilityDecisionGroupCount(
                capability_type=capability_type,
                decision_code=decision_code,
                count=sum(
                    item.count
                    for report in reports
                    for item in report.grouped_decision_counts
                    if item.capability_type == capability_type and item.decision_code == decision_code
                ),
            )
            for capability_type, decision_code in grouped_keys
        ),
    )


def extract_project_evidence_capabilities_by_project(
    facts: Iterable[ProjectEvidenceFact],
) -> tuple[dict[str, list[ProjectCapabilityFact]], ProjectCapabilityExtractionReport]:
    records = list(facts)
    if any(not isinstance(fact, ProjectEvidenceFact) for fact in records):
        raise TypeError("extract_project_evidence_capabilities_by_project expects ProjectEvidenceFact values")
    if not records:
        return {}, _empty_capability_report()
    by_project: dict[str, list[ProjectEvidenceFact]] = {}
    for fact in records:
        by_project.setdefault(fact.project_id, []).append(fact)

    duplicate_projects: dict[str, set[str]] = {}
    for fact in records:
        duplicate_projects.setdefault(fact.evidence_fact_id, set()).add(fact.project_id)
    global_conflicts: dict[str, list[ProjectCapabilityExtractionConflict]] = {}
    for fact_id, projects in sorted(duplicate_projects.items()):
        if len(projects) <= 1:
            continue
        for project_id in sorted(projects):
            global_conflicts.setdefault(project_id, []).append(ProjectCapabilityExtractionConflict(
                project_id=project_id,
                conflict_code="duplicate_fact_id_across_projects",
                evidence_fact_ids=(fact_id,),
            ))

    output: dict[str, list[ProjectCapabilityFact]] = {}
    reports: list[ProjectCapabilityExtractionReport] = []
    for project_id in sorted(by_project):
        capabilities, report = _extract_one_project(
            project_id,
            by_project[project_id],
            external_conflicts=global_conflicts.get(project_id, ()),
        )
        output[project_id] = capabilities
        reports.append(report)
    return output, _aggregate_reports(reports, fact_count=len(records))


__all__ = [
    "CAPABILITY_CONTEXT_SOURCE_TYPES",
    "DIRECT_SOURCE_TYPES",
    "HIGH_CONFIDENCE_SCORE_MARGIN",
    "MAX_DECISION_SAMPLES",
    "PROJECT_CONTEXT_SOURCE_TYPES",
    "SIGNAL_EXTRACTION_RULES",
    "ProjectCapabilityExtractionConflict",
    "ProjectCapabilityExtractionDecision",
    "ProjectCapabilityExtractionReport",
    "ProjectCapabilityDecisionGroupCount",
    "ProjectCapabilityOverlapDecision",
    "ProjectFactSignalExtraction",
    "ProjectSignalEvidenceBinding",
    "ProjectSignalExtractionDecision",
    "ProjectSignalExtractionReport",
    "ProjectSignalExtractionRule",
    "classify_project_capability_evidence_kind",
    "classify_project_evidence_source_category",
    "derive_project_capability_confidence",
    "extract_project_evidence_capabilities_by_project",
    "extract_project_evidence_fact_signals",
    "extract_project_evidence_fact_signals_many",
    "extract_project_capabilities",
    "get_independent_project_evidence_identity",
    "get_project_evidence_quality_score",
    "list_project_evidence_signal_extraction_rules",
    "select_project_evidence_mechanisms",
    "select_project_evidence_technical_tags",
    "validate_project_evidence_signal_extraction_rules",
]
