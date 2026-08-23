"""Bounded semantic intents for a future project-evidence retrieval adapter.

The structures in this module describe evidence goals only.  They contain no
queries or retrieval controls and perform no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import re
from typing import Any

from backend.project_evidence_coverage import (
    MAX_PRIORITIZED_FOLLOWUP_GAPS,
    MAX_SUPPORTING_REFS_PER_DIMENSION,
    CoverageCategory,
    CoverageEvidenceRef,
    CoverageReasonCode,
    CoverageRequirement,
    CoverageState,
    GapPriority,
    GapPriorityReasonCode,
    PrioritizedCoverageGap,
)
from backend.project_evidence_models import EvidenceType
from backend.project_repository_identity import normalize_project_id


MAX_FOLLOWUP_INTENT_CATEGORIES = MAX_PRIORITIZED_FOLLOWUP_GAPS
MAX_FOLLOWUP_INTENT_GOALS = 6
MAX_FOLLOWUP_INTENT_EVIDENCE_TYPES = 4
MAX_FOLLOWUP_INTENT_REQUIREMENTS = 16
MAX_FOLLOWUP_INTENT_SUPPORTING_REFS = MAX_SUPPORTING_REFS_PER_DIMENSION

_SAFE_REFERENCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,299}$")


class FollowupEvidenceGoal(str, Enum):
    CONCRETE_IMPLEMENTATION_MECHANISM = "concrete_implementation_mechanism"
    TECHNICAL_WORKFLOW = "technical_workflow"
    ALGORITHM_OR_PROCESSING_STEP = "algorithm_or_processing_step"
    SYSTEM_COMPONENTS = "system_components"
    COMPONENT_RELATIONSHIPS = "component_relationships"
    DATA_OR_CONTROL_FLOW = "data_or_control_flow"
    SERVICE_BOUNDARIES = "service_boundaries"
    PERSISTENCE_MECHANISM = "persistence_mechanism"
    DATABASE_OR_CACHE_MECHANISM = "database_or_cache_mechanism"
    STORAGE_LIFECYCLE = "storage_lifecycle"
    RETRIEVAL_MECHANISM = "retrieval_mechanism"
    INDEXING_MECHANISM = "indexing_mechanism"
    RANKING_OR_RERANKING_MECHANISM = "ranking_or_reranking_mechanism"
    EVIDENCE_SELECTION_MECHANISM = "evidence_selection_mechanism"
    VALIDATION_MECHANISM = "validation_mechanism"
    REPAIR_MECHANISM = "repair_mechanism"
    RETRY_OR_FALLBACK_BEHAVIOR = "retry_or_fallback_behavior"
    FAILURE_HANDLING = "failure_handling"
    DETERMINISTIC_VERIFICATION = "deterministic_verification"
    OUTPUT_VALIDATION = "output_validation"
    QUALITY_GATE = "quality_gate"
    SCHEMA_VALIDATION = "schema_validation"
    CONSISTENCY_ENFORCEMENT = "consistency_enforcement"
    EVIDENCE_GROUNDING = "evidence_grounding"
    CONFIDENCE_HANDLING = "confidence_handling"
    CLAIM_VALIDATION = "claim_validation"
    SOURCE_ENFORCEMENT = "source_enforcement"
    FACTUALITY_EVALUATION = "factuality_evaluation"
    METRIC_OR_IMPACT_EVIDENCE = "metric_or_impact_evidence"
    JD_REQUIREMENT_EVIDENCE = "jd_requirement_evidence"


_CATEGORY_INDEX = {category: index for index, category in enumerate(CoverageCategory)}
_PRIORITY_INDEX = {
    GapPriority.HIGH: 0,
    GapPriority.MEDIUM: 1,
    GapPriority.LOW: 2,
    GapPriority.IGNORE: 3,
}

_CATEGORY_TARGETS: dict[
    CoverageCategory,
    tuple[tuple[FollowupEvidenceGoal, ...], tuple[EvidenceType, ...]],
] = {
    CoverageCategory.IMPLEMENTATION_MECHANISM: (
        (
            FollowupEvidenceGoal.CONCRETE_IMPLEMENTATION_MECHANISM,
            FollowupEvidenceGoal.TECHNICAL_WORKFLOW,
            FollowupEvidenceGoal.ALGORITHM_OR_PROCESSING_STEP,
        ),
        (EvidenceType.FEATURE, EvidenceType.WORKFLOW, EvidenceType.INTEGRATION),
    ),
    CoverageCategory.ARCHITECTURE: (
        (
            FollowupEvidenceGoal.SYSTEM_COMPONENTS,
            FollowupEvidenceGoal.COMPONENT_RELATIONSHIPS,
            FollowupEvidenceGoal.DATA_OR_CONTROL_FLOW,
            FollowupEvidenceGoal.SERVICE_BOUNDARIES,
        ),
        (EvidenceType.ARCHITECTURE, EvidenceType.WORKFLOW),
    ),
    CoverageCategory.DATA_STORAGE: (
        (
            FollowupEvidenceGoal.PERSISTENCE_MECHANISM,
            FollowupEvidenceGoal.DATABASE_OR_CACHE_MECHANISM,
            FollowupEvidenceGoal.STORAGE_LIFECYCLE,
        ),
        (
            EvidenceType.DATA_PERSISTENCE,
            EvidenceType.ARCHITECTURE,
            EvidenceType.WORKFLOW,
        ),
    ),
    CoverageCategory.RETRIEVAL_RANKING: (
        (
            FollowupEvidenceGoal.RETRIEVAL_MECHANISM,
            FollowupEvidenceGoal.INDEXING_MECHANISM,
            FollowupEvidenceGoal.RANKING_OR_RERANKING_MECHANISM,
            FollowupEvidenceGoal.EVIDENCE_SELECTION_MECHANISM,
        ),
        (
            EvidenceType.RETRIEVAL,
            EvidenceType.OPTIMIZATION,
            EvidenceType.WORKFLOW,
        ),
    ),
    CoverageCategory.VALIDATION_REPAIR: (
        (
            FollowupEvidenceGoal.VALIDATION_MECHANISM,
            FollowupEvidenceGoal.REPAIR_MECHANISM,
            FollowupEvidenceGoal.RETRY_OR_FALLBACK_BEHAVIOR,
            FollowupEvidenceGoal.FAILURE_HANDLING,
            FollowupEvidenceGoal.DETERMINISTIC_VERIFICATION,
        ),
        (
            EvidenceType.VALIDATION,
            EvidenceType.FAILURE_RECOVERY,
            EvidenceType.TESTING,
            EvidenceType.BUG_FIX,
        ),
    ),
    CoverageCategory.OUTPUT_QUALITY: (
        (
            FollowupEvidenceGoal.OUTPUT_VALIDATION,
            FollowupEvidenceGoal.QUALITY_GATE,
            FollowupEvidenceGoal.SCHEMA_VALIDATION,
            FollowupEvidenceGoal.CONSISTENCY_ENFORCEMENT,
        ),
        (
            EvidenceType.VALIDATION,
            EvidenceType.TESTING,
            EvidenceType.CONFIGURATION,
        ),
    ),
    CoverageCategory.RELIABILITY: (
        (
            FollowupEvidenceGoal.EVIDENCE_GROUNDING,
            FollowupEvidenceGoal.RETRY_OR_FALLBACK_BEHAVIOR,
            FollowupEvidenceGoal.CONFIDENCE_HANDLING,
            FollowupEvidenceGoal.CLAIM_VALIDATION,
            FollowupEvidenceGoal.SOURCE_ENFORCEMENT,
            FollowupEvidenceGoal.FACTUALITY_EVALUATION,
        ),
        (
            EvidenceType.VALIDATION,
            EvidenceType.FAILURE_RECOVERY,
            EvidenceType.TESTING,
            EvidenceType.CONFIGURATION,
        ),
    ),
    CoverageCategory.METRIC_IMPACT: (
        (FollowupEvidenceGoal.METRIC_OR_IMPACT_EVIDENCE,),
        (
            EvidenceType.OPTIMIZATION,
            EvidenceType.TESTING,
        ),
    ),
    CoverageCategory.JD_MUST_HAVE: (
        (FollowupEvidenceGoal.JD_REQUIREMENT_EVIDENCE,),
        (),
    ),
}


def _normalized_exact_project_id(value: Any) -> str:
    normalized = normalize_project_id(value)
    if not normalized or normalized != value:
        raise ValueError("project_id must be an exact normalized project identifier")
    return normalized


def _stable_enum_values(
    values: Sequence[Any],
    enum_type: type[Enum],
    *,
    maximum: int,
    name: str,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    index = {item: position for position, item in enumerate(enum_type)}
    normalized = {enum_type(value) for value in values}
    return tuple(sorted(normalized, key=index.__getitem__)[:maximum])


def _stable_requirement_ids(values: Sequence[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("requirement_ids must be a sequence")
    normalized: dict[str, str] = {}
    for value in values:
        try:
            requirement = CoverageRequirement(value, ("semantic_evidence_target",))
        except (TypeError, ValueError):
            continue
        if requirement.requirement_id != value:
            continue
        key = value.casefold()
        if key not in normalized or value < normalized[key]:
            normalized[key] = value
    return tuple(normalized[key] for key in sorted(normalized))[:MAX_FOLLOWUP_INTENT_REQUIREMENTS]


def _ref_sort_key(ref: CoverageEvidenceRef) -> tuple[str, ...]:
    return (
        ref.project_id,
        ref.evidence_fact_id or "",
        ref.capability_fact_id or "",
        ref.claim_boundary_id or "",
        ref.source_id or "",
        ref.chunk_id or "",
    )


def _safe_reference_id(value: str | None) -> str | None:
    return value if value is not None and _SAFE_REFERENCE_ID_RE.fullmatch(value) else None


def _sanitized_ref(ref: CoverageEvidenceRef) -> CoverageEvidenceRef | None:
    identifiers = {
        "evidence_fact_id": _safe_reference_id(ref.evidence_fact_id),
        "capability_fact_id": _safe_reference_id(ref.capability_fact_id),
        "claim_boundary_id": _safe_reference_id(ref.claim_boundary_id),
        "source_id": _safe_reference_id(ref.source_id),
        "chunk_id": _safe_reference_id(ref.chunk_id),
    }
    if not any(identifiers.values()):
        return None
    return CoverageEvidenceRef(project_id=ref.project_id, **identifiers)


def _stable_refs(
    values: Sequence[CoverageEvidenceRef],
    *,
    project_id: str,
) -> tuple[CoverageEvidenceRef, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("supporting_refs must be a sequence")
    if any(not isinstance(ref, CoverageEvidenceRef) for ref in values):
        raise TypeError("supporting_refs must contain CoverageEvidenceRef values")
    if any(ref.project_id != project_id for ref in values):
        raise ValueError("supporting_refs conflict with intent project_id")
    sanitized = {
        candidate
        for ref in values
        if (candidate := _sanitized_ref(ref)) is not None
    }
    return tuple(sorted(sanitized, key=_ref_sort_key))[:MAX_FOLLOWUP_INTENT_SUPPORTING_REFS]


@dataclass(frozen=True, slots=True)
class FollowupRetrievalIntent:
    project_id: str
    target_categories: tuple[CoverageCategory, ...]
    priority: GapPriority
    requirement_ids: tuple[str, ...]
    evidence_goals: tuple[FollowupEvidenceGoal, ...]
    preferred_evidence_types: tuple[EvidenceType, ...]
    supporting_refs: tuple[CoverageEvidenceRef, ...]
    reason_codes: tuple[GapPriorityReasonCode, ...]

    def __post_init__(self) -> None:
        project_id = _normalized_exact_project_id(self.project_id)
        categories = _stable_enum_values(
            self.target_categories,
            CoverageCategory,
            maximum=MAX_FOLLOWUP_INTENT_CATEGORIES,
            name="target_categories",
        )
        if not categories or any(category not in _CATEGORY_TARGETS for category in categories):
            raise ValueError("intent requires at least one supported target category")
        priority = GapPriority(self.priority)
        if priority is GapPriority.IGNORE:
            raise ValueError("ignored gaps cannot create follow-up intents")
        goals = _stable_enum_values(
            self.evidence_goals,
            FollowupEvidenceGoal,
            maximum=MAX_FOLLOWUP_INTENT_GOALS,
            name="evidence_goals",
        )
        evidence_types = _stable_enum_values(
            self.preferred_evidence_types,
            EvidenceType,
            maximum=MAX_FOLLOWUP_INTENT_EVIDENCE_TYPES,
            name="preferred_evidence_types",
        )
        if not goals or (not evidence_types and categories != (CoverageCategory.JD_MUST_HAVE,)):
            raise ValueError("intent requires canonical semantic goals and evidence types")
        canonical_targets = {
            (
                _stable_enum_values(
                    _CATEGORY_TARGETS[category][0],
                    FollowupEvidenceGoal,
                    maximum=MAX_FOLLOWUP_INTENT_GOALS,
                    name="canonical_evidence_goals",
                ),
                _stable_enum_values(
                    _CATEGORY_TARGETS[category][1],
                    EvidenceType,
                    maximum=MAX_FOLLOWUP_INTENT_EVIDENCE_TYPES,
                    name="canonical_evidence_types",
                ),
            )
            for category in categories
        }
        if len(canonical_targets) != 1 or (goals, evidence_types) not in canonical_targets:
            raise ValueError("intent goals and evidence types must match canonical category targets")
        reason_codes = _stable_enum_values(
            self.reason_codes,
            GapPriorityReasonCode,
            maximum=MAX_FOLLOWUP_INTENT_CATEGORIES,
            name="reason_codes",
        )
        if not reason_codes:
            raise ValueError("intent requires an upstream priority reason")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "target_categories", categories)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "requirement_ids", _stable_requirement_ids(self.requirement_ids))
        object.__setattr__(self, "evidence_goals", goals)
        object.__setattr__(self, "preferred_evidence_types", evidence_types)
        object.__setattr__(
            self,
            "supporting_refs",
            _stable_refs(self.supporting_refs, project_id=project_id),
        )
        object.__setattr__(self, "reason_codes", reason_codes)

    @property
    def target_category(self) -> CoverageCategory:
        return self.target_categories[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "target_categories": [item.value for item in self.target_categories],
            "priority": self.priority.value,
            "requirement_ids": list(self.requirement_ids),
            "evidence_goals": [item.value for item in self.evidence_goals],
            "preferred_evidence_types": [item.value for item in self.preferred_evidence_types],
            "supporting_refs": [item.to_dict() for item in self.supporting_refs],
            "reason_codes": [item.value for item in self.reason_codes],
        }


def _intent_from_gap(
    *,
    project_id: str,
    prioritized_gap: PrioritizedCoverageGap,
) -> FollowupRetrievalIntent | None:
    if prioritized_gap.priority is GapPriority.IGNORE or not prioritized_gap.searchable:
        return None
    gap = prioritized_gap.gap
    if (
        gap.state not in {CoverageState.MISSING, CoverageState.PARTIAL}
        or gap.reason_code in {
            CoverageReasonCode.CLAIM_BOUNDARY_RESTRICTED,
            CoverageReasonCode.WRONG_PROJECT,
        }
        or gap.category not in _CATEGORY_TARGETS
    ):
        return None
    requirements = _stable_requirement_ids(gap.related_requirement_ids)
    if gap.category is CoverageCategory.JD_MUST_HAVE and not requirements:
        return None
    if gap.category is CoverageCategory.METRIC_IMPACT and (
        gap.state is not CoverageState.PARTIAL or not requirements
    ):
        return None
    if any(ref.project_id != project_id for ref in gap.current_support_refs):
        return None
    safe_refs = _stable_refs(gap.current_support_refs, project_id=project_id)
    if gap.category is CoverageCategory.METRIC_IMPACT and not safe_refs:
        return None
    goals, kinds = _CATEGORY_TARGETS[gap.category]
    return FollowupRetrievalIntent(
        project_id=project_id,
        target_categories=(gap.category,),
        priority=prioritized_gap.priority,
        requirement_ids=requirements,
        evidence_goals=goals,
        preferred_evidence_types=kinds,
        supporting_refs=safe_refs,
        reason_codes=(prioritized_gap.reason_code,),
    )


def _merge_intents(
    first: FollowupRetrievalIntent,
    second: FollowupRetrievalIntent,
) -> FollowupRetrievalIntent:
    priority = min((first.priority, second.priority), key=_PRIORITY_INDEX.__getitem__)
    return FollowupRetrievalIntent(
        project_id=first.project_id,
        target_categories=(*first.target_categories, *second.target_categories),
        priority=priority,
        requirement_ids=(*first.requirement_ids, *second.requirement_ids),
        evidence_goals=first.evidence_goals,
        preferred_evidence_types=first.preferred_evidence_types,
        supporting_refs=(*first.supporting_refs, *second.supporting_refs),
        reason_codes=(*first.reason_codes, *second.reason_codes),
    )


def build_followup_retrieval_intents(
    *,
    project_id: str,
    prioritized_gaps: Sequence[PrioritizedCoverageGap],
) -> tuple[FollowupRetrievalIntent, ...]:
    """Translate actionable gaps into canonical semantic evidence goals only."""

    normalized_project_id = _normalized_exact_project_id(project_id)
    if isinstance(prioritized_gaps, (str, bytes)) or not isinstance(prioritized_gaps, Sequence):
        raise TypeError("prioritized_gaps must be a sequence")
    if any(not isinstance(item, PrioritizedCoverageGap) for item in prioritized_gaps):
        raise TypeError("prioritized_gaps must contain PrioritizedCoverageGap values")

    drafts = tuple(
        intent
        for item in prioritized_gaps
        if (
            intent := _intent_from_gap(
                project_id=normalized_project_id,
                prioritized_gap=item,
            )
        ) is not None
    )
    merged: dict[
        tuple[tuple[FollowupEvidenceGoal, ...], tuple[EvidenceType, ...]],
        FollowupRetrievalIntent,
    ] = {}
    for intent in drafts:
        key = (intent.evidence_goals, intent.preferred_evidence_types)
        merged[key] = _merge_intents(merged[key], intent) if key in merged else intent
    ordered = tuple(sorted(
        merged.values(),
        key=lambda item: (
            _PRIORITY_INDEX[item.priority],
            _CATEGORY_INDEX[item.target_category],
            tuple(_CATEGORY_INDEX[category] for category in item.target_categories),
        ),
    ))
    return ordered[:MAX_PRIORITIZED_FOLLOWUP_GAPS]


def validate_followup_retrieval_intents(
    *,
    project_id: Any,
    retrieval_intents: Any = None,
) -> tuple[FollowupRetrievalIntent, ...]:
    """Validate a bounded intent sequence for one exact project.

    ``None`` and an empty sequence deliberately bypass project validation so
    existing no-intent retrieval calls retain their pre-contract behavior.
    """

    if retrieval_intents is None:
        return ()
    if isinstance(retrieval_intents, (str, bytes)) or not isinstance(
        retrieval_intents, Sequence
    ):
        raise TypeError("retrieval_intents must be a sequence")
    if not retrieval_intents:
        return ()
    if len(retrieval_intents) > MAX_PRIORITIZED_FOLLOWUP_GAPS:
        raise ValueError("retrieval_intents exceeds the accepted follow-up bound")
    normalized_project_id = _normalized_exact_project_id(project_id)
    if any(not isinstance(item, FollowupRetrievalIntent) for item in retrieval_intents):
        raise TypeError("retrieval_intents must contain FollowupRetrievalIntent values")
    if any(item.project_id != normalized_project_id for item in retrieval_intents):
        raise ValueError("retrieval_intents conflict with the requested project_id")
    return tuple(sorted(
        retrieval_intents,
        key=lambda item: (
            _PRIORITY_INDEX[item.priority],
            tuple(_CATEGORY_INDEX[category] for category in item.target_categories),
            tuple(goal.value for goal in item.evidence_goals),
            tuple(evidence_type.value for evidence_type in item.preferred_evidence_types),
            item.requirement_ids,
            tuple(_ref_sort_key(ref) for ref in item.supporting_refs),
            tuple(reason.value for reason in item.reason_codes),
        ),
    ))


__all__ = [
    "FollowupEvidenceGoal",
    "FollowupRetrievalIntent",
    "MAX_FOLLOWUP_INTENT_CATEGORIES",
    "MAX_FOLLOWUP_INTENT_EVIDENCE_TYPES",
    "MAX_FOLLOWUP_INTENT_GOALS",
    "MAX_FOLLOWUP_INTENT_REQUIREMENTS",
    "MAX_FOLLOWUP_INTENT_SUPPORTING_REFS",
    "build_followup_retrieval_intents",
    "validate_followup_retrieval_intents",
]
