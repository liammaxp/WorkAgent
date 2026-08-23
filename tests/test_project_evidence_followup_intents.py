from __future__ import annotations

import ast
import builtins
from dataclasses import FrozenInstanceError
import inspect
import json
import os
from pathlib import Path

import pytest

import backend.project_evidence_followup_intents as intent_module
from backend.project_evidence_coverage import (
    MAX_PRIORITIZED_FOLLOWUP_GAPS,
    CoverageCategory,
    CoverageEvidenceRef,
    CoverageGap,
    CoverageReasonCode,
    CoverageState,
    GapPriority,
    GapPriorityReasonCode,
    PrioritizedCoverageGap,
)
from backend.project_evidence_followup_intents import (
    MAX_FOLLOWUP_INTENT_CATEGORIES,
    MAX_FOLLOWUP_INTENT_EVIDENCE_TYPES,
    MAX_FOLLOWUP_INTENT_GOALS,
    MAX_FOLLOWUP_INTENT_REQUIREMENTS,
    MAX_FOLLOWUP_INTENT_SUPPORTING_REFS,
    FollowupEvidenceGoal,
    FollowupRetrievalIntent,
    build_followup_retrieval_intents,
)
from backend.project_evidence_models import EvidenceType


PROJECT_ID = "workagent"


def _ref(
    identifier: str,
    *,
    project_id: str = PROJECT_ID,
    source_id: str | None = None,
) -> CoverageEvidenceRef:
    return CoverageEvidenceRef(
        project_id=project_id,
        evidence_fact_id=identifier,
        source_id=source_id,
    )


def _prioritized(
    category: CoverageCategory,
    *,
    state: CoverageState = CoverageState.MISSING,
    priority: GapPriority = GapPriority.HIGH,
    requirement_ids: tuple[str, ...] = (),
    refs: tuple[CoverageEvidenceRef, ...] = (),
    coverage_reason: CoverageReasonCode | None = None,
    priority_reason: GapPriorityReasonCode | None = None,
) -> PrioritizedCoverageGap:
    if coverage_reason is None:
        coverage_reason = (
            CoverageReasonCode.PARTIALLY_SUPPORTED
            if state is CoverageState.PARTIAL
            else CoverageReasonCode.CLAIM_BOUNDARY_RESTRICTED
            if state is CoverageState.BLOCKED
            else CoverageReasonCode.UNSUPPORTED
        )
    if priority_reason is None:
        priority_reason = (
            GapPriorityReasonCode.PARTIAL_SUPPORT_WORTH_EXPANDING
            if state is CoverageState.PARTIAL
            else GapPriorityReasonCode.JD_MUST_HAVE_GAP
        )
    ignored = priority is GapPriority.IGNORE
    return PrioritizedCoverageGap(
        gap=CoverageGap(
            category=category,
            state=state,
            reason_code=coverage_reason,
            related_requirement_ids=requirement_ids,
            current_support_refs=refs,
        ),
        priority=priority,
        searchable=not ignored,
        reason_code=priority_reason,
    )


@pytest.mark.parametrize(
    ("category", "expected_goal", "expected_type"),
    (
        (
            CoverageCategory.VALIDATION_REPAIR,
            FollowupEvidenceGoal.VALIDATION_MECHANISM,
            EvidenceType.VALIDATION,
        ),
        (
            CoverageCategory.RETRIEVAL_RANKING,
            FollowupEvidenceGoal.RETRIEVAL_MECHANISM,
            EvidenceType.RETRIEVAL,
        ),
        (
            CoverageCategory.ARCHITECTURE,
            FollowupEvidenceGoal.SYSTEM_COMPONENTS,
            EvidenceType.ARCHITECTURE,
        ),
        (
            CoverageCategory.DATA_STORAGE,
            FollowupEvidenceGoal.PERSISTENCE_MECHANISM,
            EvidenceType.DATA_PERSISTENCE,
        ),
        (
            CoverageCategory.OUTPUT_QUALITY,
            FollowupEvidenceGoal.OUTPUT_VALIDATION,
            EvidenceType.VALIDATION,
        ),
        (
            CoverageCategory.IMPLEMENTATION_MECHANISM,
            FollowupEvidenceGoal.CONCRETE_IMPLEMENTATION_MECHANISM,
            EvidenceType.FEATURE,
        ),
    ),
)
def test_actionable_categories_create_fixed_semantic_intents(category, expected_goal, expected_type):
    gap = _prioritized(category)

    intents = build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=(gap,),
    )

    assert len(intents) == 1
    intent = intents[0]
    assert intent.project_id == PROJECT_ID
    assert intent.priority is GapPriority.HIGH
    assert intent.target_categories == (category,)
    assert expected_goal in intent.evidence_goals
    assert expected_type in intent.preferred_evidence_types
    assert not ({"query", "raw_query", "vector_query", "keyword_query", "symbol_query"} & intent.to_dict().keys())


def test_reliability_intent_is_mechanism_oriented_without_unsupported_guarantees():
    intent = build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=(_prioritized(CoverageCategory.RELIABILITY),),
    )[0]
    serialized = json.dumps(intent.to_dict(), sort_keys=True).casefold()

    assert FollowupEvidenceGoal.EVIDENCE_GROUNDING in intent.evidence_goals
    assert FollowupEvidenceGoal.CLAIM_VALIDATION in intent.evidence_goals
    assert "hallucination" not in serialized
    assert "eliminate" not in serialized
    assert "guarantee" not in serialized


def test_legitimate_partial_metric_preserves_requirement_without_numeric_target():
    metric = _prioritized(
        CoverageCategory.METRIC_IMPACT,
        state=CoverageState.PARTIAL,
        priority=GapPriority.MEDIUM,
        requirement_ids=("req-latency",),
        refs=(_ref("pef-approximate-metric"),),
        priority_reason=GapPriorityReasonCode.JD_RELEVANT_PARTIAL_SUPPORT,
    )

    intent = build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=(metric,),
    )[0]
    goal_values = " ".join(goal.value for goal in intent.evidence_goals)

    assert intent.priority is GapPriority.MEDIUM
    assert intent.requirement_ids == ("req-latency",)
    assert intent.evidence_goals == (FollowupEvidenceGoal.METRIC_OR_IMPACT_EVIDENCE,)
    assert not any(character.isdigit() for character in goal_values)
    assert "%" not in goal_values


@pytest.mark.parametrize(
    "metric",
    (
        _prioritized(CoverageCategory.METRIC_IMPACT),
        _prioritized(
            CoverageCategory.METRIC_IMPACT,
            state=CoverageState.PARTIAL,
            priority=GapPriority.MEDIUM,
            requirement_ids=("req-latency",),
        ),
    ),
)
def test_missing_or_untraceable_metric_does_not_create_an_intent(metric):
    assert build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=(metric,),
    ) == ()


def test_ignore_blocked_and_project_identity_do_not_create_intents():
    ignored = _prioritized(
        CoverageCategory.VALIDATION_REPAIR,
        priority=GapPriority.IGNORE,
        priority_reason=GapPriorityReasonCode.INSUFFICIENT_PROJECT_RELEVANCE,
    )
    blocked = _prioritized(
        CoverageCategory.VALIDATION_REPAIR,
        state=CoverageState.BLOCKED,
        priority=GapPriority.IGNORE,
        priority_reason=GapPriorityReasonCode.BLOCKED_UPSTREAM,
    )
    identity = _prioritized(CoverageCategory.PROJECT_IDENTITY)

    assert build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=(ignored, blocked, identity),
    ) == ()


@pytest.mark.parametrize("state", (CoverageState.COVERED, CoverageState.NOT_APPLICABLE))
def test_covered_and_not_applicable_fail_at_the_coverage_gap_model_boundary(state):
    with pytest.raises(ValueError, match="coverage gap"):
        CoverageGap(
            category=CoverageCategory.ARCHITECTURE,
            state=state,
            reason_code=CoverageReasonCode.NO_RELEVANT_REQUIREMENT,
        )


def test_mixed_or_foreign_project_gap_is_skipped_while_safe_sibling_survives():
    foreign = _prioritized(
        CoverageCategory.VALIDATION_REPAIR,
        refs=(
            _ref("pef-local"),
            _ref("pef-foreign", project_id="other-project"),
        ),
    )
    safe = _prioritized(CoverageCategory.ARCHITECTURE)

    intents = build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=(foreign, safe),
    )

    assert len(intents) == 1
    assert intents[0].target_categories == (CoverageCategory.ARCHITECTURE,)
    assert all(ref.project_id == PROJECT_ID for ref in intents[0].supporting_refs)


@pytest.mark.parametrize("project_id", ("", " workagent ", "../workagent", "owner/workagent"))
def test_project_id_must_be_exactly_normalized(project_id):
    with pytest.raises(ValueError, match="exact normalized"):
        build_followup_retrieval_intents(project_id=project_id, prioritized_gaps=())


def test_requirement_ids_are_preserved_deduplicated_sorted_and_injection_safe():
    gap = _prioritized(
        CoverageCategory.VALIDATION_REPAIR,
        requirement_ids=(
            "req-z",
            "req-a",
            "req-a",
            "../../secret",
            "search_query=x",
            "req\nraw_query=x",
        ),
    )

    intent = build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=(gap,),
    )[0]

    assert intent.requirement_ids == ("req-a", "req-z")
    assert "secret" not in json.dumps(intent.to_dict(), sort_keys=True)
    assert "search_query" not in json.dumps(intent.to_dict(), sort_keys=True)


def test_path_or_query_shaped_reference_fields_do_not_survive_serialization():
    unsafe_ref = CoverageEvidenceRef(
        project_id=PROJECT_ID,
        evidence_fact_id="pef-safe",
        source_id="../../private/file.py",
        chunk_id="search_query=secrets",
    )
    gap = _prioritized(CoverageCategory.ARCHITECTURE, refs=(unsafe_ref,))

    intent = build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=(gap,),
    )[0]
    serialized = json.dumps(intent.to_dict(), sort_keys=True)

    assert intent.supporting_refs[0].evidence_fact_id == "pef-safe"
    assert intent.supporting_refs[0].source_id is None
    assert intent.supporting_refs[0].chunk_id is None
    assert "private/file.py" not in serialized
    assert "search_query" not in serialized


def test_equivalent_duplicate_gaps_merge_before_budget_and_preserve_highest_priority():
    duplicates = (
        _prioritized(
            CoverageCategory.VALIDATION_REPAIR,
            priority=GapPriority.LOW,
            requirement_ids=("req-z",),
            refs=(_ref("pef-z"),),
            priority_reason=GapPriorityReasonCode.HIGH_VALUE_MECHANISM_GAP,
        ),
        _prioritized(
            CoverageCategory.VALIDATION_REPAIR,
            priority=GapPriority.HIGH,
            requirement_ids=("req-a",),
            refs=(_ref("pef-a"),),
        ),
        _prioritized(CoverageCategory.VALIDATION_REPAIR),
        _prioritized(CoverageCategory.VALIDATION_REPAIR),
    )
    architecture = _prioritized(CoverageCategory.ARCHITECTURE)

    intents = build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=(*duplicates, architecture),
    )

    assert len(intents) == 2
    validation = next(
        item for item in intents if item.target_categories == (CoverageCategory.VALIDATION_REPAIR,)
    )
    assert validation.priority is GapPriority.HIGH
    assert validation.requirement_ids == ("req-a", "req-z")
    assert tuple(ref.evidence_fact_id for ref in validation.supporting_refs) == ("pef-a", "pef-z")


def test_architecture_and_retrieval_remain_distinct_despite_workflow_overlap():
    intents = build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=(
            _prioritized(CoverageCategory.ARCHITECTURE),
            _prioritized(CoverageCategory.RETRIEVAL_RANKING),
        ),
    )

    assert len(intents) == 2
    assert {item.target_category for item in intents} == {
        CoverageCategory.ARCHITECTURE,
        CoverageCategory.RETRIEVAL_RANKING,
    }


def test_priority_is_preserved_without_vocabulary_based_escalation():
    low = _prioritized(CoverageCategory.ARCHITECTURE, priority=GapPriority.LOW)
    medium = _prioritized(
        CoverageCategory.VALIDATION_REPAIR,
        state=CoverageState.PARTIAL,
        priority=GapPriority.MEDIUM,
        refs=(_ref("pef-partial"),),
    )

    intents = build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=(low, medium),
    )

    assert tuple(item.priority for item in intents) == (GapPriority.MEDIUM, GapPriority.LOW)


def test_intent_count_and_every_field_remain_bounded():
    categories = tuple(intent_module._CATEGORY_TARGETS)
    many_refs = tuple(_ref(f"pef-{index:02d}") for index in range(20))
    many_requirements = tuple(f"req-{index:02d}" for index in range(30))
    gaps = tuple(
        _prioritized(
            category,
            requirement_ids=many_requirements,
            refs=many_refs,
            state=(CoverageState.PARTIAL if category is CoverageCategory.METRIC_IMPACT else CoverageState.MISSING),
        )
        for category in categories
    )

    intents = build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=gaps,
    )

    assert len(intents) <= MAX_PRIORITIZED_FOLLOWUP_GAPS
    for intent in intents:
        assert len(intent.target_categories) <= MAX_FOLLOWUP_INTENT_CATEGORIES
        assert len(intent.evidence_goals) <= MAX_FOLLOWUP_INTENT_GOALS
        assert len(intent.preferred_evidence_types) <= MAX_FOLLOWUP_INTENT_EVIDENCE_TYPES
        assert len(intent.requirement_ids) <= MAX_FOLLOWUP_INTENT_REQUIREMENTS
        assert len(intent.supporting_refs) <= MAX_FOLLOWUP_INTENT_SUPPORTING_REFS


def test_input_permutations_produce_identical_intent_order_and_serialization():
    validation = _prioritized(
        CoverageCategory.VALIDATION_REPAIR,
        requirement_ids=("req-z", "req-a"),
        refs=(_ref("pef-z"), _ref("pef-a")),
    )
    architecture = _prioritized(CoverageCategory.ARCHITECTURE, priority=GapPriority.LOW)

    forward = build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=(validation, architecture),
    )
    reverse = build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=(architecture, validation),
    )

    assert forward == reverse
    assert [item.to_dict() for item in forward] == [item.to_dict() for item in reverse]


def test_model_accepts_only_canonical_goals_and_evidence_types():
    kwargs = {
        "project_id": PROJECT_ID,
        "target_categories": (CoverageCategory.ARCHITECTURE,),
        "priority": GapPriority.HIGH,
        "requirement_ids": (),
        "supporting_refs": (),
        "reason_codes": (GapPriorityReasonCode.JD_MUST_HAVE_GAP,),
    }
    with pytest.raises(ValueError):
        FollowupRetrievalIntent(
            **kwargs,
            evidence_goals=("raw search query",),
            preferred_evidence_types=(EvidenceType.ARCHITECTURE,),
        )
    with pytest.raises(ValueError):
        FollowupRetrievalIntent(
            **kwargs,
            evidence_goals=(FollowupEvidenceGoal.SYSTEM_COMPONENTS,),
            preferred_evidence_types=("vector_query",),
        )
    with pytest.raises(ValueError, match="canonical category targets"):
        FollowupRetrievalIntent(
            **kwargs,
            evidence_goals=(FollowupEvidenceGoal.VALIDATION_MECHANISM,),
            preferred_evidence_types=(EvidenceType.VALIDATION,),
        )


def test_intent_serialization_has_no_query_planner_or_raw_evidence_fields():
    intent = build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=(_prioritized(CoverageCategory.RETRIEVAL_RANKING),),
    )[0]
    forbidden = {
        "raw_query", "search_query", "vector_query", "keyword_query", "symbol_query",
        "top_k", "weights", "filter", "path", "file_path", "raw_text", "raw_patch",
        "document", "embedding", "prompt", "target_terms",
    }

    assert not (forbidden & set(intent.to_dict()))
    assert not (forbidden & set(intent.__dataclass_fields__))


def test_builder_is_pure_immutable_and_does_not_mutate_inputs(monkeypatch):
    gap = _prioritized(
        CoverageCategory.VALIDATION_REPAIR,
        requirement_ids=("req-validation",),
        refs=(_ref("pef-validation"),),
    )
    before_gap = json.dumps(gap.gap.to_dict(), sort_keys=True)
    before_environment = dict(os.environ)

    def _forbid_file_access(*args, **kwargs):
        raise AssertionError("intent construction must not access the filesystem")

    monkeypatch.setattr(builtins, "open", _forbid_file_access)
    intents = build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=(gap,),
    )

    assert json.dumps(gap.gap.to_dict(), sort_keys=True) == before_gap
    assert dict(os.environ) == before_environment
    assert isinstance(intents, tuple)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        intents[0].priority = GapPriority.LOW  # type: ignore[misc]


def test_intent_module_has_no_retrieval_chroma_planner_storage_or_environment_dependency():
    source_path = Path(inspect.getsourcefile(intent_module) or "")
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    forbidden_prefixes = (
        "chromadb",
        "backend.memory_store",
        "backend.chroma",
        "backend.project_retrieval",
        "backend.evidence_hybrid_retrieval",
        "backend.evidence_vector_search",
        "backend.evidence_chunk_search",
        "backend.project_query_planner",
        "backend.github_raw_storage",
    )

    assert not any(
        module_name.startswith(prefix)
        for module_name in imported_modules
        for prefix in forbidden_prefixes
    )
    assert "PersistentClient" not in source_text
    assert "os.environ" not in source_text


def test_empty_input_returns_an_empty_tuple():
    assert build_followup_retrieval_intents(
        project_id=PROJECT_ID,
        prioritized_gaps=(),
    ) == ()
