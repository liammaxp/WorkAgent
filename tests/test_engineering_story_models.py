from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from backend.engineering_story_models import (
    MAX_STORY_FIELD_PROVENANCE_IDS,
    MAX_STORY_FIELD_VALUE_LENGTH,
    MAX_STORY_PROVENANCE_IDS,
    ClaimSufficiency,
    EngineeringStory,
    EngineeringStoryField,
    EngineeringStoryFieldName,
    EngineeringStoryLifecycle,
    EngineeringStoryStatus,
    EngineeringStoryType,
    StoryContextGap,
    StoryFieldEvidenceState,
    StoryOpportunity,
    StoryOpportunityLevel,
    StoryOpportunitySignal,
    StorySufficiency,
    SufficiencyLevel,
    validate_engineering_story_id,
)


PEF_A = "pef_" + "a" * 24
PEF_B = "pef_" + "b" * 24
PEF_C = "pef_" + "c" * 24
PCF_A = "pcf_" + "a" * 24
PCB_A = "pcb_" + "a" * 24
STORY_ID = "engineering_story_" + "a" * 24
SUCCESSOR_ID = "engineering_story_" + "b" * 24


def confirmed(
    value: str,
    *,
    evidence_ids: tuple[str, ...] = (PEF_A,),
    capability_ids: tuple[str, ...] = (),
    boundary_ids: tuple[str, ...] = (),
) -> EngineeringStoryField:
    return EngineeringStoryField(
        value=value,
        evidence_state=StoryFieldEvidenceState.CONFIRMED,
        evidence_fact_ids=evidence_ids,
        capability_fact_ids=capability_ids,
        claim_boundary_ids=boundary_ids,
    )


def supported(
    value: str,
    *,
    evidence_ids: tuple[str, ...] = (PEF_A, PEF_B),
) -> EngineeringStoryField:
    return EngineeringStoryField(
        value=value,
        evidence_state=StoryFieldEvidenceState.SUPPORTED,
        evidence_fact_ids=evidence_ids,
    )


def plausible_missing() -> EngineeringStoryField:
    return EngineeringStoryField(
        value=None,
        evidence_state=StoryFieldEvidenceState.PLAUSIBLE_MISSING,
    )


def unsupported() -> EngineeringStoryField:
    return EngineeringStoryField(
        value=None,
        evidence_state=StoryFieldEvidenceState.UNSUPPORTED,
    )


def story(**overrides) -> EngineeringStory:
    values = {
        "story_id": STORY_ID,
        "project_id": "workagent",
        "story_type": EngineeringStoryType.VALIDATION_AND_QUALITY,
        "problem_context": plausible_missing(),
        "trigger": unsupported(),
        "before_state": unsupported(),
        "decision": confirmed("Adopted a strict evidence validation boundary.", evidence_ids=(PEF_B,)),
        "mechanism": confirmed(
            "Validated structured evidence before claim generation.",
            boundary_ids=(PCB_A,),
        ),
        "implementation": confirmed("Added deterministic schema and provenance checks."),
        "tradeoff": unsupported(),
        "validation": supported("Exercised accepted, rejected, and malformed evidence paths."),
        "after_state": unsupported(),
        "observable_outcome": supported("Unsupported claims are blocked before generation."),
        "ownership": plausible_missing(),
        "stakeholder_context": unsupported(),
        "evidence_fact_ids": (PEF_B, PEF_A),
        "capability_fact_ids": (),
        "claim_boundary_ids": (PCB_A,),
        "lifecycle": EngineeringStoryLifecycle(EngineeringStoryStatus.ACTIVE),
        "claim_sufficiency": ClaimSufficiency(
            level=SufficiencyLevel.HIGH,
            supported_fields=(
                EngineeringStoryFieldName.MECHANISM,
                EngineeringStoryFieldName.IMPLEMENTATION,
                EngineeringStoryFieldName.VALIDATION,
            ),
        ),
        "story_sufficiency": StorySufficiency(
            level=SufficiencyLevel.LOW,
            supported_fields=(EngineeringStoryFieldName.DECISION,),
            missing_fields=(
                EngineeringStoryFieldName.PROBLEM_CONTEXT,
                EngineeringStoryFieldName.TRIGGER,
                EngineeringStoryFieldName.OWNERSHIP,
            ),
        ),
        "opportunity": StoryOpportunity(
            level=StoryOpportunityLevel.HIGH,
            signals=(StoryOpportunitySignal.MAJOR_DESIGN_DECISION,),
            missing_context=(
                StoryContextGap.PROBLEM_CONTEXT,
                StoryContextGap.OWNERSHIP,
            ),
        ),
    }
    values.update(overrides)
    return EngineeringStory(**values)


def test_basic_story_preserves_field_level_authority_and_nonclaimable_gaps():
    model = story()
    assert model.mechanism.evidence_state is StoryFieldEvidenceState.CONFIRMED
    assert model.observable_outcome.evidence_state is StoryFieldEvidenceState.SUPPORTED
    assert model.problem_context.evidence_state is StoryFieldEvidenceState.PLAUSIBLE_MISSING
    assert model.before_state.evidence_state is StoryFieldEvidenceState.UNSUPPORTED
    assert model.problem_context.value is None
    assert not model.problem_context.has_positive_value
    assert not model.before_state.has_positive_value
    assert model.mechanism.has_positive_value
    assert model.capability_fact_ids == ()


def test_claim_sufficiency_and_story_sufficiency_are_independent():
    model = story()
    assert model.claim_sufficiency.level is SufficiencyLevel.HIGH
    assert model.story_sufficiency.level is SufficiencyLevel.LOW


@pytest.mark.parametrize("state", [StoryFieldEvidenceState.CONFIRMED, StoryFieldEvidenceState.SUPPORTED])
def test_claimable_fields_require_value_and_positive_provenance(state):
    with pytest.raises(ValueError, match="positive authority references"):
        EngineeringStoryField(value="Implemented validation.", evidence_state=state)
    with pytest.raises((TypeError, ValueError), match="must not be blank|must be a string"):
        EngineeringStoryField(
            value=" ",
            evidence_state=state,
            evidence_fact_ids=(PEF_A,),
        )
    with pytest.raises(ValueError, match="positive authority references"):
        EngineeringStoryField(
            value="Implemented validation.",
            evidence_state=state,
            claim_boundary_ids=(PCB_A,),
        )


def test_supported_field_accepts_authority_reference_and_defers_source_lineage_resolution():
    field = EngineeringStoryField(
        value="Validation is supported across evidence.",
        evidence_state=StoryFieldEvidenceState.SUPPORTED,
        evidence_fact_ids=(PEF_A,),
    )
    assert field.has_positive_value
    assert field.evidence_fact_ids == (PEF_A,)


@pytest.mark.parametrize(
    "state",
    [StoryFieldEvidenceState.PLAUSIBLE_MISSING, StoryFieldEvidenceState.UNSUPPORTED],
)
def test_nonclaimable_fields_cannot_store_speculative_positive_values(state):
    with pytest.raises(ValueError, match="cannot carry a claimable value"):
        EngineeringStoryField(
            value="Improved throughput by 50%.",
            evidence_state=state,
        )


@pytest.mark.parametrize("placeholder", ["unknown", "TBD", "not sure", "n/a"])
def test_claimable_fields_reject_placeholder_values(placeholder):
    with pytest.raises(ValueError, match="concrete value"):
        confirmed(placeholder)


def test_story_level_provenance_contains_every_field_reference():
    with pytest.raises(ValueError, match="field evidence IDs"):
        story(evidence_fact_ids=(PEF_A,))
    mechanism = confirmed(
        "Validated evidence.",
        evidence_ids=(),
        capability_ids=(PCF_A,),
    )
    model = story(
        mechanism=mechanism,
        capability_fact_ids=(PCF_A,),
    )
    assert model.mechanism.capability_fact_ids == (PCF_A,)
    capability_field = confirmed(
        "Resolved capability provenance.",
        evidence_ids=(),
        capability_ids=(PCF_A,),
    )
    with pytest.raises(ValueError, match="field capability IDs"):
        story(mechanism=capability_field)
    boundary_field = confirmed(
        "Retained a claim boundary.",
        boundary_ids=("pcb_" + "b" * 24,),
    )
    with pytest.raises(ValueError, match="field claim-boundary IDs"):
        story(mechanism=boundary_field)


def test_story_requires_positive_authority_and_a_claimable_field():
    with pytest.raises(ValueError, match="positive authority references"):
        story(
            evidence_fact_ids=(),
            mechanism=EngineeringStoryField(
                value=None,
                evidence_state=StoryFieldEvidenceState.UNSUPPORTED,
            ),
        )
    all_missing = {name.value: unsupported() for name in EngineeringStoryFieldName}
    with pytest.raises(ValueError, match="at least one positive story field"):
        story(
            **all_missing,
            claim_sufficiency=ClaimSufficiency(SufficiencyLevel.LOW),
            story_sufficiency=StorySufficiency(SufficiencyLevel.LOW),
            opportunity=StoryOpportunity(StoryOpportunityLevel.NONE),
        )


def test_lifecycle_states_and_revalidation_contract():
    assert EngineeringStoryLifecycle(EngineeringStoryStatus.ACTIVE).status is EngineeringStoryStatus.ACTIVE
    assert EngineeringStoryLifecycle(
        EngineeringStoryStatus.ACTIVE,
        requires_revalidation=True,
    ).requires_revalidation
    assert EngineeringStoryLifecycle(
        EngineeringStoryStatus.SUPERSEDED,
        superseded_by_story_id=SUCCESSOR_ID,
    ).superseded_by_story_id == SUCCESSOR_ID
    assert EngineeringStoryLifecycle(
        EngineeringStoryStatus.STALE,
        requires_revalidation=True,
    ).status is EngineeringStoryStatus.STALE
    assert EngineeringStoryLifecycle(
        EngineeringStoryStatus.CONFLICTED,
        requires_revalidation=True,
    ).status is EngineeringStoryStatus.CONFLICTED


def test_lifecycle_fails_closed_on_inconsistent_state():
    with pytest.raises(ValueError, match="successor"):
        EngineeringStoryLifecycle(EngineeringStoryStatus.SUPERSEDED)
    with pytest.raises(ValueError, match="only superseded"):
        EngineeringStoryLifecycle(
            EngineeringStoryStatus.ACTIVE,
            superseded_by_story_id=SUCCESSOR_ID,
        )
    for status in (EngineeringStoryStatus.STALE, EngineeringStoryStatus.CONFLICTED):
        with pytest.raises(ValueError, match="require revalidation"):
            EngineeringStoryLifecycle(status)
    with pytest.raises(ValueError, match="cannot supersede itself"):
        story(
            lifecycle=EngineeringStoryLifecycle(
                EngineeringStoryStatus.SUPERSEDED,
                superseded_by_story_id=STORY_ID,
            )
        )
    with pytest.raises(TypeError, match="boolean"):
        EngineeringStoryLifecycle(EngineeringStoryStatus.ACTIVE, requires_revalidation=1)


def test_sufficiency_contract_is_typed_and_consistent_with_story_fields():
    with pytest.raises(ValueError, match="both supported and missing"):
        ClaimSufficiency(
            SufficiencyLevel.LOW,
            supported_fields=(EngineeringStoryFieldName.MECHANISM,),
            missing_fields=(EngineeringStoryFieldName.MECHANISM,),
        )
    with pytest.raises(ValueError, match="unassessed"):
        StorySufficiency(
            SufficiencyLevel.UNASSESSED,
            missing_fields=(EngineeringStoryFieldName.TRIGGER,),
        )
    with pytest.raises(ValueError, match="at least one supported"):
        ClaimSufficiency(SufficiencyLevel.HIGH)
    with pytest.raises(ValueError, match="positive evidence state"):
        story(
            claim_sufficiency=ClaimSufficiency(
                SufficiencyLevel.MEDIUM,
                supported_fields=(EngineeringStoryFieldName.PROBLEM_CONTEXT,),
            )
        )
    with pytest.raises(ValueError, match="must not have positive evidence state"):
        story(
            story_sufficiency=StorySufficiency(
                SufficiencyLevel.LOW,
                missing_fields=(EngineeringStoryFieldName.MECHANISM,),
            )
        )


def test_story_opportunity_is_bounded_typed_and_jd_independent():
    assert StoryOpportunity(StoryOpportunityLevel.NONE).to_dict() == {
        "level": "none",
        "signals": [],
        "missing_context": [],
    }
    with pytest.raises(ValueError, match="cannot carry"):
        StoryOpportunity(
            StoryOpportunityLevel.NONE,
            missing_context=(StoryContextGap.TRIGGER,),
        )
    with pytest.raises(ValueError, match="requires a signal"):
        StoryOpportunity(StoryOpportunityLevel.HIGH)
    partial_context = story(
        opportunity=StoryOpportunity(
            StoryOpportunityLevel.HIGH,
            missing_context=(StoryContextGap.DECISION_REASON,),
        )
    )
    assert partial_context.decision.has_positive_value
    assert partial_context.opportunity.missing_context == (StoryContextGap.DECISION_REASON,)
    assert "jd" not in json.dumps(story().opportunity.to_dict()).casefold()


def test_unknown_enums_fail_closed():
    with pytest.raises(ValueError):
        EngineeringStoryField(value=None, evidence_state="maybe")
    with pytest.raises(ValueError):
        EngineeringStoryLifecycle(status="deleted")
    with pytest.raises(ValueError):
        ClaimSufficiency(level="complete")
    with pytest.raises(ValueError):
        StoryOpportunity(level="urgent", signals=())
    with pytest.raises(ValueError):
        story(story_type="resume_story")


def test_project_and_authority_identifiers_are_strict():
    with pytest.raises(ValueError, match="canonical project"):
        story(project_id=" workagent ")
    with pytest.raises(ValueError, match="engineering_story_"):
        story(story_id="story-a")
    with pytest.raises(ValueError, match="pef_"):
        confirmed("Implemented validation.", evidence_ids=(PCF_A,))
    with pytest.raises(ValueError, match="normalized"):
        confirmed("Implemented validation.", evidence_ids=("pef_../../secret",))
    with pytest.raises(ValueError, match="normalized"):
        confirmed("Implemented validation.", evidence_ids=("pef_a\nquery=x",))
    for bare in ("pef_", "pcf_", "pcb_", "engineering_story_"):
        with pytest.raises(ValueError, match="normalized"):
            if bare == "pef_":
                confirmed("Implemented validation.", evidence_ids=(bare,))
            elif bare == "pcf_":
                confirmed(
                    "Implemented validation.",
                    evidence_ids=(),
                    capability_ids=(bare,),
                )
            elif bare == "pcb_":
                confirmed("Implemented validation.", boundary_ids=(bare,))
            else:
                validate_engineering_story_id(bare)
    assert validate_engineering_story_id(STORY_ID) == STORY_ID


def test_collections_are_copied_deduplicated_and_deterministic():
    evidence_ids = [PEF_B, PEF_A, PEF_B]
    field = EngineeringStoryField(
        value="Implemented validation.",
        evidence_state=StoryFieldEvidenceState.CONFIRMED,
        evidence_fact_ids=evidence_ids,
    )
    evidence_ids.append(PEF_C)
    assert field.evidence_fact_ids == (PEF_A, PEF_B)
    first = story(evidence_fact_ids=(PEF_B, PEF_A, PEF_B))
    second = story(evidence_fact_ids=(PEF_A, PEF_B))
    assert first.to_json() == second.to_json()


def test_bounds_reject_instead_of_truncating():
    too_many_field_ids = tuple(
        f"pef_{index:024x}" for index in range(MAX_STORY_FIELD_PROVENANCE_IDS + 1)
    )
    with pytest.raises(ValueError, match="maximum item count"):
        confirmed("Implemented validation.", evidence_ids=too_many_field_ids)
    too_many_story_ids = tuple(
        f"pef_{index:024x}" for index in range(MAX_STORY_PROVENANCE_IDS + 1)
    )
    with pytest.raises(ValueError, match="maximum item count"):
        story(evidence_fact_ids=too_many_story_ids)
    with pytest.raises(ValueError, match="maximum length"):
        confirmed("x" * (MAX_STORY_FIELD_VALUE_LENGTH + 1))


def test_round_trip_restores_nested_types_and_semantic_equality():
    model = story()
    serialized = model.to_json()
    restored = EngineeringStory.from_dict(json.loads(serialized))
    assert restored == model
    assert restored.to_json() == serialized
    assert isinstance(restored.evidence_fact_ids, tuple)
    assert isinstance(restored.mechanism, EngineeringStoryField)
    assert restored.story_type is EngineeringStoryType.VALIDATION_AND_QUALITY


def test_unknown_top_level_and_nested_fields_are_rejected():
    payload = story().to_dict()
    with pytest.raises(ValueError, match="unknown EngineeringStory fields"):
        EngineeringStory.from_dict({**payload, "raw_patch": "private"})
    payload["mechanism"] = {**payload["mechanism"], "source_document": "private"}
    with pytest.raises(ValueError, match="unknown EngineeringStoryField fields"):
        EngineeringStory.from_dict(payload)
    with pytest.raises(ValueError, match="unknown EngineeringStoryLifecycle fields"):
        EngineeringStoryLifecycle.from_dict({"status": "active", "extra": True})
    with pytest.raises(ValueError, match="unknown ClaimSufficiency fields"):
        ClaimSufficiency.from_dict({"level": "low", "unexpected": []})
    with pytest.raises(ValueError, match="unknown StoryOpportunity fields"):
        StoryOpportunity.from_dict({"level": "none", "jd_relevance": "high"})
    with pytest.raises(TypeError, match="field names must be strings"):
        EngineeringStoryField.from_dict({
            "value": None,
            "evidence_state": "unsupported",
            1: "invalid-key",
        })


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "diff --git a/private.py b/private.py + changed source",
        "@@ -1,2 +1,3 @@ changed source",
        "-----BEGIN PRIVATE KEY----- private material",
        "access_token=topsecret",
        "Secret: 'topsecretvalue'",
        "password: topsecretvalue",
        "secret: abcdefghijklmnop",
        "credential: tokenvalue12345",
        "Authorization: Bearer abcdefghijklmnop",
    ],
)
def test_story_fields_reject_raw_or_sensitive_source_content(unsafe_value):
    with pytest.raises(ValueError, match="raw or sensitive"):
        confirmed(unsafe_value)


@pytest.mark.parametrize(
    "safe_value",
    [
        "Authorization: enforced role-based access control at the API boundary.",
        "Rotated the secret: stored credentials in a managed vault.",
        "Password: validation rejects weak credentials before persistence.",
        "Secret: configuration moved into a managed vault.",
        "Password: authentication validation rejects weak input.",
        "Credential: authorization boundaries are enforced.",
    ],
)
def test_story_fields_allow_safe_security_engineering_prose(safe_value):
    assert confirmed(safe_value).value == safe_value


def test_models_are_frozen_slotted_and_do_not_share_mutable_state():
    model = story()
    assert not hasattr(model, "__dict__")
    assert not hasattr(model.mechanism, "__dict__")
    with pytest.raises(FrozenInstanceError):
        model.project_id = "other"
    with pytest.raises(FrozenInstanceError):
        model.mechanism.value = "changed"


def test_story_identity_is_not_derived_from_mutable_prose():
    changed = story(
        implementation=confirmed("Added a different supported implementation detail."),
    )
    assert changed.story_id == STORY_ID
    assert changed.implementation.value != story().implementation.value


def test_enum_taxonomies_are_exact_and_conservative():
    assert {item.value for item in StoryFieldEvidenceState} == {
        "confirmed",
        "supported",
        "plausible_missing",
        "unsupported",
    }
    assert {item.value for item in EngineeringStoryStatus} == {
        "active",
        "superseded",
        "stale",
        "conflicted",
    }
    assert {item.value for item in EngineeringStoryType} == {
        "architecture_change",
        "reliability_hardening",
        "debugging_and_repair",
        "retrieval_redesign",
        "validation_and_quality",
        "data_or_memory_system",
        "workflow_automation",
        "performance_or_efficiency",
        "integration",
        "other",
    }
    assert {item.value for item in SufficiencyLevel} == {
        "unassessed",
        "low",
        "medium",
        "high",
    }
    assert {item.value for item in StoryOpportunityLevel} == {
        "none",
        "low",
        "medium",
        "high",
    }
    assert {item.value for item in EngineeringStoryFieldName} == {
        "problem_context",
        "trigger",
        "before_state",
        "decision",
        "mechanism",
        "implementation",
        "tradeoff",
        "validation",
        "after_state",
        "observable_outcome",
        "ownership",
        "stakeholder_context",
    }
    assert {item.value for item in StoryContextGap} == {
        "problem_context",
        "trigger",
        "decision_reason",
        "tradeoff",
        "ownership",
        "stakeholder_context",
        "observable_outcome_context",
    }
    assert StoryContextGap.DECISION_REASON.value == "decision_reason"


def test_contract_has_no_runtime_integration_or_unsafe_content_fields():
    module_path = Path(__file__).resolve().parents[1] / "backend" / "engineering_story_models.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    forbidden_imports = {
        "backend.api_server",
        "backend.evidence_hybrid_retrieval",
        "backend.memory_store",
        "backend.project_query_planner",
        "backend.project_retrieval_v2",
        "chromadb",
    }
    assert imports.isdisjoint(forbidden_imports)
    serialized = story().to_json().casefold()
    for forbidden_key in (
        "raw_patch",
        "raw_text",
        "source_document",
        "embedding",
        "vector_score",
        "query_group",
        "jd_relevance",
    ):
        assert forbidden_key not in serialized


def test_production_contract_uses_semantic_naming_only():
    module_path = Path(__file__).resolve().parents[1] / "backend" / "engineering_story_models.py"
    source = module_path.read_text(encoding="utf-8").casefold()
    roadmap_tokens = {
        "phase" + "6_75",
        "phase_" + "6_75",
        "phase" + "675",
        "phase" + "6.75",
    }
    assert not roadmap_tokens.intersection(source)
    assert module_path.name == "engineering_story_models.py"
