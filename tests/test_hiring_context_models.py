from __future__ import annotations

import ast
import builtins
from dataclasses import FrozenInstanceError, fields
import hashlib
import json
from pathlib import Path

import pytest

from backend.engineering_story_models import (
    EngineeringStoryField,
    StoryFieldEvidenceState,
)
from backend.hiring_context_models import (
    MAX_HIRING_CONTEXT_HIGH_VALUE_TRAITS,
    MAX_HIRING_CONTEXT_RANKING_EFFECTS,
    MAX_HIRING_CONTEXT_SECONDARY_ROLE_FAMILIES,
    MAX_HIRING_CONTEXT_SIGNALS,
    MAX_HIRING_CONTEXT_SIGNAL_SOURCE_REFS,
    MAX_HIRING_CONTEXT_SOURCE_REFS,
    HiringContextConfidence,
    HiringContextProfile,
    HiringContextSignal,
    HiringContextSignalKind,
    HiringContextSourceKind,
    HiringContextSourceRef,
    RankingEffect,
    RoleFamily,
)
from backend.project_evidence_models import (
    Confidence as CandidateEvidenceConfidence,
    EvidenceSourceRef,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source(
    kind: HiringContextSourceKind = HiringContextSourceKind.JOB_DESCRIPTION,
    *,
    seed: str = "job-description",
) -> HiringContextSourceRef:
    return HiringContextSourceRef(
        source_kind=kind,
        source_fingerprint=digest(seed),
    )


def signal(
    value: str = "REST API development",
    *,
    kind: HiringContextSignalKind = HiringContextSignalKind.EXPLICIT_JD,
    confidence: HiringContextConfidence = HiringContextConfidence.HIGH,
    effects: tuple[RankingEffect, ...] = (RankingEffect.EXPLICIT_ALIGNMENT,),
    source_refs: tuple[HiringContextSourceRef, ...] | None = None,
) -> HiringContextSignal:
    refs = source_refs if source_refs is not None else (source(),)
    return HiringContextSignal(
        value=value,
        kind=kind,
        confidence=confidence,
        ranking_effects=effects,
        source_refs=refs,
    )


def profile(**overrides) -> HiringContextProfile:
    values = {
        "source_refs": (source(),),
        "company": "Example Systems",
        "role_title": "Backend Engineer",
        "primary_role_family": RoleFamily.BACKEND_ENGINEERING,
        "confidence": HiringContextConfidence.HIGH,
    }
    values.update(overrides)
    return HiringContextProfile(**values)


def full_profile() -> HiringContextProfile:
    jd = source(seed="coalition-jd")
    company = source(
        HiringContextSourceKind.COMPANY_IDENTITY,
        seed="the-coalition",
    )
    team = source(
        HiringContextSourceKind.TEAM_IDENTITY,
        seed="online-systems-team",
    )
    parent = source(
        HiringContextSourceKind.PARENT_ORGANIZATION_IDENTITY,
        seed="xbox-game-studios",
    )
    taxonomy = source(
        HiringContextSourceKind.INTERNAL_TAXONOMY,
        seed="role-taxonomy-v1",
    )
    return HiringContextProfile(
        source_refs=(taxonomy, parent, team, company, jd),
        company="The Coalition",
        team="Online Systems",
        parent_organization="Xbox Game Studios",
        role_title="Senior Backend Engineer",
        primary_role_family=RoleFamily.BACKEND_ENGINEERING,
        secondary_role_families=(
            RoleFamily.GAME_DEVELOPMENT,
            RoleFamily.SOFTWARE_ENGINEERING,
        ),
        signals=(
            signal(source_refs=(jd,)),
            signal(
                "Game development",
                kind=HiringContextSignalKind.COMPANY_DOMAIN,
                effects=(RankingEffect.DOMAIN_ALIGNMENT,),
                source_refs=(company,),
            ),
            signal(
                "Reliability",
                kind=HiringContextSignalKind.ENGINEERING_TRAIT,
                effects=(RankingEffect.TRANSFERABLE_ENGINEERING_ALIGNMENT,),
                source_refs=(jd, taxonomy),
            ),
        ),
        high_value_traits=("Reliability", "Testing", "Performance"),
        confidence=HiringContextConfidence.HIGH,
    )


def candidate_source() -> EvidenceSourceRef:
    return EvidenceSourceRef(
        source_type="file_snapshot",
        source_id="raw_" + "a" * 32,
        project_id="workagent",
        content_hash="a" * 64,
        repo="example/workagent",
        file_path="backend/api_server.py",
    )


def test_minimal_valid_hiring_context_profile():
    model = HiringContextProfile(
        source_refs=(source(),),
        role_title="Software Engineer",
    )
    assert model.role_title == "Software Engineer"
    assert model.primary_role_family is RoleFamily.UNKNOWN
    assert model.profile_id.startswith("hiring_context_")
    assert len(model.fingerprint) == 64


def test_full_valid_offline_profile():
    model = full_profile()
    assert model.company == "The Coalition"
    assert model.team == "Online Systems"
    assert model.parent_organization == "Xbox Game Studios"
    assert len(model.signals) == 3
    assert model.high_value_traits == ("Performance", "Reliability", "Testing")


def test_models_are_frozen():
    model = full_profile()
    with pytest.raises(FrozenInstanceError):
        model.company = "Changed"
    with pytest.raises(FrozenInstanceError):
        model.signals[0].value = "Changed"
    with pytest.raises(FrozenInstanceError):
        model.source_refs[0].source_fingerprint = "0" * 64


def test_models_use_slots():
    model = full_profile()
    assert not hasattr(model, "__dict__")
    assert not hasattr(model.signals[0], "__dict__")
    assert not hasattr(model.source_refs[0], "__dict__")


def test_company_normalization_collapses_whitespace():
    assert profile(company="  Example   Systems\nInc. ").company == "Example Systems Inc."


def test_role_title_normalization_collapses_whitespace():
    assert profile(role_title="  Senior   Backend\tEngineer ").role_title == "Senior Backend Engineer"


def test_empty_team_is_allowed_and_removed():
    assert profile(team=" \n\t ").team is None


def test_supported_parent_organization_is_preserved():
    assert profile(parent_organization="  Parent   Group ").parent_organization == "Parent Group"


def test_primary_role_family_can_be_explicit_or_unknown():
    assert profile().primary_role_family is RoleFamily.BACKEND_ENGINEERING
    assert profile(primary_role_family="unknown").primary_role_family is RoleFamily.UNKNOWN


def test_secondary_role_families_are_bounded():
    values = tuple(list(RoleFamily)[: MAX_HIRING_CONTEXT_SECONDARY_ROLE_FAMILIES + 1])
    with pytest.raises(ValueError, match="secondary_role_families exceeds"):
        profile(secondary_role_families=values)


def test_role_family_deduplication_removes_primary_and_orders_by_taxonomy():
    model = profile(
        secondary_role_families=(
            RoleFamily.SECURITY,
            RoleFamily.SOFTWARE_ENGINEERING,
            RoleFamily.SECURITY,
            RoleFamily.BACKEND_ENGINEERING,
        )
    )
    assert model.secondary_role_families == (
        RoleFamily.SOFTWARE_ENGINEERING,
        RoleFamily.SECURITY,
    )


def test_normalized_explicit_jd_signal():
    model = signal("  REST   API\ndevelopment ")
    assert model.value == "REST API development"
    assert model.kind is HiringContextSignalKind.EXPLICIT_JD
    assert model.ranking_effects == (RankingEffect.EXPLICIT_ALIGNMENT,)


def test_normalized_company_domain_signal():
    model = signal(
        " Game   development ",
        kind=HiringContextSignalKind.COMPANY_DOMAIN,
        effects=(RankingEffect.DOMAIN_ALIGNMENT,),
    )
    assert model.value == "Game development"
    assert model.kind is HiringContextSignalKind.COMPANY_DOMAIN


def test_normalized_engineering_trait_signal():
    model = signal(
        " Reliability ",
        kind=HiringContextSignalKind.ENGINEERING_TRAIT,
        effects=(RankingEffect.TRANSFERABLE_ENGINEERING_ALIGNMENT,),
    )
    assert model.value == "Reliability"
    assert model.ranking_effects == (RankingEffect.TRANSFERABLE_ENGINEERING_ALIGNMENT,)


def test_exact_normalized_signals_are_deduplicated():
    jd = source()
    first = signal("REST  API development", source_refs=(jd,))
    second = signal(" REST API\ndevelopment ", source_refs=(jd,))
    model = profile(source_refs=(jd,), signals=(first, second))
    assert len(model.signals) == 1


def test_signals_are_bounded():
    jd = source()
    values = tuple(
        signal(f"Requirement {index}", source_refs=(jd,))
        for index in range(MAX_HIRING_CONTEXT_SIGNALS + 1)
    )
    with pytest.raises(ValueError, match="signals exceeds"):
        profile(source_refs=(jd,), signals=values)


def test_source_provenance_is_required_for_profiles_and_signals():
    with pytest.raises(ValueError, match="at least one hiring-context source"):
        HiringContextProfile(source_refs=(), role_title="Engineer")
    with pytest.raises(ValueError, match="at least one hiring-context source"):
        signal(source_refs=())


def test_hiring_context_source_ref_is_distinct_from_candidate_evidence_source_ref():
    context_ref = source()
    evidence_ref = candidate_source()
    assert type(context_ref) is HiringContextSourceRef
    assert not isinstance(context_ref, EvidenceSourceRef)
    assert not isinstance(evidence_ref, HiringContextSourceRef)
    with pytest.raises(ValueError, match="unknown HiringContextSourceRef fields"):
        HiringContextSourceRef.from_dict(evidence_ref.to_dict())


@pytest.mark.parametrize(
    "forbidden_field",
    (
        "source_id",
        "project_id",
        "evidence_fact_id",
        "capability_fact_id",
        "story_id",
        "repository",
        "file_path",
        "label",
        "metadata",
    ),
)
def test_hiring_context_source_ref_cannot_carry_candidate_or_story_ids(forbidden_field):
    payload = source().to_dict()
    payload[forbidden_field] = "pef_" + "a" * 24
    with pytest.raises(ValueError, match="unknown HiringContextSourceRef fields"):
        HiringContextSourceRef.from_dict(payload)


def test_profile_cannot_carry_candidate_evidence_refs():
    with pytest.raises(TypeError, match="HiringContextSourceRef"):
        HiringContextProfile(
            source_refs=(candidate_source(),),
            role_title="Engineer",
        )


def test_profile_cannot_carry_capability_fact_ids():
    payload = profile().to_dict()
    payload["capability_fact_ids"] = ["pcf_" + "a" * 24]
    with pytest.raises(ValueError, match="unknown HiringContextProfile fields"):
        HiringContextProfile.from_dict(payload)


def test_profile_cannot_carry_engineering_story_evidence_ids():
    payload = profile().to_dict()
    payload["story_evidence_ids"] = ["pef_" + "a" * 24]
    with pytest.raises(ValueError, match="unknown HiringContextProfile fields"):
        HiringContextProfile.from_dict(payload)


def test_profile_cannot_contain_candidate_identity_fields():
    for field_name in (
        "candidate_identity",
        "candidate_specialization",
        "candidate_domain_expertise",
        "candidate_professional_title",
        "candidate_persona",
    ):
        payload = profile().to_dict()
        payload[field_name] = "Technology Risk specialist"
        with pytest.raises(ValueError, match="unknown HiringContextProfile fields"):
            HiringContextProfile.from_dict(payload)


def test_role_domain_signal_does_not_become_candidate_domain_expertise():
    jd = source()
    domain = signal(
        "Technology risk",
        kind=HiringContextSignalKind.COMPANY_DOMAIN,
        effects=(RankingEffect.DOMAIN_ALIGNMENT,),
        source_refs=(jd,),
    )
    serialized = profile(source_refs=(jd,), signals=(domain,)).to_dict()
    assert serialized["signals"][0]["value"] == "Technology risk"
    assert "candidate_domain_expertise" not in serialized
    assert "candidate_identity" not in serialized


def test_technology_risk_jd_does_not_create_candidate_specialist_identity():
    jd = source(seed="risk-jd")
    model = HiringContextProfile(
        source_refs=(jd,),
        role_title="Technology Risk Consultant",
        primary_role_family=RoleFamily.CONSULTING_STRATEGY,
        signals=(
            signal(
                "Privacy and data controls",
                kind=HiringContextSignalKind.EXPLICIT_JD,
                effects=(RankingEffect.EXPLICIT_ALIGNMENT,),
                source_refs=(jd,),
            ),
        ),
    )
    serialized = model.to_json()
    assert "Privacy and data controls" in serialized
    for forbidden in ("candidate_specialization", "candidate_persona", "candidate_domain"):
        assert forbidden not in serialized


def test_confidence_is_a_context_only_enum():
    assert HiringContextConfidence is not CandidateEvidenceConfidence
    assert type(HiringContextConfidence.HIGH) is not type(CandidateEvidenceConfidence.HIGH)
    with pytest.raises(TypeError, match="HiringContextConfidence"):
        profile(confidence=CandidateEvidenceConfidence.HIGH)
    with pytest.raises(TypeError, match="HiringContextConfidence"):
        signal(confidence=CandidateEvidenceConfidence.HIGH)


@pytest.mark.parametrize(
    "forbidden_effect",
    (
        "evidence_strength",
        "claim_sufficiency",
        "story_sufficiency",
        "technology_support",
    ),
)
def test_ranking_effect_rejects_candidate_truth_dimensions(forbidden_effect):
    with pytest.raises(ValueError, match="ranking_effects must be one of"):
        signal(effects=(forbidden_effect,))


def test_set_like_collections_have_deterministic_ordering():
    jd = source(seed="jd")
    company = source(
        HiringContextSourceKind.COMPANY_IDENTITY,
        seed="company",
    )
    explicit = signal("REST API", source_refs=(jd,))
    domain = signal(
        "Game development",
        kind=HiringContextSignalKind.COMPANY_DOMAIN,
        effects=(RankingEffect.DOMAIN_ALIGNMENT,),
        source_refs=(company,),
    )
    first = profile(
        source_refs=(jd, company),
        signals=(explicit, domain),
        secondary_role_families=(RoleFamily.SECURITY, RoleFamily.SOFTWARE_ENGINEERING),
        high_value_traits=("Testing", "Reliability"),
    )
    second = profile(
        source_refs=(company, jd),
        signals=(domain, explicit),
        secondary_role_families=(RoleFamily.SOFTWARE_ENGINEERING, RoleFamily.SECURITY),
        high_value_traits=("Reliability", "Testing"),
    )
    assert first.to_dict() == second.to_dict()


def test_normalization_is_deterministic():
    first = profile(company="Example   Systems", role_title="Backend\nEngineer")
    second = profile(company=" Example Systems ", role_title=" Backend Engineer ")
    assert first == second


def test_fingerprint_is_deterministic():
    first = full_profile()
    second = full_profile()
    assert first.fingerprint == second.fingerprint
    assert first.profile_id == second.profile_id


def test_equivalent_normalized_input_has_the_same_fingerprint():
    first = profile(company="Example   Systems", role_title="Backend\nEngineer")
    second = profile(company=" Example Systems ", role_title="Backend Engineer")
    assert first.fingerprint == second.fingerprint


def test_material_company_change_changes_fingerprint():
    assert profile(company="Example Systems").fingerprint != profile(company="Other Systems").fingerprint


def test_material_role_family_change_changes_fingerprint():
    assert profile(primary_role_family=RoleFamily.BACKEND_ENGINEERING).fingerprint != profile(
        primary_role_family=RoleFamily.SECURITY
    ).fingerprint


def test_empty_optional_collections_normalize_safely():
    model = profile(
        secondary_role_families=[],
        signals=[],
        high_value_traits=[],
    )
    assert model.secondary_role_families == ()
    assert model.signals == ()
    assert model.high_value_traits == ()


@pytest.mark.parametrize(
    ("constructor", "match"),
    (
        (lambda: profile(primary_role_family="wizard"), "primary_role_family must be one of"),
        (lambda: signal(kind="candidate_fact"), "kind must be one of"),
        (lambda: signal(confidence="certain"), "confidence must be one of"),
    ),
)
def test_malformed_enums_fail_closed(constructor, match):
    with pytest.raises(ValueError, match=match):
        constructor()


def test_malformed_source_kind_fails_closed():
    with pytest.raises(ValueError, match="source_kind must be one of"):
        HiringContextSourceRef(
            source_kind="github_repository",
            source_fingerprint=digest("source"),
        )


@pytest.mark.parametrize("collection", ("source_refs", "signal_source_refs", "traits", "effects"))
def test_oversized_collections_fail_closed_deterministically(collection):
    refs = tuple(
        source(seed=f"source-{index}")
        for index in range(MAX_HIRING_CONTEXT_SOURCE_REFS + 1)
    )
    if collection == "source_refs":
        with pytest.raises(ValueError, match="source_refs exceeds"):
            profile(source_refs=refs)
    elif collection == "signal_source_refs":
        with pytest.raises(ValueError, match="source_refs exceeds"):
            signal(source_refs=refs[: MAX_HIRING_CONTEXT_SIGNAL_SOURCE_REFS + 1])
    elif collection == "traits":
        traits = tuple(
            f"Trait {index}"
            for index in range(MAX_HIRING_CONTEXT_HIGH_VALUE_TRAITS + 1)
        )
        with pytest.raises(ValueError, match="high_value_traits exceeds"):
            profile(high_value_traits=traits)
    else:
        effects = tuple(list(RankingEffect)[: MAX_HIRING_CONTEXT_RANKING_EFFECTS + 1])
        with pytest.raises(ValueError, match="ranking_effects exceeds"):
            signal(effects=effects)


@pytest.mark.parametrize(
    "forbidden_field",
    (
        "project_id",
        "repository",
        "repo",
        "github_chunk_id",
        "github_evidence_id",
        "raw_github_evidence",
    ),
)
def test_profile_has_no_raw_github_or_candidate_evidence_field(forbidden_field):
    payload = profile().to_dict()
    payload[forbidden_field] = "unsafe"
    with pytest.raises(ValueError, match="unknown HiringContextProfile fields"):
        HiringContextProfile.from_dict(payload)


def test_raw_patch_field_and_content_are_rejected():
    payload = profile().to_dict()
    payload["raw_patch"] = "diff --git a/file b/file"
    with pytest.raises(ValueError, match="unknown HiringContextProfile fields"):
        HiringContextProfile.from_dict(payload)
    with pytest.raises(ValueError, match="raw or sensitive"):
        signal("diff --git a/file b/file")


@pytest.mark.parametrize("forbidden_field", ("chroma_id", "collection_name", "file_path", "storage_path"))
def test_profile_has_no_chroma_or_path_field(forbidden_field):
    payload = profile().to_dict()
    payload[forbidden_field] = "unsafe"
    with pytest.raises(ValueError, match="unknown HiringContextProfile fields"):
        HiringContextProfile.from_dict(payload)


def test_fingerprint_contains_no_runtime_timestamp():
    model = profile()
    names = {item.name for item in fields(HiringContextProfile)}
    assert not names.intersection({"created_at", "updated_at", "timestamp", "generated_at"})
    assert profile().fingerprint == model.fingerprint


def test_serialization_contains_only_hiring_context_domain_data():
    payload = full_profile().to_dict()
    assert set(payload) == {
        "source_refs",
        "confidence",
        "company",
        "team",
        "parent_organization",
        "role_title",
        "primary_role_family",
        "secondary_role_families",
        "signals",
        "high_value_traits",
        "profile_id",
        "fingerprint",
    }
    serialized = json.dumps(payload, sort_keys=True).casefold()
    for forbidden in (
        "project_id",
        "evidence_fact_id",
        "capability_fact_id",
        "claim_boundary_id",
        "story_evidence",
        "claim_sufficiency",
        "story_sufficiency",
        "story_opportunity",
        "candidate_identity",
        "raw_patch",
        "embedding",
    ):
        assert forbidden not in serialized


def test_model_construction_has_no_io_side_effects(monkeypatch):
    def blocked_open(*_args, **_kwargs):
        raise AssertionError("model construction attempted file I/O")

    monkeypatch.setattr(builtins, "open", blocked_open)
    assert profile().profile_id.startswith("hiring_context_")


def test_production_module_has_no_runtime_or_candidate_authority_imports():
    module_path = Path(__file__).resolve().parents[1] / "backend" / "hiring_context_models.py"
    source_text = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
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
    forbidden = {
        "backend.api_server",
        "backend.chroma_http_client_factory",
        "backend.engineering_story_reconstruction",
        "backend.evidence_hybrid_retrieval",
        "backend.evidence_vector_search",
        "backend.github_raw_storage",
        "backend.memory_store",
        "backend.project_capability_backfill",
        "backend.project_evidence_models",
        "backend.project_query_planner",
        "chromadb",
        "httpx",
        "requests",
    }
    assert imports.isdisjoint(forbidden)


def test_strict_serialization_round_trip_preserves_types_and_fingerprint():
    original = full_profile()
    restored = HiringContextProfile.from_dict(original.to_dict())
    assert restored == original
    assert isinstance(restored.signals[0], HiringContextSignal)
    assert isinstance(restored.source_refs[0], HiringContextSourceRef)
    assert restored.to_json() == original.to_json()


def test_source_refs_are_deduplicated_and_sorted():
    first = source(seed="first")
    second = source(seed="second")
    model = profile(source_refs=(second, first, second))
    assert model.source_refs == tuple(sorted((first, second), key=lambda item: item.reference_id))


def test_profile_rejects_signal_source_not_in_profile_provenance():
    jd = source(seed="jd")
    company = source(HiringContextSourceKind.COMPANY_IDENTITY, seed="company")
    domain = signal(
        "Game development",
        kind=HiringContextSignalKind.COMPANY_DOMAIN,
        effects=(RankingEffect.DOMAIN_ALIGNMENT,),
        source_refs=(company,),
    )
    with pytest.raises(ValueError, match="signal source_refs"):
        profile(source_refs=(jd,), signals=(domain,))


def test_input_collections_are_copied_to_immutable_tuples():
    jd = source()
    sources = [jd]
    signals = [signal(source_refs=(jd,))]
    traits = ["Reliability"]
    model = profile(source_refs=sources, signals=signals, high_value_traits=traits)
    sources.append(source(seed="later"))
    signals.clear()
    traits.append("Testing")
    assert model.source_refs == (jd,)
    assert len(model.signals) == 1
    assert model.high_value_traits == ("Reliability",)


def test_context_source_and_signal_ids_are_deterministic():
    assert source().reference_id == source().reference_id
    assert signal().signal_id == signal().signal_id


def test_material_signal_change_changes_profile_fingerprint():
    jd = source()
    first = profile(source_refs=(jd,), signals=(signal("REST API", source_refs=(jd,)),))
    second = profile(source_refs=(jd,), signals=(signal("Data pipelines", source_refs=(jd,)),))
    assert first.fingerprint != second.fingerprint


def test_supplied_ids_and_fingerprint_must_match_normalized_content():
    with pytest.raises(ValueError, match="reference_id"):
        HiringContextSourceRef(
            source_kind=HiringContextSourceKind.JOB_DESCRIPTION,
            source_fingerprint=digest("jd"),
            reference_id="hiring_context_source_" + "0" * 24,
        )
    jd = source()
    with pytest.raises(ValueError, match="signal_id"):
        HiringContextSignal(
            value="REST API",
            kind=HiringContextSignalKind.EXPLICIT_JD,
            confidence=HiringContextConfidence.HIGH,
            ranking_effects=(RankingEffect.EXPLICIT_ALIGNMENT,),
            source_refs=(jd,),
            signal_id="hiring_context_signal_" + "0" * 24,
        )
    with pytest.raises(ValueError, match="fingerprint"):
        profile(fingerprint="0" * 64)


def test_raw_job_description_is_not_a_profile_field():
    payload = profile().to_dict()
    payload["raw_job_description"] = "Complete raw posting"
    with pytest.raises(ValueError, match="unknown HiringContextProfile fields"):
        HiringContextProfile.from_dict(payload)


def test_model_layer_contains_no_ranking_scores_or_selection_logic():
    module_path = Path(__file__).resolve().parents[1] / "backend" / "hiring_context_models.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    function_names = {
        node.name.casefold()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not any("score" in name or "select_project" in name or "rank_story" in name for name in function_names)


def test_role_family_vocabulary_is_bounded_and_semantic():
    assert {item.value for item in RoleFamily} == {
        "software_engineering",
        "backend_engineering",
        "frontend_engineering",
        "full_stack_engineering",
        "data_engineering",
        "data_analytics",
        "machine_learning_ai",
        "devops_cloud",
        "game_development",
        "mobile",
        "embedded_systems",
        "security",
        "consulting_strategy",
        "general",
        "unknown",
    }


def test_hiring_context_ref_cannot_be_used_as_story_evidence_provenance():
    with pytest.raises(TypeError, match="must be a string"):
        EngineeringStoryField(
            value="Implemented validation.",
            evidence_state=StoryFieldEvidenceState.SUPPORTED,
            evidence_fact_ids=(source(),),
        )


def test_story_truth_state_fields_are_not_present_in_profile_contract():
    names = {item.name for item in fields(HiringContextProfile)}
    assert names.isdisjoint({
        "evidence_state",
        "claim_sufficiency",
        "story_sufficiency",
        "story_opportunity",
        "lifecycle",
        "story_provenance",
    })
