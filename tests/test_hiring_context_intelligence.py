from __future__ import annotations

import ast
import builtins
from dataclasses import fields
import inspect
import os
from pathlib import Path
import socket
import subprocess

import pytest

from backend.hiring_context_intelligence import (
    MAX_NORMALIZED_JOB_CONTEXT_VALUES,
    MAX_NORMALIZED_JOB_VALUES_PER_FIELD,
    build_hiring_context_profile,
    classify_hiring_context_role_families,
)
from backend.hiring_context_models import (
    MAX_HIRING_CONTEXT_SIGNALS,
    HiringContextConfidence,
    HiringContextProfile,
    HiringContextSignalKind,
    HiringContextSourceKind,
    HiringContextSourceRef,
    RankingEffect,
    RoleFamily,
)


def build(
    *,
    title: str | None = "Software Engineer",
    context: dict | None = None,
    company: str | None = "Example Systems",
    team: str | None = None,
    parent: str | None = None,
) -> HiringContextProfile:
    return build_hiring_context_profile(
        company=company,
        team=team,
        parent_organization=parent,
        role_title=title,
        normalized_job_context=context or {},
    )


def signal_values(profile: HiringContextProfile) -> set[str]:
    return {signal.value for signal in profile.signals}


def signals_of_kind(
    profile: HiringContextProfile,
    kind: HiringContextSignalKind,
):
    return [signal for signal in profile.signals if signal.kind is kind]


@pytest.mark.parametrize(
    ("title", "expected"),
    (
        ("Software Engineer", RoleFamily.SOFTWARE_ENGINEERING),
        ("Backend Engineer", RoleFamily.BACKEND_ENGINEERING),
        ("Frontend Engineer", RoleFamily.FRONTEND_ENGINEERING),
        ("Full-Stack Engineer", RoleFamily.FULL_STACK_ENGINEERING),
        ("Data Engineer", RoleFamily.DATA_ENGINEERING),
        ("Data Analyst", RoleFamily.DATA_ANALYTICS),
        ("Machine Learning Engineer", RoleFamily.MACHINE_LEARNING_AI),
        ("DevOps Engineer", RoleFamily.DEVOPS_CLOUD),
        ("Game Developer", RoleFamily.GAME_DEVELOPMENT),
        ("Mobile Engineer", RoleFamily.MOBILE),
        ("Embedded Systems Engineer", RoleFamily.EMBEDDED_SYSTEMS),
        ("Security Engineer", RoleFamily.SECURITY),
        ("Strategy Consultant", RoleFamily.CONSULTING_STRATEGY),
    ),
)
def test_explicit_role_title_family(title, expected):
    profile = build(title=title)
    assert profile.primary_role_family is expected


def test_generic_software_engineer_remains_generic_and_low_confidence():
    profile = build(title="Software Engineer")
    assert profile.primary_role_family is RoleFamily.SOFTWARE_ENGINEERING
    assert profile.secondary_role_families == ()
    assert profile.confidence is HiringContextConfidence.LOW


def test_software_engineering_intern_remains_software_without_company_domain_inference():
    profile = build(
        title="Software Engineering Intern",
        company="Unregistered Game Studio",
    )
    assert profile.primary_role_family is RoleFamily.SOFTWARE_ENGINEERING
    assert RoleFamily.GAME_DEVELOPMENT not in profile.secondary_role_families
    assert "game development" not in signal_values(profile)


def test_low_information_developer_becomes_general_safely():
    profile = build(title="Developer")
    assert profile.primary_role_family is RoleFamily.GENERAL
    assert profile.confidence is HiringContextConfidence.LOW


def test_vague_title_becomes_unknown_safely():
    profile = build(title="Associate")
    assert profile.primary_role_family is RoleFamily.UNKNOWN
    assert profile.confidence is HiringContextConfidence.LOW


@pytest.mark.parametrize(
    ("responsibilities", "expected"),
    (
        (("Build REST APIs and backend services", "Own FastAPI database integrations"), RoleFamily.BACKEND_ENGINEERING),
        (("Build React and TypeScript browser UI", "Own frontend interfaces"), RoleFamily.FRONTEND_ENGINEERING),
        (("Build ETL data pipelines", "Maintain the data warehouse with Airflow"), RoleFamily.DATA_ENGINEERING),
        (("Operate Kubernetes infrastructure", "Build CI/CD with Terraform and observability"), RoleFamily.DEVOPS_CLOUD),
    ),
)
def test_generic_title_is_narrowed_by_strong_explicit_jd(responsibilities, expected):
    profile = build(context={"core_responsibilities": responsibilities})
    assert profile.primary_role_family is expected
    assert RoleFamily.SOFTWARE_ENGINEERING in profile.secondary_role_families
    assert profile.confidence is HiringContextConfidence.MEDIUM


def test_explicit_game_development_jd_can_narrow_generic_title():
    profile = build(context={
        "core_responsibilities": (
            "Own game development and gameplay systems",
            "Build real-time interactive features in a game engine",
        ),
    })
    assert profile.primary_role_family is RoleFamily.GAME_DEVELOPMENT


def test_full_stack_preserves_backend_and_frontend_secondaries():
    profile = build(
        title="Full-Stack Engineer",
        context={
            "core_responsibilities": (
                "Own React frontend interfaces",
                "Build REST API backend services",
            ),
        },
    )
    assert profile.primary_role_family is RoleFamily.FULL_STACK_ENGINEERING
    assert set(profile.secondary_role_families) == {
        RoleFamily.BACKEND_ENGINEERING,
        RoleFamily.FRONTEND_ENGINEERING,
    }


def test_balanced_explicit_frontend_and_backend_duties_infer_full_stack():
    profile = build(
        context={
            "core_responsibilities": (
                "Own frontend React browser UI",
                "Own backend REST API services",
            ),
        },
    )
    assert profile.primary_role_family is RoleFamily.FULL_STACK_ENGINEERING


def test_role_classification_is_deterministic_under_value_reordering():
    values = ["Build ETL data pipelines", "Maintain a data warehouse", "Operate Airflow"]
    first = classify_hiring_context_role_families(
        role_title="Software Engineer",
        normalized_job_context={"core_responsibilities": values},
    )
    second = classify_hiring_context_role_families(
        role_title=" Software   Engineer ",
        normalized_job_context={"core_responsibilities": list(reversed(values))},
    )
    assert first == second


def test_strong_title_plus_supporting_jd_has_high_context_confidence():
    profile = build(
        title="Backend Engineer",
        context={"core_responsibilities": ("Build backend REST API services",)},
    )
    assert profile.confidence is HiringContextConfidence.HIGH


def test_required_qualification_becomes_high_confidence_explicit_signal():
    profile = build(context={"required_qualifications": ("Five years of API development",)})
    matching = [signal for signal in profile.signals if signal.value == "Five years of API development"]
    assert len(matching) == 1
    assert matching[0].kind is HiringContextSignalKind.EXPLICIT_JD
    assert matching[0].confidence is HiringContextConfidence.HIGH
    assert matching[0].ranking_effects == (RankingEffect.EXPLICIT_ALIGNMENT,)


def test_preferred_qualification_is_lower_priority_context_signal():
    profile = build(context={"preferred_qualifications": ("Experience with Kubernetes",)})
    matching = [signal for signal in profile.signals if signal.value == "Experience with Kubernetes"]
    assert len(matching) == 1
    assert matching[0].kind is HiringContextSignalKind.PREFERRED_QUALIFICATION
    assert matching[0].confidence is HiringContextConfidence.MEDIUM


def test_legacy_nice_to_have_cue_phrases_do_not_become_signals():
    profile = build(context={
        "requirements": {"nice_to_have_skills": ("preferred", "experience with")},
    })
    assert not {"preferred", "experience with"}.intersection(signal_values(profile))


def test_responsibility_becomes_explicit_signal():
    value = "Design reliable data pipelines"
    profile = build(context={"core_responsibilities": (value,)})
    assert value in signal_values(profile)


def test_known_skill_requirement_values_become_canonical_signals():
    profile = build(context={
        "skill_requirements": {
            "languages": ("JavaScript",),
            "frameworks": ("restful APIs",),
            "cloudInfra": ("Google Cloud Platform",),
        },
    })
    assert {"JavaScript", "REST API", "GCP"}.issubset(signal_values(profile))


def test_engineering_traits_are_derived_only_from_explicit_values():
    profile = build(context={
        "core_responsibilities": (
            "Own unit and integration testing",
            "Improve reliability and performance",
        ),
    })
    assert set(profile.high_value_traits) == {"Performance", "Reliability", "Testing"}


@pytest.mark.parametrize(
    "control_value",
    (
        "skills",
        "requirements",
        "technologies",
        "responsibilities",
        "preferred",
        "metadata",
        "project",
        "description",
        "value",
    ),
)
def test_schema_or_control_value_does_not_become_signal(control_value):
    profile = build(context={
        "skills": (control_value,),
        "metadata": {control_value: "hidden"},
    })
    assert control_value not in {value.casefold() for value in signal_values(profile)}


def test_schema_keys_alone_do_not_become_signals():
    profile = build(context={
        "skills": (),
        "requirements": {},
        "technologies": (),
        "metadata": {"project": "description", "value": "requirements"},
    })
    assert signal_values(profile) == {"software engineering"}


def test_prompt_control_and_candidate_fields_cannot_influence_profile():
    baseline = build(context={"technologies": ("Python",)})
    polluted = build(context={
        "technologies": ("Python",),
        "candidate_identity": "Technology Risk specialist",
        "role_profile": {"resume_strategy": "SENTINEL"},
        "target_role": {
            "candidate_positioning": ("SENTINEL",),
            "important_ats_keywords": ("SENTINEL",),
        },
        "requirements": {
            "evidence_types_to_emphasize": ("SENTINEL",),
            "repeated_ats_keywords": ("SENTINEL",),
            "action_verbs": ("SENTINEL",),
        },
        "tech_ontology": {
            "safe_resume_phrases": ("SENTINEL",),
            "do_not_claim_without_evidence": ("SENTINEL",),
        },
    })
    assert polluted == baseline
    assert "SENTINEL" not in polluted.to_json()


def test_unknown_nested_mappings_are_not_stringified():
    profile = build(context={"metadata": {"skills": "SENTINEL"}})
    assert "SENTINEL" not in profile.to_json()


def test_non_string_semantic_collection_item_fails_closed():
    with pytest.raises(TypeError, match="must be a string"):
        build(context={"technologies": ({"project_id": "WorkAgent"},)})


@pytest.mark.parametrize(
    ("values", "present", "absent"),
    (
        (("JavaScript",), "JavaScript", "Java"),
        (("Java",), "Java", "JavaScript"),
        (("C++",), "C++", "C"),
        (("C",), "C", "C++"),
        (("Google",), "Google", "Go"),
        (("Go",), "Go", "Google"),
        (("React",), "React", "R"),
        (("R",), "R", "React"),
        (("Rails",), "Rails", "AI"),
        (("AI",), "AI", "Rails"),
    ),
)
def test_atomic_technology_collision_safety(values, present, absent):
    profile = build(context={"technologies": values})
    assert present in signal_values(profile)
    assert absent not in signal_values(profile)


def test_java_does_not_leak_from_javascript_responsibility_text():
    profile = build(context={"core_responsibilities": ("Build JavaScript browser interfaces",)})
    assert "JavaScript" in signal_values(profile)
    assert "Java" not in signal_values(profile)


def test_c_does_not_leak_from_cpp_responsibility_text():
    profile = build(context={"core_responsibilities": ("Develop performance-sensitive C++ services",)})
    assert "C++" in signal_values(profile)
    assert "C" not in signal_values(profile)


def test_google_does_not_emit_go_from_responsibility_text():
    profile = build(context={"core_responsibilities": ("Deploy services to Google Cloud",)})
    assert "Go" not in signal_values(profile)


def test_react_does_not_emit_r_from_responsibility_text():
    profile = build(context={"core_responsibilities": ("Build React interfaces",)})
    assert "React" in signal_values(profile)
    assert "R" not in signal_values(profile)


def test_rails_does_not_emit_ai_from_responsibility_text():
    profile = build(context={"core_responsibilities": ("Maintain Ruby on Rails services",)})
    assert "Rails" in signal_values(profile)
    assert "AI" not in signal_values(profile)


def test_duplicate_normalized_signals_are_deduplicated():
    profile = build(context={
        "technologies": ("REST API", "restful APIs"),
        "skills": (" REST   API ",),
    })
    assert [value.casefold() for value in signal_values(profile)].count("rest api") == 1


def test_signal_selection_and_profile_fingerprint_are_deterministic():
    values = ("Python", "React", "REST API", "PostgreSQL")
    first = build(context={"technologies": values})
    second = build(
        title=" Software   Engineer ",
        company=" Example   Systems ",
        context={"technologies": tuple(reversed(values))},
    )
    assert first.to_dict() == second.to_dict()
    assert first.fingerprint == second.fingerprint


def test_material_role_title_change_changes_profile_fingerprint():
    assert build(title="Backend Engineer").fingerprint != build(title="Frontend Engineer").fingerprint


def test_material_jd_change_changes_profile_fingerprint():
    first = build(context={"technologies": ("Python",)})
    second = build(context={"technologies": ("Java",)})
    assert first.fingerprint != second.fingerprint


@pytest.mark.parametrize(
    "company",
    (
        "Amazon",
        "Google",
        "Shopify",
        "Security Finance",
        "Healthcare Systems",
        "AI Platform Team",
        "GameStop",
        "Unregistered Platform Company",
    ),
)
def test_company_identity_alone_is_domain_neutral(company):
    profile = build(company=company)
    assert profile.primary_role_family is RoleFamily.SOFTWARE_ENGINEERING
    assert not any(
        signal.kind in {
            HiringContextSignalKind.COMPANY_DOMAIN,
            HiringContextSignalKind.TEAM_DOMAIN,
            HiringContextSignalKind.PARENT_ORGANIZATION_DOMAIN,
        }
        for signal in profile.signals
    )


def test_company_identity_does_not_create_technology_assumptions():
    expected_absent = {
        "C#",
        "Azure",
        "AWS",
        "GCP",
        "Rails",
        "Machine learning",
        "AI",
    }
    for company in ("The Coalition", "Microsoft", "Amazon", "Google", "Shopify"):
        assert expected_absent.isdisjoint(signal_values(build(company=company)))


def test_team_and_parent_identity_are_carried_without_domain_inference():
    profile = build(team="AI Platform Team", parent="Unknown Parent Organization")
    assert profile.team == "AI Platform Team"
    assert profile.parent_organization == "Unknown Parent Organization"
    assert not any("game" in value.casefold() or value == "AI" for value in signal_values(profile))


def test_technology_risk_and_privacy_remain_role_context_only():
    profile = build(
        title="Technology Risk Consultant",
        context={
            "core_responsibilities": (
                "Advise on privacy, cyber transformation, and data controls",
            ),
        },
    )
    serialized = profile.to_json()
    assert "privacy" in serialized.casefold()
    for forbidden in (
        "candidate_identity",
        "candidate_domain_expertise",
        "candidate_specialization",
        "candidate_professional_title",
        "candidate_persona",
    ):
        assert forbidden not in serialized


def test_service_creates_only_hiring_context_contracts():
    profile = build(context={"technologies": ("Python",)})
    assert type(profile) is HiringContextProfile
    assert all(type(signal).__name__ == "HiringContextSignal" for signal in profile.signals)
    assert all(type(source) is HiringContextSourceRef for source in profile.source_refs)
    serialized = profile.to_json()
    for forbidden in (
        "project_id",
        "evidence_fact_id",
        "capability_fact_id",
        "engineering_story",
        "claim_sufficiency",
        "story_sufficiency",
    ):
        assert forbidden not in serialized


def test_context_confidence_is_the_accepted_context_enum():
    profile = build(context={"core_responsibilities": ("Build backend REST API services",)})
    assert type(profile.confidence) is HiringContextConfidence
    assert all(type(signal.confidence) is HiringContextConfidence for signal in profile.signals)


def test_explicit_jd_provenance_is_context_specific_and_contains_no_raw_jd():
    raw_like_value = "Build deterministic API validation"
    profile = build(context={"core_responsibilities": (raw_like_value,)})
    explicit = signals_of_kind(profile, HiringContextSignalKind.EXPLICIT_JD)
    assert explicit
    assert all(
        type(source) is HiringContextSourceRef
        and source.source_kind is HiringContextSourceKind.JOB_DESCRIPTION
        for signal in explicit
        for source in signal.source_refs
    )
    for source in profile.source_refs:
        assert set(source.to_dict()) == {"source_kind", "source_fingerprint", "reference_id"}
        assert raw_like_value not in source.to_json()


def test_role_family_signal_uses_only_role_family_alignment():
    profile = build(title="Backend Engineer")
    family_signals = signals_of_kind(profile, HiringContextSignalKind.ROLE_FAMILY)
    assert family_signals
    assert all(
        signal.ranking_effects == (RankingEffect.ROLE_FAMILY_ALIGNMENT,)
        for signal in family_signals
    )


def test_required_signal_wins_when_same_value_is_also_preferred():
    profile = build(context={
        "required_skills": ("Python",),
        "preferred_qualifications": ("Python",),
    })
    matching = [signal for signal in profile.signals if signal.value == "Python"]
    assert len(matching) == 1
    assert matching[0].kind is HiringContextSignalKind.EXPLICIT_JD


def test_over_bound_signal_candidates_are_reduced_deterministically():
    values = tuple(f"Required qualification {index:03d}" for index in range(80))
    first = build(context={"required_qualifications": values})
    second = build(context={"required_qualifications": tuple(reversed(values))})
    assert len(first.signals) == MAX_HIRING_CONTEXT_SIGNALS
    assert first.to_dict() == second.to_dict()
    assert all(signal.kind is HiringContextSignalKind.EXPLICIT_JD for signal in first.signals)


def test_preferred_signals_cannot_displace_required_signals_on_overflow():
    required = tuple(f"Required item {index:03d}" for index in range(MAX_HIRING_CONTEXT_SIGNALS))
    preferred = tuple(f"Preferred item {index:03d}" for index in range(20))
    profile = build(context={
        "required_qualifications": required,
        "preferred_qualifications": preferred,
    })
    assert len(profile.signals) == MAX_HIRING_CONTEXT_SIGNALS
    assert all(signal.kind is HiringContextSignalKind.EXPLICIT_JD for signal in profile.signals)


def test_model_field_bound_still_fails_closed():
    values = tuple(f"Technology {index}" for index in range(MAX_NORMALIZED_JOB_VALUES_PER_FIELD + 1))
    with pytest.raises(ValueError, match="maximum item count"):
        build(context={"technologies": values})


def test_total_input_context_bound_fails_closed():
    values = tuple(f"Value {index}" for index in range(MAX_NORMALIZED_JOB_VALUES_PER_FIELD))
    with pytest.raises(ValueError, match=str(MAX_NORMALIZED_JOB_CONTEXT_VALUES)):
        build(context={
            "core_responsibilities": values,
            "required_qualifications": tuple(f"Qualification {index}" for index in range(128)),
            "required_skills": tuple(f"Skill {index}" for index in range(128)),
            "technologies": tuple(f"Technology {index}" for index in range(128)),
            "preferred_qualifications": ("One extra value",),
        })


def test_empty_optional_team_remains_none():
    assert build(team=" \n\t ").team is None


def test_current_core_responsibility_boundary_is_consumed():
    profile = build(context={
        "target_role": {"core_responsibilities": ("Build reliable APIs",)},
        "requirements": {"responsibilities": ()},
    })
    assert "Build reliable APIs" in signal_values(profile)


def test_repaired_responsibilities_alias_is_consumed_when_core_is_empty():
    profile = build(context={
        "target_role": {"core_responsibilities": ()},
        "requirements": {"responsibilities": ("Build reliable APIs",)},
    })
    assert "Build reliable APIs" in signal_values(profile)


def test_equivalent_responsibility_aliases_are_accepted_without_merging():
    profile = build(context={
        "target_role": {"core_responsibilities": (" Build reliable APIs ",)},
        "requirements": {"responsibilities": ("build   reliable APIs",)},
    })
    assert len([value for value in signal_values(profile) if value.casefold() == "build reliable apis"]) == 1


def test_conflicting_responsibility_aliases_fail_closed():
    with pytest.raises(ValueError, match="conflicting responsibilities aliases"):
        build(context={
            "target_role": {"core_responsibilities": ("Build APIs",)},
            "requirements": {"responsibilities": ("Build dashboards",)},
        })


def test_alias_conflict_behavior_is_dictionary_order_independent():
    first = {
        "target_role": {"core_responsibilities": ("Build APIs",)},
        "requirements": {"responsibilities": ("Build dashboards",)},
    }
    second = dict(reversed(tuple(first.items())))
    for context in (first, second):
        with pytest.raises(ValueError, match="conflicting responsibilities aliases"):
            build(context=context)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("company", "pef_a"),
        ("team", "pcf_a"),
        ("parent", "pcb_a"),
        ("title", "engineering_story_a"),
    ),
)
def test_shortest_candidate_authority_ids_are_rejected_from_identity_text(field_name, value):
    kwargs = {field_name: value}
    with pytest.raises(ValueError, match="authority identifiers"):
        build(**kwargs)


@pytest.mark.parametrize("value", ("pef_a", "pcf_a", "pcb_a", "engineering_story_a"))
def test_shortest_candidate_authority_ids_are_rejected_from_signals(value):
    with pytest.raises(ValueError, match="authority identifiers"):
        build(context={"required_qualifications": (value,)})


def test_service_public_signature_has_no_candidate_or_story_inputs():
    names = set(inspect.signature(build_hiring_context_profile).parameters)
    assert names == {
        "company",
        "role_title",
        "normalized_job_context",
        "team",
        "parent_organization",
        "organization_registry",
    }
    assert not any(
        forbidden in name
        for name in names
        for forbidden in ("candidate", "resume", "project", "story", "evidence")
    )


def test_service_imports_only_pure_hiring_context_dependencies():
    module_path = Path(__file__).resolve().parents[1] / "backend" / "hiring_context_intelligence.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
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
    forbidden_prefixes = (
        "backend.api_server",
        "backend.main",
        "backend.tech_ontology",
        "backend.project_evidence",
        "backend.project_capability",
        "backend.project_claim",
        "backend.project_repository",
        "backend.engineering_story",
        "backend.memory_store",
        "backend.project_query_planner",
        "backend.project_retrieval",
        "backend.evidence_",
        "backend.github_",
        "backend.chroma_",
        "chromadb",
        "fastapi",
        "httpx",
        "openai",
        "os",
        "pathlib",
        "requests",
        "socket",
        "sqlite3",
        "subprocess",
        "urllib",
    )
    assert not any(name.startswith(forbidden_prefixes) for name in imports)
    assert imports.intersection({"backend.hiring_context_models"})


def test_service_has_no_random_time_environment_or_cache_dependency():
    module_path = Path(__file__).resolve().parents[1] / "backend" / "hiring_context_intelligence.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports.isdisjoint({"os", "random", "time", "datetime", "functools"})
    assert "getenv" not in source
    assert "lru_cache" not in source


def test_service_construction_has_no_io_side_effects(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("hiring context intelligence attempted I/O")

    monkeypatch.setattr(builtins, "open", blocked)
    monkeypatch.setattr(Path, "read_text", blocked)
    monkeypatch.setattr(Path, "write_text", blocked)
    monkeypatch.setattr(os, "getenv", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)
    profile = build(context={"technologies": ("Python",)})
    assert "Python" in signal_values(profile)


def test_service_does_not_construct_official_web_sources():
    profile = build(
        company="The Coalition",
        team="Online Systems",
        parent="Xbox Game Studios",
        context={"technologies": ("Python",)},
    )
    assert all("official_" not in source.source_kind.value for source in profile.source_refs)


def test_every_signal_source_is_owned_by_profile():
    profile = build(
        title="Backend Engineer",
        context={"core_responsibilities": ("Build REST API services",)},
    )
    profile_source_ids = {source.reference_id for source in profile.source_refs}
    assert all(
        source.reference_id in profile_source_ids
        for signal in profile.signals
        for source in signal.source_refs
    )


def test_output_contract_has_no_story_or_candidate_truth_fields():
    profile_fields = {item.name for item in fields(HiringContextProfile)}
    assert profile_fields.isdisjoint({
        "candidate_identity",
        "candidate_domain_expertise",
        "project_evidence_facts",
        "project_capability_facts",
        "story_evidence_ids",
        "claim_sufficiency",
        "story_sufficiency",
    })
