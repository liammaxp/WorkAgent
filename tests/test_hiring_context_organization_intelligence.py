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

from backend.hiring_context_intelligence import build_hiring_context_profile
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
from backend.hiring_context_organization import (
    DEFAULT_ORGANIZATION_CONTEXT_REGISTRY,
    MAX_ORGANIZATION_ALIASES,
    MAX_ORGANIZATION_CONTEXT_RULES,
    MAX_ORGANIZATION_DOMAINS,
    MAX_ORGANIZATION_TEAM_RULES,
    OrganizationContextRegistry,
    OrganizationContextRule,
    OrganizationContextScope,
    OrganizationDomainSignal,
    OrganizationTeamContextRule,
    resolve_organization_context,
)


DOMAIN_KINDS = {
    HiringContextSignalKind.COMPANY_DOMAIN,
    HiringContextSignalKind.TEAM_DOMAIN,
    HiringContextSignalKind.PARENT_ORGANIZATION_DOMAIN,
}


def build(
    *,
    company: str | None = "ABC Systems",
    team: str | None = None,
    parent: str | None = None,
    title: str | None = "Software Engineer",
    context: dict | None = None,
    registry: OrganizationContextRegistry = DEFAULT_ORGANIZATION_CONTEXT_REGISTRY,
) -> HiringContextProfile:
    return build_hiring_context_profile(
        company=company,
        team=team,
        parent_organization=parent,
        role_title=title,
        normalized_job_context=context or {},
        organization_registry=registry,
    )


def domain_signals(profile: HiringContextProfile):
    return [signal for signal in profile.signals if signal.kind in DOMAIN_KINDS]


def values(profile: HiringContextProfile) -> set[str]:
    return {signal.value for signal in profile.signals}


def analytics_registry(*, company_domain: str = "Business intelligence") -> OrganizationContextRegistry:
    return OrganizationContextRegistry(rules=(
        OrganizationContextRule(
            canonical_name="Example Holdings",
            aliases=("Example Holdings Incorporated",),
            domains=(
                OrganizationDomainSignal(
                    value="Data analytics",
                    confidence=HiringContextConfidence.MEDIUM,
                ),
            ),
        ),
        OrganizationContextRule(
            canonical_name="Example Analytics",
            aliases=("Example Analytics Incorporated", "Example Analytics Inc."),
            parent_organization="Example Holdings",
            domains=(
                OrganizationDomainSignal(
                    value="Data analytics",
                    confidence=HiringContextConfidence.MEDIUM,
                ),
                OrganizationDomainSignal(
                    value=company_domain,
                    confidence=HiringContextConfidence.MEDIUM,
                ),
            ),
            teams=(
                OrganizationTeamContextRule(
                    canonical_name="Insights Team",
                    aliases=("Insights",),
                    domains=(
                        OrganizationDomainSignal(
                            value="Data analytics",
                            confidence=HiringContextConfidence.HIGH,
                        ),
                        OrganizationDomainSignal(
                            value="Decision support",
                            confidence=HiringContextConfidence.HIGH,
                        ),
                    ),
                ),
            ),
        ),
    ))


def test_unknown_company_produces_no_domain_signal():
    profile = build(company="ABC Systems")
    assert domain_signals(profile) == []


def test_empty_company_is_safe():
    profile = build(company="")
    assert profile.company is None
    assert domain_signals(profile) == []


def test_empty_team_is_safe():
    profile = build(company="The Coalition", team=" \n\t ")
    assert profile.team is None
    assert not any(signal.kind is HiringContextSignalKind.TEAM_DOMAIN for signal in profile.signals)


def test_unknown_company_identity_is_preserved():
    profile = build(company=" ABC   Systems ")
    assert profile.company == "ABC Systems"
    assert any(
        source.source_kind is HiringContextSourceKind.COMPANY_IDENTITY
        for source in profile.source_refs
    )


def test_canonical_organization_exact_match():
    profile = build(company="The Coalition")
    assert profile.company == "The Coalition"
    assert "Game development" in values(profile)


def test_registered_alias_exact_normalized_match():
    canonical = build(company="The Coalition")
    alias = build(company="  coalition  ")
    assert alias.company == "The Coalition"
    assert alias == canonical


@pytest.mark.parametrize(
    "company",
    (
        "The Coalitio",
        "The Coalition Technologies",
        "CoalitionSomethingUnrelated",
        "XboxSomethingUnrelated",
        "Microsoftish",
        "Micro Softworks",
    ),
)
def test_partial_substring_or_fuzzy_organization_does_not_match(company):
    profile = build(company=company)
    assert profile.company == company
    assert domain_signals(profile) == []


def test_known_parent_organization_is_inferred_from_accepted_rule():
    profile = build(company="The Coalition")
    assert profile.parent_organization == "Xbox Game Studios"
    assert any(
        source.source_kind is HiringContextSourceKind.PARENT_ORGANIZATION_IDENTITY
        for source in profile.source_refs
    )


def test_equivalent_explicit_parent_alias_normalizes_to_canonical_parent():
    profile = build(company="The Coalition", parent=" Xbox Studios ")
    assert profile.parent_organization == "Xbox Game Studios"


def test_conflicting_parent_mapping_fails_closed():
    with pytest.raises(ValueError, match="conflicts with accepted organization taxonomy"):
        build(company="The Coalition", parent="Amazon")


def test_specific_organization_domain_signal():
    signals = domain_signals(build(company="The Coalition"))
    game = next(signal for signal in signals if signal.value == "Game development")
    assert game.kind is HiringContextSignalKind.COMPANY_DOMAIN
    assert game.confidence is HiringContextConfidence.HIGH


def test_team_specific_domain_signal():
    profile = build(company="The Coalition", team="Online Systems")
    online = next(signal for signal in profile.signals if signal.value == "Online game systems")
    assert online.kind is HiringContextSignalKind.TEAM_DOMAIN
    assert online.confidence is HiringContextConfidence.HIGH


def test_team_alias_is_scoped_to_its_registered_company():
    matching = build(company="The Coalition", team="Online Systems Team")
    unrelated = build(company="Microsoft", team="Online Systems Team")
    assert "Online game systems" in values(matching)
    assert "Online game systems" not in values(unrelated)


def test_arbitrary_team_name_substring_does_not_match():
    profile = build(company="The Coalition", team="Online Systems Extended")
    assert not any(signal.kind is HiringContextSignalKind.TEAM_DOMAIN for signal in profile.signals)


def test_team_precedence_wins_duplicate_domain_and_preserves_all_provenance():
    profile = build(
        company="Example Analytics",
        team="Insights Team",
        registry=analytics_registry(),
    )
    matching = [
        signal
        for signal in profile.signals
        if signal.value.casefold() == "data analytics"
        and signal.kind is not HiringContextSignalKind.ROLE_FAMILY
    ]
    assert len(matching) == 1
    signal = matching[0]
    assert signal.kind is HiringContextSignalKind.TEAM_DOMAIN
    assert signal.confidence is HiringContextConfidence.HIGH
    source_kinds = {source.source_kind for source in signal.source_refs}
    assert {
        HiringContextSourceKind.TEAM_IDENTITY,
        HiringContextSourceKind.COMPANY_IDENTITY,
        HiringContextSourceKind.PARENT_ORGANIZATION_IDENTITY,
        HiringContextSourceKind.INTERNAL_TAXONOMY,
    }.issubset(source_kinds)


def test_broad_parent_context_does_not_erase_specific_company_context():
    profile = build(company="Example Analytics", registry=analytics_registry())
    assert {"Data analytics", "Business intelligence"}.issubset(values(profile))


def test_parent_context_without_known_company_is_supported():
    profile = build(
        company="Unknown Subsidiary",
        parent="Example Holdings",
        registry=analytics_registry(),
    )
    signal = next(signal for signal in profile.signals if signal.value == "Data analytics")
    assert signal.kind is HiringContextSignalKind.PARENT_ORGANIZATION_DOMAIN


def test_the_coalition_regression_resolves_through_registry_rule():
    rule = DEFAULT_ORGANIZATION_CONTEXT_REGISTRY.match("The Coalition")
    assert rule is not None
    assert any(domain.value == "Game development" for domain in rule.domains)
    profile = build(company="The Coalition", title="Software Engineering Intern")
    assert "Game development" in values(profile)


def test_the_coalition_does_not_rewrite_explicit_role_family():
    profile = build(company="The Coalition", title="Software Engineering Intern")
    assert profile.primary_role_family is RoleFamily.SOFTWARE_ENGINEERING
    assert RoleFamily.GAME_DEVELOPMENT not in profile.secondary_role_families


@pytest.mark.parametrize(
    "forbidden_technology",
    ("Unity", "C++", "Unreal Engine", "DirectX"),
)
def test_the_coalition_context_does_not_create_candidate_technology(forbidden_technology):
    assert forbidden_technology not in values(build(company="The Coalition"))


@pytest.mark.parametrize(
    ("company", "forbidden"),
    (
        ("Microsoft", "C#"),
        ("Microsoft", "Azure"),
        ("Amazon", "AWS"),
        ("Google", "GCP"),
        ("Shopify", "Rails"),
    ),
)
def test_company_stereotypes_do_not_create_technology(company, forbidden):
    assert forbidden not in values(build(company=company))


def test_broad_registered_company_domain_does_not_rewrite_role_family():
    profile = build(company="Microsoft", title="Software Engineer")
    assert "Software platforms" in values(profile)
    assert profile.primary_role_family is RoleFamily.SOFTWARE_ENGINEERING


def test_organization_context_sources_are_hiring_context_refs():
    profile = build(company="The Coalition", team="Online Systems")
    for signal in domain_signals(profile):
        assert all(type(source) is HiringContextSourceRef for source in signal.source_refs)
        assert any(
            source.source_kind is HiringContextSourceKind.INTERNAL_TAXONOMY
            for source in signal.source_refs
        )


def test_no_candidate_evidence_provenance_is_created():
    serialized = build(company="The Coalition").to_json()
    for forbidden in (
        "project_id",
        "evidence_fact_id",
        "capability_fact_id",
        "story_evidence_id",
        "repository",
        "file_path",
    ):
        assert forbidden not in serialized


def test_organization_signal_confidence_is_context_confidence():
    assert all(
        type(signal.confidence) is HiringContextConfidence
        for signal in domain_signals(build(company="The Coalition"))
    )


def test_unknown_organization_does_not_create_speculative_low_signal():
    assert domain_signals(build(company="Unknown Organization")) == []


def test_company_domain_uses_only_domain_alignment():
    assert all(
        signal.ranking_effects == (RankingEffect.DOMAIN_ALIGNMENT,)
        for signal in domain_signals(build(company="The Coalition"))
    )


@pytest.mark.parametrize(
    "forbidden_effect",
    ("evidence_strength", "claim_sufficiency", "story_sufficiency", "technology_support"),
)
def test_organization_signals_cannot_create_candidate_truth_effects(forbidden_effect):
    serialized = build(company="The Coalition").to_json()
    assert forbidden_effect not in serialized


@pytest.mark.parametrize(
    ("organization", "domain"),
    (
        ("Risk Advisory Group", "Technology risk"),
        ("Privacy Advisory Group", "Privacy"),
        ("Strategy Advisory Group", "Consulting strategy"),
    ),
)
def test_organization_context_does_not_create_candidate_persona(organization, domain):
    registry = OrganizationContextRegistry(rules=(
        OrganizationContextRule(
            canonical_name=organization,
            domains=(
                OrganizationDomainSignal(
                    value=domain,
                    confidence=HiringContextConfidence.HIGH,
                ),
            ),
        ),
    ))
    serialized = build(company=organization, registry=registry).to_json()
    assert domain in serialized
    for forbidden in (
        "candidate_identity",
        "candidate_specialization",
        "candidate_domain_expertise",
        "candidate_professional_title",
        "candidate_persona",
    ):
        assert forbidden not in serialized


def test_duplicate_jd_and_organization_domain_signal_deduplicates_semantically():
    profile = build(
        company="The Coalition",
        context={"requirements": {"domain_knowledge": ("Game development",)}},
    )
    matching = [
        signal
        for signal in profile.signals
        if signal.value.casefold() == "game development"
        and signal.kind is not HiringContextSignalKind.ROLE_FAMILY
    ]
    assert len(matching) == 1
    assert matching[0].kind is HiringContextSignalKind.EXPLICIT_JD


def test_jd_and_organization_dedupe_retains_both_provenance_domains():
    profile = build(
        company="The Coalition",
        context={"requirements": {"domain_knowledge": ("Game development",)}},
    )
    signal = next(
        signal
        for signal in profile.signals
        if signal.value.casefold() == "game development"
        and signal.kind is HiringContextSignalKind.EXPLICIT_JD
    )
    source_kinds = {source.source_kind for source in signal.source_refs}
    assert HiringContextSourceKind.JOB_DESCRIPTION in source_kinds
    assert HiringContextSourceKind.COMPANY_IDENTITY in source_kinds
    assert HiringContextSourceKind.INTERNAL_TAXONOMY in source_kinds
    assert signal.ranking_effects == (
        RankingEffect.EXPLICIT_ALIGNMENT,
        RankingEffect.DOMAIN_ALIGNMENT,
    )
    assert signal.confidence is HiringContextConfidence.HIGH


def test_role_family_truth_remains_separate_from_same_value_domain_signal():
    profile = build(
        company="The Coalition",
        title="Game Developer",
    )
    matching = [signal for signal in profile.signals if signal.value.casefold() == "game development"]
    assert {signal.kind for signal in matching} == {
        HiringContextSignalKind.ROLE_FAMILY,
        HiringContextSignalKind.COMPANY_DOMAIN,
    }


def test_conflicting_duplicate_organization_rules_fail_closed():
    first = OrganizationContextRule(canonical_name="Example")
    second = OrganizationContextRule(
        canonical_name="Example",
        domains=(
            OrganizationDomainSignal(
                value="Data analytics",
                confidence=HiringContextConfidence.HIGH,
            ),
        ),
    )
    with pytest.raises(ValueError, match="conflicting organization context rules"):
        OrganizationContextRegistry(rules=(first, second))


def test_exact_duplicate_rules_are_deduplicated_deterministically():
    rule = OrganizationContextRule(
        canonical_name="Example",
        aliases=("Example Incorporated",),
    )
    registry = OrganizationContextRegistry(rules=(rule, rule))
    assert registry.rules == (rule,)


def test_registry_order_is_deterministic():
    first = analytics_registry()
    second = OrganizationContextRegistry(rules=tuple(reversed(first.rules)))
    assert first.rules == second.rules
    assert build(company="Example Analytics", registry=first) == build(
        company="Example Analytics",
        registry=second,
    )


def test_equivalent_alias_has_identical_semantic_profile_and_fingerprint():
    registry = analytics_registry()
    canonical = build(company="Example Analytics", registry=registry)
    alias = build(company=" example   analytics inc. ", registry=registry)
    assert alias == canonical
    assert alias.fingerprint == canonical.fingerprint


def test_material_matched_domain_rule_change_changes_profile_fingerprint():
    first = build(company="Example Analytics", registry=analytics_registry())
    second = build(
        company="Example Analytics",
        registry=analytics_registry(company_domain="Decision intelligence"),
    )
    assert first.fingerprint != second.fingerprint


def test_unrelated_rule_does_not_change_matched_profile_fingerprint():
    registry = analytics_registry()
    extra = OrganizationContextRule(canonical_name="Unrelated Organization")
    expanded = OrganizationContextRegistry(rules=registry.rules + (extra,))
    assert build(company="Example Analytics", registry=registry) == build(
        company="Example Analytics",
        registry=expanded,
    )


def test_organization_identity_matches_only_its_identity_scope():
    profile = build(
        company=None,
        title="Amazon Connect Engineer",
        context={"core_responsibilities": ("Integrate with Microsoft services",)},
    )
    assert domain_signals(profile) == []


def test_no_raw_web_candidate_project_or_story_fields_exist():
    for contract in (
        OrganizationDomainSignal,
        OrganizationTeamContextRule,
        OrganizationContextRule,
    ):
        names = {item.name for item in fields(contract)}
        assert names.isdisjoint({
            "raw_web_content",
            "url",
            "project_id",
            "candidate_id",
            "story_id",
            "evidence_fact_id",
            "ranking_score",
        })


def test_organization_module_has_no_story_candidate_ranking_or_runtime_imports():
    module_path = Path(__file__).resolve().parents[1] / "backend" / "hiring_context_organization.py"
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
        "backend.tech_ontology",
        "backend.project_",
        "backend.engineering_story",
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


def test_organization_public_signatures_have_no_candidate_story_or_project_inputs():
    names = set(inspect.signature(resolve_organization_context).parameters)
    assert names == {"company", "team", "parent_organization", "registry"}
    assert not any(
        forbidden in name
        for name in names
        for forbidden in ("candidate", "resume", "project", "story", "evidence")
    )


def test_organization_resolution_has_no_io_side_effects(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("organization intelligence attempted I/O")

    monkeypatch.setattr(builtins, "open", blocked)
    monkeypatch.setattr(Path, "read_text", blocked)
    monkeypatch.setattr(Path, "write_text", blocked)
    monkeypatch.setattr(os, "getenv", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)
    profile = build(company="The Coalition", team="Online Systems")
    assert "Game development" in values(profile)


def test_offline_resolution_never_constructs_official_web_sources():
    profile = build(company="The Coalition", team="Online Systems")
    assert all("official_" not in source.source_kind.value for source in profile.source_refs)


def test_unknown_organization_preserves_step2_behavior_exactly():
    empty_registry = OrganizationContextRegistry(rules=())
    default = build(
        company="Unknown Organization",
        context={"technologies": ("Python",)},
    )
    without_rules = build(
        company="Unknown Organization",
        context={"technologies": ("Python",)},
        registry=empty_registry,
    )
    assert default == without_rules


def test_full_explicit_signal_capacity_is_not_displaced_by_organization_context():
    required = tuple(f"Required item {index:03d}" for index in range(MAX_HIRING_CONTEXT_SIGNALS))
    profile = build(
        company="The Coalition",
        context={"required_qualifications": required},
    )
    assert len(profile.signals) == MAX_HIRING_CONTEXT_SIGNALS
    assert all(signal.kind is HiringContextSignalKind.EXPLICIT_JD for signal in profile.signals)
    selected_source_ids = {
        source.reference_id for signal in profile.signals for source in signal.source_refs
    }
    assert all(
        source.source_kind is not HiringContextSourceKind.INTERNAL_TAXONOMY
        or source.reference_id in selected_source_ids
        for source in profile.source_refs
    )


def test_every_organization_signal_source_is_owned_by_profile():
    profile = build(company="The Coalition", team="Online Systems")
    owned = {source.reference_id for source in profile.source_refs}
    assert all(
        source.reference_id in owned
        for signal in domain_signals(profile)
        for source in signal.source_refs
    )


@pytest.mark.parametrize(
    "technology",
    ("C#", "Azure cloud", "AWS", "GCP platform", "Ruby on Rails", "Unity games", "C++"),
)
def test_registry_rejects_stereotyped_technology_domain_values(technology):
    with pytest.raises(ValueError, match="stereotyped technologies"):
        OrganizationDomainSignal(
            value=technology,
            confidence=HiringContextConfidence.HIGH,
        )


@pytest.mark.parametrize("authority_id", ("pef_a", "pcf_a", "pcb_a", "engineering_story_a"))
def test_registry_rejects_candidate_or_story_authority_ids(authority_id):
    with pytest.raises(ValueError, match="authority identifiers"):
        OrganizationDomainSignal(
            value=authority_id,
            confidence=HiringContextConfidence.HIGH,
        )


def test_registry_rejects_alias_collision_between_organizations():
    with pytest.raises(ValueError, match="conflicting organization alias"):
        OrganizationContextRegistry(rules=(
            OrganizationContextRule(canonical_name="First", aliases=("Shared",)),
            OrganizationContextRule(canonical_name="Second", aliases=("Shared",)),
        ))


def test_rule_rejects_conflicting_team_aliases():
    with pytest.raises(ValueError, match="conflicting team alias"):
        OrganizationContextRule(
            canonical_name="Example",
            teams=(
                OrganizationTeamContextRule(canonical_name="First Team", aliases=("Shared",)),
                OrganizationTeamContextRule(canonical_name="Second Team", aliases=("Shared",)),
            ),
        )


def test_registry_rejects_unknown_parent_reference():
    with pytest.raises(ValueError, match="unknown parent organization"):
        OrganizationContextRegistry(rules=(
            OrganizationContextRule(
                canonical_name="Child",
                parent_organization="Missing Parent",
            ),
        ))


def test_registry_rejects_parent_cycle():
    with pytest.raises(ValueError, match="must not contain cycles"):
        OrganizationContextRegistry(rules=(
            OrganizationContextRule(canonical_name="First", parent_organization="Second"),
            OrganizationContextRule(canonical_name="Second", parent_organization="First"),
        ))


def test_registry_rule_count_is_bounded():
    rules = tuple(
        OrganizationContextRule(canonical_name=f"Organization {index}")
        for index in range(MAX_ORGANIZATION_CONTEXT_RULES + 1)
    )
    with pytest.raises(ValueError, match="maximum item count"):
        OrganizationContextRegistry(rules=rules)


def test_rule_alias_count_is_bounded():
    aliases = tuple(f"Alias {index}" for index in range(MAX_ORGANIZATION_ALIASES + 1))
    with pytest.raises(ValueError, match="maximum item count"):
        OrganizationContextRule(canonical_name="Example", aliases=aliases)


def test_rule_domain_count_is_bounded():
    domains = tuple(
        OrganizationDomainSignal(
            value=f"Domain {index}",
            confidence=HiringContextConfidence.MEDIUM,
        )
        for index in range(MAX_ORGANIZATION_DOMAINS + 1)
    )
    with pytest.raises(ValueError, match="maximum item count"):
        OrganizationContextRule(canonical_name="Example", domains=domains)


def test_rule_team_count_is_bounded():
    teams = tuple(
        OrganizationTeamContextRule(canonical_name=f"Team {index}")
        for index in range(MAX_ORGANIZATION_TEAM_RULES + 1)
    )
    with pytest.raises(ValueError, match="maximum item count"):
        OrganizationContextRule(canonical_name="Example", teams=teams)


def test_default_registry_adds_no_high_value_traits_by_stereotype():
    assert build(company="The Coalition").high_value_traits == ()


def test_organization_trait_requires_explicit_rule_declaration():
    registry = OrganizationContextRegistry(rules=(
        OrganizationContextRule(
            canonical_name="Reliability Organization",
            domains=(
                OrganizationDomainSignal(
                    value="Reliability engineering",
                    confidence=HiringContextConfidence.HIGH,
                    high_value_traits=("Reliability",),
                ),
            ),
        ),
    ))
    assert build(
        company="Reliability Organization",
        registry=registry,
    ).high_value_traits == ("Reliability",)


def test_registry_contains_no_company_specific_candidate_boost_or_ranking_output():
    module_path = Path(__file__).resolve().parents[1] / "backend" / "hiring_context_organization.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name.casefold()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not any(
        "rank_project" in name
        or "rank_story" in name
        or "select_project" in name
        or "boost_project" in name
        for name in function_names
    )
    assert "rank_projects_for_resume" not in source
    assert "select_staged_projects_with_ranking" not in source


def test_no_second_hiring_context_profile_schema_is_defined():
    module_path = Path(__file__).resolve().parents[1] / "backend" / "hiring_context_organization.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    class_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }
    assert "HiringContextProfile" not in class_names
    assert type(build(company="The Coalition")) is HiringContextProfile
