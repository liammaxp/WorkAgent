"""Bounded offline organization-domain context for hiring intelligence.

Rules in this module describe employer, team, and parent-organization context.
They never describe candidate experience, candidate technology support, or
project and Engineering Story truth.  Resolution is exact, deterministic, and
pure: there is no web, filesystem, database, retrieval, environment, or model
access.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
import types
import unicodedata
from typing import Any

from backend.hiring_context_models import (
    MAX_HIRING_CONTEXT_HIGH_VALUE_TRAITS,
    MAX_HIRING_CONTEXT_NAME_LENGTH,
    MAX_HIRING_CONTEXT_SIGNAL_VALUE_LENGTH,
    MAX_HIRING_CONTEXT_TRAIT_LENGTH,
    HiringContextConfidence,
    HiringContextSignal,
    HiringContextSignalKind,
    HiringContextSourceKind,
    HiringContextSourceRef,
    RankingEffect,
)


MAX_ORGANIZATION_CONTEXT_RULES = 32
MAX_ORGANIZATION_ALIASES = 8
MAX_ORGANIZATION_DOMAINS = 8
MAX_ORGANIZATION_TEAM_RULES = 8
MAX_ORGANIZATION_TEAM_ALIASES = 8
MAX_ORGANIZATION_DOMAIN_TRAITS = 4
MAX_RESOLVED_ORGANIZATION_TRAITS = 8

_FORBIDDEN_TECHNOLOGY_DOMAIN_VALUES = frozenset({
    ".net",
    "aws",
    "azure",
    "c#",
    "c++",
    "directx",
    "gcp",
    "google cloud",
    "google cloud platform",
    "rails",
    "ruby on rails",
    "unity",
    "unreal",
    "unreal engine",
})
_CANDIDATE_AUTHORITY_ID_RE = re.compile(
    r"(?<![a-z0-9])(?:pei|pef|pcf|pcb|pem|esr|engineering_story|"
    r"engineering_story_revision)_[a-z0-9][a-z0-9_.:-]*(?![a-z0-9])",
    re.IGNORECASE,
)


class OrganizationContextScope(str, Enum):
    COMPANY = "company"
    TEAM = "team"
    PARENT_ORGANIZATION = "parent_organization"


def _normalized_text(value: Any, name: str, maximum: int, *, required: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if required and not normalized:
        raise ValueError(f"{name} must not be blank")
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds maximum length {maximum}")
    if _CANDIDATE_AUTHORITY_ID_RE.search(normalized):
        raise ValueError(f"{name} cannot carry candidate or story authority identifiers")
    return normalized


def _optional_text(value: Any, name: str, maximum: int) -> str | None:
    if value is None:
        return None
    normalized = _normalized_text(value, name, maximum)
    return normalized or None


def _identity_key(value: str) -> str:
    return _normalized_text(
        value,
        "organization identity",
        MAX_HIRING_CONTEXT_NAME_LENGTH,
        required=True,
    ).casefold()


def _stable_texts(
    values: Sequence[Any],
    name: str,
    *,
    maximum: int,
    item_maximum: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be an array")
    if len(values) > maximum:
        raise ValueError(f"{name} exceeds maximum item count {maximum}")
    normalized: dict[str, str] = {}
    for value in values:
        text = _normalized_text(value, name, item_maximum, required=True)
        key = text.casefold()
        current = normalized.get(key)
        if current is None or text < current:
            normalized[key] = text
    return tuple(normalized[key] for key in sorted(normalized))


def _confidence(value: Any, name: str) -> HiringContextConfidence:
    if isinstance(value, Enum) and not isinstance(value, HiringContextConfidence):
        raise TypeError(f"{name} must use HiringContextConfidence")
    try:
        return HiringContextConfidence(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be low, medium, or high") from exc


def _contains_forbidden_technology(value: str) -> bool:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return any(
        re.search(
            rf"(?<![a-z0-9]){re.escape(technology)}(?![a-z0-9])",
            normalized,
        )
        for technology in _FORBIDDEN_TECHNOLOGY_DOMAIN_VALUES
    )


def _stable_typed_values(
    values: Sequence[Any],
    expected_type: type,
    name: str,
    *,
    maximum: int,
    key,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be an array")
    if len(values) > maximum:
        raise ValueError(f"{name} exceeds maximum item count {maximum}")
    normalized = {}
    for value in values:
        if not isinstance(value, expected_type):
            raise TypeError(f"{name} must contain only {expected_type.__name__} values")
        item_key = key(value)
        current = normalized.get(item_key)
        if current is not None and current != value:
            raise ValueError(f"conflicting {name} entry: {item_key}")
        normalized[item_key] = value
    return tuple(normalized[item_key] for item_key in sorted(normalized))


@dataclass(frozen=True, slots=True)
class OrganizationDomainSignal:
    value: str
    confidence: HiringContextConfidence
    high_value_traits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        value = _normalized_text(
            self.value,
            "organization domain value",
            MAX_HIRING_CONTEXT_SIGNAL_VALUE_LENGTH,
            required=True,
        )
        if _contains_forbidden_technology(value):
            raise ValueError(
                "organization domain values cannot encode stereotyped technologies"
            )
        confidence = _confidence(self.confidence, "organization domain confidence")
        traits = _stable_texts(
            self.high_value_traits,
            "organization domain high_value_traits",
            maximum=MAX_ORGANIZATION_DOMAIN_TRAITS,
            item_maximum=MAX_HIRING_CONTEXT_TRAIT_LENGTH,
        )
        if any(_contains_forbidden_technology(trait) for trait in traits):
            raise ValueError(
                "organization traits cannot encode stereotyped technologies"
            )
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "high_value_traits", traits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "confidence": self.confidence.value,
            "high_value_traits": list(self.high_value_traits),
        }


@dataclass(frozen=True, slots=True)
class OrganizationTeamContextRule:
    canonical_name: str
    aliases: tuple[str, ...] = ()
    domains: tuple[OrganizationDomainSignal, ...] = ()

    def __post_init__(self) -> None:
        canonical = _normalized_text(
            self.canonical_name,
            "team canonical_name",
            MAX_HIRING_CONTEXT_NAME_LENGTH,
            required=True,
        )
        aliases = _stable_texts(
            self.aliases,
            "team aliases",
            maximum=MAX_ORGANIZATION_TEAM_ALIASES,
            item_maximum=MAX_HIRING_CONTEXT_NAME_LENGTH,
        )
        aliases = tuple(alias for alias in aliases if _identity_key(alias) != _identity_key(canonical))
        domains = _stable_typed_values(
            self.domains,
            OrganizationDomainSignal,
            "team domains",
            maximum=MAX_ORGANIZATION_DOMAINS,
            key=lambda item: item.value.casefold(),
        )
        object.__setattr__(self, "canonical_name", canonical)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "domains", domains)

    def identity_keys(self) -> tuple[str, ...]:
        return tuple(sorted({_identity_key(self.canonical_name), *(_identity_key(alias) for alias in self.aliases)}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "domains": [domain.to_dict() for domain in self.domains],
        }


@dataclass(frozen=True, slots=True)
class OrganizationContextRule:
    canonical_name: str
    aliases: tuple[str, ...] = ()
    parent_organization: str | None = None
    domains: tuple[OrganizationDomainSignal, ...] = ()
    teams: tuple[OrganizationTeamContextRule, ...] = ()

    def __post_init__(self) -> None:
        canonical = _normalized_text(
            self.canonical_name,
            "organization canonical_name",
            MAX_HIRING_CONTEXT_NAME_LENGTH,
            required=True,
        )
        aliases = _stable_texts(
            self.aliases,
            "organization aliases",
            maximum=MAX_ORGANIZATION_ALIASES,
            item_maximum=MAX_HIRING_CONTEXT_NAME_LENGTH,
        )
        aliases = tuple(alias for alias in aliases if _identity_key(alias) != _identity_key(canonical))
        parent = _optional_text(
            self.parent_organization,
            "parent_organization",
            MAX_HIRING_CONTEXT_NAME_LENGTH,
        )
        domains = _stable_typed_values(
            self.domains,
            OrganizationDomainSignal,
            "organization domains",
            maximum=MAX_ORGANIZATION_DOMAINS,
            key=lambda item: item.value.casefold(),
        )
        teams = _stable_typed_values(
            self.teams,
            OrganizationTeamContextRule,
            "organization team rules",
            maximum=MAX_ORGANIZATION_TEAM_RULES,
            key=lambda item: _identity_key(item.canonical_name),
        )
        team_aliases: dict[str, str] = {}
        for team in teams:
            for alias_key in team.identity_keys():
                current = team_aliases.get(alias_key)
                canonical_key = _identity_key(team.canonical_name)
                if current is not None and current != canonical_key:
                    raise ValueError(f"conflicting team alias: {alias_key}")
                team_aliases[alias_key] = canonical_key
        object.__setattr__(self, "canonical_name", canonical)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "parent_organization", parent)
        object.__setattr__(self, "domains", domains)
        object.__setattr__(self, "teams", teams)

    def identity_keys(self) -> tuple[str, ...]:
        return tuple(sorted({_identity_key(self.canonical_name), *(_identity_key(alias) for alias in self.aliases)}))

    def match_team(self, value: str | None) -> OrganizationTeamContextRule | None:
        if not value:
            return None
        target = _identity_key(value)
        for team in self.teams:
            if target in team.identity_keys():
                return team
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "parent_organization": self.parent_organization,
            "domains": [domain.to_dict() for domain in self.domains],
            "teams": [team.to_dict() for team in self.teams],
        }


@dataclass(frozen=True, slots=True)
class OrganizationContextRegistry:
    rules: tuple[OrganizationContextRule, ...]
    _lookup: Mapping[str, OrganizationContextRule] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        rules = _stable_typed_values(
            self.rules,
            OrganizationContextRule,
            "organization context rules",
            maximum=MAX_ORGANIZATION_CONTEXT_RULES,
            key=lambda item: _identity_key(item.canonical_name),
        )
        lookup: dict[str, OrganizationContextRule] = {}
        for rule in rules:
            for alias_key in rule.identity_keys():
                current = lookup.get(alias_key)
                if current is not None and current != rule:
                    raise ValueError(f"conflicting organization alias: {alias_key}")
                lookup[alias_key] = rule
        for rule in rules:
            if rule.parent_organization and _identity_key(rule.parent_organization) not in lookup:
                raise ValueError(
                    f"unknown parent organization for {rule.canonical_name}: "
                    f"{rule.parent_organization}"
                )
        for rule in rules:
            seen = set()
            current = rule
            while current.parent_organization:
                current_key = _identity_key(current.canonical_name)
                if current_key in seen:
                    raise ValueError("organization parent relationships must not contain cycles")
                seen.add(current_key)
                current = lookup[_identity_key(current.parent_organization)]
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "_lookup", types.MappingProxyType(lookup))

    def match(self, value: str | None) -> OrganizationContextRule | None:
        if not value:
            return None
        return self._lookup.get(_identity_key(value))

    def canonical_parent(self, rule: OrganizationContextRule) -> str | None:
        if not rule.parent_organization:
            return None
        parent = self.match(rule.parent_organization)
        return parent.canonical_name if parent is not None else None


@dataclass(frozen=True, slots=True)
class ResolvedOrganizationSignal:
    scope: OrganizationContextScope
    signal: HiringContextSignal


@dataclass(frozen=True, slots=True)
class OrganizationContextResolution:
    company: str | None
    team: str | None
    parent_organization: str | None
    source_refs: tuple[HiringContextSourceRef, ...]
    signals: tuple[ResolvedOrganizationSignal, ...]
    high_value_traits: tuple[str, ...]


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_ref(source_kind: HiringContextSourceKind, payload: Any) -> HiringContextSourceRef:
    return HiringContextSourceRef(
        source_kind=source_kind,
        source_fingerprint=_digest(payload),
    )


def _rule_source(rule: OrganizationContextRule) -> HiringContextSourceRef:
    return _source_ref(
        HiringContextSourceKind.INTERNAL_TAXONOMY,
        {
            "schema": "organization_context_rule_v1",
            "rule": rule.to_dict(),
        },
    )


def _resolved_signal(
    *,
    scope: OrganizationContextScope,
    domain: OrganizationDomainSignal,
    sources: Sequence[HiringContextSourceRef],
) -> ResolvedOrganizationSignal:
    kind = {
        OrganizationContextScope.COMPANY: HiringContextSignalKind.COMPANY_DOMAIN,
        OrganizationContextScope.TEAM: HiringContextSignalKind.TEAM_DOMAIN,
        OrganizationContextScope.PARENT_ORGANIZATION: HiringContextSignalKind.PARENT_ORGANIZATION_DOMAIN,
    }[scope]
    return ResolvedOrganizationSignal(
        scope=scope,
        signal=HiringContextSignal(
            value=domain.value,
            kind=kind,
            confidence=domain.confidence,
            ranking_effects=(RankingEffect.DOMAIN_ALIGNMENT,),
            source_refs=tuple(sources),
        ),
    )


def resolve_organization_context(
    *,
    company: str | None,
    team: str | None,
    parent_organization: str | None,
    registry: OrganizationContextRegistry,
) -> OrganizationContextResolution:
    """Resolve exact registered organization identities into context signals."""

    if not isinstance(registry, OrganizationContextRegistry):
        raise TypeError("registry must be an OrganizationContextRegistry")
    input_company = _optional_text(company, "company", MAX_HIRING_CONTEXT_NAME_LENGTH)
    input_team = _optional_text(team, "team", MAX_HIRING_CONTEXT_NAME_LENGTH)
    input_parent = _optional_text(
        parent_organization,
        "parent_organization",
        MAX_HIRING_CONTEXT_NAME_LENGTH,
    )
    company_rule = registry.match(input_company)
    canonical_company = company_rule.canonical_name if company_rule else input_company

    expected_parent = registry.canonical_parent(company_rule) if company_rule else None
    supplied_parent_rule = registry.match(input_parent)
    canonical_supplied_parent = (
        supplied_parent_rule.canonical_name if supplied_parent_rule else input_parent
    )
    if expected_parent and canonical_supplied_parent:
        if _identity_key(expected_parent) != _identity_key(canonical_supplied_parent):
            raise ValueError(
                "supplied parent organization conflicts with accepted organization taxonomy"
            )
    canonical_parent = expected_parent or canonical_supplied_parent
    parent_rule = registry.match(canonical_parent)

    team_rule = company_rule.match_team(input_team) if company_rule else None
    canonical_team = team_rule.canonical_name if team_rule else input_team

    source_refs = []
    company_source = None
    if canonical_company:
        company_source = _source_ref(
            HiringContextSourceKind.COMPANY_IDENTITY,
            {"company": canonical_company},
        )
        source_refs.append(company_source)
    team_source = None
    if canonical_team:
        team_source = _source_ref(
            HiringContextSourceKind.TEAM_IDENTITY,
            {"team": canonical_team},
        )
        source_refs.append(team_source)
    parent_source = None
    if canonical_parent:
        parent_source = _source_ref(
            HiringContextSourceKind.PARENT_ORGANIZATION_IDENTITY,
            {"parent_organization": canonical_parent},
        )
        source_refs.append(parent_source)

    resolved_signals = []
    traits = []
    if company_rule and company_source:
        taxonomy_source = _rule_source(company_rule)
        source_refs.append(taxonomy_source)
        for domain in company_rule.domains:
            resolved_signals.append(_resolved_signal(
                scope=OrganizationContextScope.COMPANY,
                domain=domain,
                sources=(company_source, taxonomy_source),
            ))
            traits.extend(domain.high_value_traits)
        if team_rule and team_source:
            for domain in team_rule.domains:
                team_sources = tuple(
                    source
                    for source in (team_source, company_source, taxonomy_source)
                    if source is not None
                )
                resolved_signals.append(_resolved_signal(
                    scope=OrganizationContextScope.TEAM,
                    domain=domain,
                    sources=team_sources,
                ))
                traits.extend(domain.high_value_traits)
    if parent_rule and parent_source:
        taxonomy_source = _rule_source(parent_rule)
        source_refs.append(taxonomy_source)
        for domain in parent_rule.domains:
            resolved_signals.append(_resolved_signal(
                scope=OrganizationContextScope.PARENT_ORGANIZATION,
                domain=domain,
                sources=(parent_source, taxonomy_source),
            ))
            traits.extend(domain.high_value_traits)

    normalized_traits = _stable_texts(
        traits,
        "resolved organization high_value_traits",
        maximum=MAX_HIRING_CONTEXT_HIGH_VALUE_TRAITS,
        item_maximum=MAX_HIRING_CONTEXT_TRAIT_LENGTH,
    )
    if len(normalized_traits) > MAX_RESOLVED_ORGANIZATION_TRAITS:
        raise ValueError(
            "resolved organization context exceeds maximum high-value trait count "
            f"{MAX_RESOLVED_ORGANIZATION_TRAITS}"
        )
    unique_sources = {source.reference_id: source for source in source_refs}
    unique_signals = {resolved.signal.signal_id: resolved for resolved in resolved_signals}
    scope_order = {
        OrganizationContextScope.TEAM: 0,
        OrganizationContextScope.COMPANY: 1,
        OrganizationContextScope.PARENT_ORGANIZATION: 2,
    }
    return OrganizationContextResolution(
        company=canonical_company,
        team=canonical_team,
        parent_organization=canonical_parent,
        source_refs=tuple(unique_sources[key] for key in sorted(unique_sources)),
        signals=tuple(sorted(
            unique_signals.values(),
            key=lambda item: (
                scope_order[item.scope],
                item.signal.value.casefold(),
                item.signal.signal_id,
            ),
        )),
        high_value_traits=normalized_traits,
    )


DEFAULT_ORGANIZATION_CONTEXT_REGISTRY = OrganizationContextRegistry(rules=(
    OrganizationContextRule(
        canonical_name="Microsoft",
        aliases=("Microsoft Corporation",),
        domains=(
            OrganizationDomainSignal(
                value="Software platforms",
                confidence=HiringContextConfidence.MEDIUM,
            ),
        ),
    ),
    OrganizationContextRule(
        canonical_name="Xbox Game Studios",
        aliases=("Xbox Studios",),
        parent_organization="Microsoft",
        domains=(
            OrganizationDomainSignal(
                value="Game development",
                confidence=HiringContextConfidence.MEDIUM,
            ),
        ),
    ),
    OrganizationContextRule(
        canonical_name="The Coalition",
        aliases=("Coalition", "The Coalition Studio"),
        parent_organization="Xbox Game Studios",
        domains=(
            OrganizationDomainSignal(
                value="Game development",
                confidence=HiringContextConfidence.HIGH,
            ),
            OrganizationDomainSignal(
                value="Real-time interactive software",
                confidence=HiringContextConfidence.MEDIUM,
            ),
        ),
        teams=(
            OrganizationTeamContextRule(
                canonical_name="Online Systems",
                aliases=("Online Systems Team",),
                domains=(
                    OrganizationDomainSignal(
                        value="Online game systems",
                        confidence=HiringContextConfidence.HIGH,
                    ),
                ),
            ),
        ),
    ),
))


__all__ = [
    "DEFAULT_ORGANIZATION_CONTEXT_REGISTRY",
    "MAX_ORGANIZATION_ALIASES",
    "MAX_ORGANIZATION_CONTEXT_RULES",
    "MAX_ORGANIZATION_DOMAINS",
    "MAX_ORGANIZATION_DOMAIN_TRAITS",
    "MAX_ORGANIZATION_TEAM_ALIASES",
    "MAX_ORGANIZATION_TEAM_RULES",
    "MAX_RESOLVED_ORGANIZATION_TRAITS",
    "OrganizationContextRegistry",
    "OrganizationContextResolution",
    "OrganizationContextRule",
    "OrganizationContextScope",
    "OrganizationDomainSignal",
    "OrganizationTeamContextRule",
    "ResolvedOrganizationSignal",
    "resolve_organization_context",
]
