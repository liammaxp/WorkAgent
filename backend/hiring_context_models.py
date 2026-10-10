"""Bounded immutable contracts describing what a hiring context appears to value.

Hiring-context sources and signals are an independent truth domain.  They may
later guide relevance or emphasis, but they do not carry candidate evidence,
capability, claim-boundary, repository, or engineering-story authority.

This module contains pure model, normalization, fingerprint, and serialization
logic only.  It performs no parsing, inference, ranking, retrieval, persistence,
network access, or model calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import json
import re
import types
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints


MAX_HIRING_CONTEXT_NAME_LENGTH = 200
MAX_HIRING_CONTEXT_ROLE_TITLE_LENGTH = 240
MAX_HIRING_CONTEXT_SIGNAL_VALUE_LENGTH = 240
MAX_HIRING_CONTEXT_TRAIT_LENGTH = 160
MAX_HIRING_CONTEXT_SECONDARY_ROLE_FAMILIES = 6
MAX_HIRING_CONTEXT_SIGNALS = 64
MAX_HIRING_CONTEXT_SOURCE_REFS = 16
MAX_HIRING_CONTEXT_SIGNAL_SOURCE_REFS = 8
MAX_HIRING_CONTEXT_HIGH_VALUE_TRAITS = 16
MAX_HIRING_CONTEXT_RANKING_EFFECTS = 3

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_CANDIDATE_AUTHORITY_ID_RE = re.compile(
    r"(?<![a-z0-9])(?:"
    r"pei|pef|pcf|pcb|pem|esr|raw|chk|hyb|capability_fact|"
    r"engineering_story|engineering_story_revision"
    r")_[a-z0-9][a-z0-9_.:-]*(?![a-z0-9])",
    re.IGNORECASE,
)
_UNSAFE_SOURCE_PATTERNS = (
    re.compile(r"\bdiff --git\b", re.IGNORECASE),
    re.compile(r"@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_ -]?key|access[_ -]?token|password|secret|credential)\s*=\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bauthorization\s*:\s*(?:bearer|basic)\s+[a-z0-9._~+/=-]{8,}",
        re.IGNORECASE,
    ),
)
_M = TypeVar("_M", bound="HiringContextContract")


class HiringContextConfidence(str, Enum):
    """Trust level for a hiring-context interpretation, never a candidate claim."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class HiringContextSourceKind(str, Enum):
    JOB_DESCRIPTION = "job_description"
    ROLE_TITLE = "role_title"
    COMPANY_IDENTITY = "company_identity"
    TEAM_IDENTITY = "team_identity"
    PARENT_ORGANIZATION_IDENTITY = "parent_organization_identity"
    INTERNAL_TAXONOMY = "internal_taxonomy"
    OFFICIAL_COMPANY_WEB = "official_company_web"
    OFFICIAL_CAREERS_WEB = "official_careers_web"
    OFFICIAL_TEAM_PRODUCT_WEB = "official_team_product_web"
    OFFICIAL_PARENT_ORGANIZATION_WEB = "official_parent_organization_web"


class HiringContextSignalKind(str, Enum):
    EXPLICIT_JD = "explicit_jd"
    ROLE_FAMILY = "role_family"
    COMPANY_DOMAIN = "company_domain"
    TEAM_DOMAIN = "team_domain"
    PARENT_ORGANIZATION_DOMAIN = "parent_organization_domain"
    ENGINEERING_TRAIT = "engineering_trait"
    PREFERRED_QUALIFICATION = "preferred_qualification"


class RoleFamily(str, Enum):
    SOFTWARE_ENGINEERING = "software_engineering"
    BACKEND_ENGINEERING = "backend_engineering"
    FRONTEND_ENGINEERING = "frontend_engineering"
    FULL_STACK_ENGINEERING = "full_stack_engineering"
    DATA_ENGINEERING = "data_engineering"
    DATA_ANALYTICS = "data_analytics"
    MACHINE_LEARNING_AI = "machine_learning_ai"
    DEVOPS_CLOUD = "devops_cloud"
    GAME_DEVELOPMENT = "game_development"
    MOBILE = "mobile"
    EMBEDDED_SYSTEMS = "embedded_systems"
    SECURITY = "security"
    CONSULTING_STRATEGY = "consulting_strategy"
    GENERAL = "general"
    UNKNOWN = "unknown"


class RankingEffect(str, Enum):
    """The only dimensions a hiring-context signal may influence later."""

    EXPLICIT_ALIGNMENT = "explicit_alignment"
    ROLE_FAMILY_ALIGNMENT = "role_family_alignment"
    DOMAIN_ALIGNMENT = "domain_alignment"
    TRANSFERABLE_ENGINEERING_ALIGNMENT = "transferable_engineering_alignment"
    EMPHASIS_ONLY = "emphasis_only"


def _enum_value(value: Any, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, Enum) and not isinstance(value, enum_type):
        raise TypeError(f"{name} must use {enum_type.__name__}, not {type(value).__name__}")
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{name} must be one of: {allowed}") from exc


def _normalized_text(
    value: Any,
    name: str,
    maximum: int,
    *,
    required: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if required and not normalized:
        raise ValueError(f"{name} must not be blank")
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds maximum length {maximum}")
    if _CONTROL_RE.search(normalized):
        raise ValueError(f"{name} contains control characters")
    if any(pattern.search(normalized) for pattern in _UNSAFE_SOURCE_PATTERNS):
        raise ValueError(f"{name} contains raw or sensitive source content")
    if _CANDIDATE_AUTHORITY_ID_RE.search(normalized):
        raise ValueError(f"{name} cannot carry candidate or story authority identifiers")
    return normalized


def _optional_text(value: Any, name: str, maximum: int) -> str | None:
    if value is None:
        return None
    normalized = _normalized_text(value, name, maximum)
    return normalized or None


def _stable_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}{_stable_digest(payload)[:24]}"


def _stable_enums(
    values: Sequence[Any],
    enum_type: type[Enum],
    name: str,
    *,
    maximum: int,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(values) > maximum:
        raise ValueError(f"{name} exceeds maximum item count {maximum}")
    order = {item: index for index, item in enumerate(enum_type)}
    normalized = {_enum_value(value, enum_type, name) for value in values}
    return tuple(sorted(normalized, key=order.__getitem__))


def _stable_texts(
    values: Sequence[Any],
    name: str,
    *,
    maximum: int,
    item_maximum: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
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


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, HiringContextContract):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported hiring-context value: {type(value).__name__}")


def _decode(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is tuple:
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise TypeError("expected an array")
        return tuple(_decode(args[0], item) for item in value)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        choices = [item for item in args if item is not type(None)]
        if len(choices) != 1:
            raise TypeError("unsupported union contract")
        return _decode(choices[0], value)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and issubclass(annotation, HiringContextContract):
        if not isinstance(value, Mapping):
            raise TypeError("expected an object")
        return annotation.from_dict(value)
    return value


class HiringContextContract:
    """Strict serializer shared only by hiring-context contracts."""

    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        return {item.name: _serialize(getattr(self, item.name)) for item in fields(self)}

    @classmethod
    def from_dict(cls: type[_M], payload: Mapping[str, Any]) -> _M:
        if not isinstance(payload, Mapping):
            raise TypeError(f"{cls.__name__}.from_dict expects an object")
        if any(not isinstance(key, str) for key in payload):
            raise TypeError(f"{cls.__name__} field names must be strings")
        allowed = {item.name for item in fields(cls)}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                f"unknown {cls.__name__} fields: {', '.join(sorted(unknown))}"
            )
        hints = get_type_hints(cls)
        return cls(**{
            key: _decode(hints[key], value)
            for key, value in payload.items()
        })

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class HiringContextSourceRef(HiringContextContract):
    source_kind: HiringContextSourceKind
    source_fingerprint: str
    reference_id: str = ""

    def __post_init__(self) -> None:
        source_kind = _enum_value(self.source_kind, HiringContextSourceKind, "source_kind")
        if not isinstance(self.source_fingerprint, str):
            raise TypeError("source_fingerprint must be a string")
        source_fingerprint = self.source_fingerprint.strip().lower()
        if not _SHA256_RE.fullmatch(source_fingerprint):
            raise ValueError("source_fingerprint must be a 64-character SHA-256 digest")
        expected_id = _stable_id(
            "hiring_context_source_",
            {
                "source_kind": source_kind.value,
                "source_fingerprint": source_fingerprint,
            },
        )
        if self.reference_id not in ("", expected_id):
            raise ValueError("reference_id does not match normalized hiring-context source")
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "source_fingerprint", source_fingerprint)
        object.__setattr__(self, "reference_id", expected_id)


def _stable_source_refs(
    values: Sequence[Any],
    name: str,
    *,
    maximum: int,
    required: bool,
) -> tuple[HiringContextSourceRef, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(values) > maximum:
        raise ValueError(f"{name} exceeds maximum item count {maximum}")
    if any(not isinstance(value, HiringContextSourceRef) for value in values):
        raise TypeError(f"{name} must contain only HiringContextSourceRef values")
    normalized = {value.reference_id: value for value in values}
    if required and not normalized:
        raise ValueError(f"{name} must contain at least one hiring-context source")
    return tuple(normalized[key] for key in sorted(normalized))


@dataclass(frozen=True, slots=True)
class HiringContextSignal(HiringContextContract):
    value: str
    kind: HiringContextSignalKind
    confidence: HiringContextConfidence
    ranking_effects: tuple[RankingEffect, ...]
    source_refs: tuple[HiringContextSourceRef, ...]
    signal_id: str = ""

    def __post_init__(self) -> None:
        value = _normalized_text(
            self.value,
            "value",
            MAX_HIRING_CONTEXT_SIGNAL_VALUE_LENGTH,
            required=True,
        )
        kind = _enum_value(self.kind, HiringContextSignalKind, "kind")
        confidence = _enum_value(
            self.confidence,
            HiringContextConfidence,
            "confidence",
        )
        effects = _stable_enums(
            self.ranking_effects,
            RankingEffect,
            "ranking_effects",
            maximum=MAX_HIRING_CONTEXT_RANKING_EFFECTS,
        )
        if not effects:
            raise ValueError("ranking_effects must contain at least one allowed effect")
        source_refs = _stable_source_refs(
            self.source_refs,
            "source_refs",
            maximum=MAX_HIRING_CONTEXT_SIGNAL_SOURCE_REFS,
            required=True,
        )
        expected_id = _stable_id(
            "hiring_context_signal_",
            {
                "value": value,
                "kind": kind.value,
                "confidence": confidence.value,
                "ranking_effects": [effect.value for effect in effects],
                "source_ref_ids": [source.reference_id for source in source_refs],
            },
        )
        if self.signal_id not in ("", expected_id):
            raise ValueError("signal_id does not match normalized hiring-context signal")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "ranking_effects", effects)
        object.__setattr__(self, "source_refs", source_refs)
        object.__setattr__(self, "signal_id", expected_id)


def _stable_signals(values: Sequence[Any]) -> tuple[HiringContextSignal, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("signals must be a sequence")
    if len(values) > MAX_HIRING_CONTEXT_SIGNALS:
        raise ValueError(
            f"signals exceeds maximum item count {MAX_HIRING_CONTEXT_SIGNALS}"
        )
    if any(not isinstance(value, HiringContextSignal) for value in values):
        raise TypeError("signals must contain only HiringContextSignal values")
    normalized = {value.signal_id: value for value in values}
    return tuple(normalized[key] for key in sorted(normalized))


@dataclass(frozen=True, slots=True)
class HiringContextProfile(HiringContextContract):
    source_refs: tuple[HiringContextSourceRef, ...]
    confidence: HiringContextConfidence = HiringContextConfidence.MEDIUM
    company: str | None = None
    team: str | None = None
    parent_organization: str | None = None
    role_title: str | None = None
    primary_role_family: RoleFamily = RoleFamily.UNKNOWN
    secondary_role_families: tuple[RoleFamily, ...] = ()
    signals: tuple[HiringContextSignal, ...] = ()
    high_value_traits: tuple[str, ...] = ()
    profile_id: str = ""
    fingerprint: str = ""

    def __post_init__(self) -> None:
        source_refs = _stable_source_refs(
            self.source_refs,
            "source_refs",
            maximum=MAX_HIRING_CONTEXT_SOURCE_REFS,
            required=True,
        )
        confidence = _enum_value(
            self.confidence,
            HiringContextConfidence,
            "confidence",
        )
        company = _optional_text(
            self.company,
            "company",
            MAX_HIRING_CONTEXT_NAME_LENGTH,
        )
        team = _optional_text(
            self.team,
            "team",
            MAX_HIRING_CONTEXT_NAME_LENGTH,
        )
        parent = _optional_text(
            self.parent_organization,
            "parent_organization",
            MAX_HIRING_CONTEXT_NAME_LENGTH,
        )
        role_title = _optional_text(
            self.role_title,
            "role_title",
            MAX_HIRING_CONTEXT_ROLE_TITLE_LENGTH,
        )
        primary = _enum_value(
            self.primary_role_family,
            RoleFamily,
            "primary_role_family",
        )
        secondary = tuple(
            role
            for role in _stable_enums(
                self.secondary_role_families,
                RoleFamily,
                "secondary_role_families",
                maximum=MAX_HIRING_CONTEXT_SECONDARY_ROLE_FAMILIES,
            )
            if role is not primary
        )
        signals = _stable_signals(self.signals)
        traits = _stable_texts(
            self.high_value_traits,
            "high_value_traits",
            maximum=MAX_HIRING_CONTEXT_HIGH_VALUE_TRAITS,
            item_maximum=MAX_HIRING_CONTEXT_TRAIT_LENGTH,
        )
        if not any((company, team, parent, role_title, signals, traits)):
            raise ValueError("a hiring context requires at least one semantic descriptor")
        profile_source_ids = {source.reference_id for source in source_refs}
        signal_source_ids = {
            source.reference_id
            for signal in signals
            for source in signal.source_refs
        }
        if not signal_source_ids.issubset(profile_source_ids):
            raise ValueError("signal source_refs must exist in profile source_refs")
        fingerprint_payload = {
            "company": company,
            "team": team,
            "parent_organization": parent,
            "role_title": role_title,
            "primary_role_family": primary.value,
            "secondary_role_families": [role.value for role in secondary],
            "signals": [signal.to_dict() for signal in signals],
            "high_value_traits": list(traits),
            "source_refs": [source.to_dict() for source in source_refs],
            "confidence": confidence.value,
        }
        expected_fingerprint = _stable_digest(fingerprint_payload)
        expected_id = f"hiring_context_{expected_fingerprint[:24]}"
        if self.fingerprint not in ("", expected_fingerprint):
            raise ValueError("fingerprint does not match normalized hiring context")
        if self.profile_id not in ("", expected_id):
            raise ValueError("profile_id does not match normalized hiring context")
        object.__setattr__(self, "source_refs", source_refs)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "company", company)
        object.__setattr__(self, "team", team)
        object.__setattr__(self, "parent_organization", parent)
        object.__setattr__(self, "role_title", role_title)
        object.__setattr__(self, "primary_role_family", primary)
        object.__setattr__(self, "secondary_role_families", secondary)
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "high_value_traits", traits)
        object.__setattr__(self, "profile_id", expected_id)
        object.__setattr__(self, "fingerprint", expected_fingerprint)


__all__ = [
    "HiringContextConfidence",
    "HiringContextProfile",
    "HiringContextSignal",
    "HiringContextSignalKind",
    "HiringContextSourceKind",
    "HiringContextSourceRef",
    "MAX_HIRING_CONTEXT_HIGH_VALUE_TRAITS",
    "MAX_HIRING_CONTEXT_NAME_LENGTH",
    "MAX_HIRING_CONTEXT_RANKING_EFFECTS",
    "MAX_HIRING_CONTEXT_ROLE_TITLE_LENGTH",
    "MAX_HIRING_CONTEXT_SECONDARY_ROLE_FAMILIES",
    "MAX_HIRING_CONTEXT_SIGNALS",
    "MAX_HIRING_CONTEXT_SIGNAL_SOURCE_REFS",
    "MAX_HIRING_CONTEXT_SIGNAL_VALUE_LENGTH",
    "MAX_HIRING_CONTEXT_SOURCE_REFS",
    "MAX_HIRING_CONTEXT_TRAIT_LENGTH",
    "RankingEffect",
    "RoleFamily",
]
