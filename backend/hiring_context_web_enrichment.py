"""Immutable official-web hiring-context enrichment and deterministic extraction.

The offline :class:`HiringContextProfile` remains authoritative and untouched.
This module produces a separately bound artifact whose provenance cannot be
used as candidate, project, Story, capability, or repository evidence.  Web
signals are not consumed by ranking in this capability.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import json
import os
import re
import types
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from backend.hiring_context_models import (
    MAX_HIRING_CONTEXT_NAME_LENGTH,
    MAX_HIRING_CONTEXT_SIGNAL_SOURCE_REFS,
    MAX_HIRING_CONTEXT_SIGNAL_VALUE_LENGTH,
    HiringContextConfidence,
    HiringContextProfile,
    HiringContextSignalKind,
    HiringContextSourceRef,
)
from backend.hiring_context_official_sources import (
    DEFAULT_OFFICIAL_HIRING_CONTEXT_SOURCE_REGISTRY,
    OfficialHiringContextSourceKind,
    OfficialHiringContextSourceRegistry,
    OfficialSourceSpecificity,
    OfficialSourceTrust,
    resolve_official_source_candidates,
)
from backend.hiring_context_web_client import (
    BoundedOfficialSourceDocument,
    OfficialSourceAcquisition,
    fetch_official_hiring_context_sources,
)


OFFICIAL_HIRING_CONTEXT_ENRICHMENT_FLAG = "USE_OFFICIAL_HIRING_CONTEXT_ENRICHMENT"
MAX_HIRING_CONTEXT_WEB_SOURCE_REFS = 8
MAX_HIRING_CONTEXT_WEB_SIGNALS = 16
MAX_HIRING_CONTEXT_WEB_CONFLICTS = 4

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]*(?::[a-z|]+)?$")
_CANDIDATE_AUTHORITY_ID_RE = re.compile(
    r"(?<![a-z0-9])(?:pei|pef|pcf|pcb|pem|esr|raw|chk|hyb|capability_fact|"
    r"engineering_story|engineering_story_revision)_[a-z0-9][a-z0-9_.:-]*",
    re.IGNORECASE,
)
_M = TypeVar("_M", bound="HiringContextWebContract")


class HiringContextEnrichmentStatus(str, Enum):
    DISABLED = "disabled"
    NO_OFFICIAL_SOURCE = "no_official_source"
    READY = "ready"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    CONFLICTED = "conflicted"


class HiringContextEnrichmentCachePolicy(str, Enum):
    NONE = "none"


def _stable_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _text(value: Any, name: str, maximum: int, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if required and not normalized:
        raise ValueError(f"{name} must not be blank")
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds maximum length {maximum}")
    if _CANDIDATE_AUTHORITY_ID_RE.search(normalized):
        raise ValueError(f"{name} cannot carry candidate or Story authority identifiers")
    return normalized


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, HiringContextWebContract):
        return value.to_dict()
    if isinstance(value, HiringContextSourceRef):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported web-enrichment value: {type(value).__name__}")


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
        choices = [choice for choice in args if choice is not type(None)]
        if len(choices) != 1:
            raise TypeError("unsupported union contract")
        return _decode(choices[0], value)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if annotation is HiringContextSourceRef:
        if not isinstance(value, Mapping):
            raise TypeError("expected a hiring-context source object")
        return HiringContextSourceRef.from_dict(value)
    if isinstance(annotation, type) and issubclass(annotation, HiringContextWebContract):
        if not isinstance(value, Mapping):
            raise TypeError("expected an object")
        return annotation.from_dict(value)
    return value


class HiringContextWebContract:
    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        return {item.name: _serialize(getattr(self, item.name)) for item in fields(self)}

    @classmethod
    def from_dict(cls: type[_M], payload: Mapping[str, Any]) -> _M:
        if not isinstance(payload, Mapping) or any(not isinstance(key, str) for key in payload):
            raise TypeError(f"{cls.__name__}.from_dict expects an object")
        allowed = {item.name for item in fields(cls)}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown {cls.__name__} fields: {', '.join(sorted(unknown))}")
        hints = get_type_hints(cls)
        return cls(**{key: _decode(hints[key], value) for key, value in payload.items()})

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class HiringContextOfficialSourceRef(HiringContextWebContract):
    normalized_url: str
    source_kind: OfficialHiringContextSourceKind
    canonical_organization: str
    content_fingerprint: str
    specificity: OfficialSourceSpecificity
    trust: OfficialSourceTrust = OfficialSourceTrust.VERIFIED_OFFICIAL
    source_id: str = ""

    def __post_init__(self) -> None:
        url = _text(self.normalized_url, "normalized_url", 2_048)
        organization = _text(
            self.canonical_organization,
            "canonical_organization",
            MAX_HIRING_CONTEXT_NAME_LENGTH,
        )
        try:
            kind = OfficialHiringContextSourceKind(self.source_kind)
            specificity = OfficialSourceSpecificity(self.specificity)
            trust = OfficialSourceTrust(self.trust)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid official source classification") from exc
        if not isinstance(self.content_fingerprint, str) or not _SHA256_RE.fullmatch(
            self.content_fingerprint
        ):
            raise ValueError("content_fingerprint must be a SHA-256 digest")
        payload = {
            "normalized_url": url,
            "source_kind": kind.value,
            "canonical_organization": organization,
            "content_fingerprint": self.content_fingerprint,
            "specificity": int(specificity),
            "trust": trust.value,
        }
        expected_id = f"hiring_context_official_source_{_stable_digest(payload)[:24]}"
        if self.source_id not in ("", expected_id):
            raise ValueError("source_id does not match official source content")
        object.__setattr__(self, "normalized_url", url)
        object.__setattr__(self, "source_kind", kind)
        object.__setattr__(self, "canonical_organization", organization)
        object.__setattr__(self, "specificity", specificity)
        object.__setattr__(self, "trust", trust)
        object.__setattr__(self, "source_id", expected_id)


def _official_source_refs(
    values: Sequence[HiringContextOfficialSourceRef],
    *,
    required: bool,
) -> tuple[HiringContextOfficialSourceRef, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("official_source_refs must be a sequence")
    if len(values) > MAX_HIRING_CONTEXT_WEB_SOURCE_REFS:
        raise ValueError("official_source_refs exceeds its bound")
    if any(not isinstance(item, HiringContextOfficialSourceRef) for item in values):
        raise TypeError("official_source_refs contains an invalid value")
    normalized = {item.source_id: item for item in values}
    if required and not normalized:
        raise ValueError("official_source_refs must not be empty")
    return tuple(normalized[key] for key in sorted(normalized))


def _offline_source_refs(
    values: Sequence[HiringContextSourceRef],
) -> tuple[HiringContextSourceRef, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("offline_source_refs must be a sequence")
    if len(values) > MAX_HIRING_CONTEXT_SIGNAL_SOURCE_REFS:
        raise ValueError("offline_source_refs exceeds its bound")
    if any(not isinstance(item, HiringContextSourceRef) for item in values):
        raise TypeError("offline_source_refs contains an invalid value")
    normalized = {item.reference_id: item for item in values}
    return tuple(normalized[key] for key in sorted(normalized))


@dataclass(frozen=True, slots=True)
class HiringContextWebSignal(HiringContextWebContract):
    value: str
    kind: HiringContextSignalKind
    confidence: HiringContextConfidence
    official_source_refs: tuple[HiringContextOfficialSourceRef, ...]
    offline_source_refs: tuple[HiringContextSourceRef, ...] = ()
    signal_id: str = ""

    def __post_init__(self) -> None:
        value = _text(
            self.value,
            "value",
            MAX_HIRING_CONTEXT_SIGNAL_VALUE_LENGTH,
        )
        try:
            kind = HiringContextSignalKind(self.kind)
            confidence = HiringContextConfidence(self.confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid hiring-context web signal classification") from exc
        allowed_kinds = {
            HiringContextSignalKind.COMPANY_DOMAIN,
            HiringContextSignalKind.TEAM_DOMAIN,
            HiringContextSignalKind.PARENT_ORGANIZATION_DOMAIN,
            HiringContextSignalKind.ENGINEERING_TRAIT,
        }
        if kind not in allowed_kinds:
            raise ValueError("web signal kind is outside the context-only contract")
        official = _official_source_refs(self.official_source_refs, required=True)
        offline = _offline_source_refs(self.offline_source_refs)
        payload = {
            "value": value,
            "kind": kind.value,
            "confidence": confidence.value,
            "official_source_ids": [item.source_id for item in official],
            "offline_source_ids": [item.reference_id for item in offline],
        }
        expected_id = f"hiring_context_web_signal_{_stable_digest(payload)[:24]}"
        if self.signal_id not in ("", expected_id):
            raise ValueError("signal_id does not match normalized web signal")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "official_source_refs", official)
        object.__setattr__(self, "offline_source_refs", offline)
        object.__setattr__(self, "signal_id", expected_id)


@dataclass(frozen=True, slots=True)
class HiringContextEnrichment(HiringContextWebContract):
    offline_profile_id: str
    offline_profile_fingerprint: str
    status: HiringContextEnrichmentStatus
    official_source_refs: tuple[HiringContextOfficialSourceRef, ...] = ()
    signals: tuple[HiringContextWebSignal, ...] = ()
    conflicts: tuple[str, ...] = ()
    cache_policy: HiringContextEnrichmentCachePolicy = HiringContextEnrichmentCachePolicy.NONE
    company: str | None = None
    team: str | None = None
    parent_organization: str | None = None
    enrichment_id: str = ""
    fingerprint: str = ""

    def __post_init__(self) -> None:
        profile_id = _text(self.offline_profile_id, "offline_profile_id", 100)
        if not isinstance(self.offline_profile_fingerprint, str) or not _SHA256_RE.fullmatch(
            self.offline_profile_fingerprint
        ):
            raise ValueError("offline_profile_fingerprint must be a SHA-256 digest")
        try:
            status = HiringContextEnrichmentStatus(self.status)
            cache_policy = HiringContextEnrichmentCachePolicy(self.cache_policy)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid enrichment state") from exc
        source_refs = _official_source_refs(self.official_source_refs, required=False)
        if isinstance(self.signals, (str, bytes)) or not isinstance(self.signals, Sequence):
            raise TypeError("signals must be a sequence")
        if len(self.signals) > MAX_HIRING_CONTEXT_WEB_SIGNALS:
            raise ValueError("signals exceeds its bound")
        if any(not isinstance(item, HiringContextWebSignal) for item in self.signals):
            raise TypeError("signals contains an invalid value")
        signals = tuple(sorted(
            {item.signal_id: item for item in self.signals}.values(),
            key=lambda item: (item.value.casefold(), item.kind.value, item.signal_id),
        ))
        source_ids = {item.source_id for item in source_refs}
        if any(
            reference.source_id not in source_ids
            for signal in signals
            for reference in signal.official_source_refs
        ):
            raise ValueError("signal source refs must exist at enrichment scope")
        if isinstance(self.conflicts, (str, bytes)) or not isinstance(self.conflicts, Sequence):
            raise TypeError("conflicts must be a sequence")
        if len(self.conflicts) > MAX_HIRING_CONTEXT_WEB_CONFLICTS:
            raise ValueError("conflicts exceeds its bound")
        conflicts = tuple(sorted({_text(item, "conflict", 120) for item in self.conflicts}))
        if any(not _TOKEN_RE.fullmatch(item) for item in conflicts):
            raise ValueError("conflicts must use bounded semantic tokens")
        empty_statuses = {
            HiringContextEnrichmentStatus.DISABLED,
            HiringContextEnrichmentStatus.NO_OFFICIAL_SOURCE,
            HiringContextEnrichmentStatus.UNAVAILABLE,
        }
        if status in empty_statuses and (source_refs or signals or conflicts):
            raise ValueError("inactive enrichment states cannot carry web evidence")
        if status is HiringContextEnrichmentStatus.CONFLICTED and (signals or not conflicts):
            raise ValueError("conflicted enrichment must fail closed without signals")
        if status is not HiringContextEnrichmentStatus.CONFLICTED and conflicts:
            raise ValueError("only conflicted enrichment may carry conflicts")
        company = None if self.company is None else _text(
            self.company, "company", MAX_HIRING_CONTEXT_NAME_LENGTH, required=False
        ) or None
        team = None if self.team is None else _text(
            self.team, "team", MAX_HIRING_CONTEXT_NAME_LENGTH, required=False
        ) or None
        parent = None if self.parent_organization is None else _text(
            self.parent_organization,
            "parent_organization",
            MAX_HIRING_CONTEXT_NAME_LENGTH,
            required=False,
        ) or None
        payload = {
            "offline_profile_id": profile_id,
            "offline_profile_fingerprint": self.offline_profile_fingerprint,
            "status": status.value,
            "official_source_refs": [item.to_dict() for item in source_refs],
            "signals": [item.to_dict() for item in signals],
            "conflicts": list(conflicts),
            "cache_policy": cache_policy.value,
            "company": company,
            "team": team,
            "parent_organization": parent,
        }
        expected_fingerprint = _stable_digest(payload)
        expected_id = f"hiring_context_enrichment_{expected_fingerprint[:24]}"
        if self.fingerprint not in ("", expected_fingerprint):
            raise ValueError("fingerprint does not match enrichment content")
        if self.enrichment_id not in ("", expected_id):
            raise ValueError("enrichment_id does not match enrichment content")
        object.__setattr__(self, "offline_profile_id", profile_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "official_source_refs", source_refs)
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "conflicts", conflicts)
        object.__setattr__(self, "cache_policy", cache_policy)
        object.__setattr__(self, "company", company)
        object.__setattr__(self, "team", team)
        object.__setattr__(self, "parent_organization", parent)
        object.__setattr__(self, "fingerprint", expected_fingerprint)
        object.__setattr__(self, "enrichment_id", expected_id)


@dataclass(frozen=True, slots=True)
class _SemanticRule:
    value: str
    patterns: tuple[re.Pattern[str], ...]
    family: str | None


def _patterns(*values: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(value, re.IGNORECASE) for value in values)


_SEMANTIC_RULES = (
    _SemanticRule(
        "Game development",
        _patterns(
            r"\bgame development\b",
            r"\bdevelop(?:s|ed|ing)?\s+(?:[a-z0-9-]+\s+){0,3}(?:video\s+)?games?\b",
            r"\bgame studio\b",
        ),
        "game",
    ),
    _SemanticRule(
        "Real-time interactive software",
        _patterns(r"\breal[- ]time interactive (?:software|systems?|experiences?)\b"),
        None,
    ),
    _SemanticRule(
        "Online game systems",
        _patterns(r"\bonline game systems?\b", r"\bonline systems?\b", r"\bmultiplayer systems?\b"),
        None,
    ),
    _SemanticRule(
        "Software platforms",
        _patterns(r"\bsoftware platforms?\b"),
        "platform",
    ),
    _SemanticRule(
        "Developer platform",
        _patterns(r"\bdeveloper platforms?\b"),
        "platform",
    ),
    _SemanticRule(
        "Data infrastructure",
        _patterns(r"\bdata infrastructure\b", r"\bdata platforms?\b"),
        "data",
    ),
    _SemanticRule(
        "Commerce platform",
        _patterns(r"\bcommerce platforms?\b"),
        "platform",
    ),
    _SemanticRule(
        "Security engineering",
        _patterns(r"\bsecurity engineering\b"),
        "security",
    ),
)


@dataclass(frozen=True, slots=True)
class _Match:
    value: str
    family: str | None
    kind: HiringContextSignalKind
    confidence: HiringContextConfidence
    source_ref: HiringContextOfficialSourceRef


def _source_ref(document: BoundedOfficialSourceDocument) -> HiringContextOfficialSourceRef:
    return HiringContextOfficialSourceRef(
        normalized_url=document.normalized_url,
        source_kind=document.source.source_kind,
        canonical_organization=document.source.canonical_organization,
        content_fingerprint=document.content_fingerprint,
        specificity=document.source.specificity,
        trust=document.source.trust,
    )


def _signal_kind(source_kind: OfficialHiringContextSourceKind) -> HiringContextSignalKind:
    if source_kind in {
        OfficialHiringContextSourceKind.OFFICIAL_TEAM_PAGE,
        OfficialHiringContextSourceKind.OFFICIAL_PRODUCT_PAGE,
    }:
        return HiringContextSignalKind.TEAM_DOMAIN
    if source_kind is OfficialHiringContextSourceKind.OFFICIAL_PARENT_ORG_PAGE:
        return HiringContextSignalKind.PARENT_ORGANIZATION_DOMAIN
    return HiringContextSignalKind.COMPANY_DOMAIN


def _confidence(specificity: OfficialSourceSpecificity) -> HiringContextConfidence:
    if specificity is OfficialSourceSpecificity.TEAM_PRODUCT:
        return HiringContextConfidence.HIGH
    return HiringContextConfidence.MEDIUM


def _empty_enrichment(
    profile: HiringContextProfile,
    status: HiringContextEnrichmentStatus,
) -> HiringContextEnrichment:
    return HiringContextEnrichment(
        offline_profile_id=profile.profile_id,
        offline_profile_fingerprint=profile.fingerprint,
        status=status,
        company=profile.company,
        team=profile.team,
        parent_organization=profile.parent_organization,
    )


def extract_hiring_context_web_enrichment(
    profile: HiringContextProfile,
    acquisition: OfficialSourceAcquisition,
) -> HiringContextEnrichment:
    """Purely extract allowlisted context semantics from bounded documents."""

    if not isinstance(profile, HiringContextProfile):
        raise TypeError("profile must be a HiringContextProfile")
    if not isinstance(acquisition, OfficialSourceAcquisition):
        raise TypeError("acquisition must be an OfficialSourceAcquisition")
    source_refs = tuple(_source_ref(document) for document in acquisition.documents)
    matches: list[_Match] = []
    for document, source_ref in zip(acquisition.documents, source_refs, strict=True):
        content = f"{document.title}\n{document.text}"
        for rule in _SEMANTIC_RULES:
            if any(pattern.search(content) for pattern in rule.patterns):
                matches.append(
                    _Match(
                        value=rule.value,
                        family=rule.family,
                        kind=_signal_kind(document.source.source_kind),
                        confidence=_confidence(document.source.specificity),
                        source_ref=source_ref,
                    )
                )

    primary = [item for item in matches if item.family is not None]
    best_primary_specificity = min(
        (int(item.source_ref.specificity) for item in primary),
        default=None,
    )
    highest_primary = [
        item
        for item in primary
        if int(item.source_ref.specificity) == best_primary_specificity
    ]
    primary_families = sorted({item.family for item in highest_primary if item.family})
    if len(primary_families) > 1:
        return HiringContextEnrichment(
            offline_profile_id=profile.profile_id,
            offline_profile_fingerprint=profile.fingerprint,
            status=HiringContextEnrichmentStatus.CONFLICTED,
            official_source_refs=source_refs,
            conflicts=(f"primary_domain:{'|'.join(primary_families)}",),
            company=profile.company,
            team=profile.team,
            parent_organization=profile.parent_organization,
        )
    effective = [
        item
        for item in matches
        if item.family is None
        or best_primary_specificity is None
        or int(item.source_ref.specificity) == best_primary_specificity
    ]
    grouped: dict[str, list[_Match]] = {}
    for item in effective:
        grouped.setdefault(item.value.casefold(), []).append(item)
    offline_by_value: dict[str, tuple[HiringContextSourceRef, ...]] = {}
    for signal in profile.signals:
        offline_by_value.setdefault(signal.value.casefold(), signal.source_refs)
    signals = []
    confidence_order = {
        HiringContextConfidence.LOW: 0,
        HiringContextConfidence.MEDIUM: 1,
        HiringContextConfidence.HIGH: 2,
    }
    for key in sorted(grouped):
        group = grouped[key]
        representative = min(
            group,
            key=lambda item: (
                int(item.source_ref.specificity),
                -confidence_order[item.confidence],
                item.kind.value,
                item.source_ref.source_id,
            ),
        )
        signals.append(
            HiringContextWebSignal(
                value=representative.value,
                kind=representative.kind,
                confidence=max(
                    (item.confidence for item in group),
                    key=confidence_order.__getitem__,
                ),
                official_source_refs=tuple(item.source_ref for item in group),
                offline_source_refs=offline_by_value.get(key, ()),
            )
        )
    status = (
        HiringContextEnrichmentStatus.PARTIAL
        if acquisition.failures and acquisition.documents
        else HiringContextEnrichmentStatus.READY
    )
    return HiringContextEnrichment(
        offline_profile_id=profile.profile_id,
        offline_profile_fingerprint=profile.fingerprint,
        status=status,
        official_source_refs=source_refs,
        signals=tuple(signals),
        company=profile.company,
        team=profile.team,
        parent_organization=profile.parent_organization,
    )


def official_hiring_context_enrichment_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if environ is None else environ
    value = source.get(OFFICIAL_HIRING_CONTEXT_ENRICHMENT_FLAG, "")
    return isinstance(value, str) and value.strip().casefold() in {"1", "true", "yes", "on"}


Acquirer = Callable[[Sequence[Any]], OfficialSourceAcquisition]
Extractor = Callable[[HiringContextProfile, OfficialSourceAcquisition], HiringContextEnrichment]


def build_hiring_context_web_enrichment(
    profile: HiringContextProfile,
    *,
    enabled: bool | None = None,
    environ: Mapping[str, str] | None = None,
    registry: OfficialHiringContextSourceRegistry = DEFAULT_OFFICIAL_HIRING_CONTEXT_SOURCE_REGISTRY,
    acquirer: Acquirer = fetch_official_hiring_context_sources,
    extractor: Extractor = extract_hiring_context_web_enrichment,
) -> HiringContextEnrichment:
    """Orchestrate optional enrichment without mutating or replacing offline context."""

    if not isinstance(profile, HiringContextProfile):
        raise TypeError("profile must be a HiringContextProfile")
    if enabled is not None and not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool or None")
    active = official_hiring_context_enrichment_enabled(environ) if enabled is None else enabled
    if not active:
        return _empty_enrichment(profile, HiringContextEnrichmentStatus.DISABLED)
    candidates = resolve_official_source_candidates(profile, registry=registry)
    if not candidates:
        return _empty_enrichment(profile, HiringContextEnrichmentStatus.NO_OFFICIAL_SOURCE)
    try:
        acquisition = acquirer(candidates)
        if not isinstance(acquisition, OfficialSourceAcquisition):
            raise TypeError("acquirer returned an invalid result")
    except Exception:
        return _empty_enrichment(profile, HiringContextEnrichmentStatus.UNAVAILABLE)
    if not acquisition.documents:
        return _empty_enrichment(profile, HiringContextEnrichmentStatus.UNAVAILABLE)
    try:
        enrichment = extractor(profile, acquisition)
    except Exception:
        return _empty_enrichment(profile, HiringContextEnrichmentStatus.UNAVAILABLE)
    if (
        enrichment.offline_profile_id != profile.profile_id
        or enrichment.offline_profile_fingerprint != profile.fingerprint
    ):
        return _empty_enrichment(profile, HiringContextEnrichmentStatus.UNAVAILABLE)
    return enrichment


__all__ = [
    "HiringContextEnrichment",
    "HiringContextEnrichmentCachePolicy",
    "HiringContextEnrichmentStatus",
    "HiringContextOfficialSourceRef",
    "HiringContextWebSignal",
    "MAX_HIRING_CONTEXT_WEB_CONFLICTS",
    "MAX_HIRING_CONTEXT_WEB_SIGNALS",
    "MAX_HIRING_CONTEXT_WEB_SOURCE_REFS",
    "OFFICIAL_HIRING_CONTEXT_ENRICHMENT_FLAG",
    "build_hiring_context_web_enrichment",
    "extract_hiring_context_web_enrichment",
    "official_hiring_context_enrichment_enabled",
]
