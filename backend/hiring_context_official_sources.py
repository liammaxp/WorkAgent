"""Exact official-source policy for optional hiring-context enrichment.

The registry in this module is deliberately small.  It binds normalized
organization identities to explicit public HTTPS pages and domains; a caller
cannot turn an arbitrary URL into an accepted source by supplying an employer
name.  DNS resolution is injected so the same fail-closed policy can be tested
without live network access.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import ipaddress
import socket
import types
import unicodedata
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

from backend.hiring_context_models import HiringContextProfile


MAX_OFFICIAL_SOURCE_RULES = 12
MAX_OFFICIAL_SOURCE_ALIASES = 8
MAX_OFFICIAL_DOMAINS_PER_RULE = 4
MAX_OFFICIAL_SOURCE_CANDIDATES_PER_RULE = 4
MAX_OFFICIAL_SOURCE_URL_LENGTH = 2_048


class OfficialHiringContextSourceKind(str, Enum):
    OFFICIAL_TEAM_PAGE = "official_team_page"
    OFFICIAL_PRODUCT_PAGE = "official_product_page"
    OFFICIAL_COMPANY_PAGE = "official_company_page"
    OFFICIAL_ENGINEERING_PAGE = "official_engineering_page"
    OFFICIAL_CAREERS_PAGE = "official_careers_page"
    OFFICIAL_PARENT_ORG_PAGE = "official_parent_org_page"
    OFFICIAL_ORG_GITHUB = "official_org_github"


class OfficialSourceSpecificity(IntEnum):
    """Lower numeric values are more specific and take precedence."""

    TEAM_PRODUCT = 1
    COMPANY_ENGINEERING = 2
    CAREERS = 3
    COMPANY = 4
    PARENT_ORGANIZATION = 5
    ORGANIZATION_GITHUB = 6


class OfficialSourceTrust(str, Enum):
    VERIFIED_OFFICIAL = "verified_official"


def _text(value: Any, name: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if not normalized:
        raise ValueError(f"{name} must not be blank")
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds maximum length {maximum}")
    return normalized


def _identity_key(value: str) -> str:
    return _text(value, "organization identity").casefold()


def _host(value: Any) -> str:
    host = _text(value, "official domain", 253).casefold().rstrip(".")
    if ":" in host or "/" in host or "@" in host or not host:
        raise ValueError("official domain must be a DNS hostname")
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("official domain is malformed") from exc
    try:
        ipaddress.ip_address(ascii_host)
    except ValueError:
        pass
    else:
        raise ValueError("official domains cannot be IP literals")
    labels = ascii_host.split(".")
    if len(labels) < 2 or any(
        not label
        or len(label) > 63
        or label.startswith("-")
        or label.endswith("-")
        or not all(character.isalnum() or character == "-" for character in label)
        for label in labels
    ):
        raise ValueError("official domain is malformed")
    return ascii_host


def _stable_texts(
    values: Sequence[Any],
    name: str,
    *,
    maximum: int,
    normalize,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(values) > maximum:
        raise ValueError(f"{name} exceeds maximum item count {maximum}")
    normalized = {normalize(value) for value in values}
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class OfficialSourceCandidate:
    canonical_organization: str
    url: str
    source_kind: OfficialHiringContextSourceKind
    specificity: OfficialSourceSpecificity
    trust: OfficialSourceTrust = OfficialSourceTrust.VERIFIED_OFFICIAL
    allowed_domains: tuple[str, ...] = ()
    allow_subdomains: bool = False

    def __post_init__(self) -> None:
        organization = _text(self.canonical_organization, "canonical_organization")
        try:
            source_kind = OfficialHiringContextSourceKind(self.source_kind)
            specificity = OfficialSourceSpecificity(self.specificity)
            trust = OfficialSourceTrust(self.trust)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid official source classification") from exc
        domains = _stable_texts(
            self.allowed_domains,
            "allowed_domains",
            maximum=MAX_OFFICIAL_DOMAINS_PER_RULE,
            normalize=_host,
        )
        if not domains:
            raise ValueError("allowed_domains must contain an official domain")
        normalized_url = _normalize_url_structure(self.url)
        parsed = urlsplit(normalized_url)
        if not any(_host_matches(parsed.hostname or "", item, self.allow_subdomains) for item in domains):
            raise ValueError("source URL is not bound to an allowed official domain")
        if source_kind is OfficialHiringContextSourceKind.OFFICIAL_ORG_GITHUB:
            if (parsed.hostname or "").casefold() != "github.com" or len(
                [part for part in parsed.path.split("/") if part]
            ) != 1:
                raise ValueError("official GitHub sources require one verified organization path")
        object.__setattr__(self, "canonical_organization", organization)
        object.__setattr__(self, "url", normalized_url)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "specificity", specificity)
        object.__setattr__(self, "trust", trust)
        object.__setattr__(self, "allowed_domains", domains)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_organization": self.canonical_organization,
            "url": self.url,
            "source_kind": self.source_kind.value,
            "specificity": int(self.specificity),
            "trust": self.trust.value,
            "allowed_domains": list(self.allowed_domains),
            "allow_subdomains": self.allow_subdomains,
        }


@dataclass(frozen=True, slots=True)
class OfficialOrganizationSourceRule:
    canonical_name: str
    aliases: tuple[str, ...] = ()
    parent_organization: str | None = None
    sources: tuple[OfficialSourceCandidate, ...] = ()

    def __post_init__(self) -> None:
        canonical = _text(self.canonical_name, "canonical_name")
        aliases = _stable_texts(
            self.aliases,
            "aliases",
            maximum=MAX_OFFICIAL_SOURCE_ALIASES,
            normalize=lambda value: _text(value, "alias"),
        )
        aliases = tuple(alias for alias in aliases if _identity_key(alias) != _identity_key(canonical))
        parent = None if self.parent_organization is None else _text(
            self.parent_organization, "parent_organization"
        )
        if isinstance(self.sources, (str, bytes)) or not isinstance(self.sources, Sequence):
            raise TypeError("sources must be a sequence")
        if len(self.sources) > MAX_OFFICIAL_SOURCE_CANDIDATES_PER_RULE:
            raise ValueError("sources exceeds the per-organization bound")
        if any(not isinstance(source, OfficialSourceCandidate) for source in self.sources):
            raise TypeError("sources must contain only OfficialSourceCandidate values")
        sources = {source.url: source for source in self.sources}
        if any(_identity_key(source.canonical_organization) != _identity_key(canonical) for source in sources.values()):
            raise ValueError("source canonical organization must match its registry rule")
        object.__setattr__(self, "canonical_name", canonical)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "parent_organization", parent)
        object.__setattr__(
            self,
            "sources",
            tuple(sorted(sources.values(), key=_candidate_sort_key)),
        )

    def identity_keys(self) -> tuple[str, ...]:
        return tuple(sorted({_identity_key(self.canonical_name), *map(_identity_key, self.aliases)}))


@dataclass(frozen=True, slots=True)
class OfficialHiringContextSourceRegistry:
    rules: tuple[OfficialOrganizationSourceRule, ...]
    _lookup: Mapping[str, OfficialOrganizationSourceRule] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if isinstance(self.rules, (str, bytes)) or not isinstance(self.rules, Sequence):
            raise TypeError("rules must be a sequence")
        if len(self.rules) > MAX_OFFICIAL_SOURCE_RULES:
            raise ValueError("official source registry exceeds its rule bound")
        if any(not isinstance(rule, OfficialOrganizationSourceRule) for rule in self.rules):
            raise TypeError("rules must contain only OfficialOrganizationSourceRule values")
        rules = {_identity_key(rule.canonical_name): rule for rule in self.rules}
        if len(rules) != len(self.rules):
            raise ValueError("official source registry has duplicate canonical organizations")
        lookup: dict[str, OfficialOrganizationSourceRule] = {}
        for rule in rules.values():
            for key in rule.identity_keys():
                existing = lookup.get(key)
                if existing is not None and existing != rule:
                    raise ValueError(f"conflicting official organization alias: {key}")
                lookup[key] = rule
        for rule in rules.values():
            if rule.parent_organization and _identity_key(rule.parent_organization) not in lookup:
                raise ValueError(f"unknown official parent organization: {rule.parent_organization}")
            seen: set[str] = set()
            current = rule
            while current.parent_organization:
                key = _identity_key(current.canonical_name)
                if key in seen:
                    raise ValueError("official organization parents must not contain cycles")
                seen.add(key)
                current = lookup[_identity_key(current.parent_organization)]
        object.__setattr__(self, "rules", tuple(rules[key] for key in sorted(rules)))
        object.__setattr__(self, "_lookup", types.MappingProxyType(lookup))

    def match(self, identity: str | None) -> OfficialOrganizationSourceRule | None:
        if identity is None:
            return None
        try:
            return self._lookup.get(_identity_key(identity))
        except (TypeError, ValueError):
            return None

    def lineage(self, identity: str | None) -> tuple[OfficialOrganizationSourceRule, ...]:
        rule = self.match(identity)
        if rule is None:
            return ()
        lineage = []
        while rule is not None:
            lineage.append(rule)
            rule = self.match(rule.parent_organization)
        return tuple(lineage)


class OfficialSourceUrlError(ValueError):
    """Sanitized fail-closed URL policy error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


DnsResolver = Callable[[str], Sequence[str]]


def _normalize_url_structure(url: Any) -> str:
    if not isinstance(url, str) or not url or len(url) > MAX_OFFICIAL_SOURCE_URL_LENGTH:
        raise OfficialSourceUrlError("malformed_url")
    if any(character.isspace() for character in url):
        raise OfficialSourceUrlError("malformed_url")
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise OfficialSourceUrlError("malformed_url") from exc
    if parsed.scheme.casefold() != "https":
        raise OfficialSourceUrlError("unsupported_scheme")
    if parsed.username is not None or parsed.password is not None:
        raise OfficialSourceUrlError("embedded_credentials")
    try:
        port = parsed.port
        host = parsed.hostname
    except ValueError as exc:
        raise OfficialSourceUrlError("malformed_url") from exc
    if not host or port not in (None, 443) or parsed.fragment:
        raise OfficialSourceUrlError("malformed_url")
    try:
        normalized_host = host.casefold().rstrip(".").encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise OfficialSourceUrlError("malformed_url") from exc
    forbidden_names = {
        "instance-data",
        "localhost",
        "localhost.localdomain",
        "metadata",
        "metadata.google.internal",
    }
    if (
        not normalized_host
        or normalized_host in forbidden_names
        or normalized_host.endswith((".localhost", ".local", ".internal", ".home.arpa"))
    ):
        raise OfficialSourceUrlError("forbidden_destination")
    try:
        ipaddress.ip_address(normalized_host)
    except ValueError:
        pass
    else:
        raise OfficialSourceUrlError("forbidden_destination")
    netloc = normalized_host
    path = parsed.path or "/"
    return urlunsplit(SplitResult("https", netloc, path, parsed.query, ""))


def _host_matches(host: str, official_domain: str, allow_subdomains: bool) -> bool:
    normalized = host.casefold().rstrip(".")
    return normalized == official_domain or (
        allow_subdomains and normalized.endswith(f".{official_domain}")
    )


def default_public_dns_resolver(host: str) -> tuple[str, ...]:
    try:
        results = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise OfficialSourceUrlError("dns_unavailable") from exc
    addresses = tuple(sorted({result[4][0] for result in results}))
    if not addresses:
        raise OfficialSourceUrlError("dns_unavailable")
    return addresses


def _require_public_addresses(addresses: Sequence[str]) -> None:
    if isinstance(addresses, (str, bytes)) or not isinstance(addresses, Sequence) or not addresses:
        raise OfficialSourceUrlError("dns_unavailable")
    for value in addresses:
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise OfficialSourceUrlError("dns_unavailable") from exc
        if (
            not address.is_global
            or address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
            or (
                isinstance(address, ipaddress.IPv6Address)
                and address.ipv4_mapped is not None
            )
        ):
            raise OfficialSourceUrlError("forbidden_destination")


def validate_official_source_url(
    candidate: OfficialSourceCandidate,
    url: str,
    *,
    resolver: DnsResolver = default_public_dns_resolver,
) -> str:
    """Normalize and validate one registry-bound URL immediately before I/O."""

    if not isinstance(candidate, OfficialSourceCandidate):
        raise TypeError("candidate must be an OfficialSourceCandidate")
    normalized = _normalize_url_structure(url)
    host = urlsplit(normalized).hostname or ""
    if not any(
        _host_matches(host, domain, candidate.allow_subdomains)
        for domain in candidate.allowed_domains
    ):
        raise OfficialSourceUrlError("unregistered_official_domain")
    if candidate.source_kind is OfficialHiringContextSourceKind.OFFICIAL_ORG_GITHUB:
        original_path = [part for part in urlsplit(candidate.url).path.split("/") if part]
        target_path = [part for part in urlsplit(normalized).path.split("/") if part]
        if not original_path or not target_path or target_path[0].casefold() != original_path[0].casefold():
            raise OfficialSourceUrlError("unregistered_official_organization")
    try:
        addresses = resolver(host)
    except OfficialSourceUrlError:
        raise
    except Exception as exc:
        raise OfficialSourceUrlError("dns_unavailable") from exc
    _require_public_addresses(addresses)
    return normalized


def _candidate_sort_key(candidate: OfficialSourceCandidate) -> tuple[int, str, str]:
    return (int(candidate.specificity), candidate.canonical_organization.casefold(), candidate.url)


def resolve_official_source_candidates(
    profile: HiringContextProfile,
    *,
    registry: OfficialHiringContextSourceRegistry,
) -> tuple[OfficialSourceCandidate, ...]:
    """Resolve exact corrected profile identity into a bounded source lineage."""

    if not isinstance(profile, HiringContextProfile):
        raise TypeError("profile must be a HiringContextProfile")
    if not isinstance(registry, OfficialHiringContextSourceRegistry):
        raise TypeError("registry must be an OfficialHiringContextSourceRegistry")
    lineage = registry.lineage(profile.company)
    if not lineage and profile.parent_organization:
        lineage = registry.lineage(profile.parent_organization)
    candidates: dict[str, OfficialSourceCandidate] = {}
    for lineage_index, rule in enumerate(lineage):
        for source in rule.sources:
            if lineage_index == 0:
                candidate = source
            else:
                candidate = OfficialSourceCandidate(
                    canonical_organization=source.canonical_organization,
                    url=source.url,
                    source_kind=OfficialHiringContextSourceKind.OFFICIAL_PARENT_ORG_PAGE,
                    specificity=OfficialSourceSpecificity.PARENT_ORGANIZATION,
                    trust=source.trust,
                    allowed_domains=source.allowed_domains,
                    allow_subdomains=source.allow_subdomains,
                )
            candidates[candidate.url] = candidate
    return tuple(sorted(candidates.values(), key=_candidate_sort_key))


DEFAULT_OFFICIAL_HIRING_CONTEXT_SOURCE_REGISTRY = OfficialHiringContextSourceRegistry(
    rules=(
        OfficialOrganizationSourceRule(
            canonical_name="Microsoft",
            aliases=("Microsoft Corporation",),
            sources=(
                OfficialSourceCandidate(
                    canonical_organization="Microsoft",
                    url="https://www.microsoft.com/",
                    source_kind=OfficialHiringContextSourceKind.OFFICIAL_COMPANY_PAGE,
                    specificity=OfficialSourceSpecificity.COMPANY,
                    allowed_domains=("microsoft.com", "www.microsoft.com"),
                ),
            ),
        ),
        OfficialOrganizationSourceRule(
            canonical_name="Xbox Game Studios",
            aliases=("Xbox Studios",),
            parent_organization="Microsoft",
            sources=(
                OfficialSourceCandidate(
                    canonical_organization="Xbox Game Studios",
                    url="https://www.xbox.com/xbox-game-studios",
                    source_kind=OfficialHiringContextSourceKind.OFFICIAL_PRODUCT_PAGE,
                    specificity=OfficialSourceSpecificity.TEAM_PRODUCT,
                    allowed_domains=("xbox.com", "www.xbox.com"),
                ),
            ),
        ),
        OfficialOrganizationSourceRule(
            canonical_name="The Coalition",
            aliases=("Coalition", "The Coalition Studio"),
            parent_organization="Xbox Game Studios",
            sources=(
                OfficialSourceCandidate(
                    canonical_organization="The Coalition",
                    url="https://www.thecoalitionstudio.com/",
                    source_kind=OfficialHiringContextSourceKind.OFFICIAL_TEAM_PAGE,
                    specificity=OfficialSourceSpecificity.TEAM_PRODUCT,
                    allowed_domains=(
                        "thecoalitionstudio.com",
                        "www.thecoalitionstudio.com",
                    ),
                ),
            ),
        ),
    )
)


__all__ = [
    "DEFAULT_OFFICIAL_HIRING_CONTEXT_SOURCE_REGISTRY",
    "DnsResolver",
    "MAX_OFFICIAL_DOMAINS_PER_RULE",
    "MAX_OFFICIAL_SOURCE_ALIASES",
    "MAX_OFFICIAL_SOURCE_CANDIDATES_PER_RULE",
    "MAX_OFFICIAL_SOURCE_RULES",
    "MAX_OFFICIAL_SOURCE_URL_LENGTH",
    "OfficialHiringContextSourceKind",
    "OfficialHiringContextSourceRegistry",
    "OfficialOrganizationSourceRule",
    "OfficialSourceCandidate",
    "OfficialSourceSpecificity",
    "OfficialSourceTrust",
    "OfficialSourceUrlError",
    "default_public_dns_resolver",
    "resolve_official_source_candidates",
    "validate_official_source_url",
]
