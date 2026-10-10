from __future__ import annotations

import hashlib

import pytest

from backend.hiring_context_models import (
    HiringContextProfile,
    HiringContextSourceKind,
    HiringContextSourceRef,
)
from backend.hiring_context_official_sources import (
    DEFAULT_OFFICIAL_HIRING_CONTEXT_SOURCE_REGISTRY,
    MAX_OFFICIAL_SOURCE_CANDIDATES_PER_RULE,
    OfficialHiringContextSourceKind,
    OfficialHiringContextSourceRegistry,
    OfficialOrganizationSourceRule,
    OfficialSourceCandidate,
    OfficialSourceSpecificity,
    OfficialSourceTrust,
    OfficialSourceUrlError,
    resolve_official_source_candidates,
    validate_official_source_url,
)


PUBLIC_IP = "93.184.216.34"


def _profile(company: str, *, parent: str | None = None) -> HiringContextProfile:
    source = HiringContextSourceRef(
        HiringContextSourceKind.COMPANY_IDENTITY,
        hashlib.sha256(company.encode()).hexdigest(),
    )
    return HiringContextProfile(
        source_refs=(source,),
        company=company,
        parent_organization=parent,
    )


def _candidate(
    url: str = "https://official.example/",
    *,
    domains: tuple[str, ...] = ("official.example",),
    allow_subdomains: bool = False,
    kind: OfficialHiringContextSourceKind = OfficialHiringContextSourceKind.OFFICIAL_COMPANY_PAGE,
) -> OfficialSourceCandidate:
    return OfficialSourceCandidate(
        canonical_organization="Example",
        url=url,
        source_kind=kind,
        specificity=OfficialSourceSpecificity.COMPANY,
        allowed_domains=domains,
        allow_subdomains=allow_subdomains,
    )


def _public(_: str) -> tuple[str, ...]:
    return (PUBLIC_IP,)


@pytest.mark.parametrize(
    ("identity", "canonical"),
    (
        ("The Coalition", "The Coalition"),
        ("Coalition", "The Coalition"),
        ("The Coalition Studio", "The Coalition"),
        ("Xbox Game Studios", "Xbox Game Studios"),
        ("Xbox Studios", "Xbox Game Studios"),
        ("Microsoft", "Microsoft"),
        ("Microsoft Corporation", "Microsoft"),
    ),
)
def test_production_registry_aliases_resolve_exactly(identity, canonical):
    rule = DEFAULT_OFFICIAL_HIRING_CONTEXT_SOURCE_REGISTRY.match(identity)
    assert rule is not None
    assert rule.canonical_name == canonical


@pytest.mark.parametrize(
    "identity",
    ("Micro soft", "Microsoft Azure", "TheCoalition", "Xbox", "Unknown Company", ""),
)
def test_unknown_or_fuzzy_organization_identity_is_not_resolved(identity):
    assert DEFAULT_OFFICIAL_HIRING_CONTEXT_SOURCE_REGISTRY.match(identity) is None


def test_coalition_candidate_lineage_preserves_exact_organization_binding():
    candidates = resolve_official_source_candidates(
        _profile("The Coalition"),
        registry=DEFAULT_OFFICIAL_HIRING_CONTEXT_SOURCE_REGISTRY,
    )
    assert [item.canonical_organization for item in candidates] == [
        "The Coalition",
        "Microsoft",
        "Xbox Game Studios",
    ]
    assert candidates[0].source_kind is OfficialHiringContextSourceKind.OFFICIAL_TEAM_PAGE
    assert all(
        item.source_kind is OfficialHiringContextSourceKind.OFFICIAL_PARENT_ORG_PAGE
        for item in candidates[1:]
    )


def test_unknown_company_produces_no_source_candidate():
    assert resolve_official_source_candidates(
        _profile("Generic Unknown Company"),
        registry=DEFAULT_OFFICIAL_HIRING_CONTEXT_SOURCE_REGISTRY,
    ) == ()


def test_known_parent_can_resolve_when_company_is_unknown():
    candidates = resolve_official_source_candidates(
        _profile("Unknown Subsidiary", parent="Microsoft"),
        registry=DEFAULT_OFFICIAL_HIRING_CONTEXT_SOURCE_REGISTRY,
    )
    assert len(candidates) == 1
    assert candidates[0].canonical_organization == "Microsoft"


@pytest.mark.parametrize(
    "url",
    (
        "https://official.example/",
        "https://official.example:443/path",
        "https://official.example/path?locale=en",
    ),
)
def test_registered_official_company_urls_are_normalized_and_accepted(url):
    normalized = validate_official_source_url(_candidate(), url, resolver=_public)
    assert normalized.startswith("https://official.example/")
    assert ":443" not in normalized


def test_explicit_registered_team_domain_is_accepted():
    candidate = _candidate(
        "https://team.example/",
        domains=("team.example",),
        kind=OfficialHiringContextSourceKind.OFFICIAL_TEAM_PAGE,
    )
    assert validate_official_source_url(candidate, candidate.url, resolver=_public) == candidate.url


def test_explicit_registered_parent_domain_is_accepted():
    candidate = _candidate(
        kind=OfficialHiringContextSourceKind.OFFICIAL_PARENT_ORG_PAGE,
    )
    assert validate_official_source_url(candidate, candidate.url, resolver=_public) == candidate.url


@pytest.mark.parametrize(
    "url",
    (
        "https://random.example/",
        "https://official.example.evil.test/",
        "https://evil-official.example/",
        "https://sub.official.example/",
    ),
)
def test_unknown_lookalike_and_unregistered_subdomains_are_rejected(url):
    with pytest.raises(OfficialSourceUrlError, match="unregistered_official_domain"):
        validate_official_source_url(_candidate(), url, resolver=_public)


def test_subdomain_policy_can_be_explicitly_enabled():
    candidate = _candidate(allow_subdomains=True)
    assert validate_official_source_url(
        candidate,
        "https://news.official.example/",
        resolver=_public,
    ) == "https://news.official.example/"


@pytest.mark.parametrize(
    "url",
    (
        "http://official.example/",
        "file:///etc/passwd",
        "ftp://official.example/file",
        "data:text/plain,hello",
        "javascript:alert(1)",
    ),
)
def test_non_https_schemes_are_rejected(url):
    with pytest.raises(OfficialSourceUrlError, match="unsupported_scheme"):
        validate_official_source_url(_candidate(), url, resolver=_public)


@pytest.mark.parametrize(
    "url",
    (
        "https://user@official.example/",
        "https://user:secret@official.example/",
    ),
)
def test_embedded_credentials_are_rejected(url):
    with pytest.raises(OfficialSourceUrlError, match="embedded_credentials"):
        validate_official_source_url(_candidate(), url, resolver=_public)


@pytest.mark.parametrize(
    "url",
    (
        "https://official.example:444/",
        "https://official.example/path#fragment",
        "https://official.example/space here",
        "https://[::1",
        "",
    ),
)
def test_malformed_urls_are_rejected(url):
    with pytest.raises(OfficialSourceUrlError, match="malformed_url"):
        validate_official_source_url(_candidate(), url, resolver=_public)


@pytest.mark.parametrize(
    "address",
    (
        "127.0.0.1",
        "::1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.1.1",
        "169.254.169.254",
        "fe80::1",
        "fc00::1",
        "0.0.0.0",
        "224.0.0.1",
        "192.0.2.1",
    ),
)
def test_non_global_resolved_destinations_are_rejected(address):
    with pytest.raises(OfficialSourceUrlError, match="forbidden_destination"):
        validate_official_source_url(
            _candidate(),
            "https://official.example/",
            resolver=lambda _: (address,),
        )


def test_mixed_public_and_private_dns_answers_fail_closed():
    with pytest.raises(OfficialSourceUrlError, match="forbidden_destination"):
        validate_official_source_url(
            _candidate(),
            "https://official.example/",
            resolver=lambda _: (PUBLIC_IP, "10.0.0.1"),
        )


@pytest.mark.parametrize(
    "host",
    (
        "localhost",
        "metadata.google.internal",
        "instance-data",
        "metadata",
        "service.local",
        "service.home.arpa",
    ),
)
def test_local_and_metadata_hostnames_cannot_be_registered(host):
    with pytest.raises((ValueError, OfficialSourceUrlError)):
        _candidate(f"https://{host}/", domains=(host,))


def test_dns_failure_is_sanitized_and_fails_closed():
    def unavailable(_: str):
        raise OSError("resolver detail")

    with pytest.raises(OfficialSourceUrlError, match="dns_unavailable") as captured:
        validate_official_source_url(_candidate(), _candidate().url, resolver=unavailable)
    assert "resolver detail" not in str(captured.value)


def test_arbitrary_frontend_url_cannot_bypass_registry_binding():
    registered = resolve_official_source_candidates(
        _profile("Microsoft"),
        registry=DEFAULT_OFFICIAL_HIRING_CONTEXT_SOURCE_REGISTRY,
    )[0]
    with pytest.raises(OfficialSourceUrlError, match="unregistered_official_domain"):
        validate_official_source_url(
            registered,
            "https://attacker.example/?company=Microsoft",
            resolver=_public,
        )


def test_verified_github_source_requires_exact_registered_org_path():
    source = _candidate(
        "https://github.com/verified-org",
        domains=("github.com",),
        kind=OfficialHiringContextSourceKind.OFFICIAL_ORG_GITHUB,
    )
    assert validate_official_source_url(source, source.url, resolver=_public) == source.url
    with pytest.raises(OfficialSourceUrlError, match="unregistered_official_organization"):
        validate_official_source_url(
            source,
            "https://github.com/different-org",
            resolver=_public,
        )


def test_registry_rejects_unknown_parent_and_alias_conflict():
    rule = OfficialOrganizationSourceRule(canonical_name="Child", parent_organization="Missing")
    with pytest.raises(ValueError, match="unknown official parent"):
        OfficialHiringContextSourceRegistry((rule,))


def test_registry_and_candidates_are_immutable_and_bounded():
    candidate = _candidate()
    with pytest.raises(AttributeError):
        candidate.url = "https://evil.example/"
    with pytest.raises(ValueError, match="per-organization bound"):
        OfficialOrganizationSourceRule(
            canonical_name="Example",
            sources=tuple(candidate for _ in range(MAX_OFFICIAL_SOURCE_CANDIDATES_PER_RULE + 1)),
        )
    assert candidate.trust is OfficialSourceTrust.VERIFIED_OFFICIAL
