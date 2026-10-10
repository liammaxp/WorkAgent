from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from backend.hiring_context_models import (
    HiringContextConfidence,
    HiringContextProfile,
    HiringContextSignal,
    HiringContextSignalKind,
    HiringContextSourceKind,
    HiringContextSourceRef,
    RankingEffect,
)
from backend.hiring_context_official_sources import (
    DEFAULT_OFFICIAL_HIRING_CONTEXT_SOURCE_REGISTRY,
    OfficialHiringContextSourceKind,
    OfficialSourceCandidate,
    OfficialSourceSpecificity,
)
from backend.hiring_context_web_client import (
    BoundedOfficialSourceDocument,
    OfficialSourceAcquisition,
    OfficialSourceAcquisitionFailure,
)
from backend.hiring_context_web_enrichment import (
    OFFICIAL_HIRING_CONTEXT_ENRICHMENT_FLAG,
    HiringContextEnrichment,
    HiringContextEnrichmentCachePolicy,
    HiringContextEnrichmentStatus,
    HiringContextOfficialSourceRef,
    HiringContextWebSignal,
    build_hiring_context_web_enrichment,
    extract_hiring_context_web_enrichment,
    official_hiring_context_enrichment_enabled,
)


ROOT = Path(__file__).resolve().parents[1]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _profile(
    company: str = "The Coalition",
    *,
    team: str | None = "Online Systems",
    parent: str | None = "Xbox Game Studios",
    offline_game_signal: bool = False,
) -> HiringContextProfile:
    company_source = HiringContextSourceRef(
        HiringContextSourceKind.COMPANY_IDENTITY,
        _digest(f"company:{company}"),
    )
    sources = [company_source]
    signals = []
    if offline_game_signal:
        taxonomy = HiringContextSourceRef(
            HiringContextSourceKind.INTERNAL_TAXONOMY,
            _digest("taxonomy:coalition"),
        )
        sources.append(taxonomy)
        signals.append(
            HiringContextSignal(
                value="Game development",
                kind=HiringContextSignalKind.COMPANY_DOMAIN,
                confidence=HiringContextConfidence.HIGH,
                ranking_effects=(RankingEffect.DOMAIN_ALIGNMENT,),
                source_refs=(company_source, taxonomy),
            )
        )
    return HiringContextProfile(
        source_refs=tuple(sources),
        company=company,
        team=team,
        parent_organization=parent,
        signals=tuple(signals),
    )


def _candidate(
    *,
    organization: str = "The Coalition",
    url: str = "https://www.thecoalitionstudio.com/",
    kind: OfficialHiringContextSourceKind = OfficialHiringContextSourceKind.OFFICIAL_TEAM_PAGE,
    specificity: OfficialSourceSpecificity = OfficialSourceSpecificity.TEAM_PRODUCT,
) -> OfficialSourceCandidate:
    host = url.split("/", 3)[2]
    return OfficialSourceCandidate(
        canonical_organization=organization,
        url=url,
        source_kind=kind,
        specificity=specificity,
        allowed_domains=(host,),
    )


def _document(
    text: str,
    *,
    title: str = "Engineering",
    candidate: OfficialSourceCandidate | None = None,
    fingerprint_seed: str | None = None,
) -> BoundedOfficialSourceDocument:
    source = candidate or _candidate()
    return BoundedOfficialSourceDocument(
        source=source,
        normalized_url=source.url,
        title=title,
        text=text,
        content_fingerprint=_digest(fingerprint_seed or f"{title}\n{text}"),
    )


def _acquisition(*documents, failures=()) -> OfficialSourceAcquisition:
    return OfficialSourceAcquisition(tuple(documents), tuple(failures))


def _extract(text: str, *, profile: HiringContextProfile | None = None):
    return extract_hiring_context_web_enrichment(
        profile or _profile(),
        _acquisition(_document(text)),
    )


@pytest.mark.parametrize("value", ("1", "true", "TRUE", "yes", "on", " On "))
def test_feature_flag_accepts_only_explicit_truthy_values(value):
    assert official_hiring_context_enrichment_enabled(
        {OFFICIAL_HIRING_CONTEXT_ENRICHMENT_FLAG: value}
    )


@pytest.mark.parametrize("value", ("", "0", "false", "off", "enabled", "2"))
def test_feature_flag_is_default_off_and_fail_closed(value):
    assert not official_hiring_context_enrichment_enabled(
        {OFFICIAL_HIRING_CONTEXT_ENRICHMENT_FLAG: value}
    )


def test_disabled_enrichment_returns_before_registry_or_network_use():
    calls = []
    profile = _profile()
    before = profile.to_json()
    result = build_hiring_context_web_enrichment(
        profile,
        enabled=False,
        acquirer=lambda candidates: calls.append(candidates),
    )
    assert result.status is HiringContextEnrichmentStatus.DISABLED
    assert calls == []
    assert profile.to_json() == before


def test_environment_flag_defaults_to_disabled():
    result = build_hiring_context_web_enrichment(_profile(), environ={})
    assert result.status is HiringContextEnrichmentStatus.DISABLED


def test_unknown_company_returns_no_source_without_network():
    calls = []
    result = build_hiring_context_web_enrichment(
        _profile("Generic Unknown Company", team=None, parent=None),
        enabled=True,
        acquirer=lambda candidates: calls.append(candidates),
    )
    assert result.status is HiringContextEnrichmentStatus.NO_OFFICIAL_SOURCE
    assert calls == []


def test_corrected_profile_identity_controls_candidate_resolution():
    seen = []

    def acquire(candidates):
        seen.extend(candidates)
        return _acquisition(_document("Game development", candidate=candidates[0]))

    result = build_hiring_context_web_enrichment(
        _profile("The Coalition"),
        enabled=True,
        registry=DEFAULT_OFFICIAL_HIRING_CONTEXT_SOURCE_REGISTRY,
        acquirer=acquire,
    )
    assert result.status is HiringContextEnrichmentStatus.READY
    assert seen[0].canonical_organization == "The Coalition"
    assert result.company == "The Coalition"


@pytest.mark.parametrize("failure", (TimeoutError("timeout"), OSError("offline"), RuntimeError("bad")))
def test_acquisition_exception_is_offline_safe_and_does_not_mutate_profile(failure):
    profile = _profile(offline_game_signal=True)
    before = profile.to_json()

    def acquire(candidates):
        raise failure

    result = build_hiring_context_web_enrichment(
        profile, enabled=True, acquirer=acquire
    )
    assert result.status is HiringContextEnrichmentStatus.UNAVAILABLE
    assert result.signals == ()
    assert profile.to_json() == before


def test_empty_failed_acquisition_is_offline_safe():
    source = _candidate()
    result = build_hiring_context_web_enrichment(
        _profile(),
        enabled=True,
        acquirer=lambda candidates: _acquisition(
            failures=(OfficialSourceAcquisitionFailure(source, "timeout"),)
        ),
    )
    assert result.status is HiringContextEnrichmentStatus.UNAVAILABLE


def test_extraction_failure_is_offline_safe():
    result = build_hiring_context_web_enrichment(
        _profile(),
        enabled=True,
        acquirer=lambda candidates: _acquisition(_document("Game development")),
        extractor=lambda profile, acquisition: (_ for _ in ()).throw(ValueError("parse")),
    )
    assert result.status is HiringContextEnrichmentStatus.UNAVAILABLE


def test_mismatched_extractor_binding_is_discarded_as_unavailable():
    other = _profile("Microsoft", team=None, parent=None)
    mismatched = HiringContextEnrichment(
        offline_profile_id=other.profile_id,
        offline_profile_fingerprint=other.fingerprint,
        status=HiringContextEnrichmentStatus.READY,
        company=other.company,
    )
    result = build_hiring_context_web_enrichment(
        _profile(),
        enabled=True,
        acquirer=lambda candidates: _acquisition(_document("Game development")),
        extractor=lambda profile, acquisition: mismatched,
    )
    assert result.status is HiringContextEnrichmentStatus.UNAVAILABLE


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("We focus on game development.", "Game development"),
        ("We build real-time interactive software.", "Real-time interactive software"),
        ("Our engineers own online game systems.", "Online game systems"),
        ("We operate data infrastructure.", "Data infrastructure"),
        ("We provide a developer platform.", "Developer platform"),
        ("We build commerce platforms.", "Commerce platform"),
        ("Our discipline is security engineering.", "Security engineering"),
    ),
)
def test_allowlisted_direct_official_context_is_extracted(text, expected):
    result = _extract(text)
    assert expected in {signal.value for signal in result.signals}


def test_official_team_signal_is_high_confidence_context_only():
    result = _extract("Game development and online systems")
    game = next(signal for signal in result.signals if signal.value == "Game development")
    assert game.kind is HiringContextSignalKind.TEAM_DOMAIN
    assert game.confidence is HiringContextConfidence.HIGH
    assert isinstance(game, HiringContextWebSignal)


def test_broad_company_signal_is_medium_confidence():
    candidate = _candidate(
        organization="Microsoft",
        url="https://www.microsoft.com/",
        kind=OfficialHiringContextSourceKind.OFFICIAL_COMPANY_PAGE,
        specificity=OfficialSourceSpecificity.COMPANY,
    )
    result = extract_hiring_context_web_enrichment(
        _profile("Microsoft", team=None, parent=None),
        _acquisition(_document("Software platforms", candidate=candidate)),
    )
    assert result.signals[0].kind is HiringContextSignalKind.COMPANY_DOMAIN
    assert result.signals[0].confidence is HiringContextConfidence.MEDIUM


def test_team_primary_domain_takes_precedence_over_broad_parent_domain():
    team = _document("Game development")
    parent_candidate = _candidate(
        organization="Microsoft",
        url="https://www.microsoft.com/",
        kind=OfficialHiringContextSourceKind.OFFICIAL_PARENT_ORG_PAGE,
        specificity=OfficialSourceSpecificity.PARENT_ORGANIZATION,
    )
    parent = _document("Software platforms", candidate=parent_candidate)
    result = extract_hiring_context_web_enrichment(
        _profile(), _acquisition(parent, team)
    )
    assert [signal.value for signal in result.signals] == ["Game development"]
    assert len(result.official_source_refs) == 2


def test_duplicate_web_signal_is_one_semantic_signal_with_all_official_provenance():
    product_candidate = _candidate(
        url="https://www.thecoalitionstudio.com/games",
        kind=OfficialHiringContextSourceKind.OFFICIAL_PRODUCT_PAGE,
    )
    result = extract_hiring_context_web_enrichment(
        _profile(),
        _acquisition(
            _document("Game development", fingerprint_seed="one"),
            _document(
                "We develop games",
                candidate=product_candidate,
                fingerprint_seed="two",
            ),
        ),
    )
    assert len(result.signals) == 1
    assert len(result.signals[0].official_source_refs) == 2


def test_duplicate_offline_and_web_signal_preserves_both_provenance_domains():
    profile = _profile(offline_game_signal=True)
    result = extract_hiring_context_web_enrichment(
        profile, _acquisition(_document("Game development"))
    )
    assert len(result.signals) == 1
    assert result.signals[0].official_source_refs
    assert result.signals[0].offline_source_refs == profile.signals[0].source_refs
    assert profile.signals == profile.signals


def test_incompatible_same_specificity_primary_sources_fail_closed():
    data_candidate = _candidate(url="https://www.thecoalitionstudio.com/data")
    result = extract_hiring_context_web_enrichment(
        _profile(),
        _acquisition(
            _document("Game development", fingerprint_seed="game"),
            _document(
                "Data infrastructure",
                candidate=data_candidate,
                fingerprint_seed="data",
            ),
        ),
    )
    assert result.status is HiringContextEnrichmentStatus.CONFLICTED
    assert result.signals == ()
    assert result.conflicts == ("primary_domain:data|game",)
    assert len(result.official_source_refs) == 2


def test_partial_acquisition_retains_successful_context_and_explicit_status():
    failed_source = _candidate(url="https://www.thecoalitionstudio.com/missing")
    result = extract_hiring_context_web_enrichment(
        _profile(),
        _acquisition(
            _document("Game development"),
            failures=(OfficialSourceAcquisitionFailure(failed_source, "timeout"),),
        ),
    )
    assert result.status is HiringContextEnrichmentStatus.PARTIAL
    assert result.signals


def test_successful_page_with_no_supported_semantics_produces_no_guess():
    result = _extract("Welcome to our company. We value people and ideas.")
    assert result.status is HiringContextEnrichmentStatus.READY
    assert result.signals == ()


@pytest.mark.parametrize(
    "text",
    (
        "Microsoft uses Azure and C#.",
        "Amazon engineers use AWS.",
        "The Coalition uses C++ and Unreal Engine.",
        "Google operates GCP.",
        "Our privacy professionals advise clients.",
        "Cyber transformation consultants deliver strategy.",
        "Technology Risk specialists improve controls.",
        "The candidate is a game developer with ten years of experience.",
        "Reduced latency by 40 percent and owned the architecture.",
    ),
)
def test_unapproved_technology_persona_candidate_and_metric_claims_are_never_extracted(text):
    result = _extract(text)
    assert result.signals == ()
    serialized = result.to_json()
    assert text not in serialized
    assert all(
        forbidden not in serialized.casefold()
        for forbidden in (
            "azure",
            "aws",
            "c++",
            "unreal",
            "consultant",
            "specialist",
            "ten years",
            "40 percent",
        )
    )


def test_raw_source_candidate_authority_id_does_not_propagate():
    result = _extract("Game development pei_candidate_secret engineering_story_123")
    serialized = result.to_json()
    assert "pei_candidate_secret" not in serialized
    assert "engineering_story_123" not in serialized
    assert [signal.value for signal in result.signals] == ["Game development"]


def test_official_source_contract_has_no_candidate_or_story_evidence_id_fields():
    reference = _extract("Game development").official_source_refs[0]
    assert set(reference.to_dict()) == {
        "normalized_url",
        "source_kind",
        "canonical_organization",
        "content_fingerprint",
        "specificity",
        "trust",
        "source_id",
    }
    assert not any("candidate" in field or "story" in field for field in reference.to_dict())


def test_web_confidence_is_hiring_context_confidence_not_candidate_confidence():
    signal = _extract("Game development").signals[0]
    assert type(signal.confidence) is HiringContextConfidence
    assert "candidate" not in type(signal.confidence).__name__.casefold()


def test_signal_order_and_enrichment_identity_are_deterministic():
    first = _document("Online systems and game development", fingerprint_seed="one")
    second_candidate = _candidate(url="https://www.thecoalitionstudio.com/realtime")
    second = _document(
        "Real-time interactive software",
        candidate=second_candidate,
        fingerprint_seed="two",
    )
    one = extract_hiring_context_web_enrichment(_profile(), _acquisition(first, second))
    two = extract_hiring_context_web_enrichment(_profile(), _acquisition(second, first))
    assert [item.value for item in one.signals] == sorted(
        item.value for item in one.signals
    )
    assert one == two
    assert one.fingerprint == two.fingerprint


def test_equivalent_bounded_semantics_have_equivalent_identity():
    one = _document("Game development", fingerprint_seed="normalized-content")
    two = _document("Game development", fingerprint_seed="normalized-content")
    first = extract_hiring_context_web_enrichment(_profile(), _acquisition(one))
    second = extract_hiring_context_web_enrichment(_profile(), _acquisition(two))
    assert first.enrichment_id == second.enrichment_id
    assert first.signals[0].signal_id == second.signals[0].signal_id


def test_material_source_content_change_changes_enrichment_identity():
    first = extract_hiring_context_web_enrichment(
        _profile(), _acquisition(_document("Game development", fingerprint_seed="version-one"))
    )
    second = extract_hiring_context_web_enrichment(
        _profile(), _acquisition(_document("Game development", fingerprint_seed="version-two"))
    )
    assert first.fingerprint != second.fingerprint
    assert first.enrichment_id != second.enrichment_id


def test_runtime_retrieval_time_is_not_part_of_semantic_contract():
    fields = _extract("Game development").to_dict()
    assert fields["cache_policy"] == "none"
    assert not any(token in fields for token in ("retrieved_at", "timestamp", "cache_age", "expires_at"))


def test_no_cache_means_no_stale_context_can_be_reused():
    result = _extract("Game development")
    assert result.cache_policy is HiringContextEnrichmentCachePolicy.NONE
    assert "stale" not in {status.value for status in HiringContextEnrichmentStatus}


def test_contract_round_trip_is_strict_and_stable():
    result = _extract("Game development and online systems")
    assert HiringContextEnrichment.from_dict(result.to_dict()) == result
    with pytest.raises(ValueError, match="unknown HiringContextEnrichment fields"):
        HiringContextEnrichment.from_dict({**result.to_dict(), "candidate_evidence": []})


def test_contracts_are_immutable():
    result = _extract("Game development")
    with pytest.raises(AttributeError):
        result.status = HiringContextEnrichmentStatus.UNAVAILABLE
    with pytest.raises(AttributeError):
        result.signals[0].value = "Candidate game developer"


def test_enrichment_failure_never_calls_legacy_ranking_fallback():
    source = inspect.getsource(build_hiring_context_web_enrichment)
    assert "legacy" not in source.casefold()
    assert "rank" not in source.casefold()
    assert "resume" not in source.casefold()


def test_enrichment_modules_are_isolated_from_candidate_story_retrieval_and_writes():
    modules = (
        ROOT / "backend" / "hiring_context_official_sources.py",
        ROOT / "backend" / "hiring_context_web_client.py",
        ROOT / "backend" / "hiring_context_web_enrichment.py",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in modules).casefold()
    forbidden_imports = (
        "from backend.memory_store import",
        "from backend.project_retrieval_v2 import",
        "from backend.chroma_write_client import",
        "from backend.engineering_story_relevance import",
        "from backend.project_story_ranking import",
        "from backend.project_portfolio_ranking import",
        "from backend.resume_emphasis_preferences import",
        "from backend.resume",
    )
    assert all(value not in source for value in forbidden_imports)
    assert "chromadb" not in source
    assert "capabilityfact(" not in source
    assert "projectevidencefact(" not in source


def test_enrichment_does_not_modify_accepted_ranking_or_preference_modules():
    production_modules = {
        "backend/engineering_story_relevance.py",
        "backend/project_story_ranking.py",
        "backend/project_portfolio_ranking.py",
        "backend/resume_emphasis_preferences.py",
    }
    import subprocess

    changed = subprocess.run(
        ["git", "diff", "--name-only", "--", *sorted(production_modules)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert changed == []


def test_enrichment_is_not_imported_by_current_review_product_boundary():
    review = (ROOT / "backend" / "hiring_context_ranking_review.py").read_text(
        encoding="utf-8"
    )
    api = (ROOT / "backend" / "hiring_context_ranking_review_api.py").read_text(
        encoding="utf-8"
    )
    assert "hiring_context_web_enrichment" not in review
    assert "hiring_context_web_enrichment" not in api
