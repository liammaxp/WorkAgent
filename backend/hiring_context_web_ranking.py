"""Pure, bounded influence of official hiring-context enrichment on ranking.

Official web enrichment is context only.  This module consumes the already
validated enrichment artifact and accepted Story relevance features; it does
not acquire pages, parse HTML, retrieve candidate evidence, or modify Story
truth.  The existing Story, project aggregation, and portfolio policies stay
unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from typing import Any

from backend.engineering_story_relevance import (
    StoryHiringRelevance,
    StoryRelevanceFeature,
)
from backend.hiring_context_models import (
    HiringContextConfidence,
    HiringContextProfile,
)
from backend.hiring_context_web_enrichment import (
    HiringContextEnrichment,
    HiringContextEnrichmentStatus,
    HiringContextOfficialSourceRef,
    HiringContextWebSignal,
)
from backend.project_portfolio_ranking import rank_project_portfolio
from backend.project_story_ranking import (
    ProjectHiringRelevance,
    ProjectStoryRelevanceInput,
    rank_projects_from_story_relevance,
)
from backend.project_story_ranking_refresh import (
    PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
    ProjectStoryRelevanceSnapshot,
)
from backend.project_story_ranking_state import (
    ProjectStoryRankingState,
    _build_project_story_ranking_state,
    build_project_story_ranking_state,
)


OFFICIAL_HIRING_CONTEXT_RANKING_INFLUENCE_FLAG = (
    "USE_OFFICIAL_HIRING_CONTEXT_RANKING_INFLUENCE"
)
OFFICIAL_HIRING_CONTEXT_WEB_RANKING_POLICY_ID = (
    "hiring_context_web_ranking.bounded_context_adjustment.v2"
)
# Maximum semantic adjustment before the authoritative candidate safety gate.
MAX_WEB_CONTEXT_ADJUSTMENT = 0.05
WEB_RANKING_SCORE_DECIMALS = 6
MAX_WEB_RANKING_INFLUENCE_TRACES = 16

_CONFIDENCE_WEIGHTS = {"low": 0.35, "medium": 0.65, "high": 1.0}
_SPECIFICITY_WEIGHTS = {
    1: 1.0,   # team/product
    2: 0.75,  # company engineering
    3: 0.55,  # careers
    4: 0.40,  # company
    5: 0.25,  # parent organization
    6: 0.20,  # organization GitHub
}
_CORROBORATED_SIGNAL_FACTOR = 0.55
_USABLE_STATUSES = frozenset({
    HiringContextEnrichmentStatus.READY,
    HiringContextEnrichmentStatus.PARTIAL,
})
_SIGNAL_FEATURES: Mapping[str, frozenset[StoryRelevanceFeature]] = {
    "game development": frozenset({StoryRelevanceFeature.GAME_DEVELOPMENT}),
    "real-time interactive software": frozenset({StoryRelevanceFeature.REAL_TIME_SYSTEMS}),
    "online game systems": frozenset({
        StoryRelevanceFeature.GAME_DEVELOPMENT,
        StoryRelevanceFeature.REAL_TIME_SYSTEMS,
    }),
    "software platforms": frozenset({StoryRelevanceFeature.PLATFORM_ENGINEERING}),
    "developer platform": frozenset({StoryRelevanceFeature.PLATFORM_ENGINEERING}),
    "data infrastructure": frozenset({StoryRelevanceFeature.DATA_ENGINEERING}),
    "commerce platform": frozenset({StoryRelevanceFeature.PLATFORM_ENGINEERING}),
    "security engineering": frozenset({StoryRelevanceFeature.SECURITY}),
}


class WebRankingInfluenceErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    ENRICHMENT_CONTEXT_MISMATCH = "enrichment_context_mismatch"
    MIXED_WEB_RANKING_GENERATION = "mixed_web_ranking_generation"
    CONTEXT_CHANGED_REQUIRES_FULL_RERANK = "context_changed_requires_full_rerank"


class WebRankingInfluenceError(ValueError):
    def __init__(self, code: WebRankingInfluenceErrorCode, message: str) -> None:
        self.code = WebRankingInfluenceErrorCode(code)
        super().__init__(message)


class WebAwarePartialRerankDecision(str, Enum):
    STORY_ONLY_ALLOWED = "story_only_allowed"
    CONTEXT_CHANGED_REQUIRES_FULL_RERANK = (
        "context_changed_requires_full_rerank"
    )


class WebEnrichmentChangeDecision(str, Enum):
    UNCHANGED = "unchanged"
    CONTEXT_CHANGED_REQUIRES_FULL_RERANK = (
        "context_changed_requires_full_rerank"
    )


def _score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return round(normalized, WEB_RANKING_SCORE_DECIMALS)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    return value


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def official_hiring_context_ranking_influence_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if environ is None else environ
    value = source.get(OFFICIAL_HIRING_CONTEXT_RANKING_INFLUENCE_FLAG, "")
    return isinstance(value, str) and value.strip().casefold() in {
        "1", "true", "yes", "on"
    }


def _stable_features(
    values: Sequence[StoryRelevanceFeature],
    name: str,
) -> tuple[StoryRelevanceFeature, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    try:
        normalized = {StoryRelevanceFeature(value) for value in values}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains an unsupported semantic feature") from exc
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    order = {item: index for index, item in enumerate(StoryRelevanceFeature)}
    return tuple(sorted(normalized, key=order.__getitem__))


@dataclass(frozen=True, slots=True)
class WebRankingInfluenceTrace:
    """One bounded, source-complete contribution to a web adjustment."""

    signal_id: str
    signal_value: str
    official_source_refs: tuple[HiringContextOfficialSourceRef, ...]
    candidate_features: tuple[StoryRelevanceFeature, ...]
    context_confidence: HiringContextConfidence
    source_specificity_weight: float
    offline_corroboration_factor: float
    raw_contribution: float
    trace_id: str = ""

    def __post_init__(self) -> None:
        signal_id = _text(self.signal_id, "signal_id")
        signal_value = _text(self.signal_value, "signal_value")
        if (
            isinstance(self.official_source_refs, (str, bytes))
            or not isinstance(self.official_source_refs, Sequence)
            or not self.official_source_refs
        ):
            raise ValueError("official_source_refs must be a non-empty sequence")
        if any(
            not isinstance(reference, HiringContextOfficialSourceRef)
            for reference in self.official_source_refs
        ):
            raise TypeError(
                "official_source_refs must contain HiringContextOfficialSourceRef values"
            )
        references = tuple(
            {reference.source_id: reference for reference in self.official_source_refs}[key]
            for key in sorted({reference.source_id for reference in self.official_source_refs})
        )
        features = _stable_features(self.candidate_features, "candidate_features")
        expected_features = _SIGNAL_FEATURES.get(signal_value.casefold())
        if expected_features is None or not set(features).issubset(expected_features):
            raise ValueError(
                "candidate_features are not compatible with the web signal"
            )
        confidence = HiringContextConfidence(self.context_confidence)
        expected_confidence = _CONFIDENCE_WEIGHTS[confidence.value]
        specificity = _score(
            self.source_specificity_weight,
            "source_specificity_weight",
        )
        expected_specificity = _specificity_weight_for_refs(references)
        if specificity != expected_specificity:
            raise ValueError("source_specificity_weight does not match provenance")
        corroboration = _score(
            self.offline_corroboration_factor,
            "offline_corroboration_factor",
        )
        if corroboration not in {1.0, _CORROBORATED_SIGNAL_FACTOR}:
            raise ValueError("offline_corroboration_factor is not supported")
        expected_raw = _score(
            expected_confidence * specificity * corroboration,
            "raw_contribution",
        )
        raw = _score(self.raw_contribution, "raw_contribution")
        if raw != expected_raw:
            raise ValueError("raw_contribution does not match the web policy")
        payload = {
            "signal_id": signal_id,
            "signal_value": signal_value,
            "official_source_ids": [item.source_id for item in references],
            "candidate_features": [item.value for item in features],
            "context_confidence": confidence.value,
            "source_specificity_weight": specificity,
            "offline_corroboration_factor": corroboration,
            "raw_contribution": raw,
        }
        expected_id = f"web_ranking_influence_trace_{_digest(payload)[:24]}"
        if self.trace_id not in ("", expected_id):
            raise ValueError("trace_id does not match normalized web influence trace")
        object.__setattr__(self, "signal_id", signal_id)
        object.__setattr__(self, "signal_value", signal_value)
        object.__setattr__(self, "official_source_refs", references)
        object.__setattr__(self, "candidate_features", features)
        object.__setattr__(self, "context_confidence", confidence)
        object.__setattr__(self, "source_specificity_weight", specificity)
        object.__setattr__(self, "offline_corroboration_factor", corroboration)
        object.__setattr__(self, "raw_contribution", raw)
        object.__setattr__(self, "trace_id", expected_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "signal_value": self.signal_value,
            "official_source_refs": [
                item.to_dict() for item in self.official_source_refs
            ],
            "candidate_features": [item.value for item in self.candidate_features],
            "context_confidence": self.context_confidence.value,
            "source_specificity_weight": self.source_specificity_weight,
            "offline_corroboration_factor": self.offline_corroboration_factor,
            "raw_contribution": self.raw_contribution,
            "trace_id": self.trace_id,
        }


@dataclass(frozen=True, slots=True)
class StoryWebContextAdjustment:
    """Auditable additive context adjustment; it contains no candidate truth."""

    base_relevance_id: str
    enrichment_fingerprint: str
    enrichment_status: HiringContextEnrichmentStatus
    matched_signal_values: tuple[str, ...]
    candidate_match_strength: float
    context_confidence_weight: float
    source_specificity_weight: float
    offline_corroboration_factor: float
    adjustment: float
    traces: tuple[WebRankingInfluenceTrace, ...] = ()
    policy_id: str = OFFICIAL_HIRING_CONTEXT_WEB_RANKING_POLICY_ID
    adjustment_id: str = ""

    def __post_init__(self) -> None:
        base_id = _text(self.base_relevance_id, "base_relevance_id")
        fingerprint = _text(
            self.enrichment_fingerprint,
            "enrichment_fingerprint",
        )
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("enrichment_fingerprint must be a SHA-256 value")
        status = HiringContextEnrichmentStatus(self.enrichment_status)
        values = tuple(sorted({
            _text(value, "matched_signal_value") for value in self.matched_signal_values
        }, key=str.casefold))
        match = _score(self.candidate_match_strength, "candidate_match_strength")
        confidence = _score(
            self.context_confidence_weight,
            "context_confidence_weight",
        )
        specificity = _score(
            self.source_specificity_weight,
            "source_specificity_weight",
        )
        corroboration = _score(
            self.offline_corroboration_factor,
            "offline_corroboration_factor",
        )
        adjustment = _score(self.adjustment, "adjustment")
        if isinstance(self.traces, (str, bytes)) or not isinstance(
            self.traces,
            Sequence,
        ):
            raise TypeError("traces must be a sequence")
        if len(self.traces) > MAX_WEB_RANKING_INFLUENCE_TRACES:
            raise ValueError("traces exceeds the web-ranking trace bound")
        if any(not isinstance(item, WebRankingInfluenceTrace) for item in self.traces):
            raise TypeError("traces must contain WebRankingInfluenceTrace values")
        traces_by_signal = {item.signal_id: item for item in self.traces}
        if len(traces_by_signal) != len(self.traces):
            raise ValueError("traces cannot repeat a web signal")
        traces = tuple(
            traces_by_signal[key]
            for key in sorted(traces_by_signal)
        )
        if adjustment > MAX_WEB_CONTEXT_ADJUSTMENT:
            raise ValueError("adjustment exceeds the hard web-context cap")
        if self.policy_id != OFFICIAL_HIRING_CONTEXT_WEB_RANKING_POLICY_ID:
            raise ValueError("unsupported web ranking policy")
        if not traces:
            if values or any(
                value != 0.0
                for value in (
                    match,
                    confidence,
                    specificity,
                    corroboration,
                    adjustment,
                )
            ):
                raise ValueError("a zero adjustment cannot carry matching metadata")
        else:
            if status not in _USABLE_STATUSES:
                raise ValueError("only usable enrichment statuses may carry traces")
            expected_values = tuple(
                sorted({item.signal_value for item in traces}, key=str.casefold)
            )
            if values != expected_values:
                raise ValueError("matched_signal_values must match the trace signals")
            expected_adjustment = _score(
                min(
                    MAX_WEB_CONTEXT_ADJUSTMENT,
                    MAX_WEB_CONTEXT_ADJUSTMENT
                    * sum(item.raw_contribution for item in traces),
                ),
                "adjustment",
            )
            if adjustment != expected_adjustment:
                raise ValueError("adjustment does not match its bounded trace sum")
            strongest = max(
                traces,
                key=lambda item: (
                    item.raw_contribution,
                    item.signal_value.casefold(),
                    item.signal_id,
                ),
            )
            if (
                match != 1.0
                or confidence
                != _CONFIDENCE_WEIGHTS[strongest.context_confidence.value]
                or specificity != strongest.source_specificity_weight
                or corroboration != strongest.offline_corroboration_factor
            ):
                raise ValueError("summary metadata does not match the strongest trace")
        payload = {
            "base_relevance_id": base_id,
            "enrichment_fingerprint": fingerprint,
            "enrichment_status": status.value,
            "matched_signal_values": list(values),
            "candidate_match_strength": match,
            "context_confidence_weight": confidence,
            "source_specificity_weight": specificity,
            "offline_corroboration_factor": corroboration,
            "adjustment": adjustment,
            "trace_ids": [item.trace_id for item in traces],
            "policy_id": self.policy_id,
        }
        expected_id = f"story_web_context_adjustment_{_digest(payload)[:24]}"
        if self.adjustment_id not in ("", expected_id):
            raise ValueError("adjustment_id does not match normalized adjustment")
        object.__setattr__(self, "base_relevance_id", base_id)
        object.__setattr__(self, "enrichment_fingerprint", fingerprint)
        object.__setattr__(self, "enrichment_status", status)
        object.__setattr__(self, "matched_signal_values", values)
        object.__setattr__(self, "candidate_match_strength", match)
        object.__setattr__(self, "context_confidence_weight", confidence)
        object.__setattr__(self, "source_specificity_weight", specificity)
        object.__setattr__(self, "offline_corroboration_factor", corroboration)
        object.__setattr__(self, "adjustment", adjustment)
        object.__setattr__(self, "traces", traces)
        object.__setattr__(self, "adjustment_id", expected_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_relevance_id": self.base_relevance_id,
            "enrichment_fingerprint": self.enrichment_fingerprint,
            "enrichment_status": self.enrichment_status.value,
            "matched_signal_values": list(self.matched_signal_values),
            "candidate_match_strength": self.candidate_match_strength,
            "context_confidence_weight": self.context_confidence_weight,
            "source_specificity_weight": self.source_specificity_weight,
            "offline_corroboration_factor": self.offline_corroboration_factor,
            "adjustment": self.adjustment,
            "traces": [item.to_dict() for item in self.traces],
            "policy_id": self.policy_id,
            "adjustment_id": self.adjustment_id,
        }


def _web_adjusted_total(base: StoryHiringRelevance, adjustment: float) -> float:
    # The offline evaluator records its authoritative multiplier in components.
    # Reuse it directly, without deriving a ratio from rounded scores.
    raw = _score(min(1.0, base.raw_relevance_score + adjustment), "adjusted_raw_relevance")
    return _score(raw * base.components.evidence_claim_safety, "adjusted_total_relevance")


@dataclass(frozen=True, slots=True)
class WebAdjustedStoryHiringRelevance:
    """Read-only Story relevance view with base relevance preserved."""

    base_relevance: StoryHiringRelevance
    web_context_adjustment: StoryWebContextAdjustment
    total_relevance_score: float
    relevance_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.base_relevance, StoryHiringRelevance):
            raise TypeError("base_relevance must be StoryHiringRelevance")
        if not isinstance(
            self.web_context_adjustment,
            StoryWebContextAdjustment,
        ):
            raise TypeError(
                "web_context_adjustment must be StoryWebContextAdjustment"
            )
        adjustment = self.web_context_adjustment
        if adjustment.base_relevance_id != self.base_relevance.relevance_id:
            raise ValueError("web adjustment is bound to a different Story result")
        expected_total = _web_adjusted_total(self.base_relevance, adjustment.adjustment)
        total = _score(self.total_relevance_score, "total_relevance_score")
        if total != expected_total:
            raise ValueError("total_relevance_score does not match web adjustment")
        payload = {
            "base_relevance_id": self.base_relevance.relevance_id,
            "adjustment_id": adjustment.adjustment_id,
            "total_relevance_score": total,
        }
        expected_id = f"web_adjusted_story_hiring_relevance_{_digest(payload)[:24]}"
        if self.relevance_id not in ("", expected_id):
            raise ValueError("relevance_id does not match web-adjusted relevance")
        object.__setattr__(self, "total_relevance_score", total)
        object.__setattr__(self, "relevance_id", expected_id)

    @property
    def project_id(self) -> str:
        return self.base_relevance.project_id

    @property
    def canonical_story_id(self) -> str:
        return self.base_relevance.canonical_story_id

    @property
    def current_revision_id(self) -> str:
        return self.base_relevance.current_revision_id

    @property
    def hiring_context_profile_id(self) -> str:
        return self.base_relevance.hiring_context_profile_id

    @property
    def hiring_context_fingerprint(self) -> str:
        return self.base_relevance.hiring_context_fingerprint

    @property
    def story_provenance_fingerprint(self) -> str:
        return self.base_relevance.story_provenance_fingerprint

    @property
    def lifecycle_status(self) -> Any:
        return self.base_relevance.lifecycle_status

    @property
    def claim_sufficiency(self) -> Any:
        return self.base_relevance.claim_sufficiency

    @property
    def story_sufficiency(self) -> Any:
        return self.base_relevance.story_sufficiency

    @property
    def story_opportunity(self) -> Any:
        return self.base_relevance.story_opportunity

    @property
    def components(self) -> Any:
        return self.base_relevance.components

    @property
    def weights(self) -> Any:
        return self.base_relevance.weights

    @property
    def raw_relevance_score(self) -> float:
        return self.base_relevance.raw_relevance_score

    @property
    def evidence_risk_adjustment(self) -> float:
        return self.base_relevance.evidence_risk_adjustment

    @property
    def clarification_value_hint(self) -> float:
        return self.base_relevance.clarification_value_hint

    @property
    def semantic_features(self) -> tuple[Any, ...]:
        return self.base_relevance.semantic_features

    @property
    def reasons(self) -> tuple[Any, ...]:
        return self.base_relevance.reasons

    @property
    def hiring_context_source_refs(self) -> tuple[Any, ...]:
        return self.base_relevance.hiring_context_source_refs

    @property
    def base_total_relevance_score(self) -> float:
        return self.base_relevance.total_relevance_score

    @property
    def adjusted_raw_relevance_score(self) -> float:
        return _score(
            min(1.0, self.base_relevance.raw_relevance_score
                + self.web_context_adjustment.adjustment),
            "adjusted_raw_relevance",
        )

    @property
    def base_relevance_id(self) -> str:
        return self.base_relevance.relevance_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_relevance": self.base_relevance.to_dict(),
            "web_context_adjustment": self.web_context_adjustment.to_dict(),
            "adjusted_raw_relevance_score": self.adjusted_raw_relevance_score,
            "total_relevance_score": self.total_relevance_score,
            "relevance_id": self.relevance_id,
        }


def _signal_candidate_features(
    story: StoryHiringRelevance,
    signal: HiringContextWebSignal,
) -> tuple[StoryRelevanceFeature, ...]:
    required = _SIGNAL_FEATURES.get(signal.value.casefold())
    if not required:
        return ()
    # Only accepted, candidate-side semantic features can satisfy this match.
    return tuple(
        feature
        for feature in StoryRelevanceFeature
        if feature in required and feature in story.semantic_features
    )


def _specificity_weight_for_refs(
    references: Sequence[HiringContextOfficialSourceRef],
) -> float:
    specificity = min(int(reference.specificity) for reference in references)
    return _SPECIFICITY_WEIGHTS.get(specificity, 0.0)


def _signal_specificity_weight(signal: HiringContextWebSignal) -> float:
    return _specificity_weight_for_refs(signal.official_source_refs)


def _signal_trace(
    story: StoryHiringRelevance,
    signal: HiringContextWebSignal,
) -> WebRankingInfluenceTrace | None:
    candidate_features = _signal_candidate_features(story, signal)
    if not candidate_features:
        return None
    specificity = _signal_specificity_weight(signal)
    corroboration = (
        _CORROBORATED_SIGNAL_FACTOR
        if signal.offline_source_refs
        else 1.0
    )
    return WebRankingInfluenceTrace(
        signal_id=signal.signal_id,
        signal_value=signal.value,
        official_source_refs=signal.official_source_refs,
        candidate_features=candidate_features,
        context_confidence=signal.confidence,
        source_specificity_weight=specificity,
        offline_corroboration_factor=corroboration,
        raw_contribution=(
            _CONFIDENCE_WEIGHTS[signal.confidence.value]
            * specificity
            * corroboration
        ),
    )


def _usable_signal_values(
    signals: Sequence[HiringContextWebSignal],
) -> tuple[HiringContextWebSignal, ...]:
    # Enrichment normally deduplicates this already.  Re-deduplicate here so
    # manually assembled artifacts cannot stack the same semantic context.
    by_value: dict[str, HiringContextWebSignal] = {}
    for signal in signals:
        key = signal.value.casefold()
        prior = by_value.get(key)
        if prior is None:
            by_value[key] = signal
            continue
        prior_key = (
            _CONFIDENCE_WEIGHTS[prior.confidence.value]
            * _signal_specificity_weight(prior)
            * (
                _CORROBORATED_SIGNAL_FACTOR
                if prior.offline_source_refs
                else 1.0
            ),
            _signal_specificity_weight(prior),
            _CONFIDENCE_WEIGHTS[prior.confidence.value],
            prior.signal_id,
        )
        current_key = (
            _CONFIDENCE_WEIGHTS[signal.confidence.value]
            * _signal_specificity_weight(signal)
            * (
                _CORROBORATED_SIGNAL_FACTOR
                if signal.offline_source_refs
                else 1.0
            ),
            _signal_specificity_weight(signal),
            _CONFIDENCE_WEIGHTS[signal.confidence.value],
            signal.signal_id,
        )
        if current_key > prior_key:
            by_value[key] = signal
    return tuple(by_value[key] for key in sorted(by_value))


def _validate_enrichment_binding(
    hiring_context: HiringContextProfile,
    enrichment: HiringContextEnrichment,
) -> None:
    if not isinstance(hiring_context, HiringContextProfile):
        raise TypeError("hiring_context must be HiringContextProfile")
    if not isinstance(enrichment, HiringContextEnrichment):
        raise TypeError("enrichment must be HiringContextEnrichment")
    if (
        enrichment.offline_profile_id != hiring_context.profile_id
        or enrichment.offline_profile_fingerprint != hiring_context.fingerprint
    ):
        raise WebRankingInfluenceError(
            WebRankingInfluenceErrorCode.ENRICHMENT_CONTEXT_MISMATCH,
            "enrichment is bound to a different offline Hiring Context",
        )


def calculate_story_web_context_adjustment(
    *,
    hiring_context: HiringContextProfile,
    enrichment: HiringContextEnrichment,
    story_relevance: StoryHiringRelevance,
) -> StoryWebContextAdjustment:
    """Calculate one deterministic, non-negative, hard-capped adjustment."""

    _validate_enrichment_binding(hiring_context, enrichment)
    if not isinstance(story_relevance, StoryHiringRelevance):
        raise TypeError("story_relevance must be StoryHiringRelevance")
    if (
        story_relevance.hiring_context_profile_id != hiring_context.profile_id
        or story_relevance.hiring_context_fingerprint != hiring_context.fingerprint
    ):
        raise WebRankingInfluenceError(
            WebRankingInfluenceErrorCode.ENRICHMENT_CONTEXT_MISMATCH,
            "Story relevance is bound to a different Hiring Context",
        )
    traces: list[WebRankingInfluenceTrace] = []
    if enrichment.status in _USABLE_STATUSES:
        for signal in _usable_signal_values(enrichment.signals):
            trace = _signal_trace(story_relevance, signal)
            if trace is not None:
                traces.append(trace)
    if not traces:
        return StoryWebContextAdjustment(
            base_relevance_id=story_relevance.relevance_id,
            enrichment_fingerprint=enrichment.fingerprint,
            enrichment_status=enrichment.status,
            matched_signal_values=(),
            candidate_match_strength=0.0,
            context_confidence_weight=0.0,
            source_specificity_weight=0.0,
            offline_corroboration_factor=0.0,
            adjustment=0.0,
        )
    # One bounded semantic signal has one bounded influence.  Several distinct
    # accepted signals may add, but the total is never above the hard cap.
    raw = sum(
        trace.raw_contribution
        for trace in traces
    )
    adjustment = min(MAX_WEB_CONTEXT_ADJUSTMENT, MAX_WEB_CONTEXT_ADJUSTMENT * raw)
    strongest = max(
        traces,
        key=lambda trace: (
            trace.raw_contribution,
            trace.signal_value.casefold(),
            trace.signal_id,
        ),
    )
    return StoryWebContextAdjustment(
        base_relevance_id=story_relevance.relevance_id,
        enrichment_fingerprint=enrichment.fingerprint,
        enrichment_status=enrichment.status,
        matched_signal_values=tuple(trace.signal_value for trace in traces),
        candidate_match_strength=1.0,
        context_confidence_weight=(
            _CONFIDENCE_WEIGHTS[strongest.context_confidence.value]
        ),
        source_specificity_weight=strongest.source_specificity_weight,
        offline_corroboration_factor=strongest.offline_corroboration_factor,
        adjustment=adjustment,
        traces=tuple(traces),
    )


def apply_hiring_context_web_influence(
    *,
    hiring_context: HiringContextProfile,
    enrichment: HiringContextEnrichment,
    story_relevance: Sequence[StoryHiringRelevance],
    enabled: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[ProjectStoryRelevanceInput, ...]:
    """Return base results unchanged unless the separate influence flag is on."""

    _validate_enrichment_binding(hiring_context, enrichment)
    if isinstance(story_relevance, (str, bytes)) or not isinstance(
        story_relevance, Sequence
    ):
        raise TypeError("story_relevance must be a sequence")
    active = (
        official_hiring_context_ranking_influence_enabled(environ)
        if enabled is None
        else enabled
    )
    if not isinstance(active, bool):
        raise TypeError("enabled must be a bool or None")
    if not active:
        return tuple(story_relevance)
    adjustments = tuple(
        calculate_story_web_context_adjustment(
            hiring_context=hiring_context,
            enrichment=enrichment,
            story_relevance=story,
        )
        for story in story_relevance
    )
    if not any(item.adjustment > 0.0 for item in adjustments):
        return tuple(story_relevance)
    return tuple(
        WebAdjustedStoryHiringRelevance(
            base_relevance=story,
            web_context_adjustment=adjustment,
            total_relevance_score=_web_adjusted_total(story, adjustment.adjustment),
        )
        for story, adjustment in zip(story_relevance, adjustments, strict=True)
    )


def _stable_adjusted_story_order(
    story_relevance: Sequence[ProjectStoryRelevanceInput],
) -> tuple[ProjectStoryRelevanceInput, ...]:
    """Normalize material web-ranking output without changing zero-effect input."""

    return tuple(sorted(
        story_relevance,
        key=lambda story: (
            -story.total_relevance_score,
            story.canonical_story_id,
            story.current_revision_id,
            story.relevance_id,
        ),
    ))


@dataclass(frozen=True, slots=True)
class WebAwareProjectStoryRankingState:
    """Ranking state bound to both offline context and web influence context."""

    ranking_state: ProjectStoryRankingState
    enrichment_fingerprint: str
    enrichment_status: HiringContextEnrichmentStatus
    web_influence_policy_id: str = OFFICIAL_HIRING_CONTEXT_WEB_RANKING_POLICY_ID
    state_fingerprint: str = ""
    state_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.ranking_state, ProjectStoryRankingState):
            raise TypeError("ranking_state must be ProjectStoryRankingState")
        fingerprint = _text(self.enrichment_fingerprint, "enrichment_fingerprint")
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("enrichment_fingerprint must be a SHA-256 value")
        status = HiringContextEnrichmentStatus(self.enrichment_status)
        if self.web_influence_policy_id != OFFICIAL_HIRING_CONTEXT_WEB_RANKING_POLICY_ID:
            raise ValueError("unsupported web influence policy")
        adjusted_story_count = 0
        for snapshot in self.ranking_state.project_snapshots:
            for story in snapshot.story_relevance:
                if not isinstance(story, WebAdjustedStoryHiringRelevance):
                    continue
                adjusted_story_count += 1
                adjustment = story.web_context_adjustment
                if (
                    adjustment.enrichment_fingerprint != fingerprint
                    or adjustment.enrichment_status is not status
                    or adjustment.policy_id != self.web_influence_policy_id
                ):
                    raise WebRankingInfluenceError(
                        WebRankingInfluenceErrorCode.MIXED_WEB_RANKING_GENERATION,
                        "web-adjusted Story relevance is bound to a different generation",
                    )
        if not adjusted_story_count:
            raise WebRankingInfluenceError(
                WebRankingInfluenceErrorCode.MIXED_WEB_RANKING_GENERATION,
                "web-aware ranking state requires web-adjusted Story relevance",
            )
        payload = {
            "ranking_state_id": self.ranking_state.state_id,
            "offline_profile_id": self.ranking_state.hiring_context_profile_id,
            "offline_profile_fingerprint": self.ranking_state.hiring_context_fingerprint,
            "enrichment_fingerprint": fingerprint,
            "enrichment_status": status.value,
            "web_influence_policy_id": self.web_influence_policy_id,
        }
        expected_fingerprint = _digest(payload)
        expected_id = f"web_aware_project_story_ranking_state_{expected_fingerprint[:24]}"
        if self.state_fingerprint not in ("", expected_fingerprint):
            raise ValueError("state_fingerprint does not match web-aware state")
        if self.state_id not in ("", expected_id):
            raise ValueError("state_id does not match web-aware state")
        object.__setattr__(self, "enrichment_fingerprint", fingerprint)
        object.__setattr__(self, "enrichment_status", status)
        object.__setattr__(self, "state_fingerprint", expected_fingerprint)
        object.__setattr__(self, "state_id", expected_id)

    @property
    def hiring_context_profile_id(self) -> str:
        return self.ranking_state.hiring_context_profile_id

    @property
    def hiring_context_fingerprint(self) -> str:
        return self.ranking_state.hiring_context_fingerprint

    @property
    def portfolio(self) -> Any:
        return self.ranking_state.portfolio

    @property
    def project_snapshots(self) -> Any:
        return self.ranking_state.project_snapshots

    def snapshot_for_project(self, project_id: str) -> Any:
        return self.ranking_state.snapshot_for_project(project_id)

    def ranking_for_project(self, project_id: str) -> Any:
        return self.ranking_state.ranking_for_project(project_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "offline_ranking_state": self.ranking_state.to_dict(),
            "enrichment_fingerprint": self.enrichment_fingerprint,
            "enrichment_status": self.enrichment_status.value,
            "web_influence_policy_id": self.web_influence_policy_id,
            "state_fingerprint": self.state_fingerprint,
            "state_id": self.state_id,
        }


@dataclass(frozen=True, slots=True)
class WebAwareProjectStoryRanking:
    story_relevance: tuple[ProjectStoryRelevanceInput, ...]
    projects: tuple[ProjectHiringRelevance, ...]
    portfolio: Any
    state: ProjectStoryRankingState | WebAwareProjectStoryRankingState
    web_influence_applied: bool


def build_web_aware_project_story_ranking_state(
    *,
    hiring_context: HiringContextProfile,
    enrichment: HiringContextEnrichment,
    portfolio: Any,
    project_snapshots: Sequence[ProjectStoryRelevanceSnapshot],
) -> WebAwareProjectStoryRankingState:
    _validate_enrichment_binding(hiring_context, enrichment)
    offline_state = _build_project_story_ranking_state(
        hiring_context=hiring_context,
        portfolio=portfolio,
        project_snapshots=project_snapshots,
        allow_web_adjusted=True,
    )
    return WebAwareProjectStoryRankingState(
        ranking_state=offline_state,
        enrichment_fingerprint=enrichment.fingerprint,
        enrichment_status=enrichment.status,
    )


def rank_projects_with_hiring_context_web_influence(
    *,
    hiring_context: HiringContextProfile,
    enrichment: HiringContextEnrichment,
    story_relevance: Sequence[StoryHiringRelevance],
    enabled: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> WebAwareProjectStoryRanking:
    """Apply optional Story adjustments, then reuse project and portfolio policies."""

    adjusted_stories = apply_hiring_context_web_influence(
        hiring_context=hiring_context,
        enrichment=enrichment,
        story_relevance=story_relevance,
        enabled=enabled,
        environ=environ,
    )
    applied = any(
        isinstance(story, WebAdjustedStoryHiringRelevance)
        and story.web_context_adjustment.adjustment > 0.0
        for story in adjusted_stories
    )
    if applied:
        adjusted_stories = _stable_adjusted_story_order(adjusted_stories)
    projects = rank_projects_from_story_relevance(adjusted_stories)
    portfolio = rank_project_portfolio(projects=projects)
    snapshots = () if portfolio is None else tuple(
        ProjectStoryRelevanceSnapshot(
            snapshot_policy_id=PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
            source_portfolio_id=portfolio.portfolio_id,
            project_relevance=project,
            story_relevance=tuple(
                story
                for story in adjusted_stories
                if story.project_id == project.project_id
            ),
        )
        for project in projects
    )
    if applied:
        state: ProjectStoryRankingState | WebAwareProjectStoryRankingState = (
            build_web_aware_project_story_ranking_state(
                hiring_context=hiring_context,
                enrichment=enrichment,
                portfolio=portfolio,
                project_snapshots=snapshots,
            )
        )
    else:
        state = build_project_story_ranking_state(
            hiring_context=hiring_context,
            portfolio=portfolio,
            project_snapshots=snapshots,
        )
    return WebAwareProjectStoryRanking(
        story_relevance=tuple(adjusted_stories),
        projects=projects,
        portfolio=portfolio,
        state=state,
        web_influence_applied=applied,
    )


def classify_web_aware_partial_rerank(
    *,
    prior_state: WebAwareProjectStoryRankingState,
    hiring_context: HiringContextProfile,
    enrichment: HiringContextEnrichment,
    policy_id: str = OFFICIAL_HIRING_CONTEXT_WEB_RANKING_POLICY_ID,
) -> WebAwarePartialRerankDecision:
    """Classify whether a one-Story refresh may reuse web context."""

    if not isinstance(prior_state, WebAwareProjectStoryRankingState):
        raise TypeError("prior_state must be WebAwareProjectStoryRankingState")
    _validate_enrichment_binding(hiring_context, enrichment)
    requested_policy = _text(policy_id, "policy_id")
    if (
        prior_state.hiring_context_profile_id != hiring_context.profile_id
        or prior_state.hiring_context_fingerprint != hiring_context.fingerprint
        or prior_state.enrichment_fingerprint != enrichment.fingerprint
        or prior_state.enrichment_status is not enrichment.status
        or prior_state.web_influence_policy_id != requested_policy
    ):
        return WebAwarePartialRerankDecision.CONTEXT_CHANGED_REQUIRES_FULL_RERANK
    return WebAwarePartialRerankDecision.STORY_ONLY_ALLOWED


def classify_web_enrichment_change(
    *,
    prior_enrichment: HiringContextEnrichment,
    current_enrichment: HiringContextEnrichment,
) -> WebEnrichmentChangeDecision:
    """Classify enrichment freshness without using retrieval-time metadata."""

    if not isinstance(prior_enrichment, HiringContextEnrichment):
        raise TypeError("prior_enrichment must be HiringContextEnrichment")
    if not isinstance(current_enrichment, HiringContextEnrichment):
        raise TypeError("current_enrichment must be HiringContextEnrichment")
    if (
        prior_enrichment.offline_profile_id
        != current_enrichment.offline_profile_id
        or prior_enrichment.offline_profile_fingerprint
        != current_enrichment.offline_profile_fingerprint
        or prior_enrichment.fingerprint != current_enrichment.fingerprint
    ):
        return WebEnrichmentChangeDecision.CONTEXT_CHANGED_REQUIRES_FULL_RERANK
    return WebEnrichmentChangeDecision.UNCHANGED


def require_web_aware_partial_rerank_context(
    *,
    prior_state: WebAwareProjectStoryRankingState,
    hiring_context: HiringContextProfile,
    enrichment: HiringContextEnrichment,
    policy_id: str = OFFICIAL_HIRING_CONTEXT_WEB_RANKING_POLICY_ID,
) -> None:
    decision = classify_web_aware_partial_rerank(
        prior_state=prior_state,
        hiring_context=hiring_context,
        enrichment=enrichment,
        policy_id=policy_id,
    )
    if decision is not WebAwarePartialRerankDecision.STORY_ONLY_ALLOWED:
        raise WebRankingInfluenceError(
            WebRankingInfluenceErrorCode.CONTEXT_CHANGED_REQUIRES_FULL_RERANK,
            "changed web or offline context requires a full Story rerank",
        )


def refresh_web_aware_story_ranking(
    *,
    prior_state: WebAwareProjectStoryRankingState,
    hiring_context: HiringContextProfile,
    enrichment: HiringContextEnrichment,
    updated_story_relevance: StoryHiringRelevance,
    enabled: bool | None = None,
    environ: Mapping[str, str] | None = None,
    policy_id: str = OFFICIAL_HIRING_CONTEXT_WEB_RANKING_POLICY_ID,
) -> WebAwareProjectStoryRanking:
    """Reconcile one accepted Story result while the web context is unchanged.

    The replacement is already an accepted offline Story relevance result; no
    Story reconstruction occurs here.  Rebuilding the project and portfolio
    from the complete candidate Story set keeps the existing aggregation and
    differentiation policies authoritative.
    """

    require_web_aware_partial_rerank_context(
        prior_state=prior_state,
        hiring_context=hiring_context,
        enrichment=enrichment,
        policy_id=policy_id,
    )
    if not isinstance(updated_story_relevance, StoryHiringRelevance):
        raise TypeError("updated_story_relevance must be StoryHiringRelevance")
    if (
        updated_story_relevance.hiring_context_profile_id != hiring_context.profile_id
        or updated_story_relevance.hiring_context_fingerprint
        != hiring_context.fingerprint
    ):
        raise WebRankingInfluenceError(
            WebRankingInfluenceErrorCode.ENRICHMENT_CONTEXT_MISMATCH,
            "updated Story relevance is bound to a different Hiring Context",
        )
    all_base_stories: list[StoryHiringRelevance] = []
    found = False
    for snapshot in prior_state.project_snapshots:
        for story in snapshot.story_relevance:
            base = (
                story.base_relevance
                if isinstance(story, WebAdjustedStoryHiringRelevance)
                else story
            )
            if (
                base.project_id == updated_story_relevance.project_id
                and base.canonical_story_id == updated_story_relevance.canonical_story_id
            ):
                all_base_stories.append(updated_story_relevance)
                found = True
            else:
                all_base_stories.append(base)
    if not found:
        raise WebRankingInfluenceError(
            WebRankingInfluenceErrorCode.INVALID_INPUT,
            "updated Story relevance is absent from the prior ranking state",
        )
    return rank_projects_with_hiring_context_web_influence(
        hiring_context=hiring_context,
        enrichment=enrichment,
        story_relevance=tuple(all_base_stories),
        enabled=enabled,
        environ=environ,
    )


__all__ = [
    "MAX_WEB_CONTEXT_ADJUSTMENT",
    "MAX_WEB_RANKING_INFLUENCE_TRACES",
    "OFFICIAL_HIRING_CONTEXT_RANKING_INFLUENCE_FLAG",
    "OFFICIAL_HIRING_CONTEXT_WEB_RANKING_POLICY_ID",
    "StoryWebContextAdjustment",
    "WebEnrichmentChangeDecision",
    "WebAdjustedStoryHiringRelevance",
    "WebAwarePartialRerankDecision",
    "WebAwareProjectStoryRanking",
    "WebAwareProjectStoryRankingState",
    "WebRankingInfluenceError",
    "WebRankingInfluenceErrorCode",
    "WebRankingInfluenceTrace",
    "apply_hiring_context_web_influence",
    "build_web_aware_project_story_ranking_state",
    "calculate_story_web_context_adjustment",
    "classify_web_aware_partial_rerank",
    "classify_web_enrichment_change",
    "official_hiring_context_ranking_influence_enabled",
    "rank_projects_with_hiring_context_web_influence",
    "refresh_web_aware_story_ranking",
    "require_web_aware_partial_rerank_context",
]
