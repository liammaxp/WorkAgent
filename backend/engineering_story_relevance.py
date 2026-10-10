"""Pure, deterministic hiring relevance for authoritative Engineering Stories.

This module evaluates existing Story truth.  It never reconstructs Story
fields, interprets provenance identifiers as semantics, upgrades candidate
evidence, aggregates projects, or performs runtime I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
import unicodedata
from typing import Any

from backend.engineering_story_memory_service import EngineeringStoryView
from backend.engineering_story_models import (
    EngineeringStory,
    EngineeringStoryField,
    EngineeringStoryStatus,
    EngineeringStoryType,
    StoryFieldEvidenceState,
    StoryOpportunityLevel,
    SufficiencyLevel,
)
from backend.hiring_context_models import (
    HiringContextConfidence,
    HiringContextProfile,
    HiringContextSignal,
    HiringContextSourceKind,
    HiringContextSourceRef,
    RankingEffect,
    RoleFamily,
)


MAX_STORY_RELEVANCE_BATCH = 512
MAX_STORY_RELEVANCE_FEATURES = 16
MAX_STORY_RELEVANCE_REASONS = 12
MAX_STORY_RELEVANT_CONTEXT_SOURCES = 32
STORY_RELEVANCE_SCORE_DECIMALS = 6


class StoryRelevanceFeature(str, Enum):
    ARCHITECTURE = "architecture"
    RELIABILITY = "reliability"
    DEBUGGING = "debugging"
    TESTING = "testing"
    VALIDATION_REPAIR = "validation_repair"
    PERFORMANCE = "performance"
    STATE_MANAGEMENT = "state_management"
    DATA_FLOW = "data_flow"
    API_SYSTEM_DESIGN = "api_system_design"
    RETRIEVAL_RANKING = "retrieval_ranking"
    OPERATIONAL_HARDENING = "operational_hardening"
    MIGRATION = "migration"
    CONCURRENCY = "concurrency"
    ALGORITHMS = "algorithms"
    BACKEND = "backend"
    FRONTEND = "frontend"
    DATA_ENGINEERING = "data_engineering"
    ANALYTICS = "analytics"
    GAME_DEVELOPMENT = "game_development"
    REAL_TIME_SYSTEMS = "real_time_systems"
    DEVOPS_CLOUD = "devops_cloud"
    SECURITY = "security"
    MACHINE_LEARNING = "machine_learning"
    MOBILE = "mobile"
    EMBEDDED = "embedded"
    INTEGRATION = "integration"
    AUTOMATION = "automation"
    STORAGE = "storage"
    DISTRIBUTED_SYSTEMS = "distributed_systems"
    PLATFORM_ENGINEERING = "platform_engineering"


class StoryRelevanceReason(str, Enum):
    EXPLICIT_JD_ALIGNMENT = "explicit_jd_alignment"
    PRIMARY_ROLE_ALIGNMENT = "primary_role_alignment"
    SECONDARY_ROLE_ALIGNMENT = "secondary_role_alignment"
    ORGANIZATION_DOMAIN_ALIGNMENT = "organization_domain_alignment"
    TRANSFERABLE_ARCHITECTURE = "transferable_architecture"
    TRANSFERABLE_RELIABILITY = "transferable_reliability"
    TRANSFERABLE_DEBUGGING_REPAIR = "transferable_debugging_repair"
    TRANSFERABLE_TESTING_VALIDATION = "transferable_testing_validation"
    TRANSFERABLE_PERFORMANCE = "transferable_performance"
    TRANSFERABLE_DATA_SYSTEMS = "transferable_data_systems"
    TRANSFERABLE_API_SYSTEMS = "transferable_api_systems"
    TRANSFERABLE_RETRIEVAL_RANKING = "transferable_retrieval_ranking"
    TRANSFERABLE_OPERATIONAL_HARDENING = "transferable_operational_hardening"
    TRANSFERABLE_MIGRATION = "transferable_migration"
    TRANSFERABLE_CONCURRENCY_ALGORITHMS = "transferable_concurrency_algorithms"
    CLAIM_EVIDENCE_RISK = "claim_evidence_risk"
    STORY_INCOMPLETE = "story_incomplete"
    STORY_COMPLETION_OPPORTUNITY = "story_completion_opportunity"


class StoryRelevanceEvaluationErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    INACTIVE_STORY = "inactive_story"
    REVALIDATION_REQUIRED = "revalidation_required"
    DUPLICATE_STORY = "duplicate_story"
    BOUND_EXCEEDED = "bound_exceeded"


class StoryRelevanceEvaluationError(ValueError):
    def __init__(self, code: StoryRelevanceEvaluationErrorCode, message: str):
        self.code = code
        super().__init__(message)


def _score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return round(normalized, STORY_RELEVANCE_SCORE_DECIMALS)


def _stable_enum_values(
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
    try:
        normalized = {enum_type(value) for value in values}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains an unsupported value") from exc
    order = {item: index for index, item in enumerate(enum_type)}
    return tuple(sorted(normalized, key=order.__getitem__))


def _stable_source_refs(
    values: Sequence[Any],
) -> tuple[HiringContextSourceRef, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("hiring_context_source_refs must be a sequence")
    if len(values) > MAX_STORY_RELEVANT_CONTEXT_SOURCES:
        raise ValueError(
            "hiring_context_source_refs exceeds maximum item count "
            f"{MAX_STORY_RELEVANT_CONTEXT_SOURCES}"
        )
    if any(not isinstance(value, HiringContextSourceRef) for value in values):
        raise TypeError(
            "hiring_context_source_refs must contain HiringContextSourceRef values"
        )
    by_id = {value.reference_id: value for value in values}
    return tuple(by_id[key] for key in sorted(by_id))


def _exact_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must be an exact non-blank value")
    return value


@dataclass(frozen=True, slots=True)
class StoryRelevanceWeights:
    explicit_jd: float
    role_family: float
    organization_domain: float
    transferable_engineering: float

    def __post_init__(self) -> None:
        explicit = _score(self.explicit_jd, "explicit_jd weight")
        role = _score(self.role_family, "role_family weight")
        domain = _score(self.organization_domain, "organization_domain weight")
        transferable = _score(
            self.transferable_engineering,
            "transferable_engineering weight",
        )
        if not math.isclose(
            explicit + role + domain + transferable,
            1.0,
            rel_tol=0.0,
            abs_tol=10 ** -STORY_RELEVANCE_SCORE_DECIMALS,
        ):
            raise ValueError("Story relevance weights must sum to 1")
        object.__setattr__(self, "explicit_jd", explicit)
        object.__setattr__(self, "role_family", role)
        object.__setattr__(self, "organization_domain", domain)
        object.__setattr__(self, "transferable_engineering", transferable)

    def to_dict(self) -> dict[str, float]:
        return {
            "explicit_jd": self.explicit_jd,
            "role_family": self.role_family,
            "organization_domain": self.organization_domain,
            "transferable_engineering": self.transferable_engineering,
        }


@dataclass(frozen=True, slots=True)
class StoryRelevanceComponents:
    explicit_jd_relevance: float
    role_family_relevance: float
    organization_domain_relevance: float
    transferable_engineering_relevance: float
    evidence_claim_safety: float
    story_completeness: float

    def __post_init__(self) -> None:
        for name in (
            "explicit_jd_relevance",
            "role_family_relevance",
            "organization_domain_relevance",
            "transferable_engineering_relevance",
            "evidence_claim_safety",
            "story_completeness",
        ):
            object.__setattr__(self, name, _score(getattr(self, name), name))

    def to_dict(self) -> dict[str, float]:
        return {
            "explicit_jd_relevance": self.explicit_jd_relevance,
            "role_family_relevance": self.role_family_relevance,
            "organization_domain_relevance": self.organization_domain_relevance,
            "transferable_engineering_relevance": (
                self.transferable_engineering_relevance
            ),
            "evidence_claim_safety": self.evidence_claim_safety,
            "story_completeness": self.story_completeness,
        }


@dataclass(frozen=True, slots=True)
class StoryHiringRelevance:
    project_id: str
    canonical_story_id: str
    current_revision_id: str
    hiring_context_profile_id: str
    hiring_context_fingerprint: str
    story_provenance_fingerprint: str
    lifecycle_status: EngineeringStoryStatus
    claim_sufficiency: SufficiencyLevel
    story_sufficiency: SufficiencyLevel
    story_opportunity: StoryOpportunityLevel
    components: StoryRelevanceComponents
    weights: StoryRelevanceWeights
    raw_relevance_score: float
    evidence_risk_adjustment: float
    total_relevance_score: float
    clarification_value_hint: float
    semantic_features: tuple[StoryRelevanceFeature, ...]
    reasons: tuple[StoryRelevanceReason, ...]
    hiring_context_source_refs: tuple[HiringContextSourceRef, ...]
    relevance_id: str = ""

    def __post_init__(self) -> None:
        project_id = _exact_text(self.project_id, "project_id")
        story_id = _exact_text(self.canonical_story_id, "canonical_story_id")
        revision_id = _exact_text(self.current_revision_id, "current_revision_id")
        profile_id = _exact_text(
            self.hiring_context_profile_id,
            "hiring_context_profile_id",
        )
        context_fingerprint = _exact_text(
            self.hiring_context_fingerprint,
            "hiring_context_fingerprint",
        )
        provenance_fingerprint = _exact_text(
            self.story_provenance_fingerprint,
            "story_provenance_fingerprint",
        )
        lifecycle = EngineeringStoryStatus(self.lifecycle_status)
        claim = SufficiencyLevel(self.claim_sufficiency)
        story = SufficiencyLevel(self.story_sufficiency)
        opportunity = StoryOpportunityLevel(self.story_opportunity)
        if not isinstance(self.components, StoryRelevanceComponents):
            raise TypeError("components must be StoryRelevanceComponents")
        if not isinstance(self.weights, StoryRelevanceWeights):
            raise TypeError("weights must be StoryRelevanceWeights")
        raw = _score(self.raw_relevance_score, "raw_relevance_score")
        risk = _score(self.evidence_risk_adjustment, "evidence_risk_adjustment")
        total = _score(self.total_relevance_score, "total_relevance_score")
        if not math.isclose(total, raw - risk, rel_tol=0.0, abs_tol=0.000002):
            raise ValueError("total_relevance_score must equal raw score minus risk")
        clarification = _score(
            self.clarification_value_hint,
            "clarification_value_hint",
        )
        features = _stable_enum_values(
            self.semantic_features,
            StoryRelevanceFeature,
            "semantic_features",
            maximum=MAX_STORY_RELEVANCE_FEATURES,
        )
        reasons = _stable_enum_values(
            self.reasons,
            StoryRelevanceReason,
            "reasons",
            maximum=MAX_STORY_RELEVANCE_REASONS,
        )
        source_refs = _stable_source_refs(self.hiring_context_source_refs)
        payload = {
            "project_id": project_id,
            "canonical_story_id": story_id,
            "current_revision_id": revision_id,
            "hiring_context_profile_id": profile_id,
            "hiring_context_fingerprint": context_fingerprint,
            "story_provenance_fingerprint": provenance_fingerprint,
            "lifecycle_status": lifecycle.value,
            "claim_sufficiency": claim.value,
            "story_sufficiency": story.value,
            "story_opportunity": opportunity.value,
            "components": self.components.to_dict(),
            "weights": self.weights.to_dict(),
            "raw_relevance_score": raw,
            "evidence_risk_adjustment": risk,
            "total_relevance_score": total,
            "clarification_value_hint": clarification,
            "semantic_features": [item.value for item in features],
            "reasons": [item.value for item in reasons],
            "hiring_context_source_ref_ids": [
                item.reference_id for item in source_refs
            ],
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        expected_id = f"story_hiring_relevance_{digest[:24]}"
        if self.relevance_id not in ("", expected_id):
            raise ValueError("relevance_id does not match normalized relevance result")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "canonical_story_id", story_id)
        object.__setattr__(self, "current_revision_id", revision_id)
        object.__setattr__(self, "hiring_context_profile_id", profile_id)
        object.__setattr__(self, "hiring_context_fingerprint", context_fingerprint)
        object.__setattr__(
            self,
            "story_provenance_fingerprint",
            provenance_fingerprint,
        )
        object.__setattr__(self, "lifecycle_status", lifecycle)
        object.__setattr__(self, "claim_sufficiency", claim)
        object.__setattr__(self, "story_sufficiency", story)
        object.__setattr__(self, "story_opportunity", opportunity)
        object.__setattr__(self, "raw_relevance_score", raw)
        object.__setattr__(self, "evidence_risk_adjustment", risk)
        object.__setattr__(self, "total_relevance_score", total)
        object.__setattr__(self, "clarification_value_hint", clarification)
        object.__setattr__(self, "semantic_features", features)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "hiring_context_source_refs", source_refs)
        object.__setattr__(self, "relevance_id", expected_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "canonical_story_id": self.canonical_story_id,
            "current_revision_id": self.current_revision_id,
            "hiring_context_profile_id": self.hiring_context_profile_id,
            "hiring_context_fingerprint": self.hiring_context_fingerprint,
            "story_provenance_fingerprint": self.story_provenance_fingerprint,
            "lifecycle_status": self.lifecycle_status.value,
            "claim_sufficiency": self.claim_sufficiency.value,
            "story_sufficiency": self.story_sufficiency.value,
            "story_opportunity": self.story_opportunity.value,
            "components": self.components.to_dict(),
            "weights": self.weights.to_dict(),
            "raw_relevance_score": self.raw_relevance_score,
            "evidence_risk_adjustment": self.evidence_risk_adjustment,
            "total_relevance_score": self.total_relevance_score,
            "clarification_value_hint": self.clarification_value_hint,
            "semantic_features": [item.value for item in self.semantic_features],
            "reasons": [item.value for item in self.reasons],
            "hiring_context_source_refs": [
                item.to_dict() for item in self.hiring_context_source_refs
            ],
            "relevance_id": self.relevance_id,
        }


@dataclass(frozen=True, slots=True)
class _SupportedStoryField:
    name: str
    value: str
    evidence_weight: float


_STORY_FIELDS: tuple[str, ...] = (
    "problem_context",
    "trigger",
    "before_state",
    "decision",
    "mechanism",
    "implementation",
    "tradeoff",
    "validation",
    "after_state",
    "observable_outcome",
    "ownership",
    "stakeholder_context",
)

_TOKEN_RE = re.compile(r"\.net|c\+\+|c#|[a-z0-9]+(?:[.-][a-z0-9]+)*")
_MATCH_STOP_WORDS = frozenset({
    "a", "an", "and", "ability", "experience", "for", "in", "knowledge",
    "of", "on", "or", "the", "to", "using", "with", "year", "years",
})
_TECHNOLOGY_TOKENS = frozenset({
    ".net",
    "aws",
    "azure",
    "c",
    "c#",
    "c++",
    "gcp",
    "java",
    "javascript",
    "rails",
    "react",
    "ruby",
    "unity",
    "unreal",
})


def _tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(_TOKEN_RE.findall(normalized))


def _content_tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in _tokens(value) if token not in _MATCH_STOP_WORDS)


def _contains_token_sequence(
    values: Sequence[str],
    phrase: Sequence[str],
) -> bool:
    if not phrase or len(phrase) > len(values):
        return False
    width = len(phrase)
    return any(tuple(values[index:index + width]) == tuple(phrase) for index in range(len(values) - width + 1))


_FEATURE_PHRASES: dict[StoryRelevanceFeature, tuple[tuple[str, ...], ...]] = {
    StoryRelevanceFeature.ARCHITECTURE: (
        _tokens("architecture"), _tokens("architectural"), _tokens("system design"),
        _tokens("redesign"), _tokens("modular"), _tokens("abstraction"),
    ),
    StoryRelevanceFeature.RELIABILITY: (
        _tokens("reliability"), _tokens("reliable"), _tokens("resilience"),
        _tokens("fault tolerance"), _tokens("failover"), _tokens("retry"),
        _tokens("idempotent"),
    ),
    StoryRelevanceFeature.DEBUGGING: (
        _tokens("debug"), _tokens("debugging"), _tokens("root cause"),
        _tokens("diagnosis"), _tokens("bug"), _tokens("defect"),
    ),
    StoryRelevanceFeature.TESTING: (
        _tokens("test"), _tokens("tests"), _tokens("testing"),
        _tokens("unit test"), _tokens("integration test"),
        _tokens("regression test"), _tokens("test coverage"),
    ),
    StoryRelevanceFeature.VALIDATION_REPAIR: (
        _tokens("validation"), _tokens("validated"), _tokens("verification"),
        _tokens("verify"), _tokens("repair"), _tokens("fail closed"),
    ),
    StoryRelevanceFeature.PERFORMANCE: (
        _tokens("performance"), _tokens("latency"), _tokens("throughput"),
        _tokens("optimize"), _tokens("optimized"), _tokens("efficiency"),
        _tokens("memory usage"),
    ),
    StoryRelevanceFeature.STATE_MANAGEMENT: (
        _tokens("state management"), _tokens("state machine"),
        _tokens("lifecycle"), _tokens("session state"), _tokens("game state"),
    ),
    StoryRelevanceFeature.DATA_FLOW: (
        _tokens("data flow"), _tokens("pipeline"), _tokens("stream"),
        _tokens("ingestion"), _tokens("transformation"),
    ),
    StoryRelevanceFeature.API_SYSTEM_DESIGN: (
        _tokens("api"), _tokens("endpoint"), _tokens("rest"),
        _tokens("graphql"), _tokens("service design"), _tokens("system api"),
    ),
    StoryRelevanceFeature.RETRIEVAL_RANKING: (
        _tokens("retrieval"), _tokens("search"), _tokens("ranking"),
        _tokens("ranker"), _tokens("vector search"), _tokens("query planner"),
    ),
    StoryRelevanceFeature.OPERATIONAL_HARDENING: (
        _tokens("operational"), _tokens("hardening"), _tokens("observability"),
        _tokens("monitoring"), _tokens("health check"), _tokens("backup"),
        _tokens("recovery"),
    ),
    StoryRelevanceFeature.MIGRATION: (
        _tokens("migration"), _tokens("migrated"), _tokens("upgrade"),
        _tokens("transition"), _tokens("schema change"),
    ),
    StoryRelevanceFeature.CONCURRENCY: (
        _tokens("concurrency"), _tokens("concurrent"), _tokens("async"),
        _tokens("asynchronous"), _tokens("parallel"), _tokens("thread"),
        _tokens("synchronization"), _tokens("race condition"),
    ),
    StoryRelevanceFeature.ALGORITHMS: (
        _tokens("algorithm"), _tokens("sorting"), _tokens("graph algorithm"),
        _tokens("dynamic programming"), _tokens("complexity"),
    ),
    StoryRelevanceFeature.BACKEND: (
        _tokens("backend"), _tokens("server"), _tokens("service"),
        _tokens("database"), _tokens("storage"),
    ),
    StoryRelevanceFeature.FRONTEND: (
        _tokens("frontend"), _tokens("user interface"), _tokens("ui"),
        _tokens("browser"), _tokens("client"), _tokens("react"),
        _tokens("component"), _tokens("accessibility"), _tokens("css"),
    ),
    StoryRelevanceFeature.DATA_ENGINEERING: (
        _tokens("data engineering"), _tokens("etl"), _tokens("data pipeline"),
        _tokens("data ingestion"), _tokens("warehouse"),
        _tokens("data transformation"), _tokens("data infrastructure"),
    ),
    StoryRelevanceFeature.ANALYTICS: (
        _tokens("analytics"), _tokens("analysis"), _tokens("dashboard"),
        _tokens("metrics"), _tokens("kpi"), _tokens("decision support"),
        _tokens("business intelligence"),
    ),
    StoryRelevanceFeature.GAME_DEVELOPMENT: (
        _tokens("game development"), _tokens("gameplay"), _tokens("game system"),
        _tokens("game engine"), _tokens("player"), _tokens("game state"),
        _tokens("interactive simulation"),
    ),
    StoryRelevanceFeature.REAL_TIME_SYSTEMS: (
        _tokens("real-time"), _tokens("real time"), _tokens("realtime"),
        _tokens("frame loop"), _tokens("low latency"),
        _tokens("interactive system"),
    ),
    StoryRelevanceFeature.DEVOPS_CLOUD: (
        _tokens("devops"), _tokens("cloud"), _tokens("deployment"),
        _tokens("ci/cd"), _tokens("infrastructure"), _tokens("container"),
        _tokens("kubernetes"), _tokens("aws"), _tokens("azure"), _tokens("gcp"),
    ),
    StoryRelevanceFeature.SECURITY: (
        _tokens("security"), _tokens("secure"), _tokens("authentication"),
        _tokens("authorization"), _tokens("vulnerability"), _tokens("privacy"),
        _tokens("encryption"), _tokens("access control"), _tokens("threat"),
    ),
    StoryRelevanceFeature.MACHINE_LEARNING: (
        _tokens("machine learning"), _tokens("ml"), _tokens("model training"),
        _tokens("inference"), _tokens("neural network"),
    ),
    StoryRelevanceFeature.MOBILE: (
        _tokens("mobile"), _tokens("android"), _tokens("ios"),
    ),
    StoryRelevanceFeature.EMBEDDED: (
        _tokens("embedded"), _tokens("firmware"), _tokens("microcontroller"),
        _tokens("hardware"),
    ),
    StoryRelevanceFeature.INTEGRATION: (
        _tokens("integration"), _tokens("integrated"), _tokens("connector"),
        _tokens("external service"),
    ),
    StoryRelevanceFeature.AUTOMATION: (
        _tokens("automation"), _tokens("automated"), _tokens("workflow"),
    ),
    StoryRelevanceFeature.STORAGE: (
        _tokens("database"), _tokens("storage"), _tokens("persistence"),
        _tokens("cache"), _tokens("memory system"),
    ),
    StoryRelevanceFeature.DISTRIBUTED_SYSTEMS: (
        _tokens("distributed"), _tokens("service architecture"),
        _tokens("message queue"), _tokens("event-driven"),
        _tokens("microservice"),
    ),
    StoryRelevanceFeature.PLATFORM_ENGINEERING: (
        _tokens("platform"), _tokens("developer platform"),
        _tokens("software platform"), _tokens("infrastructure platform"),
    ),
}

_STORY_TYPE_FEATURES: dict[EngineeringStoryType, tuple[StoryRelevanceFeature, ...]] = {
    EngineeringStoryType.ARCHITECTURE_CHANGE: (
        StoryRelevanceFeature.ARCHITECTURE,
    ),
    EngineeringStoryType.RELIABILITY_HARDENING: (
        StoryRelevanceFeature.RELIABILITY,
        StoryRelevanceFeature.OPERATIONAL_HARDENING,
    ),
    EngineeringStoryType.DEBUGGING_AND_REPAIR: (
        StoryRelevanceFeature.DEBUGGING,
        StoryRelevanceFeature.VALIDATION_REPAIR,
    ),
    EngineeringStoryType.RETRIEVAL_REDESIGN: (
        StoryRelevanceFeature.RETRIEVAL_RANKING,
        StoryRelevanceFeature.ARCHITECTURE,
    ),
    EngineeringStoryType.VALIDATION_AND_QUALITY: (
        StoryRelevanceFeature.TESTING,
        StoryRelevanceFeature.VALIDATION_REPAIR,
    ),
    EngineeringStoryType.DATA_OR_MEMORY_SYSTEM: (
        StoryRelevanceFeature.DATA_ENGINEERING,
        StoryRelevanceFeature.STORAGE,
        StoryRelevanceFeature.STATE_MANAGEMENT,
    ),
    EngineeringStoryType.WORKFLOW_AUTOMATION: (
        StoryRelevanceFeature.AUTOMATION,
    ),
    EngineeringStoryType.PERFORMANCE_OR_EFFICIENCY: (
        StoryRelevanceFeature.PERFORMANCE,
    ),
    EngineeringStoryType.INTEGRATION: (
        StoryRelevanceFeature.INTEGRATION,
    ),
    EngineeringStoryType.OTHER: (),
}

_ROLE_FEATURES: dict[RoleFamily, frozenset[StoryRelevanceFeature]] = {
    RoleFamily.BACKEND_ENGINEERING: frozenset({
        StoryRelevanceFeature.BACKEND,
        StoryRelevanceFeature.API_SYSTEM_DESIGN,
        StoryRelevanceFeature.STORAGE,
        StoryRelevanceFeature.RELIABILITY,
        StoryRelevanceFeature.ARCHITECTURE,
        StoryRelevanceFeature.DISTRIBUTED_SYSTEMS,
    }),
    RoleFamily.FRONTEND_ENGINEERING: frozenset({
        StoryRelevanceFeature.FRONTEND,
        StoryRelevanceFeature.STATE_MANAGEMENT,
        StoryRelevanceFeature.PERFORMANCE,
        StoryRelevanceFeature.TESTING,
    }),
    RoleFamily.FULL_STACK_ENGINEERING: frozenset({
        StoryRelevanceFeature.FRONTEND,
        StoryRelevanceFeature.BACKEND,
        StoryRelevanceFeature.API_SYSTEM_DESIGN,
        StoryRelevanceFeature.INTEGRATION,
        StoryRelevanceFeature.STATE_MANAGEMENT,
    }),
    RoleFamily.DATA_ENGINEERING: frozenset({
        StoryRelevanceFeature.DATA_ENGINEERING,
        StoryRelevanceFeature.DATA_FLOW,
        StoryRelevanceFeature.STORAGE,
        StoryRelevanceFeature.DISTRIBUTED_SYSTEMS,
    }),
    RoleFamily.DATA_ANALYTICS: frozenset({
        StoryRelevanceFeature.ANALYTICS,
        StoryRelevanceFeature.DATA_FLOW,
        StoryRelevanceFeature.DATA_ENGINEERING,
    }),
    RoleFamily.MACHINE_LEARNING_AI: frozenset({
        StoryRelevanceFeature.MACHINE_LEARNING,
        StoryRelevanceFeature.DATA_ENGINEERING,
        StoryRelevanceFeature.ALGORITHMS,
        StoryRelevanceFeature.PERFORMANCE,
    }),
    RoleFamily.DEVOPS_CLOUD: frozenset({
        StoryRelevanceFeature.DEVOPS_CLOUD,
        StoryRelevanceFeature.OPERATIONAL_HARDENING,
        StoryRelevanceFeature.RELIABILITY,
        StoryRelevanceFeature.MIGRATION,
        StoryRelevanceFeature.DISTRIBUTED_SYSTEMS,
    }),
    RoleFamily.GAME_DEVELOPMENT: frozenset({
        StoryRelevanceFeature.GAME_DEVELOPMENT,
        StoryRelevanceFeature.REAL_TIME_SYSTEMS,
        StoryRelevanceFeature.STATE_MANAGEMENT,
        StoryRelevanceFeature.PERFORMANCE,
        StoryRelevanceFeature.ALGORITHMS,
    }),
    RoleFamily.MOBILE: frozenset({
        StoryRelevanceFeature.MOBILE,
        StoryRelevanceFeature.FRONTEND,
        StoryRelevanceFeature.STATE_MANAGEMENT,
        StoryRelevanceFeature.PERFORMANCE,
    }),
    RoleFamily.EMBEDDED_SYSTEMS: frozenset({
        StoryRelevanceFeature.EMBEDDED,
        StoryRelevanceFeature.REAL_TIME_SYSTEMS,
        StoryRelevanceFeature.PERFORMANCE,
        StoryRelevanceFeature.CONCURRENCY,
    }),
    RoleFamily.SECURITY: frozenset({
        StoryRelevanceFeature.SECURITY,
        StoryRelevanceFeature.RELIABILITY,
        StoryRelevanceFeature.VALIDATION_REPAIR,
    }),
    RoleFamily.CONSULTING_STRATEGY: frozenset({
        StoryRelevanceFeature.ANALYTICS,
        StoryRelevanceFeature.ARCHITECTURE,
        StoryRelevanceFeature.INTEGRATION,
    }),
}

_ROLE_ANCHORS: dict[RoleFamily, frozenset[StoryRelevanceFeature]] = {
    RoleFamily.BACKEND_ENGINEERING: frozenset({
        StoryRelevanceFeature.BACKEND,
        StoryRelevanceFeature.API_SYSTEM_DESIGN,
        StoryRelevanceFeature.STORAGE,
        StoryRelevanceFeature.DISTRIBUTED_SYSTEMS,
    }),
    RoleFamily.FRONTEND_ENGINEERING: frozenset({
        StoryRelevanceFeature.FRONTEND,
    }),
    RoleFamily.FULL_STACK_ENGINEERING: frozenset({
        StoryRelevanceFeature.BACKEND,
        StoryRelevanceFeature.FRONTEND,
        StoryRelevanceFeature.API_SYSTEM_DESIGN,
    }),
    RoleFamily.DATA_ENGINEERING: frozenset({
        StoryRelevanceFeature.DATA_ENGINEERING,
        StoryRelevanceFeature.DATA_FLOW,
    }),
    RoleFamily.DATA_ANALYTICS: frozenset({
        StoryRelevanceFeature.ANALYTICS,
    }),
    RoleFamily.MACHINE_LEARNING_AI: frozenset({
        StoryRelevanceFeature.MACHINE_LEARNING,
    }),
    RoleFamily.DEVOPS_CLOUD: frozenset({
        StoryRelevanceFeature.DEVOPS_CLOUD,
    }),
    RoleFamily.GAME_DEVELOPMENT: frozenset({
        StoryRelevanceFeature.GAME_DEVELOPMENT,
        StoryRelevanceFeature.REAL_TIME_SYSTEMS,
    }),
    RoleFamily.MOBILE: frozenset({StoryRelevanceFeature.MOBILE}),
    RoleFamily.EMBEDDED_SYSTEMS: frozenset({StoryRelevanceFeature.EMBEDDED}),
    RoleFamily.SECURITY: frozenset({StoryRelevanceFeature.SECURITY}),
    RoleFamily.CONSULTING_STRATEGY: frozenset({
        StoryRelevanceFeature.ANALYTICS,
    }),
}

_CONTEXT_ANCHOR_FEATURES = frozenset({
    StoryRelevanceFeature.ANALYTICS,
    StoryRelevanceFeature.GAME_DEVELOPMENT,
    StoryRelevanceFeature.DEVOPS_CLOUD,
    StoryRelevanceFeature.SECURITY,
    StoryRelevanceFeature.MACHINE_LEARNING,
    StoryRelevanceFeature.MOBILE,
    StoryRelevanceFeature.EMBEDDED,
})

_TRANSFERABLE_FEATURES = frozenset({
    StoryRelevanceFeature.ARCHITECTURE,
    StoryRelevanceFeature.RELIABILITY,
    StoryRelevanceFeature.DEBUGGING,
    StoryRelevanceFeature.TESTING,
    StoryRelevanceFeature.VALIDATION_REPAIR,
    StoryRelevanceFeature.PERFORMANCE,
    StoryRelevanceFeature.STATE_MANAGEMENT,
    StoryRelevanceFeature.DATA_FLOW,
    StoryRelevanceFeature.API_SYSTEM_DESIGN,
    StoryRelevanceFeature.RETRIEVAL_RANKING,
    StoryRelevanceFeature.OPERATIONAL_HARDENING,
    StoryRelevanceFeature.MIGRATION,
    StoryRelevanceFeature.CONCURRENCY,
    StoryRelevanceFeature.ALGORITHMS,
    StoryRelevanceFeature.INTEGRATION,
    StoryRelevanceFeature.AUTOMATION,
    StoryRelevanceFeature.STORAGE,
    StoryRelevanceFeature.DISTRIBUTED_SYSTEMS,
})

_EVIDENCE_WEIGHTS = {
    StoryFieldEvidenceState.CONFIRMED: 1.0,
    StoryFieldEvidenceState.SUPPORTED: 0.75,
}
_CONTEXT_CONFIDENCE_WEIGHTS = {
    HiringContextConfidence.HIGH: 1.0,
    HiringContextConfidence.MEDIUM: 0.8,
    HiringContextConfidence.LOW: 0.6,
}
_ORGANIZATION_SOURCE_KINDS = frozenset({
    HiringContextSourceKind.COMPANY_IDENTITY,
    HiringContextSourceKind.TEAM_IDENTITY,
    HiringContextSourceKind.PARENT_ORGANIZATION_IDENTITY,
    HiringContextSourceKind.INTERNAL_TAXONOMY,
})
_CLAIM_SAFETY = {
    SufficiencyLevel.HIGH: 1.0,
    SufficiencyLevel.MEDIUM: 0.92,
    SufficiencyLevel.LOW: 0.75,
    SufficiencyLevel.UNASSESSED: 0.68,
}
_STORY_COMPLETENESS = {
    SufficiencyLevel.HIGH: 1.0,
    SufficiencyLevel.MEDIUM: 0.67,
    SufficiencyLevel.LOW: 0.33,
    SufficiencyLevel.UNASSESSED: 0.0,
}
_OPPORTUNITY_HINT = {
    StoryOpportunityLevel.NONE: 0.0,
    StoryOpportunityLevel.LOW: 0.33,
    StoryOpportunityLevel.MEDIUM: 0.67,
    StoryOpportunityLevel.HIGH: 1.0,
}


def _supported_story_fields(story: EngineeringStory) -> tuple[_SupportedStoryField, ...]:
    values = []
    for name in _STORY_FIELDS:
        field = getattr(story, name)
        if not isinstance(field, EngineeringStoryField):
            raise TypeError(f"current_story.{name} must be EngineeringStoryField")
        weight = _EVIDENCE_WEIGHTS.get(field.evidence_state)
        if weight is None:
            continue
        if field.value is None:
            raise StoryRelevanceEvaluationError(
                StoryRelevanceEvaluationErrorCode.INVALID_INPUT,
                "positive Story fields must carry factual values",
            )
        values.append(_SupportedStoryField(name, field.value, weight))
    return tuple(values)


def _features_for_text(value: str) -> frozenset[StoryRelevanceFeature]:
    tokens = _tokens(value)
    return frozenset(
        feature
        for feature, phrases in _FEATURE_PHRASES.items()
        if any(_contains_token_sequence(tokens, phrase) for phrase in phrases)
    )


def _story_feature_strengths(
    story: EngineeringStory,
    contents: Sequence[_SupportedStoryField],
) -> dict[StoryRelevanceFeature, float]:
    strengths = {
        feature: 0.7 for feature in _STORY_TYPE_FEATURES[story.story_type]
    }
    for item in contents:
        for feature in _features_for_text(item.value):
            strengths[feature] = max(strengths.get(feature, 0.0), item.evidence_weight)
        if item.name == "validation":
            feature = StoryRelevanceFeature.VALIDATION_REPAIR
            strengths[feature] = max(
                strengths.get(feature, 0.0),
                item.evidence_weight * 0.65,
            )
    return strengths


def _lexical_match(signal_value: str, story_value: str) -> float:
    signal_tokens = _content_tokens(signal_value)
    story_tokens = _content_tokens(story_value)
    if not signal_tokens or not story_tokens:
        return 0.0
    if _contains_token_sequence(story_tokens, signal_tokens):
        return 1.0
    signal_set = set(signal_tokens)
    overlap = signal_set & set(story_tokens)
    coverage = len(overlap) / len(signal_set)
    if coverage >= 0.75 and len(overlap) >= 2:
        return 0.8
    if coverage >= 0.5 and len(overlap) >= 2:
        return 0.6
    return 0.0


def _semantic_match(
    signal_value: str,
    story_value: str,
) -> float:
    if set(_content_tokens(signal_value)) & _TECHNOLOGY_TOKENS:
        return 0.0
    signal_features = _features_for_text(signal_value)
    if not signal_features:
        return 0.0
    story_features = _features_for_text(story_value)
    signal_anchors = signal_features & _CONTEXT_ANCHOR_FEATURES
    if signal_anchors and not signal_anchors & story_features:
        return 0.0
    overlap = signal_features & story_features
    if not overlap:
        return 0.0
    return min(1.0, 0.85 + 0.15 * len(overlap) / len(signal_features))


def _signal_match(
    signal: HiringContextSignal,
    contents: Sequence[_SupportedStoryField],
) -> float:
    matches = [
        max(
            _lexical_match(signal.value, item.value),
            _semantic_match(signal.value, item.value),
        ) * item.evidence_weight
        for item in contents
    ]
    return _score(
        (max(matches) if matches else 0.0)
        * _CONTEXT_CONFIDENCE_WEIGHTS[signal.confidence],
        "signal_match",
    )


def _aggregate_signal_matches(values: Sequence[float]) -> float:
    positive = sorted((value for value in values if value > 0.0), reverse=True)
    if not positive:
        return 0.0
    breadth = sum(positive[:3]) / 3.0
    return _score(0.7 * positive[0] + 0.3 * breadth, "signal relevance")


def _family_score(
    family: RoleFamily,
    feature_strengths: dict[StoryRelevanceFeature, float],
) -> float:
    transferable = [
        strength
        for feature, strength in feature_strengths.items()
        if feature in _TRANSFERABLE_FEATURES
    ]
    if family is RoleFamily.UNKNOWN:
        return 0.0
    if family is RoleFamily.GENERAL:
        return _score(min(0.6, 0.25 + 0.07 * len(transferable)), "general role") if transferable else 0.0
    if family is RoleFamily.SOFTWARE_ENGINEERING:
        return _score(min(0.85, 0.45 + 0.08 * len(transferable)), "software role") if transferable else 0.0
    desired = _ROLE_FEATURES.get(family, frozenset())
    anchors = _ROLE_ANCHORS.get(family, frozenset())
    if anchors and not anchors & feature_strengths.keys():
        return 0.0
    matches = sorted(
        (feature_strengths[feature] for feature in desired if feature in feature_strengths),
        reverse=True,
    )
    if not matches:
        return 0.0
    breadth = min(1.0, len(matches) / 3.0)
    return _score(0.55 * matches[0] + 0.45 * breadth, "role relevance")


def _transferable_score(
    feature_strengths: dict[StoryRelevanceFeature, float],
) -> float:
    strengths = sorted(
        (
            strength
            for feature, strength in feature_strengths.items()
            if feature in _TRANSFERABLE_FEATURES
        ),
        reverse=True,
    )
    if not strengths:
        return 0.0
    return _score(min(1.0, 0.2 + 0.16 * sum(strengths[:5])), "transferable relevance")


def _weights_for_context(
    hiring_context: HiringContextProfile,
    domain_signals: Sequence[HiringContextSignal],
) -> StoryRelevanceWeights:
    if not domain_signals:
        return StoryRelevanceWeights(0.38, 0.22, 0.0, 0.40)
    domain_specific_roles = {
        RoleFamily.DATA_ENGINEERING,
        RoleFamily.DATA_ANALYTICS,
        RoleFamily.MACHINE_LEARNING_AI,
        RoleFamily.DEVOPS_CLOUD,
        RoleFamily.GAME_DEVELOPMENT,
        RoleFamily.MOBILE,
        RoleFamily.EMBEDDED_SYSTEMS,
        RoleFamily.SECURITY,
    }
    if (
        hiring_context.primary_role_family in domain_specific_roles
        and any(
            signal.confidence is HiringContextConfidence.HIGH
            for signal in domain_signals
        )
    ):
        return StoryRelevanceWeights(0.27, 0.20, 0.25, 0.28)
    return StoryRelevanceWeights(0.30, 0.20, 0.20, 0.30)


def _transferable_reasons(
    features: set[StoryRelevanceFeature],
) -> set[StoryRelevanceReason]:
    reasons = set()
    mappings = (
        ({StoryRelevanceFeature.ARCHITECTURE}, StoryRelevanceReason.TRANSFERABLE_ARCHITECTURE),
        ({StoryRelevanceFeature.RELIABILITY}, StoryRelevanceReason.TRANSFERABLE_RELIABILITY),
        ({StoryRelevanceFeature.DEBUGGING, StoryRelevanceFeature.VALIDATION_REPAIR}, StoryRelevanceReason.TRANSFERABLE_DEBUGGING_REPAIR),
        ({StoryRelevanceFeature.TESTING, StoryRelevanceFeature.VALIDATION_REPAIR}, StoryRelevanceReason.TRANSFERABLE_TESTING_VALIDATION),
        ({StoryRelevanceFeature.PERFORMANCE}, StoryRelevanceReason.TRANSFERABLE_PERFORMANCE),
        ({StoryRelevanceFeature.DATA_FLOW, StoryRelevanceFeature.DATA_ENGINEERING, StoryRelevanceFeature.STORAGE}, StoryRelevanceReason.TRANSFERABLE_DATA_SYSTEMS),
        ({StoryRelevanceFeature.API_SYSTEM_DESIGN, StoryRelevanceFeature.DISTRIBUTED_SYSTEMS}, StoryRelevanceReason.TRANSFERABLE_API_SYSTEMS),
        ({StoryRelevanceFeature.RETRIEVAL_RANKING}, StoryRelevanceReason.TRANSFERABLE_RETRIEVAL_RANKING),
        ({StoryRelevanceFeature.OPERATIONAL_HARDENING}, StoryRelevanceReason.TRANSFERABLE_OPERATIONAL_HARDENING),
        ({StoryRelevanceFeature.MIGRATION}, StoryRelevanceReason.TRANSFERABLE_MIGRATION),
        ({StoryRelevanceFeature.CONCURRENCY, StoryRelevanceFeature.ALGORITHMS}, StoryRelevanceReason.TRANSFERABLE_CONCURRENCY_ALGORITHMS),
    )
    for accepted, reason in mappings:
        if features & accepted:
            reasons.add(reason)
    return reasons


def _validate_inputs(
    hiring_context: HiringContextProfile,
    story_view: EngineeringStoryView,
) -> None:
    if not isinstance(hiring_context, HiringContextProfile):
        raise TypeError("hiring_context must be HiringContextProfile")
    if not isinstance(story_view, EngineeringStoryView):
        raise TypeError("story_view must be EngineeringStoryView")
    if story_view.lifecycle.status is not EngineeringStoryStatus.ACTIVE:
        raise StoryRelevanceEvaluationError(
            StoryRelevanceEvaluationErrorCode.INACTIVE_STORY,
            "only active authoritative Engineering Stories may be evaluated",
        )
    if story_view.lifecycle.requires_revalidation:
        raise StoryRelevanceEvaluationError(
            StoryRelevanceEvaluationErrorCode.REVALIDATION_REQUIRED,
            "Stories requiring revalidation are not rankable",
        )
    story = story_view.current_story
    if (
        story_view.evidence_fact_ids != story.evidence_fact_ids
        or story_view.capability_fact_ids != story.capability_fact_ids
        or story_view.claim_boundary_ids != story.claim_boundary_ids
    ):
        raise StoryRelevanceEvaluationError(
            StoryRelevanceEvaluationErrorCode.INVALID_INPUT,
            "Story View authority references must match the current Story",
        )


def evaluate_engineering_story_relevance(
    *,
    hiring_context: HiringContextProfile,
    story_view: EngineeringStoryView,
) -> StoryHiringRelevance:
    """Evaluate one active authoritative Story without changing Story truth."""

    _validate_inputs(hiring_context, story_view)
    story = story_view.current_story
    contents = _supported_story_fields(story)
    feature_strengths = _story_feature_strengths(story, contents)

    explicit_signals = tuple(
        signal
        for signal in hiring_context.signals
        if RankingEffect.EXPLICIT_ALIGNMENT in signal.ranking_effects
        and any(
            source.source_kind is HiringContextSourceKind.JOB_DESCRIPTION
            for source in signal.source_refs
        )
    )
    domain_signals = tuple(
        signal
        for signal in hiring_context.signals
        if RankingEffect.DOMAIN_ALIGNMENT in signal.ranking_effects
        and any(
            source.source_kind in _ORGANIZATION_SOURCE_KINDS
            for source in signal.source_refs
        )
    )
    explicit_matches = {
        signal.signal_id: _signal_match(signal, contents)
        for signal in explicit_signals
    }
    domain_matches = {
        signal.signal_id: _signal_match(signal, contents)
        for signal in domain_signals
    }
    explicit_score = _aggregate_signal_matches(tuple(explicit_matches.values()))
    domain_score = _aggregate_signal_matches(tuple(domain_matches.values()))

    primary_role_score = _family_score(
        hiring_context.primary_role_family,
        feature_strengths,
    )
    secondary_role_score = max(
        (
            _family_score(family, feature_strengths) * 0.8
            for family in hiring_context.secondary_role_families
        ),
        default=0.0,
    )
    role_score = _score(
        max(primary_role_score, secondary_role_score),
        "role_family_relevance",
    )
    transferable_score = _transferable_score(feature_strengths)
    evidence_safety = _CLAIM_SAFETY[story_view.claim_sufficiency.level]
    story_completeness = _STORY_COMPLETENESS[story_view.story_sufficiency.level]
    components = StoryRelevanceComponents(
        explicit_jd_relevance=explicit_score,
        role_family_relevance=role_score,
        organization_domain_relevance=domain_score,
        transferable_engineering_relevance=transferable_score,
        evidence_claim_safety=evidence_safety,
        story_completeness=story_completeness,
    )
    weights = _weights_for_context(hiring_context, domain_signals)
    raw_score = _score(
        explicit_score * weights.explicit_jd
        + role_score * weights.role_family
        + domain_score * weights.organization_domain
        + transferable_score * weights.transferable_engineering,
        "raw_relevance_score",
    )
    total_score = _score(raw_score * evidence_safety, "total_relevance_score")
    risk_adjustment = _score(raw_score - total_score, "evidence_risk_adjustment")

    reasons: set[StoryRelevanceReason] = set()
    if explicit_score > 0.0:
        reasons.add(StoryRelevanceReason.EXPLICIT_JD_ALIGNMENT)
    if primary_role_score > 0.0:
        reasons.add(StoryRelevanceReason.PRIMARY_ROLE_ALIGNMENT)
    elif secondary_role_score > 0.0:
        reasons.add(StoryRelevanceReason.SECONDARY_ROLE_ALIGNMENT)
    if domain_score > 0.0:
        reasons.add(StoryRelevanceReason.ORGANIZATION_DOMAIN_ALIGNMENT)
    reasons.update(_transferable_reasons(set(feature_strengths)))
    if story_view.claim_sufficiency.level in {
        SufficiencyLevel.LOW,
        SufficiencyLevel.UNASSESSED,
    }:
        reasons.add(StoryRelevanceReason.CLAIM_EVIDENCE_RISK)
    if story_view.story_sufficiency.level in {
        SufficiencyLevel.LOW,
        SufficiencyLevel.UNASSESSED,
    }:
        reasons.add(StoryRelevanceReason.STORY_INCOMPLETE)
    if story_view.opportunity.level is not StoryOpportunityLevel.NONE:
        reasons.add(StoryRelevanceReason.STORY_COMPLETION_OPPORTUNITY)
    reason_priority = (
        StoryRelevanceReason.EXPLICIT_JD_ALIGNMENT,
        StoryRelevanceReason.PRIMARY_ROLE_ALIGNMENT,
        StoryRelevanceReason.SECONDARY_ROLE_ALIGNMENT,
        StoryRelevanceReason.ORGANIZATION_DOMAIN_ALIGNMENT,
        StoryRelevanceReason.CLAIM_EVIDENCE_RISK,
        StoryRelevanceReason.STORY_INCOMPLETE,
        StoryRelevanceReason.STORY_COMPLETION_OPPORTUNITY,
        *tuple(
            item
            for item in StoryRelevanceReason
            if item not in {
                StoryRelevanceReason.EXPLICIT_JD_ALIGNMENT,
                StoryRelevanceReason.PRIMARY_ROLE_ALIGNMENT,
                StoryRelevanceReason.SECONDARY_ROLE_ALIGNMENT,
                StoryRelevanceReason.ORGANIZATION_DOMAIN_ALIGNMENT,
                StoryRelevanceReason.CLAIM_EVIDENCE_RISK,
                StoryRelevanceReason.STORY_INCOMPLETE,
                StoryRelevanceReason.STORY_COMPLETION_OPPORTUNITY,
            }
        ),
    )
    reason_order = {item: index for index, item in enumerate(reason_priority)}
    stable_reasons = tuple(sorted(reasons, key=reason_order.__getitem__))[
        :MAX_STORY_RELEVANCE_REASONS
    ]

    source_refs = {
        source.reference_id: source
        for signal in (*explicit_signals, *domain_signals)
        if (
            explicit_matches.get(signal.signal_id, 0.0) > 0.0
            or domain_matches.get(signal.signal_id, 0.0) > 0.0
        )
        for source in signal.source_refs
    }
    if role_score > 0.0:
        for signal in hiring_context.signals:
            if RankingEffect.ROLE_FAMILY_ALIGNMENT in signal.ranking_effects:
                for source in signal.source_refs:
                    source_refs[source.reference_id] = source

    feature_order = {item: index for index, item in enumerate(StoryRelevanceFeature)}
    stable_features = tuple(
        sorted(feature_strengths, key=feature_order.__getitem__)
    )[:MAX_STORY_RELEVANCE_FEATURES]
    return StoryHiringRelevance(
        project_id=story_view.project_id,
        canonical_story_id=story_view.canonical_story_id,
        current_revision_id=story_view.current_revision_id,
        hiring_context_profile_id=hiring_context.profile_id,
        hiring_context_fingerprint=hiring_context.fingerprint,
        story_provenance_fingerprint=story_view.provenance_fingerprint,
        lifecycle_status=story_view.lifecycle.status,
        claim_sufficiency=story_view.claim_sufficiency.level,
        story_sufficiency=story_view.story_sufficiency.level,
        story_opportunity=story_view.opportunity.level,
        components=components,
        weights=weights,
        raw_relevance_score=raw_score,
        evidence_risk_adjustment=risk_adjustment,
        total_relevance_score=total_score,
        clarification_value_hint=_OPPORTUNITY_HINT[story_view.opportunity.level],
        semantic_features=stable_features,
        reasons=stable_reasons,
        hiring_context_source_refs=tuple(source_refs.values()),
    )


def rank_engineering_stories_for_hiring_context(
    *,
    hiring_context: HiringContextProfile,
    story_views: Sequence[EngineeringStoryView],
) -> tuple[StoryHiringRelevance, ...]:
    """Evaluate and stably sort Stories without aggregating them by project."""

    if not isinstance(hiring_context, HiringContextProfile):
        raise TypeError("hiring_context must be HiringContextProfile")
    if isinstance(story_views, (str, bytes)) or not isinstance(story_views, Sequence):
        raise TypeError("story_views must be a sequence")
    if len(story_views) > MAX_STORY_RELEVANCE_BATCH:
        raise StoryRelevanceEvaluationError(
            StoryRelevanceEvaluationErrorCode.BOUND_EXCEEDED,
            f"story_views exceeds maximum item count {MAX_STORY_RELEVANCE_BATCH}",
        )
    seen = set()
    results = []
    for story_view in story_views:
        if not isinstance(story_view, EngineeringStoryView):
            raise TypeError("story_views must contain EngineeringStoryView values")
        identity = (story_view.project_id, story_view.canonical_story_id)
        if identity in seen:
            raise StoryRelevanceEvaluationError(
                StoryRelevanceEvaluationErrorCode.DUPLICATE_STORY,
                "story_views cannot contain duplicate canonical Story identities",
            )
        seen.add(identity)
        results.append(evaluate_engineering_story_relevance(
            hiring_context=hiring_context,
            story_view=story_view,
        ))
    return tuple(sorted(
        results,
        key=lambda item: (
            -item.total_relevance_score,
            item.canonical_story_id,
            item.current_revision_id,
            item.project_id,
        ),
    ))


__all__ = [
    "MAX_STORY_RELEVANCE_BATCH",
    "MAX_STORY_RELEVANCE_FEATURES",
    "MAX_STORY_RELEVANCE_REASONS",
    "MAX_STORY_RELEVANT_CONTEXT_SOURCES",
    "STORY_RELEVANCE_SCORE_DECIMALS",
    "StoryHiringRelevance",
    "StoryRelevanceComponents",
    "StoryRelevanceEvaluationError",
    "StoryRelevanceEvaluationErrorCode",
    "StoryRelevanceFeature",
    "StoryRelevanceReason",
    "StoryRelevanceWeights",
    "evaluate_engineering_story_relevance",
    "rank_engineering_stories_for_hiring_context",
]
