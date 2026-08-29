"""Read-only product presentation for hiring-context ranking review.

The service composes the accepted offline ranking chain and projects it into a
small UI contract.  Candidate evidence, raw Stories, ranking scores, policy
identities, and persistence concerns deliberately remain outside this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.engineering_story_memory_service import (
    AuthoritativeEngineeringStoryMemory,
    StoryMemoryArtifactStatus,
    StoryMemoryReadinessState,
    get_active_engineering_stories_for_project,
    inspect_authoritative_engineering_story_memory_readiness,
    load_authoritative_engineering_story_memory,
)
from backend.engineering_story_relevance import (
    StoryHiringRelevance,
    StoryRelevanceFeature,
    rank_engineering_stories_for_hiring_context,
)
from backend.hiring_context_intelligence import build_hiring_context_profile
from backend.hiring_context_models import (
    HiringContextConfidence,
    HiringContextProfile,
    HiringContextSignalKind,
    RoleFamily,
)
from backend.project_portfolio_ranking import rank_project_portfolio
from backend.project_story_ranking import (
    ProjectHiringRelevance,
    aggregate_project_story_relevance,
)
from backend.project_story_ranking_refresh import (
    PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
    ProjectStoryRelevanceSnapshot,
)
from backend.project_story_ranking_state import (
    ProjectStoryRankingState,
    build_project_story_ranking_state,
    derive_story_clarification_handoffs,
)
from backend.story_clarification_handoff import StoryClarificationReason


MAX_REVIEW_CONTEXT_SIGNALS = 8
MAX_REVIEW_REASON_LABELS = 3


class HiringContextRankingReviewStatus(str, Enum):
    READY = "ready"
    EMPTY = "empty"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class HiringContextReviewSummary:
    company: str | None
    team: str | None
    role_title: str | None
    primary_role_family: str
    secondary_role_families: tuple[str, ...]
    context_signals: tuple[str, ...]
    confidence: str

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class StoryRankingReviewItem:
    story_id: str
    label: str
    relevance_reasons: tuple[str, ...]
    notices: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class ProjectRankingReviewItem:
    project_id: str
    display_name: str
    position: int
    relevance_reasons: tuple[str, ...]
    strongest_stories: tuple[StoryRankingReviewItem, ...]
    additional_stories: tuple[StoryRankingReviewItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class HiringContextRankingReview:
    status: HiringContextRankingReviewStatus
    hiring_context: HiringContextReviewSummary
    projects: tuple[ProjectRankingReviewItem, ...] = ()
    corrections_persisted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "hiring_context": self.hiring_context.to_dict(),
            "projects": [item.to_dict() for item in self.projects],
            "corrections_persisted": self.corrections_persisted,
        }


_ROLE_LABELS = {
    "en": {
        RoleFamily.SOFTWARE_ENGINEERING: "Software engineering",
        RoleFamily.BACKEND_ENGINEERING: "Backend engineering",
        RoleFamily.FRONTEND_ENGINEERING: "Frontend engineering",
        RoleFamily.FULL_STACK_ENGINEERING: "Full-stack engineering",
        RoleFamily.DATA_ENGINEERING: "Data engineering",
        RoleFamily.DATA_ANALYTICS: "Data analysis",
        RoleFamily.MACHINE_LEARNING_AI: "Machine learning and AI",
        RoleFamily.DEVOPS_CLOUD: "Infrastructure and delivery",
        RoleFamily.GAME_DEVELOPMENT: "Game development",
        RoleFamily.MOBILE: "Mobile engineering",
        RoleFamily.EMBEDDED_SYSTEMS: "Embedded systems",
        RoleFamily.SECURITY: "Software security",
        RoleFamily.CONSULTING_STRATEGY: "Consulting and strategy",
        RoleFamily.GENERAL: "General engineering",
        RoleFamily.UNKNOWN: "Role context still forming",
    },
    "zh": {
        RoleFamily.SOFTWARE_ENGINEERING: "软件工程",
        RoleFamily.BACKEND_ENGINEERING: "后端工程",
        RoleFamily.FRONTEND_ENGINEERING: "前端工程",
        RoleFamily.FULL_STACK_ENGINEERING: "全栈工程",
        RoleFamily.DATA_ENGINEERING: "数据工程",
        RoleFamily.DATA_ANALYTICS: "数据分析",
        RoleFamily.MACHINE_LEARNING_AI: "机器学习与 AI",
        RoleFamily.DEVOPS_CLOUD: "基础设施与交付",
        RoleFamily.GAME_DEVELOPMENT: "游戏开发",
        RoleFamily.MOBILE: "移动端工程",
        RoleFamily.EMBEDDED_SYSTEMS: "嵌入式系统",
        RoleFamily.SECURITY: "软件安全",
        RoleFamily.CONSULTING_STRATEGY: "咨询与战略",
        RoleFamily.GENERAL: "通用工程",
        RoleFamily.UNKNOWN: "职位重点仍在形成",
    },
}

_FEATURE_LABELS = {
    "en": {
        StoryRelevanceFeature.ARCHITECTURE: "System architecture",
        StoryRelevanceFeature.RELIABILITY: "Reliability",
        StoryRelevanceFeature.DEBUGGING: "Debugging",
        StoryRelevanceFeature.TESTING: "Testing",
        StoryRelevanceFeature.VALIDATION_REPAIR: "Validation and repair",
        StoryRelevanceFeature.PERFORMANCE: "Performance",
        StoryRelevanceFeature.STATE_MANAGEMENT: "State management",
        StoryRelevanceFeature.DATA_FLOW: "Data flow",
        StoryRelevanceFeature.API_SYSTEM_DESIGN: "API and system design",
        StoryRelevanceFeature.RETRIEVAL_RANKING: "Retrieval and ranking",
        StoryRelevanceFeature.OPERATIONAL_HARDENING: "Operational hardening",
        StoryRelevanceFeature.MIGRATION: "Migration",
        StoryRelevanceFeature.CONCURRENCY: "Concurrency",
        StoryRelevanceFeature.ALGORITHMS: "Algorithms",
        StoryRelevanceFeature.BACKEND: "Backend systems",
        StoryRelevanceFeature.FRONTEND: "User interface",
        StoryRelevanceFeature.DATA_ENGINEERING: "Data engineering",
        StoryRelevanceFeature.ANALYTICS: "Analysis and decision support",
        StoryRelevanceFeature.GAME_DEVELOPMENT: "Game systems",
        StoryRelevanceFeature.REAL_TIME_SYSTEMS: "Real-time systems",
        StoryRelevanceFeature.DEVOPS_CLOUD: "Infrastructure and delivery",
        StoryRelevanceFeature.SECURITY: "Software security",
        StoryRelevanceFeature.MACHINE_LEARNING: "Machine learning",
        StoryRelevanceFeature.MOBILE: "Mobile engineering",
        StoryRelevanceFeature.EMBEDDED: "Embedded systems",
        StoryRelevanceFeature.INTEGRATION: "System integration",
        StoryRelevanceFeature.AUTOMATION: "Automation",
        StoryRelevanceFeature.STORAGE: "Data storage",
        StoryRelevanceFeature.DISTRIBUTED_SYSTEMS: "Distributed systems",
        StoryRelevanceFeature.PLATFORM_ENGINEERING: "Platform engineering",
    },
    "zh": {
        StoryRelevanceFeature.ARCHITECTURE: "系统架构",
        StoryRelevanceFeature.RELIABILITY: "可靠性",
        StoryRelevanceFeature.DEBUGGING: "调试与排障",
        StoryRelevanceFeature.TESTING: "测试",
        StoryRelevanceFeature.VALIDATION_REPAIR: "验证与修复",
        StoryRelevanceFeature.PERFORMANCE: "性能",
        StoryRelevanceFeature.STATE_MANAGEMENT: "状态管理",
        StoryRelevanceFeature.DATA_FLOW: "数据流",
        StoryRelevanceFeature.API_SYSTEM_DESIGN: "API 与系统设计",
        StoryRelevanceFeature.RETRIEVAL_RANKING: "检索与排序",
        StoryRelevanceFeature.OPERATIONAL_HARDENING: "运行稳健性",
        StoryRelevanceFeature.MIGRATION: "迁移",
        StoryRelevanceFeature.CONCURRENCY: "并发",
        StoryRelevanceFeature.ALGORITHMS: "算法",
        StoryRelevanceFeature.BACKEND: "后端系统",
        StoryRelevanceFeature.FRONTEND: "用户界面",
        StoryRelevanceFeature.DATA_ENGINEERING: "数据工程",
        StoryRelevanceFeature.ANALYTICS: "分析与决策支持",
        StoryRelevanceFeature.GAME_DEVELOPMENT: "游戏系统",
        StoryRelevanceFeature.REAL_TIME_SYSTEMS: "实时系统",
        StoryRelevanceFeature.DEVOPS_CLOUD: "基础设施与交付",
        StoryRelevanceFeature.SECURITY: "软件安全",
        StoryRelevanceFeature.MACHINE_LEARNING: "机器学习",
        StoryRelevanceFeature.MOBILE: "移动端工程",
        StoryRelevanceFeature.EMBEDDED: "嵌入式系统",
        StoryRelevanceFeature.INTEGRATION: "系统集成",
        StoryRelevanceFeature.AUTOMATION: "自动化",
        StoryRelevanceFeature.STORAGE: "数据存储",
        StoryRelevanceFeature.DISTRIBUTED_SYSTEMS: "分布式系统",
        StoryRelevanceFeature.PLATFORM_ENGINEERING: "平台工程",
    },
}

_CONFIDENCE_LABELS = {
    "en": {
        HiringContextConfidence.HIGH: "High confidence",
        HiringContextConfidence.MEDIUM: "Moderate confidence",
        HiringContextConfidence.LOW: "Based on limited context",
    },
    "zh": {
        HiringContextConfidence.HIGH: "理解较明确",
        HiringContextConfidence.MEDIUM: "理解程度适中",
        HiringContextConfidence.LOW: "基于有限职位信息",
    },
}

_NOTICE_LABELS = {
    "en": {
        StoryClarificationReason.CLAIM_SAFETY_GAP: "Some claims may need confirmation",
        StoryClarificationReason.STORY_COMPLETENESS_GAP: "More context could strengthen this story",
    },
    "zh": {
        StoryClarificationReason.CLAIM_SAFETY_GAP: "部分表述可能需要确认",
        StoryClarificationReason.STORY_COMPLETENESS_GAP: "补充背景可让这个故事更完整",
    },
}


def _language(value: str) -> str:
    return "zh" if str(value or "").strip().casefold().startswith("zh") else "en"


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def _stable_labels(values: Sequence[str], *, maximum: int) -> tuple[str, ...]:
    selected: list[str] = []
    seen: set[str] = set()
    for value in values:
        label = " ".join(str(value or "").split())
        key = label.casefold()
        if not label or key in seen:
            continue
        seen.add(key)
        selected.append(label)
        if len(selected) == maximum:
            break
    return tuple(selected)


def _role_label(role: RoleFamily, language: str) -> str:
    return _ROLE_LABELS[language][role]


def _feature_labels(
    features: Sequence[StoryRelevanceFeature],
    language: str,
    *,
    maximum: int = MAX_REVIEW_REASON_LABELS,
) -> tuple[str, ...]:
    return _stable_labels(
        [_FEATURE_LABELS[language][StoryRelevanceFeature(item)] for item in features],
        maximum=maximum,
    )


def _context_summary(
    profile: HiringContextProfile,
    language: str,
) -> HiringContextReviewSummary:
    context_values: list[str] = list(profile.high_value_traits)
    visible_signal_kinds = {
        HiringContextSignalKind.ROLE_FAMILY,
        HiringContextSignalKind.COMPANY_DOMAIN,
        HiringContextSignalKind.TEAM_DOMAIN,
        HiringContextSignalKind.PARENT_ORGANIZATION_DOMAIN,
        HiringContextSignalKind.ENGINEERING_TRAIT,
    }
    for signal in profile.signals:
        if signal.kind not in visible_signal_kinds:
            continue
        try:
            value = _role_label(RoleFamily(signal.value), language)
        except ValueError:
            value = signal.value
        context_values.append(value)
    return HiringContextReviewSummary(
        company=profile.company,
        team=profile.team,
        role_title=profile.role_title,
        primary_role_family=_role_label(profile.primary_role_family, language),
        secondary_role_families=tuple(
            _role_label(item, language) for item in profile.secondary_role_families
        ),
        context_signals=_stable_labels(
            context_values,
            maximum=MAX_REVIEW_CONTEXT_SIGNALS,
        ),
        confidence=_CONFIDENCE_LABELS[language][profile.confidence],
    )


def _project_display_names(project_memory: Mapping[str, Any] | None) -> dict[str, str]:
    projects = project_memory.get("projects") if isinstance(project_memory, Mapping) else None
    if not isinstance(projects, (list, tuple)):
        return {}
    names: dict[str, str] = {}
    for item in projects:
        if not isinstance(item, Mapping):
            continue
        project_id = item.get("project_id")
        project_name = item.get("project_name")
        if not isinstance(project_id, str) or not project_id.strip() or project_id != project_id.strip():
            continue
        if not isinstance(project_name, str):
            continue
        display_name = " ".join(project_name.split())
        if not display_name or len(display_name) > 300:
            continue
        names.setdefault(project_id, display_name)
    return names


def _story_label(story: StoryHiringRelevance, language: str) -> str:
    labels = _feature_labels(story.semantic_features, language, maximum=2)
    if not labels:
        return "工程实践" if language == "zh" else "Engineering story"
    separator = "与" if language == "zh" else " and "
    return separator.join(labels)


def _project_reasons(
    project: ProjectHiringRelevance,
    language: str,
) -> tuple[str, ...]:
    features: list[StoryRelevanceFeature] = []
    for contribution in project.contributions:
        features.extend(contribution.story_relevance.semantic_features)
    labels = _feature_labels(features, language)
    if not labels:
        return ("相关工程实践" if language == "zh" else "Relevant engineering evidence",)
    prefix = "突出实践：" if language == "zh" else "Strong evidence in "
    joiner = "、" if language == "zh" else ", "
    return (f"{prefix}{joiner.join(labels)}",)


def _story_item(
    story: StoryHiringRelevance,
    language: str,
    handoff_reasons: Sequence[StoryClarificationReason],
) -> StoryRankingReviewItem:
    return StoryRankingReviewItem(
        story_id=story.canonical_story_id,
        label=_story_label(story, language),
        relevance_reasons=_feature_labels(story.semantic_features, language),
        notices=tuple(
            _NOTICE_LABELS[language][StoryClarificationReason(reason)]
            for reason in handoff_reasons
        ),
    )


def present_hiring_context_ranking_review(
    *,
    hiring_context: HiringContextProfile,
    ranking_state: ProjectStoryRankingState,
    project_memory: Mapping[str, Any] | None,
    language: str = "en",
) -> HiringContextRankingReview:
    """Project one complete, validated ranking state into bounded product copy."""

    selected_language = _language(language)
    context = _context_summary(hiring_context, selected_language)
    if ranking_state.portfolio is None:
        return HiringContextRankingReview(
            status=HiringContextRankingReviewStatus.EMPTY,
            hiring_context=context,
        )
    if (
        ranking_state.hiring_context_profile_id != hiring_context.profile_id
        or ranking_state.hiring_context_fingerprint != hiring_context.fingerprint
    ):
        return HiringContextRankingReview(
            status=HiringContextRankingReviewStatus.ERROR,
            hiring_context=context,
        )
    display_names = _project_display_names(project_memory)
    handoffs = derive_story_clarification_handoffs(ranking_state=ranking_state)
    notices_by_story = {
        (item.project_id, item.canonical_story_id): item.reasons
        for item in handoffs
    }
    projects: list[ProjectRankingReviewItem] = []
    for ranking in ranking_state.portfolio.ranked_projects:
        snapshot = ranking_state.snapshot_for_project(ranking.project_id)
        if snapshot is None:
            return HiringContextRankingReview(
                status=HiringContextRankingReviewStatus.ERROR,
                hiring_context=context,
            )
        contributor_ids = {
            item.canonical_story_id
            for item in snapshot.project_relevance.contributions
        }
        strongest: list[StoryRankingReviewItem] = []
        additional: list[StoryRankingReviewItem] = []
        for story in snapshot.story_relevance:
            item = _story_item(
                story,
                selected_language,
                notices_by_story.get((story.project_id, story.canonical_story_id), ()),
            )
            (strongest if story.canonical_story_id in contributor_ids else additional).append(item)
        fallback = (
            f"项目 {ranking.position}"
            if selected_language == "zh"
            else f"Project {ranking.position}"
        )
        projects.append(ProjectRankingReviewItem(
            project_id=ranking.project_id,
            display_name=display_names.get(ranking.project_id, fallback),
            position=ranking.position,
            relevance_reasons=_project_reasons(snapshot.project_relevance, selected_language),
            strongest_stories=tuple(strongest),
            additional_stories=tuple(additional),
        ))
    return HiringContextRankingReview(
        status=HiringContextRankingReviewStatus.READY,
        hiring_context=context,
        projects=tuple(projects),
    )


def _active_story_views(
    memory: AuthoritativeEngineeringStoryMemory,
) -> dict[str, tuple[Any, ...]]:
    by_project: dict[str, tuple[Any, ...]] = {}
    for project_id in sorted({history.project_id for history in memory.histories}):
        views = get_active_engineering_stories_for_project(memory, project_id)
        if views:
            by_project[project_id] = views
    return by_project


def build_hiring_context_ranking_review(
    *,
    company: str | None,
    team: str | None,
    role_title: str | None,
    normalized_job_context: Mapping[str, Any],
    project_memory: Mapping[str, Any] | None = None,
    language: str = "en",
    story_memory_path: str | Path | None = None,
) -> HiringContextRankingReview:
    """Read Story Memory, run accepted rankers, and return a fail-closed review."""

    selected_language = _language(language)
    profile = build_hiring_context_profile(
        company=company,
        team=team,
        role_title=role_title,
        normalized_job_context=normalized_job_context,
    )
    context = _context_summary(profile, selected_language)
    load_result = load_authoritative_engineering_story_memory(story_memory_path)
    if load_result.status is StoryMemoryArtifactStatus.EMPTY:
        return HiringContextRankingReview(
            status=HiringContextRankingReviewStatus.EMPTY,
            hiring_context=context,
        )
    if load_result.status is StoryMemoryArtifactStatus.INTEGRITY_MISMATCH:
        return HiringContextRankingReview(
            status=HiringContextRankingReviewStatus.ERROR,
            hiring_context=context,
        )
    if load_result.status is not StoryMemoryArtifactStatus.READY or load_result.memory is None:
        return HiringContextRankingReview(
            status=HiringContextRankingReviewStatus.UNAVAILABLE,
            hiring_context=context,
        )
    try:
        readiness = inspect_authoritative_engineering_story_memory_readiness(
            path=story_memory_path,
            compare_upstream=True,
        )
    except Exception:
        return HiringContextRankingReview(
            status=HiringContextRankingReviewStatus.ERROR,
            hiring_context=context,
        )
    if readiness.state is not StoryMemoryReadinessState.READY:
        return HiringContextRankingReview(
            status=HiringContextRankingReviewStatus.UNAVAILABLE,
            hiring_context=context,
        )
    try:
        relevance_by_project: dict[str, tuple[StoryHiringRelevance, ...]] = {}
        project_results: dict[str, ProjectHiringRelevance] = {}
        for project_id, views in _active_story_views(load_result.memory).items():
            ranked_stories = rank_engineering_stories_for_hiring_context(
                hiring_context=profile,
                story_views=views,
            )
            if not ranked_stories:
                continue
            relevance_by_project[project_id] = ranked_stories
            project = aggregate_project_story_relevance(
                project_id=project_id,
                story_relevance=ranked_stories,
            )
            if project is not None:
                project_results[project_id] = project
        portfolio = rank_project_portfolio(projects=tuple(project_results.values()))
        if portfolio is None:
            state = build_project_story_ranking_state(
                hiring_context=profile,
                portfolio=None,
                project_snapshots=(),
            )
        else:
            snapshots = tuple(
                ProjectStoryRelevanceSnapshot(
                    snapshot_policy_id=PROJECT_STORY_RELEVANCE_SNAPSHOT_POLICY_ID,
                    source_portfolio_id=portfolio.portfolio_id,
                    project_relevance=project_results[ranking.project_id],
                    story_relevance=relevance_by_project[ranking.project_id],
                )
                for ranking in portfolio.ranked_projects
            )
            state = build_project_story_ranking_state(
                hiring_context=profile,
                portfolio=portfolio,
                project_snapshots=snapshots,
            )
        return present_hiring_context_ranking_review(
            hiring_context=profile,
            ranking_state=state,
            project_memory=project_memory,
            language=selected_language,
        )
    except Exception:
        return HiringContextRankingReview(
            status=HiringContextRankingReviewStatus.ERROR,
            hiring_context=context,
        )


__all__ = [
    "HiringContextRankingReview",
    "HiringContextRankingReviewStatus",
    "HiringContextReviewSummary",
    "ProjectRankingReviewItem",
    "StoryRankingReviewItem",
    "build_hiring_context_ranking_review",
    "present_hiring_context_ranking_review",
]
