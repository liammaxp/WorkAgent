"""Immutable, non-authoritative resume emphasis preferences.

The overlay records user intent for one Hiring Context and one complete
``ProjectStoryRankingState`` generation.  It never rewrites ranking, evidence,
Story Memory, sufficiency, or candidate identity.  A later resume-space
consumer may read the two contracts together, but this module performs no
selection or allocation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any

from backend.project_story_ranking_state import (
    MAX_RANKING_STATE_PROJECTS,
    MAX_RANKING_STATE_STORIES,
    ProjectStoryRankingState,
)


RESUME_EMPHASIS_PREFERENCE_POLICY_ID = (
    "resume_emphasis_preference_overlay.non_authoritative.v1"
)
MAX_RESUME_PROJECT_PREFERENCES = MAX_RANKING_STATE_PROJECTS
MAX_RESUME_STORY_PREFERENCES = MAX_RANKING_STATE_STORIES
MAX_RESUME_EMPHASIS_PREFERENCES = (
    MAX_RESUME_PROJECT_PREFERENCES + MAX_RESUME_STORY_PREFERENCES
)


class EmphasisPreferenceAction(str, Enum):
    DEFAULT = "default"
    EMPHASIZE_MORE = "emphasize_more"
    EMPHASIZE_LESS = "emphasize_less"
    KEEP = "keep"
    EXCLUDE = "exclude"


class ResumeEmphasisPreferenceErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    BOUND_EXCEEDED = "bound_exceeded"
    UNKNOWN_PROJECT = "unknown_project"
    UNKNOWN_STORY = "unknown_story"
    CONFLICTING_DUPLICATE = "conflicting_duplicate"
    STALE_RANKING_STATE = "stale_ranking_state"
    CONTEXT_MISMATCH = "context_mismatch"


class ResumeEmphasisPreferenceError(ValueError):
    def __init__(
        self,
        code: ResumeEmphasisPreferenceErrorCode,
        message: str,
    ) -> None:
        self.code = ResumeEmphasisPreferenceErrorCode(code)
        super().__init__(message)


class ResumeEmphasisReconciliationStatus(str, Enum):
    CURRENT = "current"
    RECONCILED = "reconciled"
    CONTEXT_STALE = "context_stale"


def _exact_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must be an exact non-blank value")
    return value


def _sha256(value: Any, name: str) -> str:
    normalized = _exact_text(value, name)
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 value")
    return normalized


def _prefixed_identity(value: Any, name: str, prefix: str) -> str:
    normalized = _exact_text(value, name)
    token = normalized.removeprefix(prefix)
    if (
        normalized != f"{prefix}{token}"
        or len(token) != 24
        or any(char not in "0123456789abcdef" for char in token)
    ):
        raise ValueError(f"{name} has an invalid identity shape")
    return normalized


def _action(value: Any) -> EmphasisPreferenceAction:
    try:
        return EmphasisPreferenceAction(value)
    except (TypeError, ValueError) as error:
        raise ResumeEmphasisPreferenceError(
            ResumeEmphasisPreferenceErrorCode.INVALID_INPUT,
            "unknown emphasis preference action",
        ) from error


@dataclass(frozen=True, slots=True)
class ProjectEmphasisPreference:
    project_id: str
    action: EmphasisPreferenceAction

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_id", _exact_text(self.project_id, "project_id"))
        object.__setattr__(self, "action", _action(self.action))

    def to_dict(self) -> dict[str, str]:
        return {"project_id": self.project_id, "action": self.action.value}


@dataclass(frozen=True, slots=True)
class StoryEmphasisPreference:
    project_id: str
    canonical_story_id: str
    current_revision_id: str
    action: EmphasisPreferenceAction

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_id", _exact_text(self.project_id, "project_id"))
        object.__setattr__(
            self,
            "canonical_story_id",
            _exact_text(self.canonical_story_id, "canonical_story_id"),
        )
        object.__setattr__(
            self,
            "current_revision_id",
            _exact_text(self.current_revision_id, "current_revision_id"),
        )
        object.__setattr__(self, "action", _action(self.action))

    def to_dict(self) -> dict[str, str]:
        return {
            "project_id": self.project_id,
            "canonical_story_id": self.canonical_story_id,
            "current_revision_id": self.current_revision_id,
            "action": self.action.value,
        }


def _overlay_payload(
    *,
    ranking_state_id: str,
    ranking_state_fingerprint: str,
    hiring_context_profile_id: str,
    hiring_context_fingerprint: str,
    project_preferences: tuple[ProjectEmphasisPreference, ...],
    story_preferences: tuple[StoryEmphasisPreference, ...],
) -> dict[str, Any]:
    return {
        "overlay_policy_id": RESUME_EMPHASIS_PREFERENCE_POLICY_ID,
        "ranking_state_id": ranking_state_id,
        "ranking_state_fingerprint": ranking_state_fingerprint,
        "hiring_context_profile_id": hiring_context_profile_id,
        "hiring_context_fingerprint": hiring_context_fingerprint,
        "project_preferences": [item.to_dict() for item in project_preferences],
        "story_preferences": [item.to_dict() for item in story_preferences],
    }


def _identity(payload: dict[str, Any]) -> tuple[str, str]:
    fingerprint = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return fingerprint, f"resume_emphasis_overlay_{fingerprint[:24]}"


@dataclass(frozen=True, slots=True)
class ResumeEmphasisPreferenceOverlay:
    """Explicit resume intent bound to one authoritative ranking generation."""

    overlay_policy_id: str
    ranking_state_id: str
    ranking_state_fingerprint: str
    hiring_context_profile_id: str
    hiring_context_fingerprint: str
    project_preferences: tuple[ProjectEmphasisPreference, ...] = ()
    story_preferences: tuple[StoryEmphasisPreference, ...] = ()
    overlay_fingerprint: str = ""
    overlay_id: str = ""

    def __post_init__(self) -> None:
        if self.overlay_policy_id != RESUME_EMPHASIS_PREFERENCE_POLICY_ID:
            raise ValueError("overlay_policy_id is not supported")
        object.__setattr__(
            self,
            "ranking_state_id",
            _prefixed_identity(
                self.ranking_state_id,
                "ranking_state_id",
                "project_story_ranking_state_",
            ),
        )
        object.__setattr__(
            self,
            "ranking_state_fingerprint",
            _sha256(self.ranking_state_fingerprint, "ranking_state_fingerprint"),
        )
        object.__setattr__(
            self,
            "hiring_context_profile_id",
            _prefixed_identity(
                self.hiring_context_profile_id,
                "hiring_context_profile_id",
                "hiring_context_",
            ),
        )
        object.__setattr__(
            self,
            "hiring_context_fingerprint",
            _sha256(self.hiring_context_fingerprint, "hiring_context_fingerprint"),
        )
        if not isinstance(self.project_preferences, tuple) or not all(
            isinstance(item, ProjectEmphasisPreference)
            for item in self.project_preferences
        ):
            raise TypeError("project_preferences must be an immutable preference tuple")
        if not isinstance(self.story_preferences, tuple) or not all(
            isinstance(item, StoryEmphasisPreference)
            for item in self.story_preferences
        ):
            raise TypeError("story_preferences must be an immutable preference tuple")
        if len(self.project_preferences) > MAX_RESUME_PROJECT_PREFERENCES:
            raise ResumeEmphasisPreferenceError(
                ResumeEmphasisPreferenceErrorCode.BOUND_EXCEEDED,
                "project preferences exceed the ranking-state project bound",
            )
        if len(self.story_preferences) > MAX_RESUME_STORY_PREFERENCES:
            raise ResumeEmphasisPreferenceError(
                ResumeEmphasisPreferenceErrorCode.BOUND_EXCEEDED,
                "Story preferences exceed the ranking-state Story bound",
            )
        if len(self.project_preferences) + len(self.story_preferences) > MAX_RESUME_EMPHASIS_PREFERENCES:
            raise ResumeEmphasisPreferenceError(
                ResumeEmphasisPreferenceErrorCode.BOUND_EXCEEDED,
                "preference entries exceed the total overlay bound",
            )
        if any(item.action is EmphasisPreferenceAction.DEFAULT for item in self.project_preferences):
            raise ValueError("DEFAULT project preferences must not be stored")
        if any(item.action is EmphasisPreferenceAction.DEFAULT for item in self.story_preferences):
            raise ValueError("DEFAULT Story preferences must not be stored")
        if tuple(sorted(self.project_preferences, key=lambda item: item.project_id)) != self.project_preferences:
            raise ValueError("project preferences must use deterministic identity order")
        if tuple(
            sorted(
                self.story_preferences,
                key=lambda item: (item.project_id, item.canonical_story_id),
            )
        ) != self.story_preferences:
            raise ValueError("Story preferences must use deterministic identity order")
        if len({item.project_id for item in self.project_preferences}) != len(self.project_preferences):
            raise ValueError("project preferences must be unique")
        story_keys = {
            (item.project_id, item.canonical_story_id)
            for item in self.story_preferences
        }
        if len(story_keys) != len(self.story_preferences):
            raise ValueError("Story preferences must be unique")
        excluded_projects = {
            item.project_id
            for item in self.project_preferences
            if item.action is EmphasisPreferenceAction.EXCLUDE
        }
        if any(item.project_id in excluded_projects for item in self.story_preferences):
            raise ValueError("excluded projects cannot retain child Story preferences")
        payload = _overlay_payload(
            ranking_state_id=self.ranking_state_id,
            ranking_state_fingerprint=self.ranking_state_fingerprint,
            hiring_context_profile_id=self.hiring_context_profile_id,
            hiring_context_fingerprint=self.hiring_context_fingerprint,
            project_preferences=self.project_preferences,
            story_preferences=self.story_preferences,
        )
        expected_fingerprint, expected_id = _identity(payload)
        if self.overlay_fingerprint not in ("", expected_fingerprint):
            raise ValueError("overlay_fingerprint does not match overlay content")
        if self.overlay_id not in ("", expected_id):
            raise ValueError("overlay_id does not match overlay content")
        object.__setattr__(self, "overlay_fingerprint", expected_fingerprint)
        object.__setattr__(self, "overlay_id", expected_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            **_overlay_payload(
                ranking_state_id=self.ranking_state_id,
                ranking_state_fingerprint=self.ranking_state_fingerprint,
                hiring_context_profile_id=self.hiring_context_profile_id,
                hiring_context_fingerprint=self.hiring_context_fingerprint,
                project_preferences=self.project_preferences,
                story_preferences=self.story_preferences,
            ),
            "overlay_fingerprint": self.overlay_fingerprint,
            "overlay_id": self.overlay_id,
        }


@dataclass(frozen=True, slots=True)
class ResumeEmphasisReconciliation:
    status: ResumeEmphasisReconciliationStatus
    overlay: ResumeEmphasisPreferenceOverlay | None
    removed_project_ids: tuple[str, ...] = ()
    removed_story_ids: tuple[tuple[str, str], ...] = ()


def _ranking_index(
    ranking_state: ProjectStoryRankingState,
) -> tuple[set[str], dict[tuple[str, str], str]]:
    if not isinstance(ranking_state, ProjectStoryRankingState):
        raise TypeError("ranking_state must be a ProjectStoryRankingState")
    project_ids: set[str] = set()
    stories: dict[tuple[str, str], str] = {}
    for snapshot in ranking_state.project_snapshots:
        project_ids.add(snapshot.project_id)
        for story in snapshot.story_relevance:
            stories[(snapshot.project_id, story.canonical_story_id)] = story.current_revision_id
    return project_ids, stories


def _bounded_sequence(value: Any, name: str, maximum: int) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(value) > maximum:
        raise ResumeEmphasisPreferenceError(
            ResumeEmphasisPreferenceErrorCode.BOUND_EXCEEDED,
            f"{name} exceed their ranking-state bound",
        )
    return value


def _normalize_projects(
    preferences: Sequence[ProjectEmphasisPreference],
    known_projects: set[str],
) -> tuple[ProjectEmphasisPreference, ...]:
    unique: dict[str, ProjectEmphasisPreference] = {}
    for raw in preferences:
        if not isinstance(raw, ProjectEmphasisPreference):
            raise TypeError("project preferences must use ProjectEmphasisPreference")
        item = ProjectEmphasisPreference(raw.project_id, raw.action)
        if item.project_id not in known_projects:
            raise ResumeEmphasisPreferenceError(
                ResumeEmphasisPreferenceErrorCode.UNKNOWN_PROJECT,
                "project preference references an unknown project",
            )
        prior = unique.get(item.project_id)
        if prior is not None and prior.action is not item.action:
            raise ResumeEmphasisPreferenceError(
                ResumeEmphasisPreferenceErrorCode.CONFLICTING_DUPLICATE,
                "project preference contains conflicting duplicate actions",
            )
        unique[item.project_id] = item
    return tuple(
        item
        for item in sorted(unique.values(), key=lambda value: value.project_id)
        if item.action is not EmphasisPreferenceAction.DEFAULT
    )


def _normalize_stories(
    preferences: Sequence[StoryEmphasisPreference],
    known_projects: set[str],
    known_stories: dict[tuple[str, str], str],
    excluded_projects: set[str],
) -> tuple[StoryEmphasisPreference, ...]:
    unique: dict[tuple[str, str], StoryEmphasisPreference] = {}
    for raw in preferences:
        if not isinstance(raw, StoryEmphasisPreference):
            raise TypeError("Story preferences must use StoryEmphasisPreference")
        item = StoryEmphasisPreference(
            raw.project_id,
            raw.canonical_story_id,
            raw.current_revision_id,
            raw.action,
        )
        key = (item.project_id, item.canonical_story_id)
        expected_revision = known_stories.get(key)
        if expected_revision is None:
            code = (
                ResumeEmphasisPreferenceErrorCode.UNKNOWN_PROJECT
                if item.project_id not in known_projects
                else ResumeEmphasisPreferenceErrorCode.UNKNOWN_STORY
            )
            raise ResumeEmphasisPreferenceError(
                code,
                "Story preference references an unknown project/Story identity",
            )
        if item.current_revision_id != expected_revision:
            raise ResumeEmphasisPreferenceError(
                ResumeEmphasisPreferenceErrorCode.STALE_RANKING_STATE,
                "Story preference revision is stale for this ranking state",
            )
        prior = unique.get(key)
        if prior is not None and prior.action is not item.action:
            raise ResumeEmphasisPreferenceError(
                ResumeEmphasisPreferenceErrorCode.CONFLICTING_DUPLICATE,
                "Story preference contains conflicting duplicate actions",
            )
        unique[key] = item
    return tuple(
        item
        for item in sorted(
            unique.values(),
            key=lambda value: (value.project_id, value.canonical_story_id),
        )
        if item.action is not EmphasisPreferenceAction.DEFAULT
        and item.project_id not in excluded_projects
    )


def build_resume_emphasis_preference_overlay(
    *,
    ranking_state: ProjectStoryRankingState,
    project_preferences: Sequence[ProjectEmphasisPreference] = (),
    story_preferences: Sequence[StoryEmphasisPreference] = (),
) -> ResumeEmphasisPreferenceOverlay:
    """Validate exact identities and build a deterministic preference set.

    Project ``EXCLUDE`` dominates and clears all explicit child Story choices.
    Other project actions do not override child actions, so a project may be
    generally lowered while a particular Story is emphasized more.
    """

    known_projects, known_stories = _ranking_index(ranking_state)
    raw_projects = _bounded_sequence(
        project_preferences,
        "project preferences",
        MAX_RESUME_PROJECT_PREFERENCES,
    )
    raw_stories = _bounded_sequence(
        story_preferences,
        "Story preferences",
        MAX_RESUME_STORY_PREFERENCES,
    )
    if len(raw_projects) + len(raw_stories) > MAX_RESUME_EMPHASIS_PREFERENCES:
        raise ResumeEmphasisPreferenceError(
            ResumeEmphasisPreferenceErrorCode.BOUND_EXCEEDED,
            "preference entries exceed the total overlay bound",
        )
    projects = _normalize_projects(raw_projects, known_projects)
    excluded_projects = {
        item.project_id
        for item in projects
        if item.action is EmphasisPreferenceAction.EXCLUDE
    }
    stories = _normalize_stories(
        raw_stories,
        known_projects,
        known_stories,
        excluded_projects,
    )
    return ResumeEmphasisPreferenceOverlay(
        overlay_policy_id=RESUME_EMPHASIS_PREFERENCE_POLICY_ID,
        ranking_state_id=ranking_state.state_id,
        ranking_state_fingerprint=ranking_state.state_fingerprint,
        hiring_context_profile_id=ranking_state.hiring_context_profile_id,
        hiring_context_fingerprint=ranking_state.hiring_context_fingerprint,
        project_preferences=projects,
        story_preferences=stories,
    )


def validate_overlay_against_ranking_state(
    *,
    overlay: ResumeEmphasisPreferenceOverlay,
    ranking_state: ProjectStoryRankingState,
) -> None:
    """Require an exact, current generation binding and all current identities."""

    if not isinstance(overlay, ResumeEmphasisPreferenceOverlay):
        raise TypeError("overlay must be a ResumeEmphasisPreferenceOverlay")
    _ranking_index(ranking_state)
    if (
        overlay.hiring_context_profile_id != ranking_state.hiring_context_profile_id
        or overlay.hiring_context_fingerprint != ranking_state.hiring_context_fingerprint
    ):
        raise ResumeEmphasisPreferenceError(
            ResumeEmphasisPreferenceErrorCode.CONTEXT_MISMATCH,
            "preference overlay belongs to another Hiring Context",
        )
    if (
        overlay.ranking_state_id != ranking_state.state_id
        or overlay.ranking_state_fingerprint != ranking_state.state_fingerprint
    ):
        raise ResumeEmphasisPreferenceError(
            ResumeEmphasisPreferenceErrorCode.STALE_RANKING_STATE,
            "preference overlay belongs to another ranking generation",
        )
    rebuilt = build_resume_emphasis_preference_overlay(
        ranking_state=ranking_state,
        project_preferences=overlay.project_preferences,
        story_preferences=overlay.story_preferences,
    )
    if rebuilt != overlay:
        raise ResumeEmphasisPreferenceError(
            ResumeEmphasisPreferenceErrorCode.INVALID_INPUT,
            "preference overlay does not match normalized current state",
        )


def reconcile_resume_emphasis_preferences(
    *,
    overlay: ResumeEmphasisPreferenceOverlay,
    ranking_state: ProjectStoryRankingState,
) -> ResumeEmphasisReconciliation:
    """Rebind canonical preferences without reranking or mutating source state."""

    if not isinstance(overlay, ResumeEmphasisPreferenceOverlay):
        raise TypeError("overlay must be a ResumeEmphasisPreferenceOverlay")
    known_projects, known_stories = _ranking_index(ranking_state)
    if (
        overlay.hiring_context_profile_id != ranking_state.hiring_context_profile_id
        or overlay.hiring_context_fingerprint != ranking_state.hiring_context_fingerprint
    ):
        return ResumeEmphasisReconciliation(
            status=ResumeEmphasisReconciliationStatus.CONTEXT_STALE,
            overlay=None,
        )
    if (
        overlay.ranking_state_id == ranking_state.state_id
        and overlay.ranking_state_fingerprint == ranking_state.state_fingerprint
    ):
        validate_overlay_against_ranking_state(
            overlay=overlay,
            ranking_state=ranking_state,
        )
        return ResumeEmphasisReconciliation(
            status=ResumeEmphasisReconciliationStatus.CURRENT,
            overlay=overlay,
        )
    kept_projects = tuple(
        item for item in overlay.project_preferences if item.project_id in known_projects
    )
    removed_projects = tuple(sorted(
        item.project_id
        for item in overlay.project_preferences
        if item.project_id not in known_projects
    ))
    kept_stories: list[StoryEmphasisPreference] = []
    removed_stories: list[tuple[str, str]] = []
    for item in overlay.story_preferences:
        key = (item.project_id, item.canonical_story_id)
        revision_id = known_stories.get(key)
        if revision_id is None:
            removed_stories.append(key)
            continue
        kept_stories.append(StoryEmphasisPreference(
            project_id=item.project_id,
            canonical_story_id=item.canonical_story_id,
            current_revision_id=revision_id,
            action=item.action,
        ))
    reconciled = build_resume_emphasis_preference_overlay(
        ranking_state=ranking_state,
        project_preferences=kept_projects,
        story_preferences=tuple(kept_stories),
    )
    return ResumeEmphasisReconciliation(
        status=ResumeEmphasisReconciliationStatus.RECONCILED,
        overlay=reconciled,
        removed_project_ids=removed_projects,
        removed_story_ids=tuple(sorted(removed_stories)),
    )


def project_preference_action(
    overlay: ResumeEmphasisPreferenceOverlay,
    project_id: str,
) -> EmphasisPreferenceAction:
    target = _exact_text(project_id, "project_id")
    return next(
        (item.action for item in overlay.project_preferences if item.project_id == target),
        EmphasisPreferenceAction.DEFAULT,
    )


def story_preference_action(
    overlay: ResumeEmphasisPreferenceOverlay,
    *,
    project_id: str,
    canonical_story_id: str,
) -> EmphasisPreferenceAction:
    project_action = project_preference_action(overlay, project_id)
    if project_action is EmphasisPreferenceAction.EXCLUDE:
        return EmphasisPreferenceAction.EXCLUDE
    target_story = _exact_text(canonical_story_id, "canonical_story_id")
    return next(
        (
            item.action
            for item in overlay.story_preferences
            if item.project_id == project_id
            and item.canonical_story_id == target_story
        ),
        EmphasisPreferenceAction.DEFAULT,
    )


def reset_project_preference(
    *,
    overlay: ResumeEmphasisPreferenceOverlay,
    ranking_state: ProjectStoryRankingState,
    project_id: str,
) -> ResumeEmphasisPreferenceOverlay:
    target = _exact_text(project_id, "project_id")
    validate_overlay_against_ranking_state(overlay=overlay, ranking_state=ranking_state)
    known_projects, _ = _ranking_index(ranking_state)
    if target not in known_projects:
        raise ResumeEmphasisPreferenceError(
            ResumeEmphasisPreferenceErrorCode.UNKNOWN_PROJECT,
            "project preference references an unknown project",
        )
    return build_resume_emphasis_preference_overlay(
        ranking_state=ranking_state,
        project_preferences=tuple(
            item for item in overlay.project_preferences if item.project_id != target
        ),
        story_preferences=overlay.story_preferences,
    )


def reset_story_preference(
    *,
    overlay: ResumeEmphasisPreferenceOverlay,
    ranking_state: ProjectStoryRankingState,
    project_id: str,
    canonical_story_id: str,
) -> ResumeEmphasisPreferenceOverlay:
    target = (
        _exact_text(project_id, "project_id"),
        _exact_text(canonical_story_id, "canonical_story_id"),
    )
    validate_overlay_against_ranking_state(overlay=overlay, ranking_state=ranking_state)
    _, known_stories = _ranking_index(ranking_state)
    if target not in known_stories:
        raise ResumeEmphasisPreferenceError(
            ResumeEmphasisPreferenceErrorCode.UNKNOWN_STORY,
            "Story preference references an unknown project/Story identity",
        )
    return build_resume_emphasis_preference_overlay(
        ranking_state=ranking_state,
        project_preferences=overlay.project_preferences,
        story_preferences=tuple(
            item
            for item in overlay.story_preferences
            if (item.project_id, item.canonical_story_id) != target
        ),
    )


def reset_all_preferences(
    *,
    overlay: ResumeEmphasisPreferenceOverlay,
    ranking_state: ProjectStoryRankingState,
) -> ResumeEmphasisPreferenceOverlay:
    validate_overlay_against_ranking_state(overlay=overlay, ranking_state=ranking_state)
    return build_resume_emphasis_preference_overlay(ranking_state=ranking_state)


__all__ = [
    "EmphasisPreferenceAction",
    "MAX_RESUME_EMPHASIS_PREFERENCES",
    "MAX_RESUME_PROJECT_PREFERENCES",
    "MAX_RESUME_STORY_PREFERENCES",
    "ProjectEmphasisPreference",
    "RESUME_EMPHASIS_PREFERENCE_POLICY_ID",
    "ResumeEmphasisPreferenceError",
    "ResumeEmphasisPreferenceErrorCode",
    "ResumeEmphasisPreferenceOverlay",
    "ResumeEmphasisReconciliation",
    "ResumeEmphasisReconciliationStatus",
    "StoryEmphasisPreference",
    "build_resume_emphasis_preference_overlay",
    "project_preference_action",
    "reconcile_resume_emphasis_preferences",
    "reset_all_preferences",
    "reset_project_preference",
    "reset_story_preference",
    "story_preference_action",
    "validate_overlay_against_ranking_state",
]
