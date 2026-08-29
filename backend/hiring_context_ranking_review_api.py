"""Semantic HTTP boundary for the read-only tailoring-context review."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.hiring_context_ranking_review import (
    HiringContextRankingReview,
    build_hiring_context_ranking_review,
)


HIRING_CONTEXT_RANKING_REVIEW_FLAG = "USE_HIRING_CONTEXT_RANKING_REVIEW"
_ENABLED_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})
_DISABLED_FLAG_VALUES = frozenset({"", "0", "false", "no", "off"})


def is_hiring_context_ranking_review_enabled() -> bool:
    """Resolve the semantic rollout guard at call time and fail closed."""

    try:
        raw = os.getenv(HIRING_CONTEXT_RANKING_REVIEW_FLAG, "")
    except Exception:
        return False
    value = raw.strip().casefold() if isinstance(raw, str) else ""
    if value in _ENABLED_FLAG_VALUES:
        return True
    if value in _DISABLED_FLAG_VALUES:
        return False
    return False


class HiringContextReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str | None = Field(default=None, max_length=200)
    team: str | None = Field(default=None, max_length=200)
    role_title: str | None = Field(default=None, max_length=240)
    language: str = Field(default="en", max_length=12)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _project_memory(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _identity_value(
    body: HiringContextReviewRequest,
    field: str,
    normalized_job_context: Mapping[str, Any],
    fallback_field: str,
) -> str | None:
    if field in body.model_fields_set:
        return getattr(body, field)
    value = normalized_job_context.get(fallback_field)
    return value if isinstance(value, str) else None


def create_hiring_context_ranking_review_router(
    *,
    read_job_description: Callable[[], str],
    normalize_job_context: Callable[[str], Mapping[str, Any]],
    read_project_memory: Callable[[], Any],
    prepare_review: Callable[..., HiringContextRankingReview] = (
        build_hiring_context_ranking_review
    ),
    feature_enabled: Callable[[], bool] = is_hiring_context_ranking_review_enabled,
) -> APIRouter:
    """Create an isolated router around existing product-state readers."""

    router = APIRouter()

    @router.get("/api/hiring-context/review/availability")
    def hiring_context_review_availability() -> dict[str, bool]:
        return {"available": bool(feature_enabled())}

    @router.post("/api/hiring-context/review")
    def hiring_context_review(body: HiringContextReviewRequest) -> dict[str, Any]:
        if not feature_enabled():
            raise HTTPException(status_code=404, detail="Not found.")
        try:
            job_description = read_job_description()
            normalized_job_context = _mapping(normalize_job_context(job_description))
        except (FileNotFoundError, TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="A saved job description is required.",
            ) from None
        if not str(job_description or "").strip():
            raise HTTPException(
                status_code=400,
                detail="A saved job description is required.",
            )
        try:
            raw_project_memory = read_project_memory()
        except Exception:
            raw_project_memory = None
        try:
            review = prepare_review(
                company=_identity_value(
                    body,
                    "company",
                    normalized_job_context,
                    "company",
                ),
                team=(
                    body.team
                    if "team" in body.model_fields_set
                    else None
                ),
                role_title=_identity_value(
                    body,
                    "role_title",
                    normalized_job_context,
                    "job_title",
                ),
                normalized_job_context=normalized_job_context,
                project_memory=_project_memory(raw_project_memory),
                language=body.language,
            )
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="The job context could not be reviewed.",
            ) from None
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Could not prepare the tailoring review.",
            ) from None
        return review.to_dict()

    return router


__all__ = [
    "HIRING_CONTEXT_RANKING_REVIEW_FLAG",
    "HiringContextReviewRequest",
    "create_hiring_context_ranking_review_router",
    "is_hiring_context_ranking_review_enabled",
]
