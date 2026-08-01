"""Default-off scaffold for project evidence retrieval v2.

This module intentionally has no capability-memory, pipeline, persistence, or
raw-content dependencies.  Retrieval behavior will be added in later steps;
steps; for now the enabled path safely returns the existing list-shaped empty
result.
"""

from __future__ import annotations

import os
from typing import Any


GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG = "USE_GITHUB_EVIDENCE_RETRIEVAL_V2"
_ENABLED_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})


def is_github_evidence_retrieval_v2_enabled() -> bool:
    """Read the flag at call time, defaulting and failing closed."""

    try:
        value = os.getenv(GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "")
    except Exception:
        return False
    return isinstance(value, str) and value.strip().casefold() in _ENABLED_FLAG_VALUES


def retrieve_evidence_for_project_v2(project: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a bounded schema-compatible placeholder without performing I/O."""

    del project
    return []
