"""Pure normalization and exact deduplication for Phase 4 evidence inputs.

This module operates only on already-safe :class:`Phase4EvidenceInput` objects.
It performs no semantic comparison, synthesis, scoring, persistence, retrieval,
or external calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Iterable, Mapping

from backend.phase4_input_adapter import build_phase4_content_hash
from backend.phase4_models import (
    Phase4EvidenceInput,
    Phase4SourceRef,
    build_phase4_stable_id,
)


MAX_DECISION_SAMPLES = 100
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:/")
_NON_IDENTITY_METADATA_KEYS = frozenset({
    "absolute_path",
    "created_at",
    "generated_at",
    "machine_path",
    "modified_at",
    "mtime",
    "timestamp",
    "timestamps",
    "updated_at",
})


@dataclass(frozen=True)
class Phase4DeduplicationDecision:
    reason: str
    kept_input_id: str
    dropped_input_id: str
    project_id: str
    input_type: str


@dataclass(frozen=True)
class Phase4DeduplicationReport:
    input_count: int
    normalized_count: int
    output_count: int
    exact_duplicates_removed: int
    repeated_source_records_removed: int
    duplicate_source_refs_removed: int
    duplicate_signal_values_removed: int
    retained_cross_source_records: int
    decisions: tuple[Phase4DeduplicationDecision, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Phase4IntegrityError(ValueError):
    """Safe integrity error containing identifiers but no evidence content."""

    def __init__(
        self,
        reason: str,
        *,
        project_id: str,
        input_type: str,
        input_id: str = "",
        content_hash: str = "",
    ) -> None:
        self.reason = reason
        self.project_id = project_id
        self.input_type = input_type
        self.input_id = input_id
        self.content_hash = content_hash
        identifier = input_id or content_hash or "unknown"
        super().__init__(
            f"Phase 4 input integrity conflict ({reason}) for "
            f"project={project_id!r}, input_type={input_type!r}, identifier={identifier!r}"
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("Phase 4 descriptive values must be strings")
    return " ".join(value.split())


def _normalize_strings(values: list[str], *, set_like: bool = False) -> tuple[list[str], int]:
    if not isinstance(values, list):
        raise TypeError("Phase 4 signal fields must be lists")
    normalized: list[str] = []
    seen: set[str] = set()
    removed = 0
    for value in values:
        item = _normalize_text(value)
        if not item:
            removed += 1
            continue
        key = item.casefold()
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        normalized.append(item)
    if set_like:
        normalized.sort(key=lambda item: (item.casefold(), item))
    return normalized, removed


def _normalize_source_refs(values: list[Phase4SourceRef]) -> tuple[list[Phase4SourceRef], int]:
    if not isinstance(values, list):
        raise TypeError("source_refs must be a list")
    normalized: list[Phase4SourceRef] = []
    seen: set[str] = set()
    removed = 0
    for value in values:
        if not isinstance(value, Phase4SourceRef):
            raise TypeError("source_refs must contain Phase4SourceRef values")
        ref = Phase4SourceRef.from_dict(value.to_dict())
        key = ref.to_json()
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        normalized.append(ref)
    return normalized, removed


def _without_non_identity_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_non_identity_metadata(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).casefold() not in _NON_IDENTITY_METADATA_KEYS
        }
    if isinstance(value, list):
        return [_without_non_identity_metadata(item) for item in value]
    return value


def _source_ref_identity(ref: Phase4SourceRef) -> dict[str, Any]:
    payload = ref.to_dict()
    file_path = payload.get("file_path")
    if isinstance(file_path, str) and (
        file_path.startswith("/") or _WINDOWS_ABSOLUTE_PATH.match(file_path)
    ):
        payload["file_path"] = None
    payload["metadata"] = _without_non_identity_metadata(payload.get("metadata", {}))
    return payload


def _normalized_content_payload(
    *,
    project_id: str,
    input_type: str,
    title: str,
    summary: str,
    problem_signal: str | None,
    mechanism_signals: list[str],
    implementation_signals: list[str],
    impact_signals: list[str],
    technical_tags: list[str],
    source_refs: list[Phase4SourceRef],
) -> dict[str, Any]:
    return {
        "project_id": project_id,
        "input_type": input_type,
        "title": title,
        "summary": summary,
        "problem_signal": problem_signal,
        "mechanism_signals": mechanism_signals,
        "implementation_signals": implementation_signals,
        "impact_signals": impact_signals,
        "technical_tags": technical_tags,
        "source_refs": [_source_ref_identity(ref) for ref in source_refs],
    }


def _stable_input_id(
    project_id: str,
    content_payload: Mapping[str, Any],
    content_hash: str,
) -> str:
    identity_payload = dict(content_payload)
    identity_payload["content_hash"] = content_hash
    return build_phase4_stable_id("p4in_", project_id, identity_payload)


def _normalize_with_counts(item: Phase4EvidenceInput) -> tuple[Phase4EvidenceInput, int, int]:
    if not isinstance(item, Phase4EvidenceInput):
        raise TypeError("normalize_phase4_input expects a Phase4EvidenceInput")
    project_id = _normalize_text(item.project_id)
    input_type = _normalize_text(item.input_type)
    title = _normalize_text(item.title)
    summary = _normalize_text(item.summary)
    problem_signal = _normalize_text(item.problem_signal) if item.problem_signal is not None else None
    problem_signal = problem_signal or None
    mechanism_signals, mechanism_removed = _normalize_strings(item.mechanism_signals)
    implementation_signals, implementation_removed = _normalize_strings(item.implementation_signals)
    impact_signals, impact_removed = _normalize_strings(item.impact_signals)
    technical_tags, tag_removed = _normalize_strings(item.technical_tags, set_like=True)
    source_refs, refs_removed = _normalize_source_refs(item.source_refs)
    payload = _normalized_content_payload(
        project_id=project_id,
        input_type=input_type,
        title=title,
        summary=summary,
        problem_signal=problem_signal,
        mechanism_signals=mechanism_signals,
        implementation_signals=implementation_signals,
        impact_signals=impact_signals,
        technical_tags=technical_tags,
        source_refs=source_refs,
    )
    content_hash = build_phase4_content_hash(payload)
    input_id = _stable_input_id(project_id, payload, content_hash)
    normalized = Phase4EvidenceInput(
        project_id=project_id,
        input_type=input_type,
        title=title,
        summary=summary,
        problem_signal=problem_signal,
        mechanism_signals=mechanism_signals,
        implementation_signals=implementation_signals,
        impact_signals=impact_signals,
        technical_tags=technical_tags,
        source_refs=source_refs,
        content_hash=content_hash,
        input_id=input_id,
    )
    signal_removed = mechanism_removed + implementation_removed + impact_removed + tag_removed
    return normalized, refs_removed, signal_removed


def normalize_phase4_input(item: Phase4EvidenceInput) -> Phase4EvidenceInput:
    """Return a new normalized model with recomputed content hash and input ID."""

    return _normalize_with_counts(item)[0]


def _sort_key(item: Phase4EvidenceInput) -> tuple[str, str, str, str, str, str]:
    primary = item.source_refs[0]
    return (
        item.project_id,
        item.input_type,
        primary.source_type,
        primary.source_id,
        item.input_id,
        item.content_hash,
    )


def normalize_phase4_inputs(items: Iterable[Phase4EvidenceInput]) -> list[Phase4EvidenceInput]:
    """Normalize inputs independently and return deterministic collection order."""

    return sorted((normalize_phase4_input(item) for item in items), key=_sort_key)


def _full_payload(item: Phase4EvidenceInput) -> str:
    return item.to_json()


def _safe_payload_without_identity(item: Phase4EvidenceInput) -> str:
    payload = item.to_dict()
    payload.pop("input_id", None)
    payload.pop("content_hash", None)
    return _canonical_json(payload)


def _check_declared_integrity(
    originals: list[Phase4EvidenceInput],
    normalized: list[Phase4EvidenceInput],
) -> None:
    by_input_id: dict[str, tuple[str, Phase4EvidenceInput]] = {}
    by_hash: dict[tuple[str, str, str, str], tuple[str, Phase4EvidenceInput]] = {}
    for original, current in zip(originals, normalized):
        safe_payload = _safe_payload_without_identity(current)
        previous_id = by_input_id.get(original.input_id)
        if previous_id is not None and previous_id[0] != safe_payload:
            raise Phase4IntegrityError(
                "same_input_id_different_payload",
                project_id=current.project_id,
                input_type=current.input_type,
                input_id=original.input_id,
            )
        by_input_id.setdefault(original.input_id, (safe_payload, current))
        source_identity = _canonical_json([
            _source_ref_identity(ref) for ref in current.source_refs
        ])
        hash_key = (
            current.project_id,
            current.input_type,
            source_identity,
            original.content_hash,
        )
        previous_hash = by_hash.get(hash_key)
        if previous_hash is not None and previous_hash[0] != safe_payload:
            raise Phase4IntegrityError(
                "same_content_hash_different_payload",
                project_id=current.project_id,
                input_type=current.input_type,
                content_hash=original.content_hash,
            )
        by_hash.setdefault(hash_key, (safe_payload, current))


def _check_normalized_integrity(items: list[Phase4EvidenceInput]) -> None:
    by_id: dict[str, str] = {}
    by_hash: dict[str, str] = {}
    for item in items:
        payload = _safe_payload_without_identity(item)
        if item.input_id in by_id and by_id[item.input_id] != payload:
            raise Phase4IntegrityError(
                "same_input_id_different_normalized_payload",
                project_id=item.project_id,
                input_type=item.input_type,
                input_id=item.input_id,
            )
        by_id.setdefault(item.input_id, payload)
        if item.content_hash in by_hash and by_hash[item.content_hash] != payload:
            raise Phase4IntegrityError(
                "same_content_hash_different_normalized_payload",
                project_id=item.project_id,
                input_type=item.input_type,
                content_hash=item.content_hash,
            )
        by_hash.setdefault(item.content_hash, payload)


def _cross_source_retained_count(items: list[Phase4EvidenceInput]) -> int:
    groups: dict[str, set[tuple[str, str, str]]] = {}
    for item in items:
        payload = {
            "project_id": item.project_id,
            "title": item.title,
            "summary": item.summary,
            "problem_signal": item.problem_signal,
            "mechanism_signals": item.mechanism_signals,
            "implementation_signals": item.implementation_signals,
            "impact_signals": item.impact_signals,
            "technical_tags": item.technical_tags,
        }
        primary = item.source_refs[0]
        groups.setdefault(_canonical_json(payload), set()).add(
            (item.input_type, primary.source_type, primary.source_id)
        )
    return sum(len(identities) - 1 for identities in groups.values() if len(identities) > 1)


def dedupe_phase4_inputs(
    items: Iterable[Phase4EvidenceInput],
) -> tuple[list[Phase4EvidenceInput], Phase4DeduplicationReport]:
    """Normalize and remove only fully equivalent or repeated-traversal inputs.

    Deterministic winner policy: normalized collection key, then normalized JSON,
    then original JSON. No semantic or fuzzy comparisons are performed.
    """

    originals = list(items)
    normalized_records: list[Phase4EvidenceInput] = []
    refs_removed = 0
    signals_removed = 0
    for item in originals:
        normalized, current_refs_removed, current_signals_removed = _normalize_with_counts(item)
        normalized_records.append(normalized)
        refs_removed += current_refs_removed
        signals_removed += current_signals_removed
    _check_declared_integrity(originals, normalized_records)
    _check_normalized_integrity(normalized_records)
    pairs = sorted(
        zip(originals, normalized_records),
        key=lambda pair: (*_sort_key(pair[1]), _full_payload(pair[1]), _full_payload(pair[0])),
    )
    kept: list[Phase4EvidenceInput] = []
    kept_original: dict[str, Phase4EvidenceInput] = {}
    exact_removed = 0
    repeated_removed = 0
    decisions: list[Phase4DeduplicationDecision] = []
    for original, normalized in pairs:
        normalized_key = _full_payload(normalized)
        winner = kept_original.get(normalized_key)
        if winner is None:
            kept_original[normalized_key] = original
            kept.append(normalized)
            continue
        if _full_payload(original) == _full_payload(winner):
            reason = "exact_duplicate"
            exact_removed += 1
        else:
            reason = "repeated_source_record"
            repeated_removed += 1
        if len(decisions) < MAX_DECISION_SAMPLES:
            decisions.append(Phase4DeduplicationDecision(
                reason=reason,
                kept_input_id=winner.input_id,
                dropped_input_id=original.input_id,
                project_id=normalized.project_id,
                input_type=normalized.input_type,
            ))
    output = sorted(kept, key=_sort_key)
    decisions.sort(key=lambda item: (
        item.project_id,
        item.input_type,
        item.reason,
        item.kept_input_id,
        item.dropped_input_id,
    ))
    report = Phase4DeduplicationReport(
        input_count=len(originals),
        normalized_count=len(normalized_records),
        output_count=len(output),
        exact_duplicates_removed=exact_removed,
        repeated_source_records_removed=repeated_removed,
        duplicate_source_refs_removed=refs_removed,
        duplicate_signal_values_removed=signals_removed,
        retained_cross_source_records=_cross_source_retained_count(output),
        decisions=tuple(decisions),
    )
    return output, report


def normalize_and_dedupe_phase4_inputs(
    items: Iterable[Phase4EvidenceInput],
) -> tuple[list[Phase4EvidenceInput], Phase4DeduplicationReport]:
    return dedupe_phase4_inputs(items)


__all__ = [
    "Phase4DeduplicationDecision",
    "Phase4DeduplicationReport",
    "Phase4IntegrityError",
    "dedupe_phase4_inputs",
    "normalize_and_dedupe_phase4_inputs",
    "normalize_phase4_input",
    "normalize_phase4_inputs",
]
