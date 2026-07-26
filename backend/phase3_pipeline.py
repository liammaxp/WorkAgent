"""Manual Phase 3 diff-memory pipeline orchestration.

This module adapts already-saved GitHub context into Phase 3 artifacts. It does
not fetch GitHub data, call LLMs, touch Chroma, or alter resume generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any

from phase3_diff_memory import (
    CAPABILITY_TYPES,
    DEFAULT_PHASE3_MEMORY_PATH,
    RawDiffInput,
    build_evidence_card,
    build_raw_change_summary,
    extract_capability_facts,
    extract_diff_units,
    get_phase3_project_memory,
    is_phase3_diff_memory_enabled,
    load_phase3_project_memory,
    persist_phase3_artifacts,
    score_evidence_card,
    stable_hash_text,
    summarize_phase3_project_memory,
)


logger = logging.getLogger(__name__)

DEFAULT_GITHUB_REPO_SCAN_STATE_PATH = Path("information/github_repo_scan_state.json")
PHASE3_DISABLED_MESSAGE = "Phase 3 diff memory is disabled. Set USE_PHASE3_DIFF_MEMORY=1 to enable it."
MAX_INSPECT_SAMPLE_LIMIT = 20
DEFAULT_INSPECT_SAMPLE_LIMIT = 5
PIPELINE_STATUSES = {"disabled", "no_source", "completed", "completed_with_skips", "failed"}


@dataclass
class Phase3PipelineResult:
    enabled: bool
    status: str
    source_context_count: int
    raw_diff_input_count: int
    diff_unit_count: int
    raw_change_summary_count: int
    evidence_card_candidate_count: int
    qualified_evidence_card_count: int
    capability_fact_count: int
    persisted_project_count: int
    skipped_source_count: int
    skipped_sources: list[dict[str, Any]]
    project_summaries: list[dict[str, Any]]
    memory_path: str
    errors: list[str]


def pipeline_result_to_dict(result: Phase3PipelineResult) -> dict[str, Any]:
    return asdict(result)


def empty_pipeline_result(
    *,
    enabled: bool,
    status: str,
    memory_path: str | Path = DEFAULT_PHASE3_MEMORY_PATH,
    source_context_count: int = 0,
    skipped_sources: list[dict[str, Any]] | None = None,
    errors: list[str] | None = None,
) -> Phase3PipelineResult:
    return Phase3PipelineResult(
        enabled=enabled,
        status=status if status in PIPELINE_STATUSES else "failed",
        source_context_count=source_context_count,
        raw_diff_input_count=0,
        diff_unit_count=0,
        raw_change_summary_count=0,
        evidence_card_candidate_count=0,
        qualified_evidence_card_count=0,
        capability_fact_count=0,
        persisted_project_count=0,
        skipped_source_count=len(skipped_sources or []),
        skipped_sources=skipped_sources or [],
        project_summaries=[],
        memory_path=str(memory_path),
        errors=errors or [],
    )


def load_saved_github_contexts_for_phase3(
    scan_state_path: str | Path = DEFAULT_GITHUB_REPO_SCAN_STATE_PATH,
) -> list[dict[str, Any]]:
    path = Path(scan_state_path)
    if not path.exists():
        return []
    raw_text = path.read_text(encoding="utf-8")
    if not raw_text.strip():
        return []
    payload = json.loads(raw_text)
    return normalize_saved_github_contexts(payload)


def normalize_saved_github_contexts(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [context for context in payload if isinstance(context, dict)]
    if not isinstance(payload, dict):
        return []
    repositories = payload.get("repositories")
    if isinstance(repositories, dict):
        contexts: list[dict[str, Any]] = []
        for repo_key in sorted(str(key) for key in repositories):
            entry = repositories.get(repo_key)
            if not isinstance(entry, dict):
                continue
            context = entry.get("context") if isinstance(entry.get("context"), dict) else entry
            if isinstance(context, dict):
                normalized = dict(context)
                normalized.setdefault("repository", entry.get("repository") or repo_key)
                normalized.setdefault("url", entry.get("url") or "")
                normalized.setdefault("latest_commit_sha", entry.get("latest_commit_sha") or "")
                contexts.append(normalized)
        return contexts
    for key in ["contexts", "repo_contexts", "github_contexts"]:
        value = payload.get(key)
        if isinstance(value, list):
            return [context for context in value if isinstance(context, dict)]
    if payload.get("repository") or payload.get("url") or payload.get("contribution_evidence"):
        return [payload]
    return []


def collect_phase3_raw_diff_inputs(
    github_contexts: list[dict[str, Any]],
) -> tuple[list[RawDiffInput], list[dict[str, Any]]]:
    raw_inputs: list[RawDiffInput] = []
    skipped_sources: list[dict[str, Any]] = []

    for context in github_contexts:
        if not isinstance(context, dict):
            skipped_sources.append(skipped_source(reason="malformed_context"))
            continue
        repo = phase3_context_repo(context)
        project_id = phase3_context_project_id(context)
        latest_commit_sha = phase3_context_latest_commit_sha(context)
        if not project_id:
            skipped_sources.append(skipped_source(repo=repo, commit_sha=latest_commit_sha, reason="missing_project_id"))
            continue
        if not repo:
            skipped_sources.append(skipped_source(project_id=project_id, commit_sha=latest_commit_sha, reason="missing_repo"))
            continue

        before_count = len(raw_inputs)
        for patch_entry in iter_context_patch_entries(context):
            raw_input, skipped = raw_diff_input_from_patch_entry(project_id, repo, patch_entry)
            if raw_input is not None:
                raw_inputs.append(raw_input)
            elif skipped is not None:
                skipped_sources.append(skipped)

        if len(raw_inputs) == before_count and not any_source_skip_for_project(skipped_sources, project_id, repo):
            skipped_sources.append(skipped_source(project_id=project_id, repo=repo, commit_sha=latest_commit_sha, reason="missing_patch_text"))

    return raw_inputs, skipped_sources


def dedupe_raw_diff_inputs(items: list[RawDiffInput]) -> list[RawDiffInput]:
    by_key: dict[str, RawDiffInput] = {}
    for item in items:
        key = raw_diff_input_key(item)
        by_key[key] = item
    return [by_key[key] for key in sorted(by_key)]


def run_phase3_diff_memory_pipeline(
    github_contexts: list[dict[str, Any]] | None = None,
    memory_path: str | Path = DEFAULT_PHASE3_MEMORY_PATH,
    min_evidence_score: int = 6,
) -> Phase3PipelineResult:
    if not is_phase3_diff_memory_enabled():
        return empty_pipeline_result(enabled=False, status="disabled", memory_path=memory_path)

    try:
        contexts = github_contexts if github_contexts is not None else load_saved_github_contexts_for_phase3()
    except Exception as error:
        logger.warning("Phase 3 saved GitHub context load failed: %s", error)
        return empty_pipeline_result(
            enabled=True,
            status="failed",
            memory_path=memory_path,
            errors=[safe_error_message(error)],
        )

    raw_inputs, skipped_sources = collect_phase3_raw_diff_inputs(contexts)
    raw_inputs = dedupe_raw_diff_inputs(raw_inputs)
    if not raw_inputs:
        status = "no_source"
        return empty_pipeline_result(
            enabled=True,
            status=status,
            memory_path=memory_path,
            source_context_count=len(contexts),
            skipped_sources=sort_skipped_sources(skipped_sources),
        )

    diff_units = []
    summaries = []
    candidate_cards = []
    processing_errors: list[str] = []
    for raw_input in raw_inputs:
        try:
            units = extract_diff_units(raw_input)
            diff_units.extend(units)
            for unit in units:
                summary = build_raw_change_summary(unit)
                summaries.append(summary)
                candidate_cards.append(build_evidence_card(summary))
        except Exception as error:
            logger.warning("Phase 3 source processing failed for %s: %s", raw_input.file_path, error)
            skipped_sources.append(
                skipped_source(
                    project_id=raw_input.project_id,
                    repo=raw_input.repo,
                    file_path=raw_input.file_path,
                    commit_sha=raw_input.commit_sha or "",
                    reason="diff_extraction_failed",
                    error_type=type(error).__name__,
                )
            )
            processing_errors.append(f"diff_extraction_failed:{type(error).__name__}")

    qualified_cards = [
        card for card in candidate_cards if score_evidence_card(card) >= min_evidence_score
    ]
    capabilities = extract_capability_facts(qualified_cards, min_evidence_score=min_evidence_score)

    try:
        persist_phase3_artifacts(
            summaries,
            candidate_cards,
            capabilities,
            path=memory_path,
            min_evidence_score=min_evidence_score,
        )
    except Exception as error:
        logger.warning("Phase 3 persistence failed: %s", error)
        return Phase3PipelineResult(
            enabled=True,
            status="failed",
            source_context_count=len(contexts),
            raw_diff_input_count=len(raw_inputs),
            diff_unit_count=len(diff_units),
            raw_change_summary_count=len(summaries),
            evidence_card_candidate_count=len(candidate_cards),
            qualified_evidence_card_count=len(qualified_cards),
            capability_fact_count=len(capabilities),
            persisted_project_count=0,
            skipped_source_count=len(skipped_sources),
            skipped_sources=sort_skipped_sources(skipped_sources),
            project_summaries=build_phase3_project_summaries(
                raw_inputs,
                diff_units,
                summaries,
                candidate_cards,
                qualified_cards,
                capabilities,
            ),
            memory_path=str(memory_path),
            errors=[*processing_errors, safe_error_message(error)],
        )

    status = "completed_with_skips" if skipped_sources or processing_errors else "completed"
    project_summaries = build_phase3_project_summaries(
        raw_inputs,
        diff_units,
        summaries,
        candidate_cards,
        qualified_cards,
        capabilities,
    )
    return Phase3PipelineResult(
        enabled=True,
        status=status,
        source_context_count=len(contexts),
        raw_diff_input_count=len(raw_inputs),
        diff_unit_count=len(diff_units),
        raw_change_summary_count=len(summaries),
        evidence_card_candidate_count=len(candidate_cards),
        qualified_evidence_card_count=len(qualified_cards),
        capability_fact_count=len(capabilities),
        persisted_project_count=len(project_summaries),
        skipped_source_count=len(skipped_sources),
        skipped_sources=sort_skipped_sources(skipped_sources),
        project_summaries=project_summaries,
        memory_path=str(memory_path),
        errors=processing_errors,
    )


def get_phase3_project_inspect(
    project_id: str | None = None,
    memory_path: str | Path = DEFAULT_PHASE3_MEMORY_PATH,
    sample_limit: int | str = DEFAULT_INSPECT_SAMPLE_LIMIT,
) -> dict[str, Any]:
    limit = safe_sample_limit(sample_limit)
    enabled = is_phase3_diff_memory_enabled()
    requested_project_id = str(project_id or "").strip()
    try:
        if requested_project_id:
            entry = get_phase3_project_memory(requested_project_id, memory_path)
            return {
                "enabled": enabled,
                "project_id": requested_project_id,
                **phase3_project_counts(entry),
                "capability_types": capability_types_from_project_entry(entry),
                "sample_limit": limit,
                "sample_raw_change_summaries": sample_raw_change_summaries(entry, limit),
                "sample_evidence_cards": sample_evidence_cards(entry, limit),
                "sample_capability_facts": sample_capability_facts(entry, limit),
                "errors": [],
            }
        memory = load_phase3_project_memory(memory_path)
        projects = [
            {
                "project_id": project_key,
                **phase3_project_counts(entry),
                "capability_types": capability_types_from_project_entry(entry),
            }
            for project_key, entry in sorted(memory.get("projects", {}).items())
            if isinstance(entry, dict)
        ]
        return {
            "enabled": enabled,
            "schema_version": memory.get("schema_version"),
            "updated_at": memory.get("updated_at"),
            "project_count": len(projects),
            "projects": projects,
            "errors": [],
        }
    except Exception as error:
        return {
            "enabled": enabled,
            "schema_version": None,
            "updated_at": None,
            "project_count": 0,
            "projects": [],
            "errors": [safe_error_message(error)],
        }


def get_phase3_pipeline_health(
    memory_path: str | Path = DEFAULT_PHASE3_MEMORY_PATH,
) -> dict[str, Any]:
    if not is_phase3_diff_memory_enabled():
        return {
            "enabled": False,
            "status": "disabled",
            "schema_version": None,
            "memory_exists": Path(memory_path).exists(),
            "memory_readable": False,
            "updated_at": None,
            "project_count": 0,
            "raw_change_summary_count": 0,
            "evidence_card_count": 0,
            "capability_fact_count": 0,
            "issues": [],
        }

    memory_exists = Path(memory_path).exists()
    try:
        memory = load_phase3_project_memory(memory_path)
    except Exception as error:
        return {
            "enabled": True,
            "status": "error",
            "schema_version": None,
            "memory_exists": memory_exists,
            "memory_readable": False,
            "updated_at": None,
            "project_count": 0,
            "raw_change_summary_count": 0,
            "evidence_card_count": 0,
            "capability_fact_count": 0,
            "issues": [safe_error_message(error)],
        }

    projects = memory.get("projects", {}) if isinstance(memory.get("projects"), dict) else {}
    totals = aggregate_memory_counts(projects)
    issues: list[str] = []
    if not projects:
        status = "empty"
    else:
        if totals["evidence_card_count"] == 0:
            issues.append("no_evidence_cards")
        if totals["capability_fact_count"] == 0:
            issues.append("no_capability_facts")
        status = "degraded" if issues else "ready"
    return {
        "enabled": True,
        "status": status,
        "schema_version": memory.get("schema_version"),
        "memory_exists": memory_exists,
        "memory_readable": True,
        "updated_at": memory.get("updated_at"),
        "project_count": len(projects),
        **totals,
        "issues": issues,
    }


def phase3_context_repo(context: dict[str, Any]) -> str:
    repo = str(context.get("repository") or "").strip()
    if repo:
        return repo
    url = str(context.get("url") or "").strip()
    if "github.com/" in url:
        parts = url.rstrip("/").split("github.com/", 1)[-1].split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1].removesuffix('.git')}"
    return url


def phase3_context_project_id(context: dict[str, Any]) -> str:
    return (
        str(context.get("project_id") or "").strip()
        or str(context.get("project_name") or "").strip()
        or phase3_context_repo(context)
    )


def phase3_context_latest_commit_sha(context: dict[str, Any]) -> str:
    incremental = context.get("incremental_update")
    if isinstance(incremental, dict) and str(incremental.get("head_sha") or "").strip():
        return str(incremental.get("head_sha") or "").strip()
    return str(context.get("latest_commit_sha") or context.get("commit_sha") or "").strip()


def iter_context_patch_entries(context: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    repo = phase3_context_repo(context)
    project_id = phase3_context_project_id(context)
    latest_sha = phase3_context_latest_commit_sha(context)

    def add_change(change: Any, *, commit: dict[str, Any] | None = None, source_group: str = "file_changes") -> None:
        if not isinstance(change, dict):
            entries.append(
                {
                    "project_id": project_id,
                    "repo": repo,
                    "commit_sha": latest_sha,
                    "reason": "malformed_file_change",
                }
            )
            return
        commit = commit or {}
        entries.append(
            {
                "project_id": project_id,
                "repo": repo,
                "commit_sha": str(commit.get("sha") or latest_sha or "").strip(),
                "commit_message": str(commit.get("message") or "").strip(),
                "file_path": str(
                    change.get("filename")
                    or change.get("file_path")
                    or change.get("path")
                    or change.get("name")
                    or ""
                ).strip(),
                "patch_text": str(change.get("patch") or change.get("patch_text") or change.get("diff") or ""),
                "source_type": str(change.get("source_type") or "commit_patch"),
                "source_group": source_group,
            }
        )

    for evidence in iter_dict_or_list(context.get("contribution_evidence")):
        if not isinstance(evidence, dict):
            continue
        for commit in iter_dict_or_list(evidence.get("commits")):
            if not isinstance(commit, dict):
                continue
            for change in iter_dict_or_list(commit.get("file_changes") or commit.get("changed_files")):
                add_change(change, commit=commit, source_group="commit_file_changes")
        compare_sha = str(evidence.get("head_sha") or latest_sha or "").strip()
        for change in iter_dict_or_list(evidence.get("compare_file_changes")):
            add_change(
                change,
                commit={"sha": compare_sha, "message": "Compare previous to latest"},
                source_group="compare_file_changes",
            )

    for commit in iter_dict_or_list(context.get("commits")):
        if isinstance(commit, dict):
            for change in iter_dict_or_list(commit.get("file_changes") or commit.get("changed_files")):
                add_change(change, commit=commit, source_group="top_level_commit_file_changes")

    for change in iter_dict_or_list(context.get("file_changes") or context.get("changed_files")):
        add_change(change, commit={"sha": latest_sha}, source_group="top_level_file_changes")

    return entries


def iter_dict_or_list(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    return []


def raw_diff_input_from_patch_entry(
    project_id: str,
    repo: str,
    patch_entry: dict[str, Any],
) -> tuple[RawDiffInput | None, dict[str, Any] | None]:
    if "reason" in patch_entry:
        return None, skipped_source(
            project_id=project_id,
            repo=repo,
            commit_sha=str(patch_entry.get("commit_sha") or ""),
            reason=str(patch_entry.get("reason") or "malformed_source"),
        )
    file_path = str(patch_entry.get("file_path") or "").strip()
    commit_sha = str(patch_entry.get("commit_sha") or "").strip()
    patch_text = str(patch_entry.get("patch_text") or "")
    if not file_path:
        return None, skipped_source(project_id=project_id, repo=repo, commit_sha=commit_sha, reason="missing_file_path")
    if not patch_text.strip():
        return None, skipped_source(project_id=project_id, repo=repo, file_path=file_path, commit_sha=commit_sha, reason="missing_patch_text")
    if not patch_has_change_lines(patch_text):
        return None, skipped_source(project_id=project_id, repo=repo, file_path=file_path, commit_sha=commit_sha, reason="metadata_only_patch")
    return (
        RawDiffInput(
            project_id=project_id,
            repo=repo,
            commit_sha=commit_sha or None,
            file_path=file_path,
            patch_text=patch_text,
            commit_message=str(patch_entry.get("commit_message") or "") or None,
            source_type=str(patch_entry.get("source_type") or "commit_patch"),
        ),
        None,
    )


def raw_diff_input_key(item: RawDiffInput) -> str:
    return "|".join(
        [
            item.project_id,
            item.repo,
            item.commit_sha or "",
            item.file_path,
            stable_hash_text(item.patch_text),
        ]
    )


def patch_has_change_lines(patch_text: str) -> bool:
    for line in str(patch_text or "").splitlines():
        if (line.startswith("+") and not line.startswith("+++")) or (
            line.startswith("-") and not line.startswith("---")
        ):
            return True
    return False


def skipped_source(
    *,
    project_id: str = "",
    repo: str = "",
    file_path: str = "",
    commit_sha: str = "",
    reason: str,
    error_type: str = "",
) -> dict[str, Any]:
    record = {
        "project_id": project_id,
        "repo": repo,
        "file_path": file_path,
        "commit_sha": commit_sha,
        "reason": reason,
    }
    if error_type:
        record["error_type"] = error_type
    return {key: value for key, value in record.items() if value != ""}


def sort_skipped_sources(skipped_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        skipped_sources,
        key=lambda item: (
            str(item.get("project_id") or ""),
            str(item.get("repo") or ""),
            str(item.get("file_path") or ""),
            str(item.get("commit_sha") or ""),
            str(item.get("reason") or ""),
        ),
    )


def any_source_skip_for_project(skipped_sources: list[dict[str, Any]], project_id: str, repo: str) -> bool:
    return any(
        str(item.get("project_id") or "") == project_id and str(item.get("repo") or "") == repo
        for item in skipped_sources
    )


def build_phase3_project_summaries(
    raw_inputs: list[RawDiffInput],
    diff_units: list[Any],
    summaries: list[Any],
    candidate_cards: list[Any],
    qualified_cards: list[Any],
    capabilities: list[Any],
) -> list[dict[str, Any]]:
    project_ids = sorted(
        {
            getattr(item, "project_id", "")
            for collection in [raw_inputs, diff_units, summaries, candidate_cards, qualified_cards, capabilities]
            for item in collection
            if getattr(item, "project_id", "")
        }
    )
    project_summaries = []
    for project_id in project_ids:
        project_capabilities = [
            capability for capability in capabilities if capability.project_id == project_id
        ]
        project_summaries.append(
            {
                "project_id": project_id,
                "raw_diff_input_count": sum(1 for item in raw_inputs if item.project_id == project_id),
                "diff_unit_count": sum(1 for item in diff_units if item.project_id == project_id),
                "raw_change_summary_count": sum(1 for item in summaries if item.project_id == project_id),
                "evidence_card_candidate_count": sum(1 for item in candidate_cards if item.project_id == project_id),
                "qualified_evidence_card_count": sum(1 for item in qualified_cards if item.project_id == project_id),
                "capability_fact_count": len(project_capabilities),
                "capability_types": sorted_capability_types(
                    [capability.capability_type for capability in project_capabilities]
                ),
            }
        )
    return project_summaries


def sorted_capability_types(values: list[str]) -> list[str]:
    unique_values = set(values)
    known = [capability_type for capability_type in CAPABILITY_TYPES if capability_type in unique_values]
    unknown = sorted(unique_values - set(CAPABILITY_TYPES))
    return known + unknown


def safe_sample_limit(value: int | str | None) -> int:
    try:
        parsed = int(value if value is not None else DEFAULT_INSPECT_SAMPLE_LIMIT)
    except (TypeError, ValueError):
        parsed = DEFAULT_INSPECT_SAMPLE_LIMIT
    return max(0, min(parsed, MAX_INSPECT_SAMPLE_LIMIT))


def phase3_project_counts(entry: dict[str, Any]) -> dict[str, int]:
    return {
        "raw_change_summary_count": len(entry.get("raw_change_summaries") or []),
        "evidence_card_count": len(entry.get("evidence_cards") or []),
        "capability_fact_count": len(entry.get("capability_facts") or []),
    }


def capability_types_from_project_entry(entry: dict[str, Any]) -> list[str]:
    return sorted_capability_types(
        [
            str(capability.get("capability_type") or "")
            for capability in entry.get("capability_facts", [])
            if isinstance(capability, dict)
        ]
    )


def sample_raw_change_summaries(entry: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    return [
        {
            "change_id": str(item.get("change_id") or ""),
            "file_path": str(item.get("file_path") or ""),
            "raw_change_types": item.get("raw_change_types") if isinstance(item.get("raw_change_types"), list) else [],
            "what_changed": str(item.get("what_changed") or ""),
            "confidence": str(item.get("confidence") or "low"),
        }
        for item in (entry.get("raw_change_summaries") or [])[:limit]
        if isinstance(item, dict)
    ]


def sample_evidence_cards(entry: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": str(item.get("evidence_id") or ""),
            "problem": str(item.get("problem") or ""),
            "mechanism": str(item.get("mechanism") or ""),
            "safe_impact": str(item.get("safe_impact") or ""),
            "resume_angle": str(item.get("resume_angle") or "implementation_change"),
            "confidence": str(item.get("confidence") or "low"),
            "metric_support": str(item.get("metric_support") or "none"),
        }
        for item in (entry.get("evidence_cards") or [])[:limit]
        if isinstance(item, dict)
    ]


def sample_capability_facts(entry: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    return [
        {
            "capability_id": str(item.get("capability_id") or ""),
            "capability_type": str(item.get("capability_type") or ""),
            "confidence": str(item.get("confidence") or "low"),
            "metric_support": str(item.get("metric_support") or "none"),
            "source_evidence_count": len(item.get("source_evidence_ids") or []),
        }
        for item in (entry.get("capability_facts") or [])[:limit]
        if isinstance(item, dict)
    ]


def aggregate_memory_counts(projects: dict[str, Any]) -> dict[str, int]:
    totals = {
        "raw_change_summary_count": 0,
        "evidence_card_count": 0,
        "capability_fact_count": 0,
    }
    for entry in projects.values():
        if not isinstance(entry, dict):
            continue
        counts = phase3_project_counts(entry)
        for key in totals:
            totals[key] += counts[key]
    return totals


def safe_error_message(error: Exception | str) -> str:
    return str(error) or type(error).__name__
