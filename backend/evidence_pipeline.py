"""Safe GitHub evidence evidence pipeline orchestration and inspection helpers."""

from __future__ import annotations

import os
import re
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import capability_extractor
import evidence_card_extractor
import evidence_change_summary
import evidence_chunker
import evidence_memory


GITHUB_EVIDENCE_MEMORY_ENV = "USE_GITHUB_EVIDENCE_MEMORY"
ENABLED_VALUES = {"1", "true", "yes", "on"}
DISABLED_MESSAGE = "GitHub context evidence memory is disabled."
DEFAULT_LIMIT = 10
MAX_LIMIT = 50
SUMMARY_CHARS = 240
PIPELINE_RUN_MANIFEST = ".pipeline_runs.json"

STAGE_ORDER = [
    "chunk",
    "summarize_changes",
    "build_evidence_cards",
    "build_capability_facts",
]
STAGE_BUILDERS: dict[str, Callable[..., dict[str, Any]]] = {
    "chunk": evidence_chunker.chunk_github_evidence_raw_sources,
    "summarize_changes": evidence_change_summary.build_github_evidence_raw_change_summaries,
    "build_evidence_cards": evidence_card_extractor.build_github_evidence_cards,
    "build_capability_facts": capability_extractor.build_github_evidence_capability_facts,
}
COUNT_KEYS = [
    "raw_sources_count",
    "chunks_count",
    "raw_change_summaries_count",
    "evidence_cards_count",
    "capability_facts_count",
]


def run_github_evidence_pipeline(
    project_id: str | None = None,
    limit: int | None = None,
    stages: list[str] | str | None = None,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    requested_project_id = normalize_project_id(project_id)
    if not github_evidence_enabled():
        return {
            "enabled": False,
            "memory_type": "github_evidence",
            "project_id": None,
            "ran_stages": [],
            "message": DISABLED_MESSAGE,
            "counts_before": {},
            "counts_after": {},
            "stage_results": [],
            "errors": [],
        }

    requested_stages, invalid_stages = normalize_stages(stages)
    counts_before = safe_counts(requested_project_id)
    if invalid_stages:
        error = {
            "type": "invalid_stage",
            "message": f"Unsupported GitHub evidence pipeline stage(s): {', '.join(invalid_stages)}",
            "invalid_stages": invalid_stages,
            "supported_stages": STAGE_ORDER,
        }
        return {
            "enabled": True,
            "ok": False,
            "memory_type": "github_evidence",
            "project_id": requested_project_id or None,
            "requested_stages": requested_stages,
            "ran_stages": [],
            "counts_before": counts_before,
            "counts_after": counts_before,
            "deltas": zero_deltas(counts_before),
            "stage_results": [],
            "project_summaries": safe_project_summaries(requested_project_id),
            "errors": [error],
            "warnings": [],
            "message": "GitHub evidence evidence pipeline did not run because one or more stages are invalid.",
        }

    safe_limit = normalize_optional_limit(limit)
    stage_results: list[dict[str, Any]] = []
    ran_stages: list[str] = []
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []

    if counts_before.get("raw_sources_count", 0) == 0:
        warnings.append("No GitHub evidence raw sources found; run GitHub context sync separately before building.")

    for stage in requested_stages:
        try:
            result = STAGE_BUILDERS[stage](project_id=requested_project_id or None, limit=safe_limit)
            stage_result = summarize_stage_result(stage, result)
        except Exception as error:  # pragma: no cover - defensive helper safety
            stage_result = {
                "stage": stage,
                "ok": False,
                "processed": 0,
                "created": 0,
                "updated": 0,
                "unchanged": 0,
                "created_or_updated": 0,
                "skipped": 0,
                "message": f"GitHub evidence pipeline stage failed: {stage}",
                "errors": [str(error)],
            }
        ran_stages.append(stage)
        stage_results.append(stage_result)
        if not stage_result["ok"]:
            errors.extend(
                {
                    "stage": stage,
                    "message": str(error_message),
                }
                for error_message in stage_result.get("errors", [])
            )
            if not continue_on_error:
                break

    counts_after = safe_counts(requested_project_id)
    if not errors and requested_stages == STAGE_ORDER:
        save_pipeline_run_manifest(requested_project_id)
    return {
        "enabled": True,
        "ok": not errors,
        "memory_type": "github_evidence",
        "project_id": requested_project_id or None,
        "requested_stages": requested_stages,
        "ran_stages": ran_stages,
        "counts_before": counts_before,
        "counts_after": counts_after,
        "deltas": count_deltas(counts_before, counts_after),
        "stage_results": stage_results,
        "project_summaries": safe_project_summaries(requested_project_id),
        "errors": errors,
        "warnings": warnings,
        "message": "GitHub evidence evidence pipeline completed." if not errors else "GitHub evidence evidence pipeline completed with errors.",
    }


def inspect_github_evidence_memory(
    project_id: str | None = None,
    limit: int = DEFAULT_LIMIT,
    include_samples: bool = True,
) -> dict[str, Any]:
    requested_project_id = normalize_project_id(project_id)
    safe_limit = safe_sample_limit(limit)
    empty_samples = empty_sample_sets()
    if not github_evidence_enabled():
        return {
            "enabled": False,
            "memory_type": "github_evidence",
            "project_id": requested_project_id or None,
            "limit": safe_limit,
            "counts": {},
            "projects": [],
            "samples": empty_samples if include_samples else {},
            "message": DISABLED_MESSAGE,
            "errors": [],
        }

    try:
        counts = evidence_memory.get_github_evidence_memory_counts(project_id=requested_project_id or None)
        stats = evidence_memory.github_raw_source_stats(project_id=requested_project_id or None)
        samples = build_safe_samples(requested_project_id, safe_limit) if include_samples else {}
        return {
            "enabled": True,
            "memory_type": "github_evidence",
            "project_id": requested_project_id or None,
            "limit": safe_limit,
            "counts": counts,
            "projects": stats.get("projects", []),
            "samples": samples,
            "errors": [],
        }
    except Exception as error:  # pragma: no cover - defensive inspect safety
        return {
            "enabled": True,
            "memory_type": "github_evidence",
            "project_id": requested_project_id or None,
            "limit": safe_limit,
            "counts": {},
            "projects": [],
            "samples": empty_samples if include_samples else {},
            "message": "GitHub evidence evidence inspect could not be read.",
            "errors": [str(error)],
        }


def get_github_evidence_health(project_id: str | None = None) -> dict[str, Any]:
    requested_project_id = normalize_project_id(project_id)
    if not github_evidence_enabled():
        return {
            "enabled": False,
            "memory_type": "github_evidence",
            "project_id": requested_project_id or None,
            "counts": {},
            "health": {
                "has_raw_sources": False,
                "has_chunks": False,
                "has_raw_change_summaries": False,
                "has_evidence_cards": False,
                "has_capability_facts": False,
            },
            "pipeline_complete": False,
            "missing_stages": [],
            "next_recommended_action": "enable_github_evidence_memory",
            "message": DISABLED_MESSAGE,
            "errors": [],
        }

    try:
        counts = evidence_memory.get_github_evidence_memory_counts(project_id=requested_project_id or None)
    except Exception as error:  # pragma: no cover - defensive health safety
        return {
            "enabled": True,
            "memory_type": "github_evidence",
            "project_id": requested_project_id or None,
            "counts": {},
            "health": {},
            "pipeline_complete": False,
            "missing_stages": [],
            "next_recommended_action": "inspect_storage_error",
            "message": "GitHub evidence pipeline health could not be read.",
            "errors": [str(error)],
        }

    health = {
        "has_raw_sources": counts.get("raw_sources_count", 0) > 0,
        "has_chunks": counts.get("chunks_count", 0) > 0,
        "has_raw_change_summaries": counts.get("raw_change_summaries_count", 0) > 0,
        "has_evidence_cards": counts.get("evidence_cards_count", 0) > 0,
        "has_capability_facts": counts.get("capability_facts_count", 0) > 0,
    }
    missing_stages = [
        name
        for name, present in [
            ("raw_sources", health["has_raw_sources"]),
            ("chunk", health["has_chunks"]),
            ("summarize_changes", health["has_raw_change_summaries"]),
            ("build_evidence_cards", health["has_evidence_cards"]),
            ("build_capability_facts", health["has_capability_facts"]),
        ]
        if not present
    ]
    next_action = next_recommended_action(health)
    records_available = all(health.values())
    lineage_current = pipeline_run_manifest_is_current(requested_project_id) if records_available else False
    if records_available and not lineage_current:
        next_action = "run_pipeline"
    return {
        "enabled": True,
        "memory_type": "github_evidence",
        "project_id": requested_project_id or None,
        "counts": counts,
        "health": health,
        "records_available": records_available,
        "pipeline_complete": lineage_current,
        "lineage_current": lineage_current,
        "missing_stages": missing_stages,
        "next_recommended_action": next_action,
        "errors": [],
    }


def github_evidence_enabled() -> bool:
    return str(os.getenv(GITHUB_EVIDENCE_MEMORY_ENV, "1")).strip().lower() in ENABLED_VALUES


def _pipeline_manifest_path() -> Path:
    return evidence_memory.get_github_evidence_memory_dir() / PIPELINE_RUN_MANIFEST


def _project_record_signatures(project_id: str) -> dict[str, str]:
    signatures = {}
    for record_type in evidence_memory.RECORD_FILES:
        records = evidence_memory.read_records(record_type)
        if project_id:
            records = [record for record in records if str(record.get("project_id") or "") == project_id]
        payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        signatures[record_type] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return signatures


def save_pipeline_run_manifest(project_id: str) -> None:
    path = _pipeline_manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        manifests = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        manifests = {}
    if not isinstance(manifests, dict):
        manifests = {}
    manifest_project_ids = [project_id] if project_id else sorted({
        str(record.get("project_id") or "")
        for record in evidence_memory.read_records(evidence_memory.GITHUB_RAW_SOURCES)
        if str(record.get("project_id") or "")
    })
    if not manifest_project_ids:
        manifest_project_ids = [""]
    for current_project_id in manifest_project_ids:
        manifests[current_project_id or "__all__"] = {
            "signatures": _project_record_signatures(current_project_id)
        }
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(manifests, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def pipeline_run_manifest_is_current(project_id: str) -> bool:
    path = _pipeline_manifest_path()
    try:
        manifests = json.loads(path.read_text(encoding="utf-8"))
        manifest = manifests.get(project_id or "__all__", {})
        return manifest.get("signatures") == _project_record_signatures(project_id)
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def normalize_project_id(project_id: str | None) -> str:
    return str(project_id or "").strip()


def normalize_optional_limit(limit: int | str | None) -> int | None:
    if limit in (None, ""):
        return None
    try:
        return max(0, int(limit))
    except (TypeError, ValueError):
        return None


def safe_sample_limit(limit: int | str | None) -> int:
    try:
        parsed = int(limit if limit not in (None, "") else DEFAULT_LIMIT)
    except (TypeError, ValueError):
        parsed = DEFAULT_LIMIT
    if parsed < 1:
        return DEFAULT_LIMIT
    return max(1, min(parsed, MAX_LIMIT))


def normalize_stages(stages: list[str] | str | None) -> tuple[list[str], list[str]]:
    if stages is None:
        requested = list(STAGE_ORDER)
    elif isinstance(stages, str):
        requested = [stage.strip() for stage in stages.split(",") if stage.strip()]
    else:
        requested = [str(stage).strip() for stage in stages if str(stage).strip()]

    invalid = [stage for stage in requested if stage not in STAGE_BUILDERS]
    deduped = []
    seen = set()
    for stage in STAGE_ORDER:
        if stage in requested and stage not in seen:
            deduped.append(stage)
            seen.add(stage)
    return deduped, invalid


def summarize_stage_result(stage: str, result: dict[str, Any]) -> dict[str, Any]:
    errors = safe_string_list(result.get("errors"))
    ok = bool(result.get("enabled", True)) and not errors
    if not result.get("enabled", True):
        errors.append(DISABLED_MESSAGE)
    return {
        "stage": stage,
        "ok": ok,
        "processed": processed_count(stage, result),
        "created": stage_count(stage, result, "created"),
        "updated": stage_count(stage, result, "updated"),
        "unchanged": stage_count(stage, result, "unchanged"),
        "created_or_updated": created_or_updated_count(stage, result),
        "skipped": skipped_count(stage, result),
        "message": str(result.get("message") or ""),
        "errors": errors,
        "skips": safe_limited_list(result.get("skips"), 10),
    }


def processed_count(stage: str, result: dict[str, Any]) -> int:
    key_by_stage = {
        "chunk": "processed_raw_sources",
        "summarize_changes": "processed_chunks",
        "build_evidence_cards": "processed_summaries",
        "build_capability_facts": "processed_evidence_cards",
    }
    return safe_int(result.get(key_by_stage.get(stage, "")))


def created_or_updated_count(stage: str, result: dict[str, Any]) -> int:
    key_by_stage = {
        "chunk": "created_or_updated_chunks",
        "summarize_changes": "created_or_updated_summaries",
        "build_evidence_cards": "created_or_updated_evidence_cards",
        "build_capability_facts": "created_or_updated_capability_facts",
    }
    return safe_int(result.get(key_by_stage.get(stage, "")))


def stage_count(stage: str, result: dict[str, Any], kind: str) -> int:
    suffix_by_stage = {
        "chunk": "chunks",
        "summarize_changes": "summaries",
        "build_evidence_cards": "evidence_cards",
        "build_capability_facts": "capability_facts",
    }
    suffix = suffix_by_stage.get(stage)
    return safe_int(result.get(f"{kind}_{suffix}")) if suffix else 0


def skipped_count(stage: str, result: dict[str, Any]) -> int:
    key_by_stage = {
        "chunk": "skipped_raw_sources",
        "summarize_changes": "skipped_chunks",
        "build_evidence_cards": "skipped_summaries",
        "build_capability_facts": "skipped_evidence_cards",
    }
    return safe_int(result.get(key_by_stage.get(stage, ""), 0))


def safe_counts(project_id: str = "") -> dict[str, int]:
    try:
        return evidence_memory.get_github_evidence_memory_counts(project_id=project_id or None)
    except Exception:
        return {key: 0 for key in COUNT_KEYS}


def safe_project_summaries(project_id: str = "") -> list[dict[str, Any]]:
    try:
        stats = evidence_memory.github_raw_source_stats(project_id=project_id or None)
        return stats.get("projects", []) if isinstance(stats.get("projects"), list) else []
    except Exception:
        return []


def count_deltas(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: safe_int(after.get(key)) - safe_int(before.get(key)) for key in COUNT_KEYS}


def zero_deltas(counts: dict[str, int]) -> dict[str, int]:
    return {key: 0 for key in counts}


def empty_sample_sets() -> dict[str, list[Any]]:
    return {
        "raw_sources": [],
        "chunks": [],
        "raw_change_summaries": [],
        "evidence_cards": [],
        "capability_facts": [],
    }


def build_safe_samples(project_id: str, limit: int) -> dict[str, list[dict[str, Any]]]:
    return {
        "raw_sources": [raw_source_sample(record) for record in limited_records(evidence_memory.GITHUB_RAW_SOURCES, project_id, limit)],
        "chunks": [chunk_sample(record) for record in limited_records(evidence_memory.EVIDENCE_CHUNKS, project_id, limit)],
        "raw_change_summaries": [
            raw_change_summary_sample(record)
            for record in limited_records(evidence_memory.RAW_CHANGE_SUMMARIES, project_id, limit)
        ],
        "evidence_cards": [evidence_card_sample(record) for record in limited_records(evidence_memory.EVIDENCE_CARDS, project_id, limit)],
        "capability_facts": [
            capability_fact_sample(record)
            for record in limited_records(evidence_memory.CAPABILITY_FACTS, project_id, limit)
        ],
    }


def limited_records(record_type: str, project_id: str, limit: int) -> list[dict[str, Any]]:
    records = (
        evidence_memory.read_records_by_project(record_type, project_id)
        if project_id
        else evidence_memory.read_records(record_type)
    )
    return sorted(records, key=record_sort_key)[:limit]


def record_sort_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("project_id") or ""),
        str(record.get("repo") or ""),
        str(
            record.get("source_id")
            or record.get("chunk_id")
            or record.get("change_id")
            or record.get("evidence_id")
            or record.get("capability_id")
            or ""
        ),
    )


def raw_source_sample(record: dict[str, Any]) -> dict[str, Any]:
    raw_text = str(record.get("raw_text") or "")
    return {
        "source_id": str(record.get("source_id") or ""),
        "project_id": str(record.get("project_id") or ""),
        "repo": str(record.get("repo") or ""),
        "source_type": str(record.get("source_type") or "unknown"),
        "path": str(record.get("path") or ""),
        "commit_sha": str(record.get("commit_sha") or ""),
        "raw_chars": len(raw_text),
        "raw_hash": str(record.get("raw_hash") or ""),
        "raw_available": bool(raw_text),
    }


def chunk_sample(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": str(record.get("chunk_id") or ""),
        "source_id": str(record.get("source_id") or ""),
        "project_id": str(record.get("project_id") or ""),
        "repo": str(record.get("repo") or ""),
        "path": str(record.get("path") or ""),
        "symbol": str(record.get("symbol") or ""),
        "chunk_type": str(record.get("chunk_type") or "unknown"),
        "summary": truncate(record.get("summary") or "", SUMMARY_CHARS),
        "keywords": safe_string_list(record.get("keywords"))[:10],
        "technical_tags": safe_string_list(record.get("technical_tags"))[:10],
        "text_chars": len(str(record.get("text") or "")),
    }


def raw_change_summary_sample(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "change_id": str(record.get("change_id") or ""),
        "project_id": str(record.get("project_id") or ""),
        "source_chunk_ids": safe_string_list(record.get("source_chunk_ids"))[:10],
        "files_changed": safe_string_list(record.get("files_changed"))[:10],
        "symbols_changed": safe_string_list(record.get("symbols_changed"))[:10],
        "raw_change_type": safe_string_list(record.get("raw_change_type"))[:10],
        "what_changed": truncate(record.get("what_changed") or "", SUMMARY_CHARS),
        "direct_code_evidence": [truncate(item, SUMMARY_CHARS) for item in safe_string_list(record.get("direct_code_evidence"))[:6]],
    }


def evidence_card_sample(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": str(record.get("evidence_id") or ""),
        "project_id": str(record.get("project_id") or ""),
        "source_chunk_ids": safe_string_list(record.get("source_chunk_ids"))[:10],
        "problem": truncate(record.get("problem") or "", SUMMARY_CHARS),
        "mechanism": truncate(record.get("mechanism") or "", SUMMARY_CHARS),
        "safe_impact": truncate(record.get("safe_impact") or "", SUMMARY_CHARS),
        "resume_angle": str(record.get("resume_angle") or ""),
        "confidence": str(record.get("confidence") or "low"),
        "metric_support": str(record.get("metric_support") or "none"),
    }


def capability_fact_sample(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "capability_id": str(record.get("capability_id") or ""),
        "project_id": str(record.get("project_id") or ""),
        "capability_type": str(record.get("capability_type") or "unknown"),
        "present": bool(record.get("present")),
        "confidence": str(record.get("confidence") or "low"),
        "mechanisms": [truncate(item, SUMMARY_CHARS) for item in safe_string_list(record.get("mechanisms"))[:8]],
        "source_evidence_ids": safe_string_list(record.get("source_evidence_ids"))[:12],
        "metric_support": str(record.get("metric_support") or "none"),
    }


def next_recommended_action(health: dict[str, bool]) -> str:
    if all(
        health.get(key)
        for key in [
            "has_raw_sources",
            "has_chunks",
            "has_raw_change_summaries",
            "has_evidence_cards",
            "has_capability_facts",
        ]
    ):
        return "inspect"
    if health.get("has_evidence_cards") and not health.get("has_capability_facts"):
        return "build_capability_facts"
    if health.get("has_raw_change_summaries") and not health.get("has_evidence_cards"):
        return "build_evidence_cards"
    if health.get("has_chunks") and not health.get("has_raw_change_summaries"):
        return "summarize_changes"
    if health.get("has_raw_sources") and not health.get("has_chunks"):
        return "run_chunk"
    return "wait_for_raw_sources"


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def safe_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def safe_limited_list(value: Any, limit: int) -> list[Any]:
    return list(value[:limit]) if isinstance(value, list) else []


def truncate(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
