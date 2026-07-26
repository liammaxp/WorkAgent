"""Read-only adapters from existing local artifacts to Phase 4 evidence inputs.

The adapters deliberately select only already-structured, bounded fields. They
never copy raw source text or patches and perform no extraction, inference,
cross-source deduplication, persistence, network access, or model calls.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from backend.phase4_models import (
    Phase4EvidenceInput,
    Phase4PipelineWarning,
    Phase4SourceRef,
    build_phase4_stable_id,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PHASE2_MEMORY_DIR = ROOT_DIR / "information" / "phase2_evidence_memory"
DEFAULT_PHASE3_MEMORY_PATH = ROOT_DIR / "information" / "project_memory_phase3.json"
DEFAULT_PROJECT_MEMORY_PATH = ROOT_DIR / "information" / "project_memory.json"
DEFAULT_COMPACT_FACTS_PATH = ROOT_DIR / "information" / "project_compact_facts.json"
PHASE3_SCHEMA_VERSION = "phase3.v1"
PROJECT_MEMORY_VERSION = 1
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class Phase4InputSourcePaths:
    """Injectable local artifact paths; ``None`` disables that source."""

    phase2_memory_dir: Path | None = DEFAULT_PHASE2_MEMORY_DIR
    phase3_memory_path: Path | None = DEFAULT_PHASE3_MEMORY_PATH
    project_memory_path: Path | None = DEFAULT_PROJECT_MEMORY_PATH
    compact_facts_path: Path | None = DEFAULT_COMPACT_FACTS_PATH


def _warning(
    code: str,
    message: str,
    *,
    project_id: str | None = None,
    source_id: str | None = None,
) -> Phase4PipelineWarning:
    safe_project_id = _warning_identifier(project_id)
    safe_source_id = _warning_identifier(source_id)
    return Phase4PipelineWarning(
        code=code,
        message=message,
        project_id=safe_project_id,
        source_id=safe_source_id,
    )


def _warning_identifier(value: str | None) -> str | None:
    """Omit malformed identifiers instead of truncating or echoing them."""

    normalized = _text(value)
    return normalized if normalized and len(normalized) <= 300 else None


def _normalize_project_id(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = _text(item)
        if normalized and normalized.casefold() not in seen:
            seen.add(normalized.casefold())
            output.append(normalized)
    return output


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_phase4_content_hash(payload: Mapping[str, Any]) -> str:
    """Build the canonical SHA-256 digest used for safe Phase 4 input payloads."""

    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


# Backward-compatible private alias used throughout the accepted Step 2 adapter.
_content_hash = build_phase4_content_hash


def _valid_sha256(value: Any) -> str:
    candidate = _text(value).lower()
    if len(candidate) == _SHA256_LENGTH and all(char in "0123456789abcdef" for char in candidate):
        return candidate
    return ""


def _positive_line(value: Any) -> int | None:
    """Translate legacy zero/negative line sentinels to absent provenance."""

    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _source_id(record: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = _text(record.get(key))
        if value:
            return value
    return ""


def _read_json(path: Path, label: str) -> tuple[Any | None, list[Phase4PipelineWarning]]:
    if not path.is_file():
        return None, [_warning("source_file_missing", f"Optional {label} artifact was not found.")]
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, [_warning("source_json_invalid", f"The {label} artifact is not valid JSON.")]


def _read_jsonl(path: Path, label: str) -> tuple[list[Any], list[Phase4PipelineWarning]]:
    if not path.is_file():
        return [], [_warning("source_file_missing", f"Optional {label} artifact was not found.")]
    records: list[Any] = []
    warnings: list[Phase4PipelineWarning] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    warnings.append(_warning("source_json_invalid", f"A {label} record contains invalid JSON.", source_id=f"line-{line_number}"))
    except (OSError, UnicodeError):
        return [], [_warning("source_json_invalid", f"The {label} artifact could not be read as UTF-8 JSONL.")]
    return records, warnings


def _validated_input(
    builder: Callable[[], Phase4EvidenceInput],
    *,
    project_id: str | None,
    source_id: str | None,
) -> tuple[Phase4EvidenceInput | None, Phase4PipelineWarning | None]:
    try:
        return builder(), None
    except (TypeError, ValueError):
        return None, _warning(
            "record_validation_failed",
            "A structured source record failed Phase 4 bounded-content validation.",
            project_id=project_id,
            source_id=source_id,
        )


def _phase2_chunk_ref(record: Mapping[str, Any], project_id: str) -> Phase4SourceRef:
    chunk_id = _source_id(record, "chunk_id")
    digest = _valid_sha256(record.get("hash"))
    if not chunk_id:
        raise ValueError("missing chunk_id")
    if not digest:
        raise ValueError("missing chunk hash")
    if _normalize_project_id(record.get("project_id")) != project_id:
        raise ValueError("project mismatch")
    upstream_source_id = _text(record.get("source_id"))
    metadata = {"chunk_type": _text(record.get("chunk_type"))}
    if upstream_source_id:
        metadata["upstream_source_id"] = upstream_source_id
    return Phase4SourceRef(
        source_type="phase2_evidence_chunk",
        source_id=chunk_id,
        project_id=project_id,
        repo=_text(record.get("repo")) or None,
        file_path=_text(record.get("path")) or None,
        symbol=_text(record.get("symbol")) or None,
        start_line=_positive_line(record.get("start_line")),
        end_line=_positive_line(record.get("end_line")),
        content_hash=digest,
        metadata=metadata,
    )


def _phase2_direct_ref(
    record: Mapping[str, Any],
    *,
    source_type: str,
    source_id: str,
    project_id: str,
    safe_payload: Mapping[str, Any],
) -> Phase4SourceRef:
    return Phase4SourceRef(
        source_type=source_type,
        source_id=source_id,
        project_id=project_id,
        content_hash=_content_hash(safe_payload),
    )


def load_phase2_inputs(
    memory_dir: Path,
    *,
    project_id: str | None = None,
) -> tuple[list[Phase4EvidenceInput], list[Phase4PipelineWarning]]:
    """Adapt stored Phase 2 summaries, evidence cards, and capability facts."""

    chunk_records, warnings = _read_jsonl(memory_dir / "evidence_chunks.jsonl", "Phase 2 evidence chunks")
    chunk_index = {
        _source_id(record, "chunk_id"): record
        for record in chunk_records
        if isinstance(record, dict) and _source_id(record, "chunk_id")
    }
    inputs: list[Phase4EvidenceInput] = []
    specifications = (
        ("raw_change_summaries.jsonl", "Phase 2 raw change summaries", _adapt_phase2_change_summary),
        ("evidence_cards.jsonl", "Phase 2 evidence cards", _adapt_phase2_evidence_card),
        ("capability_facts.jsonl", "Phase 2 capability facts", _adapt_phase2_capability_fact),
    )
    for filename, label, adapter in specifications:
        records, current_warnings = _read_jsonl(memory_dir / filename, label)
        warnings.extend(current_warnings)
        for record in records:
            if not isinstance(record, dict):
                warnings.append(_warning("record_not_object", f"A {label} record is not a JSON object."))
                continue
            record_project = _normalize_project_id(record.get("project_id"))
            record_id = _source_id(record, "change_id", "evidence_id", "capability_id")
            if not record_project:
                warnings.append(_warning("missing_project_id", f"A {label} record has no project ID.", source_id=record_id or None))
                continue
            if project_id is not None and record_project != project_id:
                continue
            if not record_id:
                warnings.append(_warning("missing_source_id", f"A {label} record has no stable source ID.", project_id=record_project))
                continue
            if adapter in (_adapt_phase2_change_summary, _adapt_phase2_evidence_card):
                chunk_ids = _strings(record.get("source_chunk_ids"))
                referenced_chunks = [chunk_index.get(chunk_id) for chunk_id in chunk_ids]
                if not chunk_ids or any(chunk is None for chunk in referenced_chunks):
                    warnings.append(_warning("source_ref_invalid", f"A {label} record has missing source provenance.", project_id=record_project, source_id=record_id))
                    continue
                if any(_normalize_project_id(chunk.get("project_id")) != record_project for chunk in referenced_chunks if chunk is not None):
                    warnings.append(_warning("project_id_mismatch", f"A {label} record has source provenance from another project.", project_id=record_project, source_id=record_id))
                    continue
                if any(not _valid_sha256(chunk.get("hash")) for chunk in referenced_chunks if chunk is not None):
                    warnings.append(_warning("missing_content_hash", f"A {label} record has source provenance without a valid content hash.", project_id=record_project, source_id=record_id))
                    continue
            result, current_warning = adapter(record, chunk_index)
            if result is not None:
                inputs.append(result)
            if current_warning is not None:
                warnings.append(current_warning)
    return inputs, warnings


def _phase2_refs(
    record: Mapping[str, Any],
    chunk_index: Mapping[str, Mapping[str, Any]],
    project_id: str,
) -> list[Phase4SourceRef]:
    source_ids = _strings(record.get("source_chunk_ids"))
    refs: list[Phase4SourceRef] = []
    seen: set[tuple[str, str]] = set()
    for source_id in source_ids:
        chunk = chunk_index.get(source_id)
        if chunk is None:
            raise ValueError("missing source chunk")
        ref = _phase2_chunk_ref(chunk, project_id)
        key = (ref.source_type, ref.source_id)
        if key not in seen:
            seen.add(key)
            refs.append(ref)
    if not refs:
        raise ValueError("missing source provenance")
    return refs


def _adapt_phase2_change_summary(
    record: Mapping[str, Any],
    chunk_index: Mapping[str, Mapping[str, Any]],
) -> tuple[Phase4EvidenceInput | None, Phase4PipelineWarning | None]:
    project = _normalize_project_id(record.get("project_id"))
    source_id = _source_id(record, "change_id")
    summary = _text(record.get("what_changed"))
    files = _strings(record.get("files_changed"))
    symbols = _strings(record.get("symbols_changed"))
    change_types = _strings(record.get("raw_change_type"))
    safe_payload = {"what_changed": summary, "files_changed": files, "symbols_changed": symbols, "raw_change_type": change_types}

    def build() -> Phase4EvidenceInput:
        return Phase4EvidenceInput(
            project_id=project,
            input_type="phase2_raw_change_summary",
            title=summary or f"Structured change {source_id}",
            summary=summary or "Structured Phase 2 change summary.",
            mechanism_signals=change_types,
            implementation_signals=[*files, *symbols],
            technical_tags=change_types,
            source_refs=_phase2_refs(record, chunk_index, project),
            content_hash=_content_hash(safe_payload),
        )
    return _validated_input(build, project_id=project, source_id=source_id)


def _adapt_phase2_evidence_card(
    record: Mapping[str, Any],
    chunk_index: Mapping[str, Mapping[str, Any]],
) -> tuple[Phase4EvidenceInput | None, Phase4PipelineWarning | None]:
    project = _normalize_project_id(record.get("project_id"))
    source_id = _source_id(record, "evidence_id")
    problem = _text(record.get("problem"))
    mechanism = _text(record.get("mechanism"))
    implementation = _strings(record.get("implementation_details"))
    impacts = [item for item in [_text(record.get("safe_impact")), *_strings(record.get("allowed_claims"))] if item]
    tags = _strings(record.get("metadata", {}).get("technical_tags")) if isinstance(record.get("metadata"), dict) else []
    safe_payload = {"problem": problem, "mechanism": mechanism, "implementation": implementation, "impacts": impacts, "technical_tags": sorted(tags, key=str.casefold)}

    def build() -> Phase4EvidenceInput:
        return Phase4EvidenceInput(
            project_id=project,
            input_type="phase2_evidence_card",
            title=_text(record.get("resume_angle")) or mechanism or f"Evidence {source_id}",
            summary=mechanism or problem or "Structured Phase 2 evidence card.",
            problem_signal=problem or None,
            mechanism_signals=[mechanism] if mechanism else [],
            implementation_signals=implementation,
            impact_signals=impacts,
            technical_tags=tags,
            source_refs=_phase2_refs(record, chunk_index, project),
            content_hash=_content_hash(safe_payload),
        )
    return _validated_input(build, project_id=project, source_id=source_id)


def _adapt_phase2_capability_fact(
    record: Mapping[str, Any],
    _chunk_index: Mapping[str, Mapping[str, Any]],
) -> tuple[Phase4EvidenceInput | None, Phase4PipelineWarning | None]:
    project = _normalize_project_id(record.get("project_id"))
    source_id = _source_id(record, "capability_id")
    capability_type = _text(record.get("capability_type"))
    mechanisms = _strings(record.get("mechanisms"))
    impacts = _strings(record.get("allowed_resume_claims"))
    safe_payload = {"capability_type": capability_type, "present": bool(record.get("present")), "mechanisms": mechanisms, "impacts": impacts, "source_evidence_ids": _strings(record.get("source_evidence_ids"))}

    def build() -> Phase4EvidenceInput:
        return Phase4EvidenceInput(
            project_id=project,
            input_type="phase2_capability_fact",
            title=capability_type,
            summary=mechanisms[0] if mechanisms else f"Structured capability: {capability_type}",
            mechanism_signals=mechanisms,
            impact_signals=impacts,
            technical_tags=[capability_type] if capability_type else [],
            source_refs=[_phase2_direct_ref(record, source_type="phase2_capability_fact", source_id=source_id, project_id=project, safe_payload=safe_payload)],
            content_hash=_content_hash(safe_payload),
        )
    return _validated_input(build, project_id=project, source_id=source_id)


def load_phase3_inputs(
    path: Path,
    *,
    project_id: str | None = None,
) -> tuple[list[Phase4EvidenceInput], list[Phase4PipelineWarning]]:
    payload, warnings = _read_json(path, "Phase 3 memory")
    if payload is None:
        return [], warnings
    if not isinstance(payload, dict):
        return [], [*warnings, _warning("source_schema_invalid", "The Phase 3 memory root must be a JSON object.")]
    if payload.get("schema_version") != PHASE3_SCHEMA_VERSION:
        return [], [*warnings, _warning("unsupported_source_schema", "The Phase 3 memory schema version is unsupported.")]
    projects = payload.get("projects")
    if not isinstance(projects, dict):
        return [], [*warnings, _warning("source_schema_invalid", "The Phase 3 projects field must be a JSON object.")]
    inputs: list[Phase4EvidenceInput] = []
    for project_key in sorted(projects, key=str):
        entry = projects[project_key]
        parent_project = _normalize_project_id(project_key)
        if not isinstance(entry, dict):
            warnings.append(_warning("record_not_object", "A Phase 3 project entry is not a JSON object.", project_id=parent_project or None))
            continue
        entry_project = _normalize_project_id(entry.get("project_id")) or parent_project
        if not entry_project:
            warnings.append(_warning("missing_project_id", "A Phase 3 project entry has no project ID."))
            continue
        if parent_project and entry_project != parent_project:
            warnings.append(_warning("project_id_mismatch", "A Phase 3 project entry does not match its project key.", project_id=entry_project))
            continue
        if project_id is not None and entry_project != project_id:
            continue
        for field_name, source_type, id_field, adapter in (
            ("raw_change_summaries", "phase3_raw_change_summary", "change_id", _adapt_phase3_change),
            ("evidence_cards", "phase3_evidence_card", "evidence_id", _adapt_phase3_card),
            ("capability_facts", "phase3_capability_fact", "capability_id", _adapt_phase3_capability),
        ):
            records = entry.get(field_name, [])
            if not isinstance(records, list):
                warnings.append(_warning("source_schema_invalid", f"A Phase 3 {field_name} collection is not a list.", project_id=entry_project))
                continue
            for record in records:
                if not isinstance(record, dict):
                    warnings.append(_warning("record_not_object", f"A Phase 3 {field_name} record is not a JSON object.", project_id=entry_project))
                    continue
                record_id = _source_id(record, id_field)
                record_project = _normalize_project_id(record.get("project_id"))
                if not record_project:
                    warnings.append(_warning("missing_project_id", f"A Phase 3 {field_name} record has no project ID.", source_id=record_id or None))
                    continue
                if record_project != entry_project:
                    warnings.append(_warning("project_id_mismatch", f"A Phase 3 {field_name} record has a mismatched project ID.", project_id=record_project, source_id=record_id or None))
                    continue
                if not record_id:
                    warnings.append(_warning("missing_source_id", f"A Phase 3 {field_name} record has no stable source ID.", project_id=record_project))
                    continue
                result, current_warning = adapter(record, source_type, record_id, record_project)
                if result is not None:
                    inputs.append(result)
                if current_warning is not None:
                    warnings.append(current_warning)
    return inputs, warnings


def _phase3_ref(record: Mapping[str, Any], source_type: str, source_id: str, project: str, safe_payload: Mapping[str, Any]) -> Phase4SourceRef:
    return Phase4SourceRef(
        source_type=source_type,
        source_id=source_id,
        project_id=project,
        repo=_text(record.get("repo")) or None,
        commit_sha=_text(record.get("commit_sha")) or None,
        file_path=_text(record.get("file_path")) or None,
        content_hash=_content_hash(safe_payload),
    )


def _adapt_phase3_change(record: Mapping[str, Any], source_type: str, source_id: str, project: str) -> tuple[Phase4EvidenceInput | None, Phase4PipelineWarning | None]:
    summary = _text(record.get("what_changed"))
    change_types = _strings(record.get("raw_change_types"))
    symbols = _strings(record.get("symbols_changed"))
    safe_payload = {"what_changed": summary, "raw_change_types": change_types, "symbols_changed": symbols, "repo": _text(record.get("repo")), "commit_sha": _text(record.get("commit_sha")), "file_path": _text(record.get("file_path"))}
    return _validated_input(lambda: Phase4EvidenceInput(project_id=project, input_type=source_type, title=summary or f"Structured change {source_id}", summary=summary or "Structured Phase 3 change summary.", mechanism_signals=change_types, implementation_signals=[*([safe_payload["file_path"]] if safe_payload["file_path"] else []), *symbols], technical_tags=change_types, source_refs=[_phase3_ref(record, source_type, source_id, project, safe_payload)], content_hash=_content_hash(safe_payload)), project_id=project, source_id=source_id)


def _adapt_phase3_card(record: Mapping[str, Any], source_type: str, source_id: str, project: str) -> tuple[Phase4EvidenceInput | None, Phase4PipelineWarning | None]:
    problem = _text(record.get("problem")); mechanism = _text(record.get("mechanism")); implementation = _strings(record.get("implementation_details")); impact = _text(record.get("safe_impact")); claims = _strings(record.get("allowed_claims"))
    safe_payload = {"problem": problem, "mechanism": mechanism, "implementation_details": implementation, "safe_impact": impact, "allowed_claims": sorted(claims, key=str.casefold), "source_change_ids": sorted(_strings(record.get("source_change_ids")), key=str.casefold)}
    return _validated_input(lambda: Phase4EvidenceInput(project_id=project, input_type=source_type, title=_text(record.get("resume_angle")) or mechanism or f"Evidence {source_id}", summary=mechanism or problem or "Structured Phase 3 evidence card.", problem_signal=problem or None, mechanism_signals=[mechanism] if mechanism else [], implementation_signals=implementation, impact_signals=[*([impact] if impact else []), *claims], source_refs=[_phase3_ref(record, source_type, source_id, project, safe_payload)], content_hash=_content_hash(safe_payload)), project_id=project, source_id=source_id)


def _adapt_phase3_capability(record: Mapping[str, Any], source_type: str, source_id: str, project: str) -> tuple[Phase4EvidenceInput | None, Phase4PipelineWarning | None]:
    capability = _text(record.get("capability_type")); mechanisms = _strings(record.get("mechanisms")); claims = _strings(record.get("allowed_resume_claims"))
    safe_payload = {"capability_type": capability, "present": bool(record.get("present")), "mechanisms": mechanisms, "allowed_resume_claims": sorted(claims, key=str.casefold), "source_evidence_ids": sorted(_strings(record.get("source_evidence_ids")), key=str.casefold)}
    return _validated_input(lambda: Phase4EvidenceInput(project_id=project, input_type=source_type, title=capability, summary=mechanisms[0] if mechanisms else f"Structured capability: {capability}", mechanism_signals=mechanisms, impact_signals=claims, technical_tags=[capability] if capability else [], source_refs=[_phase3_ref(record, source_type, source_id, project, safe_payload)], content_hash=_content_hash(safe_payload)), project_id=project, source_id=source_id)


def load_project_memory_inputs(
    path: Path,
    *,
    project_id: str | None = None,
) -> tuple[list[Phase4EvidenceInput], list[Phase4PipelineWarning], dict[str, str]]:
    payload, warnings = _read_json(path, "project memory")
    if payload is None:
        return [], warnings, {}
    if not isinstance(payload, dict):
        return [], [*warnings, _warning("source_schema_invalid", "The project-memory root must be a JSON object.")], {}
    if payload.get("version") != PROJECT_MEMORY_VERSION:
        return [], [*warnings, _warning("unsupported_source_schema", "The project-memory schema version is unsupported.")], {}
    projects = payload.get("projects")
    if isinstance(projects, dict):
        project_records: list[Any] = [projects]
    elif isinstance(projects, list):
        project_records = projects
    else:
        return [], [*warnings, _warning("source_schema_invalid", "The project-memory projects field must be a list or object.")], {}
    inputs: list[Phase4EvidenceInput] = []
    project_map: dict[str, str] = {}
    for record in project_records:
        if not isinstance(record, dict):
            warnings.append(_warning("record_not_object", "A project-memory record is not a JSON object."))
            continue
        current_project = _normalize_project_id(record.get("project_id"))
        project_name = _text(record.get("project_name"))
        if not current_project:
            warnings.append(_warning("missing_project_id", "A project-memory record has no project ID."))
            continue
        project_map[current_project] = current_project
        if project_name:
            project_map[project_name] = current_project
        if project_id is not None and current_project != project_id:
            continue
        identity = record.get("identity") if isinstance(record.get("identity"), dict) else {}
        problem = _text(identity.get("core_problem"))
        summary = _text(identity.get("core_value")) or _text(identity.get("positioning")) or problem
        workflows = _strings(record.get("workflows"))
        features = _strings(record.get("confirmed_features"))
        modules = _strings(record.get("key_modules"))
        impacts = _strings(record.get("resume_relevant_claims"))
        tags = _strings(record.get("tech_stack"))
        safe_payload = {"project_id": current_project, "project_name": project_name, "summary": summary, "problem": problem, "workflows": workflows, "confirmed_features": features, "key_modules": modules, "resume_relevant_claims": sorted(impacts, key=str.casefold), "tech_stack": sorted(tags, key=str.casefold)}
        source_id = build_phase4_stable_id("p4src_", current_project, {"source_type": "project_memory", **safe_payload})

        def build() -> Phase4EvidenceInput:
            return Phase4EvidenceInput(project_id=current_project, input_type="project_memory", title=project_name or current_project, summary=summary or f"Structured project memory for {project_name or current_project}.", problem_signal=problem or None, mechanism_signals=workflows, implementation_signals=[*features, *modules], impact_signals=impacts, technical_tags=tags, source_refs=[Phase4SourceRef(source_type="project_memory", source_id=source_id, project_id=current_project, content_hash=_content_hash(safe_payload), metadata={"schema_version": PROJECT_MEMORY_VERSION})], content_hash=_content_hash(safe_payload))
        result, current_warning = _validated_input(build, project_id=current_project, source_id=source_id)
        if result is not None:
            inputs.append(result)
        if current_warning is not None:
            warnings.append(current_warning)
    return inputs, warnings, project_map


def load_compact_fact_inputs(
    path: Path,
    *,
    project_id_map: Mapping[str, str],
    project_id: str | None = None,
) -> tuple[list[Phase4EvidenceInput], list[Phase4PipelineWarning]]:
    payload, warnings = _read_json(path, "project compact facts")
    if payload is None:
        return [], warnings
    if not isinstance(payload, dict):
        return [], [*warnings, _warning("source_schema_invalid", "The compact-facts root must be a JSON object.")]
    inputs: list[Phase4EvidenceInput] = []
    for entry_key in sorted(payload, key=str):
        record = payload[entry_key]
        if not isinstance(record, dict):
            warnings.append(_warning("record_not_object", "A compact-facts record is not a JSON object."))
            continue
        source_id = _source_id(record, "id") or _text(entry_key)
        explicit_project = _normalize_project_id(record.get("project_id"))
        project_name = _text(record.get("project_name"))
        current_project = explicit_project or project_id_map.get(project_name, "")
        if not current_project:
            warnings.append(_warning("missing_project_id", "A compact-facts record has no established project-ID mapping.", source_id=source_id or None))
            continue
        if explicit_project and project_name in project_id_map and project_id_map[project_name] != explicit_project:
            warnings.append(_warning("project_id_mismatch", "A compact-facts record conflicts with the established project mapping.", project_id=explicit_project, source_id=source_id or None))
            continue
        if project_id is not None and current_project != project_id:
            continue
        facts = record.get("compact_facts_json")
        if not isinstance(facts, dict):
            warnings.append(_warning("source_schema_invalid", "A compact-facts record has no structured facts object.", project_id=current_project, source_id=source_id or None))
            continue
        summary = _text(facts.get("projectSummary")); modules = _strings(facts.get("keyModules")); impacts = _strings(facts.get("resumeRelevantClaims")); tags = _strings(facts.get("technicalStack"))
        safe_payload = {"project_id": current_project, "project_name": project_name, "repo_name": _text(record.get("repo_name")), "projectSummary": summary, "keyModules": modules, "resumeRelevantClaims": sorted(impacts, key=str.casefold), "technicalStack": sorted(tags, key=str.casefold)}
        digest = _valid_sha256(record.get("source_hash")) or _content_hash(safe_payload)

        def build() -> Phase4EvidenceInput:
            return Phase4EvidenceInput(project_id=current_project, input_type="project_compact_facts", title=_text(facts.get("projectName")) or project_name or current_project, summary=summary or f"Structured compact facts for {project_name or current_project}.", implementation_signals=modules, impact_signals=impacts, technical_tags=tags, source_refs=[Phase4SourceRef(source_type="project_compact_facts", source_id=source_id, project_id=current_project, repo=_text(record.get("repo_name")) or None, content_hash=digest)], content_hash=_content_hash(safe_payload))
        result, current_warning = _validated_input(build, project_id=current_project, source_id=source_id)
        if result is not None:
            inputs.append(result)
        if current_warning is not None:
            warnings.append(current_warning)
    return inputs, warnings


def _input_sort_key(item: Phase4EvidenceInput) -> tuple[str, str, str, str, str]:
    first_ref = item.source_refs[0]
    return (item.project_id, item.input_type, first_ref.source_type, first_ref.source_id, item.input_id)


def _warning_sort_key(item: Phase4PipelineWarning) -> tuple[str, str, str, str, str]:
    return (item.project_id or "", item.code, item.source_id or "", item.severity.value, item.message)


def load_phase4_inputs(
    project_id: str | None = None,
    *,
    source_paths: Phase4InputSourcePaths | None = None,
) -> tuple[list[Phase4EvidenceInput], list[Phase4PipelineWarning]]:
    """Load all enabled local sources without writing files or merging evidence."""

    paths = source_paths or Phase4InputSourcePaths()
    requested_project = _normalize_project_id(project_id) if project_id is not None else None
    if project_id is not None and not requested_project:
        return [], [_warning("missing_project_id", "The requested project filter is blank.")]
    inputs: list[Phase4EvidenceInput] = []
    warnings: list[Phase4PipelineWarning] = []
    project_map: dict[str, str] = {}
    if paths.project_memory_path is not None:
        current, current_warnings, project_map = load_project_memory_inputs(Path(paths.project_memory_path), project_id=requested_project)
        inputs.extend(current); warnings.extend(current_warnings)
    if paths.phase2_memory_dir is not None:
        current, current_warnings = load_phase2_inputs(Path(paths.phase2_memory_dir), project_id=requested_project)
        inputs.extend(current); warnings.extend(current_warnings)
    if paths.phase3_memory_path is not None:
        current, current_warnings = load_phase3_inputs(Path(paths.phase3_memory_path), project_id=requested_project)
        inputs.extend(current); warnings.extend(current_warnings)
    if paths.compact_facts_path is not None:
        current, current_warnings = load_compact_fact_inputs(Path(paths.compact_facts_path), project_id_map=project_map, project_id=requested_project)
        inputs.extend(current); warnings.extend(current_warnings)
    return sorted(inputs, key=_input_sort_key), sorted(warnings, key=_warning_sort_key)


__all__ = [
    "Phase4InputSourcePaths",
    "build_phase4_content_hash",
    "load_compact_fact_inputs",
    "load_phase2_inputs",
    "load_phase3_inputs",
    "load_phase4_inputs",
    "load_project_memory_inputs",
]
