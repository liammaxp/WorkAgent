"""project change memory models and feature flag helpers.

This module is intentionally standalone. It defines data shapes and small
deterministic helpers only; it does not run diff extraction or write memory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_CHANGE_MEMORY_ENV = "USE_PROJECT_CHANGE_MEMORY"
PROJECT_CHANGE_MEMORY_SCHEMA_VERSION = "project_change_memory.v1"
DEFAULT_PROJECT_CHANGE_MEMORY_PATH = Path("information/project_change_memory.json")

RAW_CHANGE_TYPES = [
    "validation_logic_update",
    "merge_logic_update",
    "retrieval_logic_update",
    "memory_storage_update",
    "chunking_update",
    "fallback_update",
    "prompt_constraint_update",
    "quality_gate_update",
    "test_update",
    "ui_debug_update",
    "unknown",
]

CAPABILITY_TYPES = [
    "evidence_grounded_generation",
    "retrieval_and_reranking",
    "local_project_memory",
    "deterministic_latex_validation",
    "template_pollution_blocking",
    "output_quality_control",
    "validation_and_repair",
    "llm_reliability",
    "token_or_cost_reduction",
]

CHANGE_HINT_RULES = [
    (
        "validation_logic",
        [
            "validate",
            "validation",
            "validator",
            "unsupported",
            "claim boundary",
            "forbidden_claims",
            "allowed_claims",
        ],
    ),
    (
        "merge_logic",
        [
            "merge",
            "merged",
            "merge_staged_resume",
            "final_bullets",
            "bullet_depth_profile",
            "star_analysis",
        ],
    ),
    (
        "retrieval_logic",
        [
            "retrieve",
            "retrieval",
            "search",
            "query",
            "rerank",
            "ranking",
            "evidence",
        ],
    ),
    (
        "chunking_logic",
        [
            "chunk",
            "hunk",
            "split_text",
            "token window",
        ],
    ),
    (
        "local_memory_or_cache",
        [
            "sqlite",
            "cache",
            "memory",
            "project_memory",
            "local db",
            "db",
        ],
    ),
    (
        "fallback_or_retry",
        [
            "fallback",
            "retry",
            "recover",
            "repair",
        ],
    ),
    (
        "latex_pipeline",
        [
            "latex",
            "tex",
            "compile",
            "pdf",
        ],
    ),
    (
        "prompt_constraint",
        [
            "prompt",
            "system message",
            "instruction",
            "constraint",
        ],
    ),
    (
        "quality_gate",
        [
            "quality_gate",
            "quality gate",
            "gate",
            "fail_quality",
            "blocker",
        ],
    ),
    (
        "test_update",
        [
            "tests/",
            "test_",
            "pytest",
            "assert",
        ],
    ),
    (
        "ui_debug",
        [
            "frontend/",
            "debug",
            "inspect",
            "preview",
            "status panel",
        ],
    ),
    (
        "api_route",
        [
            "@app.get",
            "@app.post",
            "@router.get",
            "@router.post",
            "api/",
        ],
    ),
    (
        "model_or_schema",
        [
            "dataclass",
            "basemodel",
            "schema",
            "field",
            "capabilityfact",
            "evidencecard",
            "rawchangesummary",
            "diffunit",
            "rawdiffinput",
        ],
    ),
]

FUNCTION_SYMBOL_RE = re.compile(r"^\s*(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
CLASS_SYMBOL_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|:)")
ROUTE_SYMBOL_RE = re.compile(r"""^\s*@(app|router)\.(get|post)\(\s*["']([^"']+)["']""")
HUNK_HEADER_RE = re.compile(r"^@@\s+.*?@@\s*(.*)$")

HINT_TO_RAW_CHANGE_TYPE = {
    "validation_logic": "validation_logic_update",
    "merge_logic": "merge_logic_update",
    "retrieval_logic": "retrieval_logic_update",
    "local_memory_or_cache": "memory_storage_update",
    "chunking_logic": "chunking_update",
    "fallback_or_retry": "fallback_update",
    "prompt_constraint": "prompt_constraint_update",
    "quality_gate": "quality_gate_update",
    "test_update": "test_update",
    "ui_debug": "ui_debug_update",
}

MEMORY_SCHEMA_TERMS = [
    "memory",
    "project_memory",
    "cache",
    "sqlite",
    "db",
    "capabilityfact",
    "evidencecard",
    "rawchangesummary",
    "diffunit",
    "rawdiffinput",
    "source_evidence_ids",
    "allowed_resume_claims",
    "forbidden_claims",
    "metric_support",
]

EVIDENCE_REFERENCE_TERMS = [
    "unsupported_metric",
    "unsupported metric",
    "unsupported claim",
    "forbidden_claims",
    "allowed_claims",
    "claim boundary",
    "validation",
    "validator",
    "merge_staged_resume",
    "final_bullets",
    "bullet_depth_profile",
    "star_analysis",
    "retrieve_evidence_for_project",
    "retrieve",
    "retrieval",
    "query",
    "rerank",
    "evidence",
    "project_memory",
    "sqlite",
    "cache",
    "fallback",
    "retry",
    "repair",
    "prompt",
    "system message",
    "instruction",
    "constraint",
    "quality_gate",
    "fail_quality",
    "dataclass",
    "basemodel",
    "schema",
    "diffunit",
    "rawdiffinput",
    "rawchangesummary",
    "evidencecard",
    "capabilityfact",
]

SECRET_LINE_RE = re.compile(
    r"(api[_-]?key|secret|token|password|credential|authorization)\s*[=:]",
    re.IGNORECASE,
)

FORBIDDEN_CONFIRMED_CLAIM_PATTERNS = [
    r"reduced hallucinations",
    r"eliminated hallucinations",
    r"guaranteed",
    r"improved accuracy",
    r"improved performance",
    r"by\s+\d+%",
    r"\d+%",
    r"ats success",
]

RESUME_ANGLE_BY_RAW_CHANGE_TYPE = {
    "validation_logic_update": "validation_and_safety",
    "merge_logic_update": "deterministic_merge",
    "retrieval_logic_update": "evidence_retrieval",
    "memory_storage_update": "project_memory",
    "chunking_update": "content_chunking",
    "fallback_update": "fallback_and_repair",
    "prompt_constraint_update": "generation_constraints",
    "quality_gate_update": "output_quality_control",
    "test_update": "testing_and_regression",
    "ui_debug_update": "developer_observability",
    "unknown": "implementation_change",
}

RESUME_ANGLE_PRIORITY = [
    "validation_logic_update",
    "merge_logic_update",
    "retrieval_logic_update",
    "memory_storage_update",
    "chunking_update",
    "fallback_update",
    "prompt_constraint_update",
    "quality_gate_update",
    "test_update",
    "ui_debug_update",
]

GLOBAL_FORBIDDEN_CLAIMS = [
    "solved hallucinations",
    "eliminated hallucinations",
    "reduced hallucinations by X%",
    "guaranteed factual correctness",
    "guaranteed correct resume output",
    "improved ATS success",
    "improved accuracy by X%",
    "improved performance by X%",
    "reduced latency by X%",
    "reduced cost by X%",
    "eliminated all failures",
]

CATEGORY_FORBIDDEN_CLAIMS = {
    "validation_logic_update": [
        "eliminated unsupported claims",
        "guaranteed factual output",
    ],
    "merge_logic_update": [
        "guaranteed optimal bullet selection",
        "eliminated generic bullets",
    ],
    "retrieval_logic_update": [
        "improved retrieval recall without measured evidence",
        "guaranteed all relevant evidence was found",
    ],
    "memory_storage_update": [
        "reduced token usage without measured evidence",
        "reduced processing cost without measured evidence",
    ],
    "fallback_update": [
        "eliminated pipeline failures",
        "guaranteed recovery from all errors",
    ],
    "prompt_constraint_update": [
        "improved LLM reliability",
        "reduced hallucinations",
    ],
    "quality_gate_update": [
        "guaranteed all generated output is correct",
    ],
    "unknown": [
        "claimed measurable impact without supporting evidence",
    ],
}

UNSUPPORTED_EVIDENCE_CLAIM_PATTERNS = [
    r"reduced hallucinations",
    r"eliminated hallucinations",
    r"solved hallucinations",
    r"guaranteed",
    r"improved accuracy",
    r"improved performance",
    r"improved recall",
    r"improved factuality",
    r"improved llm reliability",
    r"guaranteed reliability",
    r"mitigated hallucinations",
    r"reduced latency",
    r"reduced cost",
    r"ats success",
    r"\boptimized\b",
    r"by\s+\d+%",
]

UNIVERSAL_GUARANTEE_RE = re.compile(
    r"\b(guaranteed|always|never|eliminated all|solved all)\b",
    re.IGNORECASE,
)

EXPLICIT_METRIC_PATTERNS = [
    r"reduced processing time from \d+(?:\.\d+)?\s*(?:seconds?|secs?|s|milliseconds?|ms) to \d+(?:\.\d+)?\s*(?:seconds?|secs?|s|milliseconds?|ms)",
    r"processed \d+(?:,\d{3})*\s+(?:repositories|repos|records|files|commits|patches)",
    r"reduced token usage by \d+(?:\.\d+)?%",
]

AMBIGUOUS_METRIC_PATTERNS = [
    r"\btop_k\s*=\s*\d+",
    r"\btimeout\s*=\s*\d+",
    r"\bchunk_size\s*=\s*\d+",
    r"\bmax_retries\s*=\s*\d+",
    r"\bretry count\s*\d+",
    r"\bthreshold\s*\d+(?:\.\d+)?",
    r"\bconfidence_threshold\s*=\s*\d+(?:\.\d+)?",
    r"\b\d+(?:\.\d+)?\b",
]

STRICT_CAPABILITY_TYPES = {
    "evidence_grounded_generation",
    "llm_reliability",
    "token_or_cost_reduction",
    "deterministic_latex_validation",
    "template_pollution_blocking",
}

CAPABILITY_FORBIDDEN_CLAIMS = {
    "evidence_grounded_generation": [
        "guaranteed all generated output is factual",
        "guaranteed all generated claims are supported",
    ],
    "retrieval_and_reranking": [
        "improved retrieval recall without measured evidence",
        "guaranteed all relevant evidence was found",
    ],
    "local_project_memory": [
        "reduced token usage by X% without evidence",
        "reduced processing cost without measured evidence",
    ],
    "deterministic_latex_validation": [
        "guaranteed LaTeX compilation never fails",
        "guaranteed correct PDF output",
    ],
    "template_pollution_blocking": [
        "eliminated all template pollution",
        "guaranteed no cross-project contamination",
    ],
    "output_quality_control": [
        "guaranteed factual correctness",
        "guaranteed correct output",
    ],
    "validation_and_repair": [
        "eliminated pipeline failures",
        "guaranteed recovery from all errors",
    ],
    "llm_reliability": [
        "solved hallucinations",
        "eliminated hallucinations",
        "reduced hallucinations by X%",
        "guaranteed factual correctness",
        "guaranteed reliable LLM output",
    ],
    "token_or_cost_reduction": [
        "reduced token usage by X% without evidence",
        "reduced cost by X% without evidence",
        "optimized token consumption",
    ],
}

CAPABILITY_GLOBAL_FORBIDDEN_CLAIMS = [
    "solved hallucinations",
    "eliminated hallucinations",
    "reduced hallucinations by X%",
    "guaranteed factual correctness",
    "guaranteed correct output",
    "improved ATS success",
    "improved accuracy by X%",
    "improved performance by X%",
    "improved recall without measured evidence",
    "reduced latency by X% without evidence",
    "reduced cost by X% without evidence",
    "reduced token usage by X% without evidence",
    "eliminated all failures",
]

CAPABILITY_ALLOWED_CLAIMS = {
    "retrieval_and_reranking": [
        "implemented project evidence retrieval and ranking logic",
        "added deterministic evidence selection rules",
    ],
    "local_project_memory": [
        "implemented structured local project-memory persistence",
        "enabled reuse of stored project-analysis state",
    ],
    "deterministic_latex_validation": [
        "implemented deterministic checks for LaTeX or PDF generation",
        "controlled LaTeX output through validation or merge logic",
    ],
    "template_pollution_blocking": [
        "blocked explicitly identified template contamination patterns",
        "added boundaries for unsupported template content",
    ],
    "output_quality_control": [
        "applied explicit checks before accepting generated output",
        "implemented deterministic controls for generated output",
    ],
    "validation_and_repair": [
        "implemented explicit validation, fallback, retry, or repair behavior",
        "added defined handling around unsupported claims or failed states",
    ],
    "evidence_grounded_generation": [
        "connected generated output to retrieved or stored project evidence",
        "applied explicit checks around evidence-backed generated claims",
    ],
    "llm_reliability": [
        "combined retrieval grounding with unsupported-claim validation",
        "added evidence-backed safeguards around generated output",
    ],
    "token_or_cost_reduction": [
        "implemented diff-based incremental analysis instead of repeated full-repository processing",
        "used cached or incremental analysis to avoid repeated full-repository processing",
    ],
}


@dataclass(frozen=True)
class RawDiffInput:
    project_id: str
    repo: str
    commit_sha: str | None
    file_path: str
    patch_text: str
    commit_message: str | None = None
    source_type: str = "commit_patch"


@dataclass(frozen=True)
class DiffUnit:
    unit_id: str
    project_id: str
    repo: str
    commit_sha: str | None
    file_path: str
    hunk_text: str
    added_lines: list[str]
    removed_lines: list[str]
    symbols_changed: list[str]
    change_hints: list[str]


@dataclass(frozen=True)
class RawChangeSummary:
    change_id: str
    project_id: str
    repo: str
    commit_sha: str | None
    file_path: str
    symbols_changed: list[str]
    raw_change_types: list[str]
    what_changed: str
    direct_code_evidence: list[str]
    uncertain_intent: list[str]
    confidence: str


@dataclass(frozen=True)
class EvidenceCard:
    evidence_id: str
    project_id: str
    source_change_ids: list[str]
    problem: str
    mechanism: str
    implementation_details: list[str]
    safe_impact: str
    resume_angle: str
    confidence: str
    metric_support: str
    allowed_claims: list[str]
    forbidden_claims: list[str]


@dataclass(frozen=True)
class CapabilityFact:
    capability_id: str
    project_id: str
    capability_type: str
    present: bool
    confidence: str
    mechanisms: list[str]
    source_evidence_ids: list[str]
    allowed_resume_claims: list[str]
    forbidden_claims: list[str]
    metric_support: str


def is_project_change_memory_enabled() -> bool:
    """Return true only when project change memory is explicitly enabled."""

    return os.getenv(PROJECT_CHANGE_MEMORY_ENV, "").strip() == "1"


def stable_hash_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def model_to_dict(obj: Any) -> dict[str, Any]:
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("model_to_dict expects a project change memory dataclass instance")
    payload = asdict(obj)
    if not isinstance(payload, dict):
        raise TypeError("model_to_dict expected dataclass serialization to produce a dict")
    return payload


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def create_empty_project_change_memory() -> dict[str, Any]:
    return {
        "schema_version": PROJECT_CHANGE_MEMORY_SCHEMA_VERSION,
        "updated_at": None,
        "projects": {},
    }


def load_project_change_memory(
    path: str | Path = DEFAULT_PROJECT_CHANGE_MEMORY_PATH,
) -> dict[str, Any]:
    memory_path = Path(path)
    if not memory_path.exists():
        return create_empty_project_change_memory()

    raw_text = memory_path.read_text(encoding="utf-8")
    if not raw_text.strip():
        return create_empty_project_change_memory()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid project change memory project memory JSON: {memory_path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Invalid project change memory project memory structure: {memory_path}")
    if data.get("schema_version") != PROJECT_CHANGE_MEMORY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported project change memory project memory schema: {data.get('schema_version')!r}"
        )
    return normalize_project_change_memory(data)


def normalize_project_change_memory(data: dict[str, Any]) -> dict[str, Any]:
    if not data:
        return create_empty_project_change_memory()
    if data.get("schema_version") != PROJECT_CHANGE_MEMORY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported project change memory project memory schema: {data.get('schema_version')!r}"
        )

    normalized = dict(data)
    normalized["schema_version"] = PROJECT_CHANGE_MEMORY_SCHEMA_VERSION
    normalized.setdefault("updated_at", None)
    projects = normalized.get("projects")
    if not isinstance(projects, dict):
        projects = {}

    normalized_projects: dict[str, dict[str, Any]] = {}
    for project_id in sorted(str(key) for key in projects):
        entry = projects.get(project_id)
        normalized_projects[project_id] = normalize_project_change_entry(
            project_id,
            entry if isinstance(entry, dict) else None,
        )
    normalized["projects"] = normalized_projects
    return normalized


def normalize_project_change_entry(
    project_id: str,
    entry: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = dict(entry or {})
    normalized["project_id"] = project_id
    normalized["raw_change_summaries"] = dedupe_raw_change_summaries(
        safe_dict_list(normalized.get("raw_change_summaries"))
    )
    normalized["evidence_cards"] = dedupe_evidence_cards(
        safe_dict_list(normalized.get("evidence_cards"))
    )
    normalized["capability_facts"] = dedupe_capability_facts(
        safe_dict_list(normalized.get("capability_facts"))
    )
    return normalized


def dedupe_raw_change_summaries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return dedupe_dicts_by_id(items, "change_id", sort_fields=("change_id",))


def dedupe_evidence_cards(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return dedupe_dicts_by_id(items, "evidence_id", sort_fields=("evidence_id",))


def dedupe_capability_facts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return dedupe_dicts_by_id(items, "capability_id", sort_fields=("capability_type", "capability_id"))


def write_project_change_memory(
    project_id: str,
    summaries: list[RawChangeSummary],
    cards: list[EvidenceCard],
    capabilities: list[CapabilityFact],
    path: str | Path = DEFAULT_PROJECT_CHANGE_MEMORY_PATH,
    min_evidence_score: int = 6,
) -> dict[str, Any]:
    normalized_project_id = str(project_id or "").strip()
    if not normalized_project_id:
        raise ValueError("project change memory project_id is required")

    validate_project_change_artifact_projects(normalized_project_id, summaries, cards, capabilities)
    memory = load_project_change_memory(path)
    projects = memory.setdefault("projects", {})
    existing_entry = projects.get(normalized_project_id)
    entry = normalize_project_change_entry(
        normalized_project_id,
        existing_entry if isinstance(existing_entry, dict) else None,
    )

    existing_summaries = safe_dict_list(entry.get("raw_change_summaries"))
    existing_cards = safe_dict_list(entry.get("evidence_cards"))
    existing_capabilities = safe_dict_list(entry.get("capability_facts"))

    new_summaries = [
        model_to_dict(summary)
        for summary in summaries
        if summary.change_id and summary.project_id == normalized_project_id
    ]
    new_cards = [
        model_to_dict(card)
        for card in cards
        if card.evidence_id
        and card.project_id == normalized_project_id
        and score_evidence_card(card) >= min_evidence_score
    ]

    merged_summaries = dedupe_raw_change_summaries([*existing_summaries, *new_summaries])
    merged_cards = dedupe_evidence_cards([*existing_cards, *new_cards])
    qualified_evidence_ids = {
        str(card.get("evidence_id") or "")
        for card in merged_cards
        if str(card.get("evidence_id") or "")
    }

    new_capabilities = [
        model_to_dict(capability)
        for capability in capabilities
        if capability_is_structurally_valid(capability)
        and capability.project_id == normalized_project_id
        and all(source_id in qualified_evidence_ids for source_id in capability.source_evidence_ids)
    ]
    merged_capabilities = [
        capability
        for capability in dedupe_capability_facts([*existing_capabilities, *new_capabilities])
        if capability_sources_exist(capability, qualified_evidence_ids)
    ]

    entry["raw_change_summaries"] = merged_summaries
    entry["evidence_cards"] = merged_cards
    entry["capability_facts"] = merged_capabilities
    projects[normalized_project_id] = entry
    memory["projects"] = {key: projects[key] for key in sorted(projects)}
    memory["updated_at"] = utc_now_iso()

    atomic_write_json(path, memory)
    return normalize_project_change_memory(memory)


def atomic_write_json(path: str | Path, data: dict[str, Any]) -> None:
    destination = Path(path)
    serialized = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def get_project_change_memory(
    project_id: str,
    path: str | Path = DEFAULT_PROJECT_CHANGE_MEMORY_PATH,
) -> dict[str, Any]:
    requested_project_id = str(project_id or "").strip()
    memory = load_project_change_memory(path)
    project_entry = memory.get("projects", {}).get(requested_project_id)
    return normalize_project_change_entry(
        requested_project_id,
        project_entry if isinstance(project_entry, dict) else None,
    )


def summarize_project_change_memory(
    project_id: str,
    path: str | Path = DEFAULT_PROJECT_CHANGE_MEMORY_PATH,
) -> dict[str, Any]:
    entry = get_project_change_memory(project_id, path)
    capability_types = sorted(
        {
            str(capability.get("capability_type") or "")
            for capability in entry["capability_facts"]
            if str(capability.get("capability_type") or "")
        }
    )
    return {
        "project_id": entry["project_id"],
        "raw_change_summary_count": len(entry["raw_change_summaries"]),
        "evidence_card_count": len(entry["evidence_cards"]),
        "capability_fact_count": len(entry["capability_facts"]),
        "capability_types": capability_types,
    }


def persist_project_change_artifacts(
    summaries: list[RawChangeSummary],
    cards: list[EvidenceCard],
    capabilities: list[CapabilityFact],
    path: str | Path = DEFAULT_PROJECT_CHANGE_MEMORY_PATH,
    min_evidence_score: int = 6,
) -> dict[str, Any]:
    project_ids = sorted(
        {
            artifact.project_id
            for artifact in [*summaries, *cards, *capabilities]
            if str(getattr(artifact, "project_id", "") or "").strip()
        }
    )
    if not project_ids:
        return load_project_change_memory(path)

    memory: dict[str, Any] = create_empty_project_change_memory()
    for project_id in project_ids:
        project_summaries = [summary for summary in summaries if summary.project_id == project_id]
        project_cards = [card for card in cards if card.project_id == project_id]
        project_capabilities = [
            capability for capability in capabilities if capability.project_id == project_id
        ]
        memory = write_project_change_memory(
            project_id,
            project_summaries,
            project_cards,
            project_capabilities,
            path=path,
            min_evidence_score=min_evidence_score,
        )
    return memory


def safe_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def dedupe_dicts_by_id(
    items: list[dict[str, Any]],
    id_field: str,
    *,
    sort_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = str(item.get(id_field) or "").strip()
        if not item_id:
            continue
        by_id[item_id] = dict(item)
    return sorted(by_id.values(), key=lambda item: tuple(str(item.get(field) or "") for field in sort_fields))


def validate_project_change_artifact_projects(
    project_id: str,
    summaries: list[RawChangeSummary],
    cards: list[EvidenceCard],
    capabilities: list[CapabilityFact],
) -> None:
    for artifact in [*summaries, *cards, *capabilities]:
        artifact_project_id = str(getattr(artifact, "project_id", "") or "").strip()
        if artifact_project_id and artifact_project_id != project_id:
            raise ValueError(
                "project change memory artifact project mismatch: "
                f"expected {project_id}, got {artifact_project_id}"
            )


def capability_is_structurally_valid(capability: CapabilityFact) -> bool:
    return bool(
        capability.capability_id
        and capability.project_id
        and capability.present is True
        and capability.source_evidence_ids
    )


def capability_sources_exist(
    capability: dict[str, Any],
    qualified_evidence_ids: set[str],
) -> bool:
    capability_id = str(capability.get("capability_id") or "").strip()
    if not capability_id:
        return False
    if capability.get("present") is not True:
        return False
    source_evidence_ids = capability.get("source_evidence_ids")
    if not isinstance(source_evidence_ids, list) or not source_evidence_ids:
        return False
    return all(str(source_id) in qualified_evidence_ids for source_id in source_evidence_ids)


def extract_diff_units(raw: RawDiffInput) -> list[DiffUnit]:
    hunks = split_patch_hunks(raw.patch_text)
    units: list[DiffUnit] = []
    for hunk_text in hunks:
        added_lines = extract_added_lines(hunk_text)
        removed_lines = extract_removed_lines(hunk_text)
        symbols_changed = extract_symbols_changed(hunk_text, added_lines, removed_lines)
        change_hints = extract_change_hints(
            hunk_text=hunk_text,
            added_lines=added_lines,
            removed_lines=removed_lines,
            file_path=raw.file_path,
            symbols_changed=symbols_changed,
        )
        units.append(
            DiffUnit(
                unit_id=make_diff_unit_id(raw, hunk_text),
                project_id=raw.project_id,
                repo=raw.repo,
                commit_sha=raw.commit_sha,
                file_path=raw.file_path,
                hunk_text=hunk_text,
                added_lines=added_lines,
                removed_lines=removed_lines,
                symbols_changed=symbols_changed,
                change_hints=change_hints,
            )
        )
    return units


def build_raw_change_summary(unit: DiffUnit) -> RawChangeSummary:
    raw_change_types = map_change_hints_to_raw_change_types(unit)
    direct_code_evidence = build_direct_code_evidence(unit)
    return RawChangeSummary(
        change_id=make_raw_change_id(unit),
        project_id=unit.project_id,
        repo=unit.repo,
        commit_sha=unit.commit_sha,
        file_path=unit.file_path,
        symbols_changed=list(unit.symbols_changed),
        raw_change_types=raw_change_types,
        what_changed=build_what_changed(unit, raw_change_types),
        direct_code_evidence=direct_code_evidence,
        uncertain_intent=build_uncertain_intent(unit, raw_change_types),
        confidence=assign_raw_change_confidence(unit, raw_change_types),
    )


def make_raw_change_id(unit: DiffUnit) -> str:
    return stable_hash_text(
        "|".join(
            [
                unit.project_id,
                unit.repo,
                unit.commit_sha or "",
                unit.file_path,
                unit.unit_id,
                unit.hunk_text,
            ]
        )
    )


def map_change_hints_to_raw_change_types(unit: DiffUnit) -> list[str]:
    raw_change_types: list[str] = []
    hints = unit.change_hints or ["unknown"]
    for hint in hints:
        raw_change_type = HINT_TO_RAW_CHANGE_TYPE.get(hint)
        if raw_change_type:
            add_unique(raw_change_types, raw_change_type)

    if "model_or_schema" in hints and schema_change_looks_memory_related(unit):
        add_unique(raw_change_types, "memory_storage_update")

    return raw_change_types or ["unknown"]


def schema_change_looks_memory_related(unit: DiffUnit) -> bool:
    haystack = normalize_hint_text(
        "\n".join(
            [
                unit.file_path,
                unit.hunk_text,
                "\n".join(unit.added_lines),
                "\n".join(unit.removed_lines),
                "\n".join(unit.symbols_changed),
            ]
        )
    )
    return any(keyword_matches(haystack, term) for term in MEMORY_SCHEMA_TERMS)


def build_what_changed(unit: DiffUnit, raw_change_types: list[str]) -> str:
    symbol = primary_symbol(unit)
    location = symbol or unit.file_path or "the source file"

    if "validation_logic_update" in raw_change_types:
        sentence = f"Added or modified validation logic in {location}."
    elif "merge_logic_update" in raw_change_types:
        sentence = f"Modified project bullet merge logic in {location}."
    elif "retrieval_logic_update" in raw_change_types:
        sentence = f"Updated evidence retrieval logic in {location}."
    elif "memory_storage_update" in raw_change_types:
        sentence = (
            f"Added or modified local project-memory handling in {location}."
            if symbol
            else "Added or modified local project-memory handling."
        )
    elif "chunking_update" in raw_change_types:
        sentence = f"Added or modified diff chunking logic in {location}."
    elif "fallback_update" in raw_change_types:
        sentence = f"Added or modified fallback or retry handling in {location}."
    elif "prompt_constraint_update" in raw_change_types:
        sentence = f"Modified prompt constraint logic in {location}."
    elif "quality_gate_update" in raw_change_types:
        sentence = f"Added or modified quality gate logic in {location}."
    elif "test_update" in raw_change_types:
        sentence = f"Added or modified tests covering {location}."
    elif "ui_debug_update" in raw_change_types:
        sentence = f"Modified frontend debugging or inspection logic in {location}."
    elif "api_route" in unit.change_hints and symbol:
        sentence = f"Modified API route handling in {symbol}."
    elif unit.file_path:
        sentence = f"Modified implementation logic in {unit.file_path}."
    else:
        sentence = "Modified implementation logic in the source file."

    return sanitize_generated_sentence(sentence)


def build_direct_code_evidence(unit: DiffUnit) -> list[str]:
    evidence: list[str] = []

    if unit.file_path:
        add_evidence(evidence, f"Changed file: {unit.file_path}.")

    for symbol in unit.symbols_changed[:3]:
        add_evidence(evidence, f"Changed symbol: {symbol}.")

    for line in unit.added_lines:
        add_line_evidence(evidence, line, "Added")
        if len(evidence) >= 8:
            return evidence[:8]

    for line in unit.removed_lines:
        add_line_evidence(evidence, line, "Removed")
        if len(evidence) >= 8:
            return evidence[:8]

    for hint in unit.change_hints:
        if hint != "unknown":
            add_evidence(evidence, f"Detected {hint} change context.")
        if len(evidence) >= 8:
            return evidence[:8]

    if not evidence and (unit.added_lines or unit.removed_lines or unit.hunk_text.strip()):
        add_evidence(evidence, "The diff contains source-code changes.")

    return evidence[:8] or ["The diff contains source-code changes."]


def add_line_evidence(evidence: list[str], line: str, action: str) -> None:
    clean = normalize_code_line_for_evidence(line)
    if not clean:
        return

    route_match = ROUTE_SYMBOL_RE.match(clean)
    if route_match:
        add_evidence(evidence, f"{action} route decorator: {route_match.group(2).upper()} {route_match.group(3)}.")
        return

    function_match = FUNCTION_SYMBOL_RE.match(clean)
    if function_match:
        add_evidence(evidence, f"{action} function definition: {function_match.group(1)}.")
        return

    class_match = CLASS_SYMBOL_RE.match(clean)
    if class_match:
        add_evidence(evidence, f"{action} class definition: {class_match.group(1)}.")
        return

    lowered = clean.lower()
    reference = evidence_reference_from_line(clean)
    if re.match(r"^(if|elif)\b", clean.strip()):
        add_evidence(
            evidence,
            f"{action} a conditional referencing {reference}."
            if reference
            else f"{action} a conditional branch.",
        )
    elif re.match(r"^else\s*:", clean.strip()):
        add_evidence(evidence, f"{action} an else branch.")
    elif re.match(r"^(except|try|finally)\b", clean.strip()):
        add_evidence(evidence, f"{action} exception-handling logic.")
    elif "return" in lowered and any(term in lowered for term in ["fail", "false", "error", "unsupported"]):
        add_evidence(evidence, f"{action} a failure return branch.")
    elif "return" in lowered and action == "Removed" and any(term in lowered for term in ["true", "success", "ok"]):
        add_evidence(evidence, "Removed a previous success return branch.")
    elif reference:
        add_evidence(evidence, f"{action} reference to {reference}.")


def normalize_code_line_for_evidence(line: str) -> str:
    clean = str(line or "").strip()
    if not clean or SECRET_LINE_RE.search(clean):
        return ""
    return clean


def evidence_reference_from_line(line: str) -> str:
    haystack = normalize_hint_text(line)
    for term in EVIDENCE_REFERENCE_TERMS:
        if keyword_matches(haystack, term):
            return term
    return ""


def build_uncertain_intent(unit: DiffUnit, raw_change_types: list[str]) -> list[str]:
    intents: list[str] = []
    if "validation_logic_update" in raw_change_types:
        add_unique(intents, "The validation change may be intended to prevent unsupported generated claims.")
    if "merge_logic_update" in raw_change_types:
        add_unique(intents, "The merge change may be intended to preserve stronger project evidence.")
    if "retrieval_logic_update" in raw_change_types:
        add_unique(intents, "The retrieval change may be intended to adjust evidence selection.")
    if "memory_storage_update" in raw_change_types:
        add_unique(intents, "The memory or cache change may be intended to avoid repeated project analysis.")
    if "chunking_update" in raw_change_types:
        add_unique(intents, "The chunking change may be intended to split diff context more consistently.")
    if "fallback_update" in raw_change_types:
        add_unique(intents, "The fallback or retry change may be intended to handle error paths more explicitly.")
    if "prompt_constraint_update" in raw_change_types:
        add_unique(intents, "The prompt constraint change may be intended to narrow model output requirements.")
    if "quality_gate_update" in raw_change_types:
        add_unique(intents, "The quality gate change may support stricter acceptance checks.")
    if "test_update" in raw_change_types:
        add_unique(intents, "The test change may support local verification of the modified behavior.")
    if "ui_debug_update" in raw_change_types:
        add_unique(intents, "The UI debug change could be related to inspecting intermediate state.")
    return [sanitize_generated_sentence(intent) for intent in intents[:3]]


def assign_raw_change_confidence(unit: DiffUnit, raw_change_types: list[str]) -> str:
    meaningful_changes = has_meaningful_code_changes(unit)
    has_symbol = bool(unit.symbols_changed)
    has_known_hint = any(hint != "unknown" for hint in unit.change_hints)
    if raw_change_types == ["unknown"]:
        return "low"
    if has_symbol and meaningful_changes and has_known_hint:
        return "high"
    if meaningful_changes and (has_symbol or has_known_hint):
        return "medium"
    return "low"


def has_meaningful_code_changes(unit: DiffUnit) -> bool:
    return any(line.strip() for line in [*unit.added_lines, *unit.removed_lines])


def primary_symbol(unit: DiffUnit) -> str:
    return unit.symbols_changed[0] if unit.symbols_changed else ""


def sanitize_generated_sentence(text: str) -> str:
    sanitized = str(text or "").strip()
    for pattern in FORBIDDEN_CONFIRMED_CLAIM_PATTERNS:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s+", " ", sanitized).strip(" .")
    return f"{sanitized}." if sanitized else "Modified implementation logic in the source file."


def add_evidence(evidence: list[str], item: str) -> None:
    clean = sanitize_generated_sentence(item)
    if clean and clean not in evidence:
        evidence.append(clean)


def build_evidence_card(summary: RawChangeSummary) -> EvidenceCard:
    problem = guard_allowed_evidence_text(
        infer_evidence_problem(summary),
        "The source implementation required a localized code change.",
    )
    mechanism = guard_allowed_evidence_text(
        infer_evidence_mechanism(summary),
        "Modified implementation logic in the affected source file.",
    )
    card = EvidenceCard(
        evidence_id=make_evidence_id(summary, problem, mechanism),
        project_id=summary.project_id,
        source_change_ids=[summary.change_id] if summary.change_id else [],
        problem=problem,
        mechanism=mechanism,
        implementation_details=build_implementation_details(summary),
        safe_impact=guard_allowed_evidence_text(
            infer_safe_impact(summary),
            "Recorded the implementation change for later project analysis.",
        ),
        resume_angle=infer_resume_angle(summary),
        confidence=infer_evidence_confidence(summary),
        metric_support=infer_metric_support(summary),
        allowed_claims=build_allowed_claims(summary),
        forbidden_claims=build_forbidden_claims(summary),
    )
    if unsafe_generated_card_language(card):
        return EvidenceCard(
            evidence_id=card.evidence_id,
            project_id=card.project_id,
            source_change_ids=card.source_change_ids,
            problem=guard_allowed_evidence_text(card.problem, "The source implementation required a localized code change."),
            mechanism=guard_allowed_evidence_text(
                card.mechanism,
                "Modified implementation logic in the affected source file.",
            ),
            implementation_details=card.implementation_details,
            safe_impact=guard_allowed_evidence_text(
                card.safe_impact,
                "Recorded the implementation change for later project analysis.",
            ),
            resume_angle=card.resume_angle,
            confidence="low" if card.confidence == "medium" else card.confidence,
            metric_support=card.metric_support,
            allowed_claims=[
                guard_allowed_evidence_text(claim, "modified implementation logic in the affected module")
                for claim in card.allowed_claims
            ],
            forbidden_claims=card.forbidden_claims,
        )
    return card


def make_evidence_id(
    summary: RawChangeSummary,
    problem: str | None = None,
    mechanism: str | None = None,
) -> str:
    resolved_problem = problem if problem is not None else infer_evidence_problem(summary)
    resolved_mechanism = mechanism if mechanism is not None else infer_evidence_mechanism(summary)
    return stable_hash_text(
        "|".join(
            [
                summary.project_id,
                summary.change_id,
                summary.file_path,
                resolved_problem,
                resolved_mechanism,
            ]
        )
    )


def infer_evidence_problem(summary: RawChangeSummary) -> str:
    primary_type = primary_raw_change_type(summary)
    if primary_type == "validation_logic_update":
        return "Generated output lacked a validation rule for unsupported claims or metrics."
    if primary_type == "merge_logic_update":
        return "Project bullet merge logic required explicit ordering or selection control."
    if primary_type == "retrieval_logic_update":
        return "Project evidence retrieval required updated query, ranking, or selection logic."
    if primary_type == "memory_storage_update":
        return "Project evidence or analysis state required structured local persistence or reuse."
    if primary_type == "chunking_update":
        return "Long source content required explicit segmentation before downstream processing."
    if primary_type == "fallback_update":
        return "The pipeline required handling for missing evidence, failed processing, or retryable states."
    if primary_type == "prompt_constraint_update":
        return "Generated output required additional explicit constraints or formatting rules."
    if primary_type == "quality_gate_update":
        return "Generated resume output required an additional post-generation quality check."
    if primary_type == "test_update":
        return "The changed behavior required direct regression or unit-test coverage."
    if primary_type == "ui_debug_update":
        return "Stored or processed project change memory state required inspectable debug visibility."
    return "The source implementation required a localized code change."


def infer_evidence_mechanism(summary: RawChangeSummary) -> str:
    primary_type = primary_raw_change_type(summary)
    if primary_type == "validation_logic_update":
        return "Added rule-based validation for unsupported claims or metrics."
    if primary_type == "merge_logic_update":
        return "Modified deterministic merge and bullet-selection logic."
    if primary_type == "retrieval_logic_update":
        return "Updated evidence query, retrieval, ranking, or filtering logic."
    if primary_type == "memory_storage_update":
        return "Added or modified structured local project-memory persistence."
    if primary_type == "chunking_update":
        return "Added or modified deterministic diff or text segmentation."
    if primary_type == "fallback_update":
        return "Added explicit fallback, retry, or repair handling."
    if primary_type == "prompt_constraint_update":
        return "Added explicit prompt constraints for generated output."
    if primary_type == "quality_gate_update":
        return "Added post-generation quality-gate checks."
    if primary_type == "test_update":
        return "Added deterministic tests for the changed behavior."
    if primary_type == "ui_debug_update":
        return "Added inspect or debug visibility for internal processing state."
    return "Modified implementation logic in the affected source file."


def build_implementation_details(summary: RawChangeSummary) -> list[str]:
    details: list[str] = []
    if summary.file_path:
        add_detail(details, f"Source file changed: {summary.file_path}.")
    for symbol in summary.symbols_changed[:3]:
        add_detail(details, f"Changed symbol: {symbol}.")
    for evidence in summary.direct_code_evidence:
        add_detail(details, evidence)
        if len(details) >= 8:
            return details[:8]
    if summary.what_changed:
        add_detail(details, summary.what_changed)
    for raw_change_type in summary.raw_change_types:
        if raw_change_type != "unknown":
            add_detail(details, f"Classified raw change type: {raw_change_type}.")
        if len(details) >= 8:
            return details[:8]
    return details[:8] or ["Source file changed: the affected source file."]


def infer_safe_impact(summary: RawChangeSummary) -> str:
    primary_type = primary_raw_change_type(summary)
    if primary_type == "validation_logic_update":
        return "Added an explicit safeguard against unsupported generated claims."
    if primary_type == "merge_logic_update":
        return "Made project bullet selection and merge behavior more explicit and deterministic."
    if primary_type == "retrieval_logic_update":
        return "Made project evidence selection logic more explicit and traceable."
    if primary_type == "memory_storage_update":
        return "Enabled structured reuse of project-analysis state."
    if primary_type == "chunking_update":
        return "Enabled deterministic processing of large or multi-hunk source changes."
    if primary_type == "fallback_update":
        return "Added defined handling for incomplete or failed processing paths."
    if primary_type == "prompt_constraint_update":
        return "Added explicit generation constraints for downstream output."
    if primary_type == "quality_gate_update":
        return "Added an additional validation stage before final output acceptance."
    if primary_type == "test_update":
        return "Added regression coverage for the changed behavior."
    if primary_type == "ui_debug_update":
        return "Made internal project change memory processing state inspectable during development."
    return "Recorded the implementation change for later project analysis."


def infer_resume_angle(summary: RawChangeSummary) -> str:
    return RESUME_ANGLE_BY_RAW_CHANGE_TYPE.get(primary_raw_change_type(summary), "implementation_change")


def build_allowed_claims(summary: RawChangeSummary) -> list[str]:
    primary_type = primary_raw_change_type(summary)
    claims: list[str] = []
    if primary_type == "validation_logic_update":
        claims.extend(
            [
                "added validation for unsupported generated claims",
                "introduced rule-based output validation",
                "added safeguards around unsupported metrics",
            ]
        )
    elif primary_type == "merge_logic_update":
        claims.extend(
            [
                "implemented deterministic project bullet merge logic",
                "made bullet-selection order explicit",
                "preserved structured merge behavior across project candidates",
            ]
        )
    elif primary_type == "retrieval_logic_update":
        claims.extend(
            [
                "implemented project evidence retrieval logic",
                "added query and ranking rules for project evidence",
                "made evidence selection traceable",
            ]
        )
    elif primary_type == "memory_storage_update":
        claims.extend(
            [
                "implemented structured local project-memory persistence",
                "enabled reuse of stored project-analysis state",
            ]
        )
    elif primary_type == "chunking_update":
        claims.extend(
            [
                "implemented deterministic diff-hunk segmentation",
                "added structured processing for multi-hunk patches",
            ]
        )
    elif primary_type == "fallback_update":
        claims.extend(
            [
                "implemented fallback or retry handling for incomplete processing",
                "added defined repair paths for failed pipeline stages",
            ]
        )
    elif primary_type == "prompt_constraint_update":
        claims.extend(
            [
                "added explicit generation constraints",
                "enforced structured output instructions",
            ]
        )
    elif primary_type == "quality_gate_update":
        claims.extend(
            [
                "implemented post-generation quality checks",
                "added a quality gate before final output acceptance",
            ]
        )
    elif primary_type == "test_update":
        claims.extend(
            [
                "added regression tests for project change memory change extraction",
                "added deterministic coverage for diff parsing behavior",
            ]
        )
    elif primary_type == "ui_debug_update":
        claims.extend(
            [
                "added inspectability for internal processing state",
                "added debug visibility for project change memory development",
            ]
        )
    else:
        claims.append("modified implementation logic in the affected module")

    safe_claims: list[str] = []
    for claim in claims:
        guarded = guard_allowed_evidence_text(claim, "modified implementation logic in the affected module")
        add_unique(safe_claims, guarded)
    return safe_claims[:5] or ["modified implementation logic in the affected module"]


def build_forbidden_claims(summary: RawChangeSummary) -> list[str]:
    forbidden_claims: list[str] = []
    for claim in GLOBAL_FORBIDDEN_CLAIMS:
        add_unique(forbidden_claims, claim)
    for raw_change_type in summary.raw_change_types or ["unknown"]:
        for claim in CATEGORY_FORBIDDEN_CLAIMS.get(raw_change_type, []):
            add_unique(forbidden_claims, claim)
    if not forbidden_claims:
        add_unique(forbidden_claims, "claimed measurable impact without supporting evidence")
    return forbidden_claims


def infer_metric_support(summary: RawChangeSummary) -> str:
    source_text = "\n".join([summary.what_changed, *summary.direct_code_evidence])
    lowered = source_text.lower()
    if any(re.search(pattern, lowered) for pattern in EXPLICIT_METRIC_PATTERNS):
        return "explicit"
    if any(re.search(pattern, lowered) for pattern in AMBIGUOUS_METRIC_PATTERNS):
        return "ambiguous"
    return "none"


def infer_evidence_confidence(summary: RawChangeSummary) -> str:
    primary_type = primary_raw_change_type(summary)
    details = build_implementation_details(summary)
    mechanism = infer_evidence_mechanism(summary)
    if (
        summary.confidence == "high"
        and primary_type != "unknown"
        and len(details) >= 2
        and mechanism_is_specific(mechanism)
    ):
        return "high"
    if summary.confidence in {"medium", "high"} and details and mechanism_is_supported(summary):
        return "medium"
    return "low"


def score_evidence_card(card: EvidenceCard) -> int:
    score = 0
    if card.problem and not is_generic_problem(card.problem):
        score += 2
    if card.mechanism and mechanism_is_specific(card.mechanism):
        score += 2
    if card.implementation_details:
        score += 2
    if card.safe_impact and not contains_unsupported_evidence_claim(card.safe_impact):
        score += 1
    if card.source_change_ids:
        score += 1
    if card.allowed_claims:
        score += 1
    if card.forbidden_claims:
        score += 1
    if card.resume_angle != "implementation_change":
        score += 1

    if not mechanism_is_specific(card.mechanism):
        score -= 3
    if not card.source_change_ids:
        score -= 3
    if not card.implementation_details:
        score -= 3
    if generated_card_fields_contain_unsafe_claim(card):
        score -= 5
    if generated_card_fields_contain_universal_guarantee(card):
        score -= 5

    return max(0, score)


def contains_unsupported_evidence_claim(text: str) -> bool:
    lowered = normalize_hint_text(text)
    return any(re.search(pattern, lowered) for pattern in UNSUPPORTED_EVIDENCE_CLAIM_PATTERNS)


def primary_raw_change_type(summary: RawChangeSummary) -> str:
    raw_change_types = summary.raw_change_types or ["unknown"]
    for raw_change_type in RESUME_ANGLE_PRIORITY:
        if raw_change_type in raw_change_types:
            return raw_change_type
    return raw_change_types[0] if raw_change_types else "unknown"


def guard_allowed_evidence_text(text: str, fallback: str) -> str:
    clean = sanitize_evidence_field(text)
    if contains_unsupported_evidence_claim(clean):
        return fallback
    return clean or fallback


def sanitize_evidence_field(text: str) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    return clean


def add_detail(details: list[str], item: str) -> None:
    clean = sanitize_evidence_field(item)
    if not clean or SECRET_LINE_RE.search(clean):
        return
    clean = truncate_text(clean, 180)
    if clean not in details:
        details.append(clean)


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def mechanism_is_specific(mechanism: str) -> bool:
    lowered = normalize_hint_text(mechanism)
    if not lowered:
        return False
    vague_terms = ["improved", "enhanced", "optimized", "powerful", "intelligent"]
    if any(term in lowered for term in vague_terms):
        return False
    return any(
        term in lowered
        for term in [
            "validation",
            "merge",
            "selection",
            "retrieval",
            "ranking",
            "project-memory",
            "persistence",
            "segmentation",
            "fallback",
            "retry",
            "repair",
            "prompt constraints",
            "quality-gate",
            "tests",
            "debug",
            "inspect",
        ]
    )


def mechanism_is_supported(summary: RawChangeSummary) -> bool:
    return bool(summary.direct_code_evidence or summary.symbols_changed or summary.file_path)


def is_generic_problem(problem: str) -> bool:
    return normalize_hint_text(problem) in {
        "the source implementation required a localized code change.",
        "source implementation required a localized code change.",
    }


def unsafe_generated_card_language(card: EvidenceCard) -> bool:
    return generated_card_fields_contain_unsafe_claim(card)


def generated_card_fields_contain_unsafe_claim(card: EvidenceCard) -> bool:
    generated_fields = [
        card.problem,
        card.mechanism,
        card.safe_impact,
        *card.allowed_claims,
    ]
    return any(contains_unsupported_evidence_claim(field) for field in generated_fields)


def generated_card_fields_contain_universal_guarantee(card: EvidenceCard) -> bool:
    generated_fields = [
        card.problem,
        card.mechanism,
        card.safe_impact,
        *card.allowed_claims,
    ]
    return any(UNIVERSAL_GUARANTEE_RE.search(str(field or "")) for field in generated_fields)


def extract_capability_facts(
    cards: list[EvidenceCard],
    min_evidence_score: int = 6,
) -> list[CapabilityFact]:
    qualified_cards = [
        card
        for card in cards
        if evidence_card_qualifies_for_capability(card, min_evidence_score)
    ]
    grouped_cards = group_cards_by_project(qualified_cards)
    facts: list[CapabilityFact] = []

    for project_id in sorted(grouped_cards):
        project_cards = deterministic_unique_cards(grouped_cards[project_id])
        for capability_type in CAPABILITY_TYPES:
            supporting_cards = select_capability_supporting_cards(capability_type, project_cards)
            supporting_cards = deterministic_unique_cards(supporting_cards)
            if not capability_requirements_met(capability_type, supporting_cards):
                continue

            source_evidence_ids = sorted({card.evidence_id for card in supporting_cards if card.evidence_id})
            mechanisms = build_capability_mechanisms(capability_type, supporting_cards)
            forbidden_claims = build_capability_forbidden_claims(capability_type, supporting_cards)
            metric_support = aggregate_capability_metric_support(capability_type, supporting_cards)
            allowed_claims = validate_capability_claim_boundaries(
                build_capability_allowed_claims(capability_type, supporting_cards),
                forbidden_claims,
                metric_support,
            )
            if not source_evidence_ids or not mechanisms or not allowed_claims:
                continue

            facts.append(
                CapabilityFact(
                    capability_id=make_capability_id(project_id, capability_type, source_evidence_ids),
                    project_id=project_id,
                    capability_type=capability_type,
                    present=True,
                    confidence=aggregate_capability_confidence(capability_type, supporting_cards),
                    mechanisms=mechanisms,
                    source_evidence_ids=source_evidence_ids,
                    allowed_resume_claims=allowed_claims,
                    forbidden_claims=forbidden_claims,
                    metric_support=metric_support,
                )
            )

    return facts


def make_capability_id(
    project_id: str,
    capability_type: str,
    source_evidence_ids: list[str],
) -> str:
    return stable_hash_text(
        "|".join(
            [
                project_id,
                capability_type,
                *sorted(set(source_evidence_ids)),
            ]
        )
    )


def group_cards_by_project(cards: list[EvidenceCard]) -> dict[str, list[EvidenceCard]]:
    grouped: dict[str, list[EvidenceCard]] = {}
    for card in cards:
        grouped.setdefault(card.project_id, []).append(card)
    return grouped


def detect_capability_types(card: EvidenceCard) -> list[str]:
    detected: list[str] = []
    if card_supports_retrieval(card):
        add_unique(detected, "retrieval_and_reranking")
    if card_supports_local_memory(card):
        add_unique(detected, "local_project_memory")
    if card_supports_latex_validation(card):
        add_unique(detected, "deterministic_latex_validation")
    if card_supports_template_pollution_blocking(card):
        add_unique(detected, "template_pollution_blocking")
    if card_supports_output_quality_control(card):
        add_unique(detected, "output_quality_control")
    if card_supports_validation_or_repair(card):
        add_unique(detected, "validation_and_repair")
    if card_supports_direct_evidence_grounded_generation(card):
        add_unique(detected, "evidence_grounded_generation")
    if card_supports_direct_llm_reliability(card):
        add_unique(detected, "llm_reliability")
    if card_supports_token_or_cost_reduction(card):
        add_unique(detected, "token_or_cost_reduction")
    return [capability_type for capability_type in CAPABILITY_TYPES if capability_type in detected]


def build_capability_mechanisms(
    capability_type: str,
    cards: list[EvidenceCard],
) -> list[str]:
    mechanisms: list[str] = []
    for card in deterministic_unique_cards(cards):
        add_capability_mechanism(mechanisms, card.mechanism)
        for detail in card.implementation_details[:3]:
            add_capability_mechanism(mechanisms, detail)
    if not mechanisms:
        for card in deterministic_unique_cards(cards):
            add_capability_mechanism(mechanisms, infer_capability_mechanism_fallback(capability_type, card))
    return deterministic_unique_text(mechanisms)[:8]


def build_capability_allowed_claims(
    capability_type: str,
    cards: list[EvidenceCard],
) -> list[str]:
    claims: list[str] = []
    for card in deterministic_unique_cards(cards):
        for claim in card.allowed_claims:
            add_unique(claims, sanitize_evidence_field(claim))
    for claim in CAPABILITY_ALLOWED_CLAIMS.get(capability_type, []):
        add_unique(claims, claim)
    return deterministic_unique_text(claims)[:8]


def build_capability_forbidden_claims(
    capability_type: str,
    cards: list[EvidenceCard],
) -> list[str]:
    forbidden_claims: list[str] = []
    for card in deterministic_unique_cards(cards):
        for claim in card.forbidden_claims:
            add_unique(forbidden_claims, sanitize_evidence_field(claim))
    for claim in CAPABILITY_GLOBAL_FORBIDDEN_CLAIMS:
        add_unique(forbidden_claims, claim)
    for claim in CAPABILITY_FORBIDDEN_CLAIMS.get(capability_type, []):
        add_unique(forbidden_claims, claim)
    return deterministic_unique_text(forbidden_claims)


def aggregate_capability_confidence(
    capability_type: str,
    cards: list[EvidenceCard],
) -> str:
    high_count = sum(1 for card in cards if card.confidence == "high")
    medium_count = sum(1 for card in cards if card.confidence == "medium")
    has_low = any(card.confidence == "low" for card in cards)
    if high_count >= 2 and not has_low:
        return "high"
    if high_count >= 1 and capability_has_direct_explicit_support(capability_type, cards) and not has_low:
        return "high"
    if high_count >= 1 or medium_count >= 2:
        return "medium"
    return "low"


def aggregate_capability_metric_support(
    capability_type: str,
    cards: list[EvidenceCard],
) -> str:
    relevant_cards = [card for card in cards if card_metric_is_relevant_to_capability(capability_type, card)]
    if any(card.metric_support == "explicit" for card in relevant_cards):
        return "explicit"
    if any(card.metric_support == "ambiguous" for card in relevant_cards):
        return "ambiguous"
    return "none"


def capability_requirements_met(
    capability_type: str,
    cards: list[EvidenceCard],
) -> bool:
    if capability_type not in CAPABILITY_TYPES or not cards:
        return False
    if capability_type == "evidence_grounded_generation":
        return evidence_grounded_generation_requirements_met(cards)
    if capability_type == "llm_reliability":
        return llm_reliability_requirements_met(cards)
    if capability_type == "token_or_cost_reduction":
        return any(card_supports_token_or_cost_reduction(card) for card in cards)
    if capability_type == "deterministic_latex_validation":
        return any(card_supports_latex_validation(card) for card in cards)
    if capability_type == "template_pollution_blocking":
        return any(card_supports_template_pollution_blocking(card) for card in cards)
    return any(capability_type in detect_capability_types(card) for card in cards)


def validate_capability_claim_boundaries(
    allowed_claims: list[str],
    forbidden_claims: list[str],
    metric_support: str,
) -> list[str]:
    safe_claims: list[str] = []
    normalized_forbidden = [normalize_hint_text(claim) for claim in forbidden_claims]
    for claim in allowed_claims:
        clean = sanitize_evidence_field(claim)
        if not clean:
            continue
        normalized_claim = normalize_hint_text(clean)
        if contains_unsupported_evidence_claim(clean):
            continue
        if UNIVERSAL_GUARANTEE_RE.search(clean):
            continue
        if metric_support != "explicit" and re.search(r"\b\d+(?:\.\d+)?%|\bby\s+\d+", normalized_claim):
            continue
        if any(
            normalized_claim == forbidden
            or (forbidden and forbidden in normalized_claim)
            for forbidden in normalized_forbidden
        ):
            continue
        add_unique(safe_claims, clean)
    return deterministic_unique_text(safe_claims)[:6]


def evidence_card_qualifies_for_capability(card: EvidenceCard, min_evidence_score: int) -> bool:
    if score_evidence_card(card) < min_evidence_score:
        return False
    if not card.source_change_ids or not card.evidence_id or not card.mechanism:
        return False
    if card.resume_angle == "implementation_change":
        return False
    if card.confidence == "low" and not concrete_implementation_details(card):
        return False
    return True


def select_capability_supporting_cards(
    capability_type: str,
    cards: list[EvidenceCard],
) -> list[EvidenceCard]:
    if capability_type == "evidence_grounded_generation":
        direct_cards = [card for card in cards if card_supports_direct_evidence_grounded_generation(card)]
        if direct_cards:
            return direct_cards
        evidence_cards = [card for card in cards if card_supports_generation_evidence_source(card)]
        guard_cards = [card for card in cards if card_supports_generation_guard(card)]
        if evidence_cards and guard_cards and cards_have_distinct_angles([*evidence_cards, *guard_cards]):
            return [*evidence_cards, *guard_cards]
        return []
    if capability_type == "llm_reliability":
        direct_cards = [card for card in cards if card_supports_direct_llm_reliability(card)]
        if direct_cards:
            return direct_cards
        retrieval_cards = [card for card in cards if card_supports_retrieval_grounding(card)]
        validation_cards = [card for card in cards if card_supports_unsupported_claim_validation_or_quality_gate(card)]
        fallback_cards = [card for card in cards if card_supports_insufficient_evidence_fallback(card)]
        if retrieval_cards and validation_cards:
            return [*retrieval_cards, *validation_cards]
        if retrieval_cards and fallback_cards:
            return [*retrieval_cards, *fallback_cards]
        return []
    if capability_type == "token_or_cost_reduction":
        return [card for card in cards if card_supports_token_or_cost_reduction(card)]
    return [card for card in cards if capability_type in detect_capability_types(card)]


def card_supports_retrieval(card: EvidenceCard) -> bool:
    if card.resume_angle == "evidence_retrieval":
        return True
    return card_has_any_term(
        card,
        [
            "retrieve",
            "retrieval",
            "query",
            "search",
            "ranking",
            "rerank",
            "filtering",
            "evidence selection",
        ],
    )


def card_supports_local_memory(card: EvidenceCard) -> bool:
    if card.resume_angle == "project_memory":
        return True
    return card_has_any_term(
        card,
        [
            "project_memory",
            "local project memory",
            "structured persistence",
            "sqlite",
            "cache",
            "stored project analysis",
            "project-analysis state",
            "reuse of project-analysis state",
        ],
    )


def card_supports_latex_validation(card: EvidenceCard) -> bool:
    has_latex = card_has_any_term(
        card,
        [
            "latex",
            ".tex",
            "pdf compilation",
            "compile validation",
            "latex structure",
            "resume rendering",
        ],
    )
    has_control = card_has_any_term(
        card,
        [
            "validation",
            "quality gate",
            "merge control",
            "repair",
            "blocking invalid output",
            "compile validation",
        ],
    )
    return has_latex and has_control


def card_supports_template_pollution_blocking(card: EvidenceCard) -> bool:
    return card_has_any_term(
        card,
        [
            "template pollution",
            "placeholder contamination",
            "standalone template technology",
            "pollution blocker",
            "forbidden template terms",
            "cross-project technology contamination",
            "unsupported template content",
        ],
    )


def card_supports_output_quality_control(card: EvidenceCard) -> bool:
    if card.resume_angle == "output_quality_control":
        return True
    if card.resume_angle in {"validation_and_safety", "deterministic_merge"}:
        return card_has_any_term(
            card,
            [
                "quality gate",
                "post-generation validation",
                "final output acceptance",
                "generated output checks",
                "deterministic selection",
                "merge validation",
                "unsupported generated claims",
                "validation",
            ],
        )
    return card_has_any_term(
        card,
        [
            "quality gate",
            "post-generation validation",
            "final output acceptance",
            "generated output checks",
        ],
    )


def card_supports_validation_or_repair(card: EvidenceCard) -> bool:
    if card.resume_angle in {"validation_and_safety", "fallback_and_repair"}:
        return True
    return card_has_any_term(
        card,
        [
            "validator",
            "validation",
            "repair path",
            "retry path",
            "unsupported claim detection",
            "failed-state handling",
            "recovery handling",
            "fallback",
        ],
    )


def card_supports_direct_evidence_grounded_generation(card: EvidenceCard) -> bool:
    return card.confidence == "high" and card_has_any_term(
        card,
        [
            "ground generated output in retrieved evidence",
            "evidence-backed generation",
            "verified project evidence used for generation",
            "generated claims checked against stored evidence",
        ],
    )


def card_supports_direct_llm_reliability(card: EvidenceCard) -> bool:
    return card.confidence == "high" and card_has_any_term(
        card,
        [
            "citation-backed output",
            "evidence-backed generation",
            "factuality validation",
            "unsupported response blocking",
            "unsupported-claim validation",
            "low-confidence fallback",
            "answer verification",
        ],
    )


def card_supports_token_or_cost_reduction(card: EvidenceCard) -> bool:
    return card_has_any_term(
        card,
        [
            "reduced token input",
            "reduced api token usage",
            "reduced processing cost",
            "avoided full repository rescans",
            "diff-based incremental analysis replacing full scans",
            "diff-based incremental analysis replaced repeated full-repository processing",
            "cached analysis explicitly used to avoid repeated model processing",
            "measured cost reduction",
            "measured token reduction",
        ],
    )


def evidence_grounded_generation_requirements_met(cards: list[EvidenceCard]) -> bool:
    if any(card_supports_direct_evidence_grounded_generation(card) for card in cards):
        return True
    evidence_cards = [card for card in cards if card_supports_generation_evidence_source(card)]
    guard_cards = [card for card in cards if card_supports_generation_guard(card)]
    return bool(evidence_cards and guard_cards and cards_have_distinct_angles([*evidence_cards, *guard_cards]))


def llm_reliability_requirements_met(cards: list[EvidenceCard]) -> bool:
    if any(card_supports_direct_llm_reliability(card) for card in cards):
        return True
    retrieval_cards = [card for card in cards if card_supports_retrieval_grounding(card)]
    validation_cards = [card for card in cards if card_supports_unsupported_claim_validation_or_quality_gate(card)]
    fallback_cards = [card for card in cards if card_supports_insufficient_evidence_fallback(card)]
    return bool(retrieval_cards and (validation_cards or fallback_cards) and len({card.evidence_id for card in cards}) >= 2)


def card_supports_generation_evidence_source(card: EvidenceCard) -> bool:
    return card_supports_retrieval(card) or card_supports_local_memory(card)


def card_supports_generation_guard(card: EvidenceCard) -> bool:
    return card.resume_angle in {
        "generation_constraints",
        "validation_and_safety",
        "output_quality_control",
    } or card_supports_output_quality_control(card)


def card_supports_retrieval_grounding(card: EvidenceCard) -> bool:
    return card_supports_retrieval(card) and card_has_any_term(
        card,
        [
            "evidence",
            "retrieved evidence",
            "evidence retrieval",
            "evidence selection",
            "query",
            "rerank",
            "ranking",
        ],
    )


def card_supports_unsupported_claim_validation_or_quality_gate(card: EvidenceCard) -> bool:
    return card_has_any_term(
        card,
        [
            "unsupported claim",
            "unsupported generated claims",
            "unsupported-claim validation",
            "quality gate",
            "post-generation validation",
            "factuality validation",
        ],
    )


def card_supports_insufficient_evidence_fallback(card: EvidenceCard) -> bool:
    return card.resume_angle == "fallback_and_repair" and card_has_any_term(
        card,
        [
            "insufficient evidence",
            "low-evidence",
            "low confidence",
            "fallback",
            "repair",
            "failed-state",
        ],
    )


def cards_have_distinct_angles(cards: list[EvidenceCard]) -> bool:
    return len({card.resume_angle for card in cards}) >= 2


def capability_has_direct_explicit_support(
    capability_type: str,
    cards: list[EvidenceCard],
) -> bool:
    if capability_type == "evidence_grounded_generation":
        return any(card_supports_direct_evidence_grounded_generation(card) for card in cards)
    if capability_type == "llm_reliability":
        return any(card_supports_direct_llm_reliability(card) for card in cards)
    if capability_type == "token_or_cost_reduction":
        return any(card_supports_token_or_cost_reduction(card) and card.confidence == "high" for card in cards)
    if capability_type in STRICT_CAPABILITY_TYPES:
        return len(cards) == 1 and cards[0].confidence == "high"
    return len(cards) == 1 and cards[0].confidence == "high"


def card_metric_is_relevant_to_capability(capability_type: str, card: EvidenceCard) -> bool:
    if card.metric_support == "none":
        return False
    if capability_type == "token_or_cost_reduction":
        return card_supports_token_or_cost_reduction(card)
    return capability_type in detect_capability_types(card) or capability_type in {
        "evidence_grounded_generation",
        "llm_reliability",
    }


def concrete_implementation_details(card: EvidenceCard) -> bool:
    return any(detail.strip() for detail in card.implementation_details)


def deterministic_unique_cards(cards: list[EvidenceCard]) -> list[EvidenceCard]:
    unique_cards: list[EvidenceCard] = []
    seen_ids: set[str] = set()
    for card in sorted(cards, key=evidence_card_sort_key):
        card_id = card.evidence_id or stable_hash_text(str(model_to_dict(card)))
        if card_id in seen_ids:
            continue
        seen_ids.add(card_id)
        unique_cards.append(card)
    return unique_cards


def evidence_card_sort_key(card: EvidenceCard) -> tuple[str, str, str, str]:
    return (
        card.project_id,
        card.evidence_id,
        card.resume_angle,
        stable_hash_text(str(model_to_dict(card))),
    )


def card_combined_text(card: EvidenceCard) -> str:
    return "\n".join(
        [
            card.resume_angle,
            card.problem,
            card.mechanism,
            "\n".join(card.implementation_details),
            card.safe_impact,
            "\n".join(card.allowed_claims),
            "\n".join(card.forbidden_claims),
        ]
    )


def card_has_any_term(card: EvidenceCard, terms: list[str]) -> bool:
    haystack = normalize_hint_text(card_combined_text(card))
    return any(keyword_matches(haystack, term) for term in terms)


def add_capability_mechanism(mechanisms: list[str], item: str) -> None:
    clean = sanitize_evidence_field(item)
    if not clean or SECRET_LINE_RE.search(clean):
        return
    if contains_unsupported_evidence_claim(clean) or UNIVERSAL_GUARANTEE_RE.search(clean):
        return
    if mechanism_contains_result_claim(clean):
        return
    add_unique(mechanisms, truncate_text(clean, 180))


def mechanism_contains_result_claim(text: str) -> bool:
    lowered = normalize_hint_text(text)
    return any(
        keyword_matches(lowered, term)
        for term in [
            "improved",
            "enhanced",
            "optimized",
            "guaranteed",
            "reduced latency",
            "reduced cost",
            "reduced token usage",
            "eliminated",
        ]
    )


def infer_capability_mechanism_fallback(capability_type: str, card: EvidenceCard) -> str:
    return card.mechanism or CAPABILITY_ALLOWED_CLAIMS.get(capability_type, ["implemented source-backed behavior"])[0]


def deterministic_unique_text(values: list[str]) -> list[str]:
    unique_by_key: dict[str, str] = {}
    for value in values:
        clean = sanitize_evidence_field(value)
        if not clean:
            continue
        unique_by_key.setdefault(normalize_hint_text(clean), clean)
    return [unique_by_key[key] for key in sorted(unique_by_key)]


def split_patch_hunks(patch_text: str) -> list[str]:
    normalized_patch = normalize_patch_text(patch_text)
    lines = normalized_patch.splitlines()
    hunks: list[str] = []
    current_hunk: list[str] = []
    in_hunk = False

    for line in lines:
        if line.startswith("@@"):
            if current_hunk:
                hunks.append("\n".join(current_hunk).strip("\n"))
            current_hunk = [line]
            in_hunk = True
        elif in_hunk:
            current_hunk.append(line)

    if current_hunk:
        hunks.append("\n".join(current_hunk).strip("\n"))

    return hunks or [normalized_patch.strip("\n")]


def extract_added_lines(hunk_text: str) -> list[str]:
    added_lines: list[str] = []
    for line in normalize_patch_text(hunk_text).splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        added_line = line[1:]
        if added_line.strip():
            added_lines.append(added_line)
    return added_lines


def extract_removed_lines(hunk_text: str) -> list[str]:
    removed_lines: list[str] = []
    for line in normalize_patch_text(hunk_text).splitlines():
        if not line.startswith("-") or line.startswith("---"):
            continue
        removed_line = line[1:]
        if removed_line.strip():
            removed_lines.append(removed_line)
    return removed_lines


def extract_symbols_changed(
    hunk_text: str,
    added_lines: list[str] | None = None,
    removed_lines: list[str] | None = None,
) -> list[str]:
    scan_lines = symbol_scan_lines(hunk_text)
    scan_lines.extend(added_lines or [])
    scan_lines.extend(removed_lines or [])

    symbols: list[str] = []
    for line in scan_lines:
        route_match = ROUTE_SYMBOL_RE.match(line)
        if route_match:
            add_unique(symbols, f"{route_match.group(2).upper()} {route_match.group(3)}")
            continue

        function_match = FUNCTION_SYMBOL_RE.match(line)
        if function_match:
            add_unique(symbols, function_match.group(1))
            continue

        class_match = CLASS_SYMBOL_RE.match(line)
        if class_match:
            add_unique(symbols, class_match.group(1))

    return symbols


def extract_change_hints(
    *,
    hunk_text: str,
    added_lines: list[str],
    removed_lines: list[str],
    file_path: str,
    symbols_changed: list[str],
) -> list[str]:
    haystack = normalize_hint_text(
        "\n".join(
            [
                file_path,
                file_path.replace("\\", "/"),
                hunk_text,
                "\n".join(added_lines),
                "\n".join(removed_lines),
                "\n".join(symbols_changed),
            ]
        )
    )

    hints: list[str] = []
    for hint, keywords in CHANGE_HINT_RULES:
        if any(keyword_matches(haystack, keyword) for keyword in keywords):
            add_unique(hints, hint)

    return hints or ["unknown"]


def make_diff_unit_id(raw: RawDiffInput, hunk_text: str) -> str:
    identity_text = "\n".join(
        [
            raw.project_id,
            raw.repo,
            raw.commit_sha or "",
            raw.file_path,
            hunk_text,
        ]
    )
    return f"diff_unit_{stable_hash_text(identity_text)[:32]}"


def symbol_scan_lines(hunk_text: str) -> list[str]:
    scan_lines: list[str] = []
    for line in normalize_patch_text(hunk_text).splitlines():
        if line.startswith("@@"):
            header_context = extract_hunk_header_context(line)
            if header_context:
                scan_lines.append(header_context)
            continue
        if line.startswith(("+++", "---", "diff --git", "index ")):
            continue
        if line[:1] in {"+", "-", " "}:
            scan_lines.append(line[1:])
        else:
            scan_lines.append(line)
    return scan_lines


def extract_hunk_header_context(line: str) -> str:
    match = HUNK_HEADER_RE.match(line)
    return match.group(1).strip() if match else ""


def normalize_patch_text(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def normalize_hint_text(text: str) -> str:
    return normalize_patch_text(text).replace("\\", "/").lower()


def keyword_matches(haystack: str, keyword: str) -> bool:
    normalized_keyword = keyword.lower()
    if not normalized_keyword:
        return False
    if normalized_keyword.endswith("_"):
        return normalized_keyword in haystack
    if re.search(r"^[a-z0-9_ ]+$", normalized_keyword):
        pattern = r"(?<![a-z0-9])" + re.escape(normalized_keyword) + r"(?![a-z0-9])"
        return re.search(pattern, haystack) is not None
    return normalized_keyword in haystack


def add_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)
