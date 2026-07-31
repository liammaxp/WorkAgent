"""Safe command-line adapter for Project Capability Memory operations.

The CLI delegates all lifecycle work to the authoritative pipeline and memory
loader.  Importing this module performs no parsing, file access, or execution.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from backend import project_capability_backfill as backfill_module
from backend import project_capability_memory as memory_module
from backend import project_capability_pipeline as pipeline_module
from backend import project_evidence_memory as evidence_memory_module


EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_ARTIFACT_MISSING = 3
EXIT_ARTIFACT_INVALID = 4
EXIT_PIPELINE_FAILED = 5
EXIT_PERSISTENCE_FAILED = 6
EXIT_SAFETY_VIOLATION = 7

SUCCESS_PIPELINE_STATUSES = frozenset({"ready", "empty"})
SUCCESS_MEMORY_STATUSES = frozenset({"ready", "empty"})


class _Parser(argparse.ArgumentParser):
    """Keep argparse conventions while allowing main(argv) to return a code."""

    def error(self, _message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, "error:invalid_arguments\n")

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if message:
            self._print_message(message, sys.stderr)
        raise SystemExit(status)


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="python -m backend.project_capability_cli",
        description="Build, inspect, validate, or backfill Project Capability Memory safely.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build",
        help="Run the authoritative pipeline in memory; writing is disabled by default.",
        description=(
            "Run the authoritative Project Capability pipeline. The command is read-only "
            "unless both --persist and an explicit safe --output are supplied."
        ),
    )
    build.add_argument(
        "--source",
        type=Path,
        default=evidence_memory_module.DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH,
        help="Project Evidence Memory source path.",
    )
    build.add_argument(
        "--persist",
        action="store_true",
        help="Persist only to the explicit non-production path supplied by --output.",
    )
    build.add_argument("--output", type=Path, help="Explicit custom JSON output path.")
    build.add_argument("--json", action="store_true", help="Print canonical JSON output.")

    inspect = commands.add_parser(
        "inspect",
        help="Run a bounded read-only pipeline inspection.",
        description="Inspect aggregate Project Capability pipeline counts without writing artifacts.",
    )
    inspect.add_argument(
        "--source",
        type=Path,
        default=evidence_memory_module.DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH,
        help="Project Evidence Memory source path.",
    )
    inspect.add_argument("--json", action="store_true", help="Print canonical JSON output.")

    validate = commands.add_parser(
        "validate",
        help="Validate an existing Project Capability Memory artifact without changing it.",
        description="Load and validate one Project Capability Memory artifact without repair or migration.",
    )
    validate.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Project Capability Memory artifact to validate.",
    )
    validate.add_argument("--json", action="store_true", help="Print canonical JSON output.")

    backfill = commands.add_parser(
        "backfill",
        help="Create or confirm the exact authoritative local capability-memory artifact.",
        description=(
            "Run the controlled authoritative backfill with fixed source and target paths. "
            "Arbitrary paths and unsafe overrides are not supported."
        ),
    )
    backfill.add_argument("--json", action="store_true", help="Print canonical JSON output.")
    return parser


def _same_path(left: Path, right: Path) -> bool:
    try:
        return os.path.normcase(str(left.resolve(strict=False))) == os.path.normcase(
            str(right.resolve(strict=False))
        )
    except OSError:
        return os.path.normcase(str(left.absolute())) == os.path.normcase(str(right.absolute()))


def _output_path_error(source: Path, output: Path | None) -> str | None:
    if output is None or not str(output).strip():
        return "output_path_required"
    if _same_path(output, memory_module.PROJECT_CAPABILITY_MEMORY_PATH):
        return "real_backfill_path_prohibited"
    if _same_path(output, evidence_memory_module.DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH):
        return "evidence_memory_overwrite_prohibited"
    if _same_path(output, source):
        return "source_output_collision"
    if output.exists() and output.is_dir():
        return "output_path_is_directory"
    return None


def _pipeline_summary(command: str, result: Any) -> dict[str, Any]:
    memory = result.memory
    diagnostics = result.diagnostics
    return {
        "command": command,
        "status": result.status,
        "source": {
            "schema_version": result.source_schema_version,
            "content_hash": result.source_content_hash,
            "file_sha256": result.source_file_sha256,
            "project_count": result.source_project_count,
            "evidence_fact_count": result.source_evidence_fact_count,
            "claim_boundary_count": result.source_claim_boundary_count,
            "capability_fact_count": result.source_capability_fact_count,
        },
        "pipeline": {
            "candidate_count": result.candidate_count,
            "matched_evidence_count": result.matched_evidence_count,
            "unmatched_evidence_count": result.unmatched_evidence_count,
            "ambiguous_evidence_count": result.ambiguous_evidence_count,
            "skipped_evidence_count": result.skipped_evidence_count,
            "assessment_count": result.assessment_count,
            "eligible_assessment_count": result.eligible_assessment_count,
            "assessment_status_counts": dict(result.assessment_status_counts),
            "policy_count": result.policy_count,
            "eligible_policy_count": result.eligible_policy_count,
            "policy_status_counts": dict(result.policy_status_counts),
            "build_result_count": result.build_result_count,
            "build_status_counts": dict(result.build_status_counts),
            "capability_fact_count": result.capability_fact_count,
            "projects_with_capabilities": int(diagnostics.get("projects_with_capabilities", 0)),
            "projects_without_capabilities": int(diagnostics.get("projects_without_capabilities", 0)),
        },
        "memory": {
            "schema_version": memory.schema_version if memory is not None else None,
            "content_hash": memory.content_hash if memory is not None else None,
            "persisted": result.persisted_path is not None,
            "persisted_path": result.persisted_path,
        },
        "warnings": list(result.warnings),
        "errors": list(result.errors),
    }


def _validation_summary(loaded: Any) -> dict[str, Any]:
    memory = loaded.memory
    error = {
        "missing": "capability_memory_missing",
        "invalid": "capability_memory_invalid",
        "unsupported_version": "capability_memory_schema_unsupported",
        "hash_mismatch": "capability_memory_hash_mismatch",
    }.get(loaded.status)
    return {
        "command": "validate",
        "status": loaded.status,
        "artifact": {
            "schema_version": memory.schema_version if memory is not None else None,
            "content_hash": memory.content_hash if memory is not None else None,
            "project_count": len(memory.projects) if memory is not None else 0,
            "capability_fact_count": len(memory.capability_facts) if memory is not None else 0,
        },
        "validation": {
            "valid": bool(loaded.validation.valid),
        },
        "warnings": [],
        "errors": [error] if error else [],
    }


def _backfill_summary(result: Any) -> dict[str, Any]:
    return {
        "command": "backfill",
        "status": result.status,
        "source": {
            "schema_version": result.source_schema_version,
            "content_hash": result.source_content_hash,
            "file_sha256": (
                result.source_file_sha256_after or result.source_file_sha256_before
            ),
            "file_sha256_before": result.source_file_sha256_before,
            "file_sha256_after": result.source_file_sha256_after,
            "project_count": result.source_project_count,
            "evidence_fact_count": result.source_evidence_fact_count,
            "claim_boundary_count": result.source_claim_boundary_count,
            "capability_fact_count": result.source_capability_fact_count,
        },
        "pipeline": {
            "status": result.pipeline_status,
            "candidate_count": result.candidate_count,
            "assessment_count": result.assessment_count,
            "eligible_assessment_count": result.eligible_assessment_count,
            "policy_count": result.policy_count,
            "build_result_count": result.build_result_count,
            "capability_fact_count": result.capability_fact_count,
            "matched_evidence_count": int(result.diagnostics.get("matched_evidence_count", 0)),
            "unmatched_evidence_count": int(result.diagnostics.get("unmatched_evidence_count", 0)),
            "ambiguous_evidence_count": int(result.diagnostics.get("ambiguous_evidence_count", 0)),
            "skipped_evidence_count": int(result.diagnostics.get("skipped_evidence_count", 0)),
            "projects_with_capabilities": int(result.diagnostics.get("projects_with_capabilities", 0)),
            "projects_without_capabilities": int(result.diagnostics.get("projects_without_capabilities", 0)),
        },
        "target": {
            "schema_version": result.target_schema_version,
            "content_hash": result.target_content_hash,
            "file_sha256": result.target_file_sha256,
            "existed_before": result.target_existed_before,
            "written": result.target_written,
            "unchanged": result.target_unchanged,
            "staging_artifact_count": int(result.diagnostics.get("staging_artifact_count", 0)),
        },
        "warnings": list(result.warnings),
        "errors": list(result.errors),
    }


def _error_summary(command: str, code: str) -> dict[str, Any]:
    return {
        "command": command,
        "status": "failed",
        "source": None,
        "pipeline": None,
        "memory": None,
        "warnings": [],
        "errors": [code],
    }


def _print_json(summary: Mapping[str, Any]) -> None:
    print(json.dumps(
        summary,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ))


def _print_human(summary: Mapping[str, Any]) -> None:
    print(f"command: {summary['command']}")
    print(f"status: {summary['status']}")
    source = summary.get("source")
    if isinstance(source, Mapping):
        for key in (
            "schema_version", "content_hash", "file_sha256", "project_count",
            "evidence_fact_count", "claim_boundary_count", "capability_fact_count",
        ):
            if key in source:
                print(f"source_{key}: {source[key]}")
    pipeline = summary.get("pipeline")
    if isinstance(pipeline, Mapping):
        for key in (
            "candidate_count", "matched_evidence_count", "unmatched_evidence_count",
            "ambiguous_evidence_count", "skipped_evidence_count", "assessment_count",
            "eligible_assessment_count", "policy_count", "eligible_policy_count",
            "build_result_count", "capability_fact_count", "projects_with_capabilities",
            "projects_without_capabilities",
        ):
            if key in pipeline:
                print(f"pipeline_{key}: {pipeline[key]}")
        for key in ("assessment_status_counts", "policy_status_counts", "build_status_counts"):
            if key in pipeline:
                print(f"pipeline_{key}: {json.dumps(pipeline[key], sort_keys=True, separators=(',', ':'))}")
    memory = summary.get("memory")
    if isinstance(memory, Mapping):
        for key in ("schema_version", "content_hash", "persisted", "persisted_path"):
            print(f"memory_{key}: {memory[key]}")
    artifact = summary.get("artifact")
    if isinstance(artifact, Mapping):
        for key in ("schema_version", "content_hash", "project_count", "capability_fact_count"):
            print(f"artifact_{key}: {artifact[key]}")
    validation = summary.get("validation")
    if isinstance(validation, Mapping):
        print(f"validation_valid: {validation['valid']}")
    target = summary.get("target")
    if isinstance(target, Mapping):
        for key in (
            "schema_version", "content_hash", "file_sha256", "existed_before",
            "written", "unchanged", "staging_artifact_count",
        ):
            print(f"target_{key}: {target[key]}")
    print(f"warnings: {','.join(summary.get('warnings', [])) or 'none'}")
    print(f"errors: {','.join(summary.get('errors', [])) or 'none'}")


def _emit(summary: Mapping[str, Any], *, json_output: bool) -> None:
    if json_output:
        _print_json(summary)
    else:
        _print_human(summary)


def _emit_error(command: str, code: str, *, json_output: bool, exit_code: int) -> int:
    summary = _error_summary(command, code)
    if json_output:
        _print_json(summary)
    print(f"error:{code}", file=sys.stderr)
    return exit_code


def _pipeline_exit_code(result: Any) -> int:
    if result.status in SUCCESS_PIPELINE_STATUSES:
        return EXIT_SUCCESS
    if result.status == "source_missing":
        return EXIT_ARTIFACT_MISSING
    if result.status == "source_invalid":
        return EXIT_ARTIFACT_INVALID
    if "persistence_validation_failed" in result.errors:
        return EXIT_PERSISTENCE_FAILED
    return EXIT_PIPELINE_FAILED


def _backfill_exit_code(result: Any) -> int:
    if result.status in {"created", "unchanged"}:
        return EXIT_SUCCESS
    if result.status == "source_missing":
        return EXIT_ARTIFACT_MISSING
    if result.status in {"source_invalid", "target_invalid"}:
        return EXIT_ARTIFACT_INVALID
    if result.status == "pipeline_failed":
        return EXIT_PIPELINE_FAILED
    if result.status == "persistence_failed":
        return EXIT_PERSISTENCE_FAILED
    return EXIT_SAFETY_VIOLATION


def _run_pipeline_command(args: argparse.Namespace) -> int:
    command = args.command
    persist = bool(getattr(args, "persist", False))
    output = getattr(args, "output", None)
    json_output = bool(args.json)
    if command == "build":
        if output is not None and not persist:
            return _emit_error(
                command,
                "output_path_without_persist",
                json_output=json_output,
                exit_code=EXIT_USAGE,
            )
        if persist:
            path_error = _output_path_error(args.source, output)
            if path_error is not None:
                exit_code = EXIT_USAGE if path_error == "output_path_required" else EXIT_SAFETY_VIOLATION
                return _emit_error(
                    command, path_error, json_output=json_output, exit_code=exit_code
                )
    try:
        result = pipeline_module.run_project_capability_pipeline(
            source_path=args.source,
            persist=persist,
            output_path=output,
        )
    except Exception:
        return _emit_error(
            command,
            "pipeline_invocation_failed",
            json_output=json_output,
            exit_code=EXIT_PIPELINE_FAILED,
        )
    summary = _pipeline_summary(command, result)
    _emit(summary, json_output=json_output)
    code = _pipeline_exit_code(result)
    if code != EXIT_SUCCESS:
        for error in result.errors or ("pipeline_failed",):
            print(f"error:{error}", file=sys.stderr)
    return code


def _run_validate_command(args: argparse.Namespace) -> int:
    try:
        loaded = memory_module.load_project_capability_memory(args.path)
    except Exception:
        return _emit_error(
            "validate",
            "capability_memory_invalid",
            json_output=bool(args.json),
            exit_code=EXIT_ARTIFACT_INVALID,
        )
    summary = _validation_summary(loaded)
    _emit(summary, json_output=bool(args.json))
    if loaded.status in SUCCESS_MEMORY_STATUSES:
        return EXIT_SUCCESS
    code = EXIT_ARTIFACT_MISSING if loaded.status == "missing" else EXIT_ARTIFACT_INVALID
    for error in summary["errors"]:
        print(f"error:{error}", file=sys.stderr)
    return code


def _run_backfill_command(args: argparse.Namespace) -> int:
    try:
        result = backfill_module.run_authoritative_project_capability_backfill()
    except Exception:
        return _emit_error(
            "backfill",
            "backfill_invocation_failed",
            json_output=bool(args.json),
            exit_code=EXIT_PIPELINE_FAILED,
        )
    summary = _backfill_summary(result)
    _emit(summary, json_output=bool(args.json))
    code = _backfill_exit_code(result)
    if code != EXIT_SUCCESS:
        for error in result.errors or ("backfill_failed",):
            print(f"error:{error}", file=sys.stderr)
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one explicit command and return its deterministic process exit code."""

    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.command in {"build", "inspect"}:
        return _run_pipeline_command(args)
    if args.command == "validate":
        return _run_validate_command(args)
    if args.command == "backfill":
        return _run_backfill_command(args)
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXIT_ARTIFACT_INVALID",
    "EXIT_ARTIFACT_MISSING",
    "EXIT_PERSISTENCE_FAILED",
    "EXIT_PIPELINE_FAILED",
    "EXIT_SAFETY_VIOLATION",
    "EXIT_SUCCESS",
    "EXIT_USAGE",
    "main",
]
