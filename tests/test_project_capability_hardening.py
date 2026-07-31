"""Final hardening contract for the isolated Project Capability subsystem.

These tests intentionally exercise only the accepted read-only lifecycle and the
controlled backfill.  They must not introduce a production consumer or rewrite
either authoritative artifact.
"""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest

from backend import project_capability_backfill as backfill
from backend import project_capability_builder as builder
from backend import project_capability_cli as capability_cli
from backend import project_capability_memory as capability_memory
from backend import project_capability_pipeline as pipeline
from backend import project_capability_reader as reader
from backend import project_evidence_pipeline as evidence_pipeline
from backend.project_capability_boundaries import inherit_capability_claim_policy
from backend.project_capability_memory import CapabilityCandidate
from backend.project_capability_scoring import assess_capability_candidate_support
from backend.project_claim_boundaries import build_project_evidence_claim_boundary
from backend.project_evidence_memory import load_project_evidence_memory
from backend.project_evidence_models import (
    Confidence,
    EvidenceSourceRef,
    EvidenceType,
    MetricSupport,
    ProjectCapabilityFact,
    ProjectEvidenceFact,
)


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
REAL_EVIDENCE = ROOT / "information" / "project_evidence_memory.json"
REAL_CAPABILITY = ROOT / "information" / "project_capability_memory.json"

EXPECTED_EVIDENCE_FILE_SHA256 = "95750df456d1fb3dea56cf40891593834a52731414a882896d99aa5a51b3f106"
EXPECTED_EVIDENCE_CONTENT_HASH = "37967289816ec13638b4b30e31a74f52688acc9bc08ff6c6faf760b2c6180fd3"
EXPECTED_CAPABILITY_FILE_SHA256 = "bf1506039141e35e1d6495c057cfffc4946e4e9b0183aa69030ac0c8b9587c4b"
EXPECTED_CAPABILITY_CONTENT_HASH = "ab588d427e883f5076ea69e04d8bd2f29d9121864a4f743db87ddee874cd5c45"

EXPECTED_COUNTS = {
    "source_project_count": 11,
    "source_evidence_fact_count": 283,
    "source_claim_boundary_count": 184,
    "candidate_count": 57,
    "matched_evidence_count": 196,
    "unmatched_evidence_count": 83,
    "ambiguous_evidence_count": 4,
    "skipped_evidence_count": 0,
    "assessment_count": 57,
    "eligible_assessment_count": 0,
    "policy_count": 0,
    "build_result_count": 0,
    "capability_fact_count": 0,
}

CAPABILITY_PRODUCTION_FILES = tuple(sorted(BACKEND.glob("project_capability*.py")))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_state(path: Path) -> tuple[bytes, int, int, str]:
    payload = path.read_bytes()
    stat = path.stat()
    return payload, stat.st_size, stat.st_mtime_ns, hashlib.sha256(payload).hexdigest()


def _module_definitions(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return classes, functions


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    return imported


def _real_rebuild():
    return pipeline.run_project_capability_pipeline(source_path=REAL_EVIDENCE, persist=False)


def _evidence_fact(
    evidence_id: str,
    *,
    project_id: str = "project-a",
    technical_tags: tuple[str, ...] = ("quality_dimensions", "FastAPI"),
) -> ProjectEvidenceFact:
    return ProjectEvidenceFact(
        project_id=project_id,
        problem="",
        mechanism="bounded deterministic quality validation",
        implementation=["Applied deterministic structured validation."],
        source_refs=[EvidenceSourceRef(
            source_type="github_evidence_card",
            source_id=f"source-{evidence_id}",
            project_id=project_id,
            content_hash=hashlib.sha256(f"{project_id}:{evidence_id}".encode()).hexdigest(),
        )],
        evidence_type=EvidenceType.VALIDATION,
        metric_support=MetricSupport.NONE,
        technical_tags=list(technical_tags),
        quality_score=90,
        evidence_fact_id=evidence_id,
    )


def _verified_inputs(fact: ProjectEvidenceFact):
    candidate = CapabilityCandidate(
        project_id=fact.project_id,
        capability_type="output_quality_control",
        supporting_evidence_ids=(fact.evidence_fact_id,),
        supporting_signals=("quality_dimensions",),
        conflicting_signals=(),
        candidate_score=0.0,
        metadata={"evaluation_state": "unscored"},
    )
    evidence_index = {fact.evidence_fact_id: fact}
    assessment = assess_capability_candidate_support(
        candidate=candidate,
        evidence_index=evidence_index,
    )
    boundary = build_project_evidence_claim_boundary(fact)
    assert boundary is not None
    policy = inherit_capability_claim_policy(
        candidate=candidate,
        assessment=assessment,
        evidence_index=evidence_index,
        claim_boundaries=(boundary,),
    )
    assert assessment.eligibility_status == "eligible"
    assert policy.policy_status == "eligible"
    return candidate, assessment, policy


def _safe_output(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str).casefold()


def test_project_capability_subsystem_has_single_authoritative_implementations():
    expected = {
        "build_project_capability_memory": "project_capability_memory.py",
        "extract_project_capabilities": "project_capability_extractor.py",
        "group_project_evidence_facts": "project_capability_grouping.py",
        "assess_project_capability_candidates": "project_capability_scoring.py",
        "inherit_project_capability_claim_policies": "project_capability_boundaries.py",
        "build_project_capability_facts": "project_capability_builder.py",
        "run_project_capability_pipeline": "project_capability_pipeline.py",
        "run_authoritative_project_capability_backfill": "project_capability_backfill.py",
        "read_project_capability_memory": "project_capability_reader.py",
    }
    definitions: dict[str, list[str]] = {name: [] for name in expected}
    for path in CAPABILITY_PRODUCTION_FILES:
        _classes, functions = _module_definitions(path)
        for name in definitions:
            if name in functions:
                definitions[name].append(path.name)
    assert definitions == {name: [module] for name, module in expected.items()}

    model_definitions = []
    for path in BACKEND.glob("*.py"):
        classes, _functions = _module_definitions(path)
        if "ProjectCapabilityFact" in classes:
            model_definitions.append(path.name)
    assert model_definitions == ["project_evidence_models.py"]


def test_all_capability_fact_paths_use_authoritative_identity():
    base = dict(
        project_id="project-a",
        capability_type="output_quality_control",
        present=True,
        source_evidence_fact_ids=["pef_b", "pef_a", "pef_a"],
    )
    first = ProjectCapabilityFact(
        **base,
        confidence=Confidence.MEDIUM,
        mechanisms=["mechanism one"],
        allowed_resume_claims=["mechanism:mechanism one"],
        forbidden_claims=["unsupported metric"],
        metric_support=MetricSupport.NONE,
        technical_tags=["Python"],
    )
    second = ProjectCapabilityFact(
        **{**base, "source_evidence_fact_ids": ["pef_a", "pef_b"]},
        confidence=Confidence.HIGH,
        mechanisms=["changed mechanism"],
        allowed_resume_claims=["changed claim"],
        forbidden_claims=["changed boundary"],
        metric_support=MetricSupport.EXPLICIT,
        technical_tags=["FastAPI"],
    )
    assert first.capability_id == second.capability_id
    assert len(first.capability_id) == 28 and first.capability_id.startswith("pcf_")
    for field, value in (
        ("project_id", "project-b"),
        ("capability_type", "failure_recovery"),
        ("present", False),
        ("source_evidence_fact_ids", ["pef_a"]),
    ):
        changed = ProjectCapabilityFact(**{**base, field: value})
        assert changed.capability_id != first.capability_id

    identity_calls = []
    for path in BACKEND.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if re.search(r"build_project_evidence_stable_id\(\s*[\"']pcf_", source):
            identity_calls.append(path.name)
    assert identity_calls == ["project_evidence_models.py"]


def test_project_capability_reader_flag_semantics_match_repository_conventions(monkeypatch):
    assert reader.ENABLED_FLAG_VALUES == evidence_pipeline.ENABLED_FLAG_VALUES
    assert reader.DISABLED_FLAG_VALUES == evidence_pipeline.DISABLED_FLAG_VALUES
    values = (*reader.ENABLED_FLAG_VALUES, *reader.DISABLED_FLAG_VALUES, "malformed")
    for value in values:
        monkeypatch.setenv(reader.PROJECT_CAPABILITY_MEMORY_FLAG, value)
        reader_value = reader.is_project_capability_memory_enabled()
        evidence_value = evidence_pipeline.is_project_evidence_memory_enabled({
            evidence_pipeline.PROJECT_EVIDENCE_MEMORY_FLAG: value,
        })
        assert reader_value == evidence_value
    monkeypatch.delenv(reader.PROJECT_CAPABILITY_MEMORY_FLAG, raising=False)
    assert reader.is_project_capability_memory_enabled() is False


def test_empty_missing_invalid_stale_and_failed_statuses_remain_distinct(tmp_path):
    loaded_real = capability_memory.load_project_capability_memory(REAL_CAPABILITY)
    missing = capability_memory.load_project_capability_memory(tmp_path / "missing.json")
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{not-json", encoding="utf-8")
    invalid = capability_memory.load_project_capability_memory(invalid_path)
    stale = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=REAL_CAPABILITY,
        evidence_memory_path=tmp_path / "missing-evidence.json",
    )
    failed = pipeline.run_project_capability_pipeline(
        source_path=REAL_EVIDENCE,
        persist=True,
        output_path=REAL_CAPABILITY,
    )
    assert (loaded_real.status, missing.status, invalid.status, stale.status, failed.status) == (
        "empty", "missing", "invalid", "stale", "failed"
    )


def test_real_capability_memory_matches_read_only_pipeline_rebuild():
    before = (_file_state(REAL_EVIDENCE), _file_state(REAL_CAPABILITY))
    result = _real_rebuild()
    loaded = capability_memory.load_project_capability_memory(REAL_CAPABILITY)
    assert result.status == "empty" and result.memory is not None
    assert loaded.status == "empty" and loaded.memory == result.memory
    assert capability_memory.serialize_project_capability_memory(result.memory) == REAL_CAPABILITY.read_bytes()
    assert result.memory.content_hash == EXPECTED_CAPABILITY_CONTENT_HASH
    assert result.source_content_hash == EXPECTED_EVIDENCE_CONTENT_HASH
    for name, expected in EXPECTED_COUNTS.items():
        assert getattr(result, name) == expected
    assert (_file_state(REAL_EVIDENCE), _file_state(REAL_CAPABILITY)) == before


def test_real_backfill_is_unchanged_and_does_not_rewrite_artifact():
    before = (_file_state(REAL_EVIDENCE), _file_state(REAL_CAPABILITY))
    result = backfill.run_authoritative_project_capability_backfill()
    assert result.status == "unchanged"
    assert result.target_existed_before is True
    assert result.target_written is False
    assert result.target_unchanged is True
    assert result.source_file_sha256_before == result.source_file_sha256_after
    assert result.target_content_hash == EXPECTED_CAPABILITY_CONTENT_HASH
    assert result.target_file_sha256 == EXPECTED_CAPABILITY_FILE_SHA256
    assert result.diagnostics["staging_artifact_count"] == 0
    assert (_file_state(REAL_EVIDENCE), _file_state(REAL_CAPABILITY)) == before


def test_generic_build_cannot_write_authoritative_capability_artifact(capsys):
    before = _file_state(REAL_CAPABILITY)
    result = pipeline.run_project_capability_pipeline(
        source_path=REAL_EVIDENCE,
        persist=True,
        output_path=REAL_CAPABILITY,
    )
    assert result.status == "failed"
    assert result.errors == ("real_output_path_forbidden",)
    exit_code = capability_cli.main([
        "build", "--source", str(REAL_EVIDENCE), "--persist",
        "--output", str(REAL_CAPABILITY), "--json",
    ])
    captured = capsys.readouterr()
    assert exit_code == capability_cli.EXIT_SAFETY_VIOLATION
    assert "real_backfill_path_prohibited" in captured.err
    assert _file_state(REAL_CAPABILITY) == before


def test_reader_backfill_pipeline_and_cli_responsibilities_remain_separated(monkeypatch):
    import_map = {path.name: _imports(path) for path in CAPABILITY_PRODUCTION_FILES}
    assert not any(
        name.endswith(("project_capability_pipeline", "project_capability_backfill", "project_capability_cli"))
        for name in import_map["project_capability_reader.py"]
    )
    assert any(name.endswith("project_capability_pipeline") for name in import_map["project_capability_backfill.py"])
    assert not any(name.endswith("project_capability_reader") for name in import_map["project_capability_backfill.py"])
    assert not any(name.endswith(("project_capability_reader", "project_capability_backfill")) for name in import_map["project_capability_pipeline.py"])
    assert not any(name.endswith(("project_capability_reader", "project_capability_pipeline", "project_capability_backfill")) for name in import_map["project_capability_memory.py"])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reader crossed into a write/build responsibility")

    monkeypatch.setattr(pipeline, "run_project_capability_pipeline", forbidden)
    monkeypatch.setattr(backfill, "run_authoritative_project_capability_backfill", forbidden)
    result = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=REAL_CAPABILITY,
        evidence_memory_path=REAL_EVIDENCE,
    )
    assert result.status == "empty"


def test_reader_returns_no_facts_for_missing_invalid_stale_or_error_states(tmp_path, monkeypatch):
    missing = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=tmp_path / "missing.json",
        evidence_memory_path=REAL_EVIDENCE,
    )
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("[]", encoding="utf-8")
    invalid = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=invalid_path,
        evidence_memory_path=REAL_EVIDENCE,
    )
    stale = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=REAL_CAPABILITY,
        evidence_memory_path=tmp_path / "missing-evidence.json",
    )
    monkeypatch.setattr(
        reader.capability_memory_module,
        "load_project_capability_memory",
        lambda _path: (_ for _ in ()).throw(RuntimeError("safe test failure")),
    )
    error = reader.read_project_capability_memory(feature_enabled=True)
    assert (missing.status, invalid.status, stale.status, error.status) == (
        "missing", "invalid", "stale", "error"
    )
    assert all(not item.facts and item.capability_fact_count == 0 for item in (missing, invalid, stale, error))


def test_no_project_capability_path_falls_back_to_unverified_data():
    result = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=REAL_CAPABILITY,
        evidence_memory_path=REAL_EVIDENCE,
    )
    assert result.status == "empty" and result.facts == ()
    source = (BACKEND / "project_capability_reader.py").read_text(encoding="utf-8")
    forbidden_dependencies = (
        "project_capability_extractor", "project_capability_grouping",
        "project_capability_scoring", "project_capability_builder",
        "project_capability_pipeline", "project_capability_backfill",
    )
    assert not any(name in source for name in forbidden_dependencies)


def test_project_capability_lifecycle_preserves_exact_project_isolation(monkeypatch):
    rebuilt = _real_rebuild()
    assert rebuilt.memory is not None
    evidence = load_project_evidence_memory(REAL_EVIDENCE)
    assert evidence.status == "ready" and evidence.snapshot is not None
    source_project = evidence.snapshot.projects[0]
    source_fact = source_project.evidence_facts[0]
    fact = ProjectCapabilityFact(
        project_id=source_project.project_id,
        capability_type="output_quality_control",
        present=True,
        source_evidence_fact_ids=[source_fact.evidence_fact_id],
        confidence=Confidence.HIGH,
        mechanisms=["bounded deterministic validation"],
        allowed_resume_claims=["mechanism:bounded deterministic validation"],
        forbidden_claims=["unsupported metric claim"],
        technical_tags=["Python"],
    )
    memory = capability_memory.build_project_capability_memory(
        source_artifact=rebuilt.memory.source_artifact,
        source_project_ids=[project.project_id for project in evidence.snapshot.projects],
        capability_facts=[fact],
    )
    monkeypatch.setattr(
        reader.capability_memory_module,
        "load_project_capability_memory",
        lambda _path: SimpleNamespace(status="ready", memory=memory),
    )
    exact = reader.get_verified_project_capabilities(
        source_project.project_id,
        feature_enabled=True,
        evidence_memory_path=REAL_EVIDENCE,
    )
    other_project_id = next(
        project.project_id for project in evidence.snapshot.projects
        if project.project_id != source_project.project_id
    )
    other = reader.get_verified_project_capabilities(
        other_project_id,
        feature_enabled=True,
        evidence_memory_path=REAL_EVIDENCE,
    )
    assert exact.status == "ready" and exact.facts == (fact,)
    assert other.status == "empty" and other.facts == ()
    with pytest.raises(capability_memory.ProjectCapabilityMemoryIntegrityError):
        capability_memory.build_project_capability_memory(
            source_artifact=replace(rebuilt.memory.source_artifact, project_count=1),
            source_project_ids=[other_project_id],
            capability_facts=[fact],
        )


def test_project_capability_facts_cannot_gain_unsupported_technologies():
    primary = _evidence_fact(
        "pef_primary",
        technical_tags=("quality_dimensions", "FastAPI", "C:\\Users\\z\\secret.py", "sk-testcredential123456789"),
    )
    unrelated = _evidence_fact(
        "pef_unrelated",
        project_id="project-b",
        technical_tags=("quality_dimensions", "Django"),
    )
    candidate, assessment, policy = _verified_inputs(primary)
    result = builder.build_project_capability_fact(
        candidate=candidate,
        assessment=assessment,
        policy=policy,
        evidence_index={
            primary.evidence_fact_id: primary,
            unrelated.evidence_fact_id: unrelated,
        },
    )
    assert result.build_status == "built" and result.fact is not None
    assert "FastAPI" in result.fact.technical_tags
    assert "Django" not in result.fact.technical_tags
    assert not any("users" in tag.casefold() or "sk-" in tag.casefold() for tag in result.fact.technical_tags)


def test_project_capability_claim_safety_contract_remains_enforced():
    fact = _evidence_fact("pef_claim")
    candidate, assessment, policy = _verified_inputs(fact)
    collision = replace(
        policy,
        allowed_claims=("mechanism:bounded deterministic quality validation",),
        forbidden_claims=("mechanism:bounded deterministic quality validation",),
    )
    collision_result = builder.build_project_capability_fact(
        candidate=candidate,
        assessment=assessment,
        policy=collision,
        evidence_index={fact.evidence_fact_id: fact},
    )
    numeric = replace(
        policy,
        allowed_claims=("impact:improved throughput by 42%",),
        metric_support=MetricSupport.NONE.value,
    )
    numeric_result = builder.build_project_capability_fact(
        candidate=candidate,
        assessment=assessment,
        policy=numeric,
        evidence_index={fact.evidence_fact_id: fact},
    )
    assert collision_result.build_status == "ineligible_claim_policy"
    assert numeric_result.build_status == "invalid_metric_support"
    assert collision_result.fact is None and numeric_result.fact is None


def test_project_capability_metric_support_values_remain_authoritative():
    assert {item.value for item in MetricSupport} == {"none", "approximate", "explicit"}
    assert capability_memory.METRIC_SUPPORT_LEVELS == {"none", "approximate", "explicit"}
    assert "qualitative" not in "\n".join(
        path.read_text(encoding="utf-8").casefold() for path in CAPABILITY_PRODUCTION_FILES
    )


def test_project_capability_outputs_contain_no_raw_or_sensitive_content():
    rebuilt = _real_rebuild()
    read_result = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=REAL_CAPABILITY,
        evidence_memory_path=REAL_EVIDENCE,
    )
    backfill_result = backfill.run_authoritative_project_capability_backfill()
    outputs = (
        json.loads(REAL_CAPABILITY.read_text(encoding="utf-8")),
        rebuilt.to_safe_dict(),
        read_result.to_safe_dict(),
        backfill_result.to_safe_dict(),
    )
    serialized = "\n".join(_safe_output(value) for value in outputs)
    forbidden = (
        "raw_patch", "raw_diff", "raw_content", "raw_text", "authorization_header",
        "chain_of_thought", "-----begin private key-----", "bearer ",
        str(ROOT).casefold(),
    )
    assert not any(value in serialized for value in forbidden)
    assert not re.search(r"(?:gh[oprsu]_|sk-)[a-z0-9_-]{16,}", serialized)


def test_project_capability_production_code_uses_semantic_naming():
    source = "\n".join(path.read_text(encoding="utf-8") for path in CAPABILITY_PRODUCTION_FILES)
    assert not re.search(r"(?i)\b(?:phase|step)[ _-]?\d+\b", source)
    assert not re.search(r"(?m)^class\s+CapabilityFact\b", source)
    assert "project_capability_memory" in source


def test_project_capability_memory_has_no_api_frontend_or_resume_integration():
    consumers = []
    for path in BACKEND.rglob("*.py"):
        if path.name.startswith("project_capability"):
            continue
        source = path.read_text(encoding="utf-8").casefold()
        if "project_capability_reader" in source or "use_project_capability_memory" in source:
            consumers.append(path.relative_to(ROOT).as_posix())
    for directory_name in ("frontend", "src"):
        directory = ROOT / directory_name
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.casefold() not in {".py", ".js", ".jsx", ".ts", ".tsx", ".vue"}:
                continue
            source = path.read_text(encoding="utf-8", errors="ignore").casefold()
            if "project_capability" in source or "/api/project-capability" in source:
                consumers.append(path.relative_to(ROOT).as_posix())
    assert consumers == []


def test_project_capability_module_dependencies_preserve_read_write_boundaries():
    graph = {path.name: _imports(path) for path in CAPABILITY_PRODUCTION_FILES}
    reader_imports = graph["project_capability_reader.py"]
    memory_imports = graph["project_capability_memory.py"]
    builder_imports = graph["project_capability_builder.py"]
    assert not any(name.endswith(("project_capability_pipeline", "project_capability_backfill", "project_capability_cli")) for name in reader_imports)
    assert not any(name.endswith(("project_capability_pipeline", "project_capability_backfill", "project_capability_reader")) for name in memory_imports)
    assert not any(name.endswith(("project_capability_pipeline", "project_capability_backfill", "project_capability_reader", "project_capability_cli")) for name in builder_imports)
    assert any(name.endswith("project_capability_pipeline") for name in graph["project_capability_backfill.py"])
    assert any(name.endswith("project_capability_backfill") for name in graph["project_capability_cli.py"])


def test_project_capability_artifact_remains_ignored_untracked_and_unstaged():
    relative = REAL_CAPABILITY.relative_to(ROOT).as_posix()
    ignored = subprocess.run(
        ["git", "check-ignore", "--", relative], cwd=ROOT,
        text=True, capture_output=True, check=False,
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--", relative], cwd=ROOT,
        text=True, capture_output=True, check=True,
    )
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--", relative], cwd=ROOT,
        text=True, capture_output=True, check=True,
    )
    assert ignored.returncode == 0 and relative in ignored.stdout.replace("\\", "/")
    assert tracked.stdout.strip() == ""
    assert staged.stdout.strip() == ""
    assert list(REAL_CAPABILITY.parent.glob(f".{REAL_CAPABILITY.name}.*.stage")) == []


def test_hardening_suite_does_not_modify_authoritative_artifacts():
    before = (_file_state(REAL_EVIDENCE), _file_state(REAL_CAPABILITY))
    rebuild = _real_rebuild()
    read_result = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=REAL_CAPABILITY,
        evidence_memory_path=REAL_EVIDENCE,
    )
    backfill_result = backfill.run_authoritative_project_capability_backfill()
    assert rebuild.status == "empty"
    assert read_result.status == "empty"
    assert backfill_result.status == "unchanged"
    assert (_file_state(REAL_EVIDENCE), _file_state(REAL_CAPABILITY)) == before


def test_project_capability_memory_final_closeout_contract(monkeypatch):
    evidence_state = _file_state(REAL_EVIDENCE)
    capability_state = _file_state(REAL_CAPABILITY)
    loaded = capability_memory.load_project_capability_memory(REAL_CAPABILITY)
    disabled = reader.read_project_capability_memory(feature_enabled=False)
    enabled = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=REAL_CAPABILITY,
        evidence_memory_path=REAL_EVIDENCE,
    )
    assert _sha256(REAL_EVIDENCE) == EXPECTED_EVIDENCE_FILE_SHA256
    assert evidence_state[3] == EXPECTED_EVIDENCE_FILE_SHA256
    assert capability_state[3] == EXPECTED_CAPABILITY_FILE_SHA256
    assert loaded.status == "empty" and loaded.memory is not None
    assert loaded.memory.content_hash == EXPECTED_CAPABILITY_CONTENT_HASH
    assert loaded.memory.source_artifact.content_hash == EXPECTED_EVIDENCE_CONTENT_HASH
    assert loaded.memory.source_artifact.file_sha256 == EXPECTED_EVIDENCE_FILE_SHA256
    assert loaded.memory.capability_facts == ()
    assert disabled.status == "disabled" and disabled.facts == ()
    assert enabled.status == "empty" and enabled.facts == ()
    monkeypatch.setenv(reader.PROJECT_CAPABILITY_MEMORY_FLAG, "malformed")
    malformed = reader.read_project_capability_memory()
    assert malformed.status == "disabled" and malformed.facts == ()
    assert (_file_state(REAL_EVIDENCE), _file_state(REAL_CAPABILITY)) == (
        evidence_state, capability_state
    )
