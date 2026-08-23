from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend import chroma_migration_baseline as baseline
from backend.chroma_baseline_models import (
    BaselineValidationError,
    add_baseline_content_hash,
    sha256_json,
    validate_chroma_migration_baseline,
)


FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def synthetic_repository(tmp_path: Path) -> tuple[Path, Path, Path, tuple[baseline.ArtifactSpec, ...]]:
    root = tmp_path / "repository"
    protected = root / "information" / "chroma"
    protected.mkdir(parents=True)
    (protected / "zeta.bin").write_bytes(b"zeta")
    nested = protected / "segments" / "alpha"
    nested.mkdir(parents=True)
    (nested / "index.bin").write_bytes(b"index")
    artifact = root / "information" / "fixture_artifact.json"
    artifact.write_text('{"schema":"fixture_artifact.v1","count":2}\n', encoding="utf-8")
    output = root / "information" / "chroma_migration_baselines" / "baseline.json"
    specs = (baseline.ArtifactSpec("fixture_artifact", "information/fixture_artifact.json"),)
    return root, protected, output, specs


def capture_synthetic(tmp_path: Path, **overrides):
    root, protected, output, specs = synthetic_repository(tmp_path)
    arguments = {
        "repository_root": root,
        "protected_root": protected,
        "output_path": output,
        "artifact_specs": specs,
        "clock": lambda: FIXED_TIME,
        "environ": {},
    }
    arguments.update(overrides)
    payload = baseline.capture_chroma_migration_baseline(**arguments)
    return root, protected, output, specs, payload


def test_file_inventory_order_hash_and_relative_path_are_deterministic(tmp_path):
    _, protected, _, _ = synthetic_repository(tmp_path)
    first = baseline.capture_protected_file_inventory(protected)
    second = baseline.capture_protected_file_inventory(protected)
    assert first == second
    assert first["files"] == sorted(first["files"], key=lambda item: item["relative_path"])
    assert [item["relative_path"] for item in first["files"]] == [
        "segments/alpha/index.bin",
        "zeta.bin",
    ]
    assert all("\\" not in item["relative_path"] for item in first["files"])
    assert first["aggregate_sha256"] == sha256_json(first["files"])


def test_aggregate_hash_changes_only_when_file_inventory_changes(tmp_path):
    _, protected, _, _ = synthetic_repository(tmp_path)
    first = baseline.capture_protected_file_inventory(protected)
    (protected / "zeta.bin").write_bytes(b"changed")
    second = baseline.capture_protected_file_inventory(protected)
    assert first["aggregate_sha256"] != second["aggregate_sha256"]
    assert first["files"][0] == second["files"][0]


def test_reparse_point_is_rejected_without_following_it(tmp_path, monkeypatch):
    _, protected, _, _ = synthetic_repository(tmp_path)
    original = baseline._is_reparse_point
    monkeypatch.setattr(
        baseline,
        "_is_reparse_point",
        lambda path: Path(path).name == "segments" or original(Path(path)),
    )
    with pytest.raises(baseline.BaselineCaptureError, match="reparse_point_rejected"):
        baseline.capture_protected_file_inventory(protected)


def test_artifact_path_escape_is_rejected_before_reading(tmp_path):
    root, _, _, _ = synthetic_repository(tmp_path)
    with pytest.raises(baseline.BaselineCaptureError, match="invalid_artifact_path"):
        baseline.capture_evidence_artifact_hashes(
            root,
            artifact_specs=(baseline.ArtifactSpec("escape", "../outside.json"),),
        )


def test_unreadable_file_fails_closed_without_omission(tmp_path):
    _, protected, _, _ = synthetic_repository(tmp_path)

    def denied(*_args, **_kwargs):
        raise PermissionError("synthetic")

    with pytest.raises(baseline.BaselineCaptureError, match="unreadable_file"):
        baseline.capture_protected_file_inventory(protected, opener=denied)


def test_inventory_reads_do_not_change_file_times(tmp_path):
    _, protected, _, _ = synthetic_repository(tmp_path)
    before = {path: path.stat().st_mtime_ns for path in protected.rglob("*") if path.is_file()}
    baseline.capture_protected_file_inventory(protected)
    after = {path: path.stat().st_mtime_ns for path in protected.rglob("*") if path.is_file()}
    assert before == after


def test_capture_writes_one_valid_baseline_atomically(tmp_path, monkeypatch):
    root, protected, output, specs = synthetic_repository(tmp_path)
    original = baseline.os.replace
    replacements = []

    def tracked(source, target):
        replacements.append((Path(source).name, Path(target).name))
        return original(source, target)

    monkeypatch.setattr(baseline.os, "replace", tracked)
    payload = baseline.capture_chroma_migration_baseline(
        repository_root=root,
        protected_root=protected,
        output_path=output,
        artifact_specs=specs,
        clock=lambda: FIXED_TIME,
        environ={},
    )
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert replacements and replacements[-1][1] == "baseline.json"
    assert not list(output.parent.glob("*.tmp"))
    validate_chroma_migration_baseline(payload)


def test_schema_and_content_digest_validation_reject_tampering(tmp_path):
    *_, payload = capture_synthetic(tmp_path)
    invalid_schema = add_baseline_content_hash({**payload, "schema": "wrong.v1"})
    with pytest.raises(BaselineValidationError, match="unsupported_baseline_schema"):
        validate_chroma_migration_baseline(invalid_schema)
    invalid_hash = copy.deepcopy(payload)
    invalid_hash["content_sha256"] = "0" * 64
    with pytest.raises(BaselineValidationError, match="baseline_content_hash_mismatch"):
        validate_chroma_migration_baseline(invalid_hash)


@pytest.mark.parametrize("field", ("documents", "embeddings", "patch", "raw_metadata", "source_body"))
def test_privacy_schema_rejects_forbidden_fields(tmp_path, field):
    *_, payload = capture_synthetic(tmp_path)
    invalid = copy.deepcopy(payload)
    invalid["logical_inventory"][field] = ["private"]
    invalid = add_baseline_content_hash(invalid)
    with pytest.raises(BaselineValidationError):
        validate_chroma_migration_baseline(invalid)


def test_absolute_windows_and_unc_paths_are_rejected(tmp_path):
    *_, payload = capture_synthetic(tmp_path)
    for unsafe in ("C:/Users/example/chroma.sqlite3", "//server/share/chroma.sqlite3"):
        invalid = copy.deepcopy(payload)
        invalid["evidence_artifacts"]["artifacts"][0]["relative_path"] = unsafe
        invalid["evidence_artifacts"]["aggregate_sha256"] = sha256_json(
            invalid["evidence_artifacts"]["artifacts"]
        )
        invalid = add_baseline_content_hash(invalid)
        with pytest.raises(BaselineValidationError):
            validate_chroma_migration_baseline(invalid)


def test_secret_values_are_rejected_even_in_allowed_string_fields(tmp_path):
    *_, payload = capture_synthetic(tmp_path)
    invalid = copy.deepcopy(payload)
    invalid["logical_inventory"]["limitations"] = ["API_KEY=private"]
    invalid = add_baseline_content_hash(invalid)
    with pytest.raises(BaselineValidationError, match="secret_value_exposure"):
        validate_chroma_migration_baseline(invalid)


def test_default_logical_inventory_is_explicitly_unavailable_and_private(tmp_path):
    *_, payload = capture_synthetic(tmp_path)
    assert payload["logical_inventory"] == baseline.unavailable_logical_inventory()
    assert payload["deployment_observation"] == {
        "configured_mode": "disabled",
        "server_reachable": False,
        "server_version": None,
    }
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in ('"documents"', '"embeddings"', '"patch"', '"raw_metadata"', "C:/", "C:\\"):
        assert forbidden not in serialized
    assert set(payload["privacy"].values()) == {False}


def test_fake_approved_http_inventory_is_allowlisted_and_hides_record_ids(tmp_path):
    ids = ["record-b", "record-a"]

    def fake_http():
        return {
            "source": "approved_http",
            "collections": [
                {
                    "semantic_name": "github_evidence",
                    "record_count": 2,
                    "record_ids_sha256": sha256_json(sorted(ids)),
                    "logical_fingerprint": hashlib.sha256(b"logical").hexdigest(),
                    "repository_count": 1,
                    "schema_marker": "github_evidence.v1",
                }
            ],
            "limitations": [],
            "server_reachable": True,
            "server_version": "1.2.3",
        }

    *_, payload = capture_synthetic(
        tmp_path,
        logical_inventory_provider=fake_http,
        environ={"GITHUB_EVIDENCE_VECTOR_QUERY_BACKEND": "chroma_http"},
    )
    assert payload["logical_inventory"]["source"] == "approved_http"
    assert payload["deployment_observation"] == {
        "configured_mode": "local_http",
        "server_reachable": True,
        "server_version": "1.2.3",
    }
    serialized = json.dumps(payload, sort_keys=True)
    assert all(record_id not in serialized for record_id in ids)


def test_artifact_hash_capture_does_not_modify_artifact(tmp_path):
    root, _, _, specs = synthetic_repository(tmp_path)
    artifact = root / "information" / "fixture_artifact.json"
    before = (artifact.read_bytes(), artifact.stat().st_mtime_ns)
    captured = baseline.capture_evidence_artifact_hashes(root, artifact_specs=specs)
    after = (artifact.read_bytes(), artifact.stat().st_mtime_ns)
    assert before == after
    assert captured["artifacts"][0]["schema_marker"] == "fixture_artifact.v1"
    assert captured["artifacts"][0]["sha256"] == hashlib.sha256(before[0]).hexdigest()


def test_static_call_inventory_classifies_test_only_calls(tmp_path):
    root = tmp_path / "repository"
    (root / "backend").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "tests" / "test_fake.py").write_text(
        "def test_fake():\n    chromadb.PersistentClient(path='synthetic')\n",
        encoding="utf-8",
    )
    result = baseline.capture_chroma_client_call_inventory(root)
    assert result["production_persistent_client_call_count"] == 0
    assert result["approved_maintenance_persistent_client_call_count"] == 0
    assert result["test_only_persistent_client_call_count"] == 1
    assert result["http_client_call_count"] == 0
    assert result["unknown_unclassified_call_count"] == 0


@pytest.mark.parametrize(
    "relative,source",
    (
        (
            "backend/unexpected.py",
            "def connect():\n    return chromadb.PersistentClient(path='synthetic')\n",
        ),
        (
            "backend/chroma_http_vector_search.py",
            "def _create_http_client():\n    return chromadb.HttpClient(host='local')\n",
        ),
    ),
)
def test_unapproved_production_client_call_blocks_capture(tmp_path, relative, source):
    root = tmp_path / "repository"
    target = root / relative
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    with pytest.raises(baseline.BaselineCaptureError, match="unclassified_chroma_call_site"):
        baseline.capture_chroma_client_call_inventory(root)


def test_current_repository_client_calls_are_fully_classified():
    result = baseline.capture_chroma_client_call_inventory(Path(__file__).resolve().parents[1])
    assert result["production_persistent_client_call_count"] == 0
    assert result["approved_maintenance_persistent_client_call_count"] == 0
    assert result["test_only_persistent_client_call_count"] == 1
    assert result["http_client_call_count"] == 2
    assert result["unknown_unclassified_call_count"] == 0


def test_capture_refuses_changed_protected_inventory_and_writes_nothing(tmp_path):
    root, protected, output, specs = synthetic_repository(tmp_path)
    original = baseline.capture_protected_file_inventory
    calls = 0

    def changing(path):
        nonlocal calls
        calls += 1
        result = original(path)
        if calls == 1:
            (protected / "zeta.bin").write_bytes(b"changed between reads")
        return result

    with pytest.raises(
        baseline.BaselineCaptureError, match="protected_storage_changed_during_capture"
    ):
        baseline.capture_chroma_migration_baseline(
            repository_root=root,
            protected_root=protected,
            output_path=output,
            artifact_specs=specs,
            inventory_reader=changing,
            clock=lambda: FIXED_TIME,
            environ={},
        )
    assert calls == 2
    assert not output.exists()


def test_verify_reports_mismatch_without_rewriting_accepted_baseline(tmp_path):
    root, protected, output, _, _ = capture_synthetic(tmp_path)
    accepted = output.read_bytes()
    accepted_mtime = output.stat().st_mtime_ns
    (protected / "zeta.bin").write_bytes(b"later change")
    result = baseline.verify_chroma_migration_baseline(
        output,
        repository_root=root,
        protected_root=protected,
        compare_protected=True,
    )
    assert result["status"] == "mismatch"
    assert result["historical_byte_baseline"] is True
    assert result["protected_storage_match"] is False
    assert output.read_bytes() == accepted
    assert output.stat().st_mtime_ns == accepted_mtime


def test_verify_can_compare_immutable_artifacts_without_rewrite(tmp_path):
    root, protected, output, _, _ = capture_synthetic(tmp_path)
    before = output.read_bytes()
    result = baseline.verify_chroma_migration_baseline(
        output,
        repository_root=root,
        protected_root=protected,
        compare_artifacts=True,
    )
    assert result["status"] == "verified"
    assert result["evidence_artifacts_match"] is True
    assert output.read_bytes() == before


def test_default_capture_performs_no_chroma_or_network_io(tmp_path, monkeypatch):
    monkeypatch.setattr(
        baseline,
        "capture_approved_http_logical_inventory",
        lambda: pytest.fail("approved HTTP must be opt-in"),
    )
    *_, payload = capture_synthetic(
        tmp_path,
        environ={"GITHUB_EVIDENCE_VECTOR_QUERY_BACKEND": "chroma_http"},
    )
    assert payload["logical_inventory"]["source"] == "unavailable"
    assert payload["deployment_observation"]["server_reachable"] is False


def test_baseline_module_has_no_embedded_client_or_collection_api_calls():
    path = Path(baseline.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "chromadb" not in imported_roots
    assert "PersistentClient" not in called_attributes
    forbidden_calls = (
        ".get_or_create_collection(",
        ".create_collection(",
        ".get_collection(",
        "collection.query(",
        "collection.add(",
        "collection.upsert(",
        "collection.update(",
        "collection.delete(",
    )
    assert all(value not in source for value in forbidden_calls)


def test_output_must_use_gitignored_runtime_boundary(tmp_path):
    root, protected, _, specs = synthetic_repository(tmp_path)
    unsafe = root / "baseline.json"
    with pytest.raises(baseline.BaselineCaptureError, match="unsafe_output_path"):
        baseline.capture_chroma_migration_baseline(
            repository_root=root,
            protected_root=protected,
            output_path=unsafe,
            artifact_specs=specs,
            clock=lambda: FIXED_TIME,
            environ={},
        )
    gitignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "information/*" in gitignore


def test_cli_uses_safe_summary_only_with_synthetic_paths(tmp_path, capsys):
    root, protected, output, _ = synthetic_repository(tmp_path)
    for _, relative_path in baseline.DEFAULT_ARTIFACT_SPECS:
        artifact = root.joinpath(*Path(relative_path).parts)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text('{"schema":"synthetic.v1"}\n', encoding="utf-8")
    exit_code = baseline.main(
        [
            "capture",
            "--repository-root",
            str(root),
            "--protected-root",
            str(protected),
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "capture succeeded files=2 artifacts=8 logical_inventory=unavailable" in captured.out
    assert str(root) not in captured.out


def test_tracked_names_are_semantic_and_no_frontend_dependency_exists():
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "backend" / "chroma_baseline_models.py",
        root / "backend" / "chroma_migration_baseline.py",
        root / "docs" / "chroma_local_server_architecture.md",
        Path(__file__),
    ]
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in sources).casefold()
    forbidden_values = (
        "phase" + "6_5",
        "phase_" + "6_5",
        "step_" + "1",
        "frontend/" + "src",
    )
    for forbidden in forbidden_values:
        assert forbidden not in serialized
    assert "fast" + "api" not in serialized
    assert "persistent" + "client(" not in Path(baseline.__file__).read_text(encoding="utf-8").casefold()


def test_ordinary_tests_reference_only_synthetic_protected_roots():
    source = Path(__file__).read_text(encoding="utf-8").replace("\\", "/").casefold()
    assert "information/" + "chroma" not in source
    assert "default_protected_" + "chroma_root" not in source
    assert "chroma" + " run" not in source
    assert "start" + "-process" not in source
