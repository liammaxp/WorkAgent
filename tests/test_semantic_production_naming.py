from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
NUMBERED_PHASE = re.compile(r"(?i)phase[ _-]?[0-9]+")


def tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def test_active_production_paths_and_source_use_semantic_names():
    assert not [path for path in tracked_paths() if NUMBERED_PHASE.search(path)]
    for directory in (ROOT / "backend", ROOT / "frontend", ROOT / "tests"):
        for path in directory.rglob("*"):
            if not path.is_file() or "node_modules" in path.parts or "dist" in path.parts:
                continue
            if path.suffix not in {".py", ".js", ".jsx", ".ts", ".tsx"}:
                continue
            assert not NUMBERED_PHASE.search(path.read_text(encoding="utf-8")), path


def test_semantic_flags_routes_schemas_and_id_prefixes_are_explicit():
    api = (ROOT / "backend" / "api_server.py").read_text(encoding="utf-8")
    models = (ROOT / "backend" / "project_evidence_models.py").read_text(encoding="utf-8")
    change = (ROOT / "backend" / "project_change_memory.py").read_text(encoding="utf-8")
    frontend = (ROOT / "frontend" / "src" / "api" / "client.js").read_text(encoding="utf-8")
    assert '"USE_GITHUB_EVIDENCE_MEMORY"' in api
    assert '"USE_PROJECT_CHANGE_MEMORY"' in change
    assert '"project_change_memory.v1"' in change
    assert '"project_evidence_memory.v1"' in models
    assert '{"esr_", "pei_", "pef_", "pcf_", "pcb_", "pem_"}' in models
    assert '/api/github/evidence/build' in api
    assert '/api/github/change-memory/build' in api
    assert 'request("/github/evidence/build"' in frontend


def test_old_production_module_paths_are_absent():
    expected = {
        "project_change_memory.py",
        "project_change_pipeline.py",
        "project_evidence_models.py",
        "project_evidence_input.py",
        "project_evidence_normalizer.py",
        "project_evidence_synthesizer.py",
        "project_evidence_scoring.py",
        "project_capability_taxonomy.py",
        "project_capability_extractor.py",
        "project_claim_boundaries.py",
        "project_evidence_memory.py",
        "project_evidence_pipeline.py",
    }
    assert expected <= {path.name for path in (ROOT / "backend").glob("*.py")}
    assert not [path for path in (ROOT / "backend").glob("*.py") if NUMBERED_PHASE.search(path.name)]
