from pathlib import Path, PurePosixPath
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def is_ignored(path: str) -> bool:
    return git("check-ignore", "--no-index", "-q", "--", path, check=False).returncode == 0


def tracked_paths() -> list[str]:
    return git("ls-files").stdout.splitlines()


def test_representative_private_and_generated_paths_are_ignored():
    paths = (
        ".env",
        "nested/.env.local",
        "config.local.json",
        "credentials.json",
        "auth/client_secret_workagent.json",
        "auth/service_account_local.json",
        "auth/token_store.json",
        "keys/id_ed25519",
        "keys/signing.pem",
        "repo/.aws/credentials",
        "uploads/resume.pdf",
        "user_data/profile.json",
        "applications/acme/notes.txt",
        "outputs/generated_resume.pdf",
        "exports/application.csv",
        "information/project_evidence_memory.json",
        "cache/project_change_memory.json",
        "github_evidence_memory/evidence_cards.jsonl",
        "github_raw_sources.jsonl",
        "runtime/local.sqlite3",
        "data/cache.db-wal",
        "chroma/index.bin",
        "logs/backend.log",
        "crash-reports/request.txt",
        "models/local.gguf",
        "frontend/node_modules/package/index.js",
    )
    assert all(is_ignored(path) for path in paths)


def test_public_templates_documentation_source_and_synthetic_fixtures_remain_trackable():
    paths = (
        ".env.example",
        "deploy/.env.sample",
        "config/.env.template",
        "docs/architecture.pdf",
        "docs/example.docx",
        "backend/secret_detection.py",
        "backend/credential_policy.py",
        "tests/fixtures/synthetic_credentials_case.json",
        "tests/fixtures/sample.sqlite3",
        "background/prompt.example.txt",
    )
    assert not any(is_ignored(path) for path in paths)


def test_no_tracked_file_is_also_ignored():
    ignored_tracked = git("ls-files", "-ci", "--exclude-standard").stdout.splitlines()
    assert ignored_tracked == []


def test_no_tracked_path_matches_high_risk_private_filename_policy():
    allowlist = {
        "background/prompt.example.txt",
    }
    private_directories = {
        "uploads",
        "attachments",
        "user_files",
        "user_data",
        "private",
        "personal",
        "application_materials",
        "applications",
        "resumes",
        "cover_letters",
        "generated_resumes",
        "information",
        "outputs",
        "exports",
        "runtime",
        "state",
    }
    risky_basename = re.compile(
        r"(?i)^(?:\.env(?:\..+)?|credentials(?:\..+)?\.json|client_secret.*\.json|"
        r"service[-_]account.*\.json|oauth_credentials.*\.json|token_store.*\.json|"
        r"auth_state.*\.json|cookies.*\.txt|session.*\.json|id_(?:rsa|dsa|ecdsa|ed25519)(?:\..+)?|"
        r".+\.(?:pem|key|p12|pfx|jks|keystore|db|sqlite|sqlite3|duckdb|rdb|log))$"
    )
    violations: list[str] = []
    for tracked in tracked_paths():
        normalized = PurePosixPath(tracked)
        if tracked in allowlist or tracked.endswith((".env.example", ".env.sample", ".env.template")):
            continue
        if private_directories.intersection(normalized.parts) or risky_basename.fullmatch(normalized.name):
            violations.append(tracked)
    assert violations == []
