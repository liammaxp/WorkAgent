import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402
import evidence_chunker  # noqa: E402
import evidence_memory  # noqa: E402


def collect_keys(value, key_name):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == key_name:
                found.append(key)
            found.extend(collect_keys(item, key_name))
    elif isinstance(value, list):
        for item in value:
            found.extend(collect_keys(item, key_name))
    return found


class Phase2EvidenceChunkerTests(unittest.TestCase):
    def setUp(self):
        self.flag_env = api_server.GITHUB_CONTEXT_PHASE2_ENV
        self.storage_env = evidence_memory.PHASE2_EVIDENCE_MEMORY_DIR_ENV
        self.original_flag = os.environ.get(self.flag_env)
        self.original_storage = os.environ.get(self.storage_env)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "phase2_memory"
        os.environ[self.storage_env] = str(self.storage_dir)
        self.client = TestClient(api_server.app)

    def tearDown(self):
        if self.original_flag is None:
            os.environ.pop(self.flag_env, None)
        else:
            os.environ[self.flag_env] = self.original_flag
        if self.original_storage is None:
            os.environ.pop(self.storage_env, None)
        else:
            os.environ[self.storage_env] = self.original_storage
        self.temp_dir.cleanup()

    def write_raw_source(
        self,
        *,
        project_id="WorkAgent",
        repo="owner/WorkAgent",
        source_type="unknown",
        path="",
        raw_text="",
        commit_sha="abc123",
    ):
        return evidence_memory.upsert_github_raw_source(
            evidence_memory.make_github_raw_source(
                project_id=project_id,
                repo=repo,
                source_type=source_type,
                path=path,
                commit_sha=commit_sha,
                raw_text=raw_text,
            )
        )

    def diff_text(self):
        return """diff --git a/backend/api_server.py b/backend/api_server.py
--- a/backend/api_server.py
+++ b/backend/api_server.py
@@ -10,6 +10,16 @@ def merge_staged_resume():
+    bullet_depth_profile = build_bullet_depth_profile(final_bullets)
+    validate_latex_output(final_bullets)
+    quality_gate = run_latex_validation()
+    return quality_gate
"""

    def test_disabled_mode_does_not_chunk(self):
        os.environ[self.flag_env] = "0"
        self.write_raw_source(source_type="commit_patch", path="backend/api_server.py", raw_text=self.diff_text())

        result = api_server.chunk_phase2_raw_sources()

        self.assertFalse(result["enabled"])
        self.assertEqual(0, result["chunks_count"])
        self.assertFalse(evidence_memory.get_record_path(evidence_memory.EVIDENCE_CHUNKS).exists())

    def test_direct_builder_disabled_mode_does_not_chunk(self):
        os.environ[self.flag_env] = "0"
        self.write_raw_source(source_type="commit_patch", path="backend/api_server.py", raw_text=self.diff_text())

        result = evidence_chunker.chunk_phase2_raw_sources()

        self.assertFalse(result["enabled"])
        self.assertEqual(0, result["chunks_count"])
        self.assertFalse(evidence_memory.get_record_path(evidence_memory.EVIDENCE_CHUNKS).exists())

    def test_missing_storage_is_safe(self):
        os.environ[self.flag_env] = "1"

        result = api_server.chunk_phase2_raw_sources()

        self.assertTrue(result["enabled"])
        self.assertEqual(0, result["processed_raw_sources"])
        self.assertEqual(0, result["chunks_count"])
        self.assertFalse(self.storage_dir.exists())

    def test_diff_hunk_chunking(self):
        os.environ[self.flag_env] = "1"
        self.write_raw_source(source_type="commit_patch", path="backend/api_server.py", raw_text=self.diff_text())

        result = api_server.chunk_phase2_raw_sources()

        self.assertGreater(result["chunks_count"], 0)
        chunks = evidence_memory.read_records(evidence_memory.EVIDENCE_CHUNKS)
        self.assertEqual({"diff_hunk"}, {chunk["chunk_type"] for chunk in chunks})
        chunk = chunks[0]
        self.assertLessEqual(len(chunk["text"]), evidence_chunker.PHASE2_CHUNK_MAX_CHARS)
        self.assertTrue(
            {"bullet_depth_profile", "final_bullets", "LaTeX", "validation"}.intersection(chunk["keywords"])
        )
        self.assertTrue({"validation", "latex"}.intersection(chunk["technical_tags"]))

    def test_readme_chunking(self):
        os.environ[self.flag_env] = "1"
        readme = """# Overview
WorkAgent stores GitHub evidence and project_memory facts for backend validation.

## Usage
Run tests before changing resume generation.
"""
        self.write_raw_source(source_type="readme", path="README.md", raw_text=readme)

        api_server.chunk_phase2_raw_sources()

        chunks = evidence_memory.read_records(evidence_memory.EVIDENCE_CHUNKS)
        self.assertGreater(len(chunks), 0)
        self.assertEqual({"readme_section"}, {chunk["chunk_type"] for chunk in chunks})
        self.assertIn("Overview", chunks[0]["summary"])

    def test_unknown_large_source_uses_safe_windows(self):
        os.environ[self.flag_env] = "1"
        raw_text = "plain serialized context with storage fallback evidence. " * 220
        self.write_raw_source(source_type="unknown", raw_text=raw_text)

        api_server.chunk_phase2_raw_sources()

        chunks = evidence_memory.read_records(evidence_memory.EVIDENCE_CHUNKS)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk["text"]) <= evidence_chunker.PHASE2_CHUNK_MAX_CHARS for chunk in chunks))
        self.assertEqual({"WorkAgent"}, {chunk["project_id"] for chunk in chunks})
        self.assertTrue(all(chunk["source_id"] for chunk in chunks))

    def test_idempotency(self):
        os.environ[self.flag_env] = "1"
        self.write_raw_source(source_type="commit_patch", path="backend/api_server.py", raw_text=self.diff_text())

        first = api_server.chunk_phase2_raw_sources()
        first_chunks = evidence_memory.read_records(evidence_memory.EVIDENCE_CHUNKS)
        first_ids = {chunk["chunk_id"] for chunk in first_chunks}
        second = api_server.chunk_phase2_raw_sources()
        second_chunks = evidence_memory.read_records(evidence_memory.EVIDENCE_CHUNKS)

        self.assertEqual(first["chunks_count"], second["chunks_count"])
        self.assertEqual(first_ids, {chunk["chunk_id"] for chunk in second_chunks})

    def test_project_filtering(self):
        os.environ[self.flag_env] = "1"
        self.write_raw_source(project_id="WorkAgent", source_type="commit_patch", path="backend/api_server.py", raw_text=self.diff_text())
        self.write_raw_source(
            project_id="Event-Lottery-System",
            repo="owner/Event-Lottery-System",
            source_type="readme",
            path="README.md",
            raw_text="# Event Lottery\nSmall readme without chunking target.",
        )

        result = api_server.chunk_phase2_raw_sources(project_id="WorkAgent")

        self.assertGreater(result["chunks_count"], 0)
        chunks = evidence_memory.read_records(evidence_memory.EVIDENCE_CHUNKS)
        self.assertEqual({"WorkAgent"}, {chunk["project_id"] for chunk in chunks})

    def test_chunk_route_and_preview(self):
        os.environ[self.flag_env] = "1"
        self.write_raw_source(source_type="commit_patch", path="backend/api_server.py", raw_text=self.diff_text())

        chunk_response = self.client.post("/api/github/context/phase2/chunk?project_id=WorkAgent")

        self.assertEqual(200, chunk_response.status_code)
        self.assertTrue(chunk_response.json()["enabled"])
        self.assertGreater(chunk_response.json()["created_or_updated_chunks"], 0)

        preview_response = self.client.get("/api/github/context/phase2/chunks/preview?project_id=WorkAgent&limit=10")
        self.assertEqual(200, preview_response.status_code)
        payload = preview_response.json()
        self.assertTrue(payload["enabled"])
        self.assertGreater(payload["count"], 0)
        self.assertEqual([], collect_keys(payload, "text"))
        item = payload["items"][0]
        for key in [
            "chunk_id",
            "source_id",
            "project_id",
            "repo",
            "path",
            "symbol",
            "chunk_type",
            "summary",
            "keywords",
            "technical_tags",
            "text_chars",
            "raw_available",
        ]:
            self.assertIn(key, item)

        missing = self.client.get("/api/github/context/phase2/chunks/preview?project_id=Missing&limit=10")
        self.assertEqual([], missing.json()["items"])

        clamped = self.client.get("/api/github/context/phase2/chunks/preview?limit=999")
        self.assertEqual(api_server.GITHUB_CONTEXT_PHASE2_PREVIEW_MAX_LIMIT, clamped.json()["limit"])

    def test_status_count_integration(self):
        os.environ[self.flag_env] = "1"
        self.write_raw_source(source_type="commit_patch", path="backend/api_server.py", raw_text=self.diff_text())
        api_server.chunk_phase2_raw_sources()

        status = self.client.get("/api/github/context/phase2/status").json()

        self.assertGreater(status["chunks_count"], 0)
        self.assertEqual([], collect_keys(status, "raw_text"))
        self.assertEqual([], collect_keys(status, "text"))
        self.assertGreater(status["projects"][0]["chunks"], 0)


if __name__ == "__main__":
    unittest.main()
