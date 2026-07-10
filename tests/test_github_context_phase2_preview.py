import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402
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


class GitHubContextPhase2PreviewTests(unittest.TestCase):
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

    def write_raw_source(self, project_id, repo, raw_text="sample raw content", path="README.md"):
        return evidence_memory.upsert_github_raw_source(
            evidence_memory.make_github_raw_source(
                project_id=project_id,
                repo=repo,
                source_type="readme",
                path=path,
                commit_sha="abc123",
                raw_text=raw_text,
                metadata={"summary": f"README for {project_id}"},
            ),
            self.storage_dir,
        )

    def test_preview_route_returns_safe_items_and_filters(self):
        os.environ[self.flag_env] = "1"
        self.write_raw_source("WorkAgent", "owner/WorkAgent", "WorkAgent README raw text")
        self.write_raw_source(
            "Event-Lottery-System",
            "owner/Event-Lottery-System",
            "Lottery README raw text",
        )

        response = self.client.get("/api/github/context/phase2/preview?project_id=WorkAgent&limit=10")

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertTrue(payload["enabled"])
        self.assertEqual("phase2", payload["phase"])
        self.assertEqual("WorkAgent", payload["project_id"])
        self.assertEqual(1, payload["count"])
        self.assertEqual([], collect_keys(payload, "raw_text"))
        item = payload["items"][0]
        for key in [
            "source_id",
            "project_id",
            "repo",
            "source_type",
            "path",
            "raw_chars",
            "raw_available",
        ]:
            self.assertIn(key, item)
        self.assertEqual("WorkAgent", item["project_id"])
        self.assertGreater(item["raw_chars"], 0)
        self.assertTrue(item["raw_available"])

        missing = self.client.get("/api/github/context/phase2/preview?project_id=Missing&limit=10")
        self.assertEqual(200, missing.status_code)
        self.assertEqual([], missing.json()["items"])

    def test_preview_limit_works_and_is_clamped(self):
        os.environ[self.flag_env] = "1"
        for index in range(3):
            self.write_raw_source(
                "WorkAgent",
                "owner/WorkAgent",
                raw_text=f"raw {index}",
                path=f"file-{index}.md",
            )

        limited = self.client.get("/api/github/context/phase2/preview?limit=1")
        self.assertEqual(200, limited.status_code)
        self.assertEqual(1, limited.json()["limit"])
        self.assertEqual(1, len(limited.json()["items"]))

        clamped = self.client.get("/api/github/context/phase2/preview?limit=999")
        self.assertEqual(200, clamped.status_code)
        self.assertEqual(api_server.GITHUB_CONTEXT_PHASE2_PREVIEW_MAX_LIMIT, clamped.json()["limit"])
        self.assertLessEqual(len(clamped.json()["items"]), api_server.GITHUB_CONTEXT_PHASE2_PREVIEW_MAX_LIMIT)

    def test_preview_disabled_and_missing_storage_are_safe(self):
        os.environ[self.flag_env] = "0"
        disabled = self.client.get("/api/github/context/phase2/preview")
        self.assertEqual(200, disabled.status_code)
        self.assertFalse(disabled.json()["enabled"])
        self.assertEqual([], disabled.json()["items"])

        os.environ[self.flag_env] = "1"
        missing = self.client.get("/api/github/context/phase2/preview")
        self.assertEqual(200, missing.status_code)
        payload = missing.json()
        self.assertTrue(payload["enabled"])
        self.assertEqual([], payload["items"])
        self.assertFalse(self.storage_dir.exists())


if __name__ == "__main__":
    unittest.main()
