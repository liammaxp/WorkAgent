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


COUNT_KEYS = [
    "raw_sources_count",
    "chunks_count",
    "raw_change_summaries_count",
    "evidence_cards_count",
    "capability_facts_count",
]


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


class ExplodingMemoryStore:
    def __getattr__(self, name):
        raise AssertionError(f"Phase 2 placeholder status must not read memory store: {name}")


class GitHubContextPhase2StatusTests(unittest.TestCase):
    def setUp(self):
        self.env_name = api_server.GITHUB_CONTEXT_PHASE2_ENV
        self.storage_env_name = evidence_memory.PHASE2_EVIDENCE_MEMORY_DIR_ENV
        self.original_present = self.env_name in os.environ
        self.original_value = os.environ.get(self.env_name)
        self.original_storage_present = self.storage_env_name in os.environ
        self.original_storage_value = os.environ.get(self.storage_env_name)
        self.original_memory_store = api_server.agent.MEMORY_STORE
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ[self.storage_env_name] = str(Path(self.temp_dir.name) / "phase2_memory")
        self.client = TestClient(api_server.app)

    def tearDown(self):
        api_server.agent.MEMORY_STORE = self.original_memory_store
        if self.original_present:
            os.environ[self.env_name] = self.original_value or ""
        else:
            os.environ.pop(self.env_name, None)
        if self.original_storage_present:
            os.environ[self.storage_env_name] = self.original_storage_value or ""
        else:
            os.environ.pop(self.storage_env_name, None)
        self.temp_dir.cleanup()

    def assert_zero_counts(self, status):
        for key in COUNT_KEYS:
            self.assertIn(key, status)
            self.assertEqual(0, status[key])

    def test_status_when_disabled(self):
        os.environ[self.env_name] = "0"
        status = api_server.get_github_context_phase2_status()

        self.assertFalse(status["enabled"])
        self.assertFalse(status["available"])
        self.assertEqual("phase2", status["phase"])
        self.assertIn("disabled", status["message"])
        self.assert_zero_counts(status)

    def test_status_when_enabled_but_storage_not_initialized(self):
        os.environ[self.env_name] = "1"
        status = api_server.get_github_context_phase2_status()

        self.assertTrue(status["enabled"])
        self.assertFalse(status["available"])
        self.assertEqual("phase2", status["phase"])
        self.assertIn("storage is not initialized yet", status["message"])
        self.assert_zero_counts(status)

    def test_status_route_when_disabled(self):
        os.environ[self.env_name] = "0"
        api_server.agent.MEMORY_STORE = ExplodingMemoryStore()

        response = self.client.get("/api/github/context/phase2/status")

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertFalse(payload["enabled"])
        self.assertFalse(payload["available"])
        self.assertEqual("phase2", payload["phase"])
        self.assert_zero_counts(payload)

    def test_status_route_when_enabled_without_storage(self):
        os.environ[self.env_name] = "1"
        api_server.agent.MEMORY_STORE = ExplodingMemoryStore()

        response = self.client.get("/api/github/context/phase2/status")

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertTrue(payload["enabled"])
        self.assertFalse(payload["available"])
        self.assertEqual("phase2", payload["phase"])
        self.assertIn("storage is not initialized yet", payload["message"])
        self.assert_zero_counts(payload)

    def test_status_reads_raw_counts_without_raw_text(self):
        os.environ[self.env_name] = "1"
        evidence_memory.upsert_github_raw_source(
            evidence_memory.make_github_raw_source(
                project_id="WorkAgent",
                repo="owner/WorkAgent",
                source_type="readme",
                path="README.md",
                commit_sha="abc123",
                raw_text="WorkAgent raw evidence text",
            )
        )

        response = self.client.get("/api/github/context/phase2/status")

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertTrue(payload["enabled"])
        self.assertTrue(payload["available"])
        self.assertGreater(payload["raw_sources_count"], 0)
        self.assertGreater(payload["raw_chars"], 0)
        self.assertGreater(payload["repos_count"], 0)
        self.assertEqual([], collect_keys(payload, "raw_text"))
        self.assertEqual(1, len(payload["projects"]))
        project = payload["projects"][0]
        self.assertEqual("WorkAgent", project["project_id"])
        self.assertEqual("owner/WorkAgent", project["repo"])
        self.assertEqual(1, project["raw_sources"])
        self.assertGreater(project["raw_chars"], 0)


if __name__ == "__main__":
    unittest.main()
