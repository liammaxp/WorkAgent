import json
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402


FORBIDDEN_STATUS_PREVIEW_KEYS = {
    "raw_text",
    "patch",
    "content",
    "full_content",
    "document",
    "documents",
    "raw_json",
    "context",
}
FORBIDDEN_RAW_KEYS = {"patch", "content", "full_content", "document", "documents", "raw_json", "context"}


def collect_forbidden_keys(value, forbidden):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in forbidden:
                found.append(key)
            found.extend(collect_forbidden_keys(item, forbidden))
    elif isinstance(value, list):
        for item in value:
            found.extend(collect_forbidden_keys(item, forbidden))
    return found


class ExplodingMemoryStore:
    def github_metadata_status(self):
        raise AssertionError("disabled routes must not read Chroma metadata")

    def github_preview_metadata(self, limit=5):
        raise AssertionError("disabled routes must not read Chroma preview metadata")

    def read_github_document(self, record_id):
        raise AssertionError("disabled routes must not read Chroma documents")


class FakeMemoryStore:
    def __init__(self):
        self.record_id = "fake-chroma-record"
        self.repository = "owner/chroma-e2e"
        self.updated_at = "2026-07-07T01:02:03"
        self.document = "Fake Chroma GitHub evidence\n" + ("raw chroma evidence " * 2000)

    def github_metadata_status(self):
        return {
            "available": True,
            "count": 1,
            "repositories": [{"repository": self.repository, "updated_at": self.updated_at}],
        }

    def github_preview_metadata(self, limit=5):
        return [
            {
                "id": self.record_id,
                "repository": self.repository,
                "updated_at": self.updated_at,
                "source": "fake-chroma",
            }
        ][:limit]

    def read_github_document(self, record_id):
        if record_id != self.record_id:
            return None
        return {
            "id": self.record_id,
            "document": self.document,
            "metadata": {"repository": self.repository, "updated_at": self.updated_at},
        }


class GitHubContextPhase1EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.original_flag = api_server.USE_GITHUB_CONTEXT_STATUS_V2
        self.original_memory_store = api_server.agent.MEMORY_STORE
        self.original_project_memory_path = api_server.agent.PROJECT_MEMORY_PATH
        self.original_compact_path = api_server.PROJECT_COMPACT_FACTS_PATH
        self.original_scan_state_path = api_server.agent.GITHUB_REPO_SCAN_STATE_PATH
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_root = Path(self.temp_dir.name)
        self.project_memory_path = temp_root / "project_memory.json"
        self.compact_path = temp_root / "project_compact_facts.json"
        self.scan_state_path = temp_root / "github_repo_scan_state.json"
        api_server.agent.PROJECT_MEMORY_PATH = self.project_memory_path
        api_server.PROJECT_COMPACT_FACTS_PATH = self.compact_path
        api_server.agent.GITHUB_REPO_SCAN_STATE_PATH = self.scan_state_path

        self.project_memory_path.write_text(
            json.dumps(
                {
                    "projects": [
                        {
                            "project_id": "pm-e2e",
                            "project_name": "Project Memory E2E",
                            "repository": "owner/pm-e2e",
                            "identity": {
                                "core_problem": "Validate bounded project memory raw inspect.",
                                "core_value": "Shows safe diagnostics.",
                            },
                            "confirmed_features": ["status", "preview", "raw inspect"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.compact_path.write_text(
            json.dumps(
                {
                    "compact-e2e-record": {
                        "id": "compact-e2e-record",
                        "project_id": "compact-e2e",
                        "project_name": "Compact Facts E2E",
                        "repo_name": "owner/compact-e2e",
                        "compact_facts_json": {
                            "summary": "Validate compact facts raw inspect.",
                            "resumeRelevantClaims": ["bounded preview", "bounded raw"],
                            "largeField": "compact raw " * 3000,
                        },
                        "updated_at": "2026-07-07T00:00:00",
                    }
                }
            ),
            encoding="utf-8",
        )
        self.scan_state_path.write_text("{}", encoding="utf-8")
        self.client = TestClient(api_server.app)

    def tearDown(self):
        api_server.USE_GITHUB_CONTEXT_STATUS_V2 = self.original_flag
        api_server.agent.MEMORY_STORE = self.original_memory_store
        api_server.agent.PROJECT_MEMORY_PATH = self.original_project_memory_path
        api_server.PROJECT_COMPACT_FACTS_PATH = self.original_compact_path
        api_server.agent.GITHUB_REPO_SCAN_STATE_PATH = self.original_scan_state_path
        self.temp_dir.cleanup()

    def test_disabled_routes_are_safe_and_do_not_touch_memory_store(self):
        api_server.USE_GITHUB_CONTEXT_STATUS_V2 = False
        api_server.agent.MEMORY_STORE = ExplodingMemoryStore()

        status = self.client.get("/api/github/context/status")
        preview = self.client.get("/api/github/context/preview?limit=5")
        raw = self.client.get("/api/github/context/raw?source_id=anything&max_chars=1000")

        self.assertEqual(status.status_code, 200)
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(raw.status_code, 200)
        self.assertFalse(status.json()["enabled"])
        self.assertFalse(preview.json()["enabled"])
        self.assertFalse(raw.json()["enabled"])
        self.assertEqual("", raw.json()["raw_text"])

    def assert_preview_raw_roundtrip(self, project_id, expected_source_type):
        api_server.USE_GITHUB_CONTEXT_STATUS_V2 = True
        preview_response = self.client.get(
            f"/api/github/context/preview?project_id={project_id}&limit=5"
        )
        self.assertEqual(preview_response.status_code, 200)
        preview_payload = preview_response.json()
        self.assertEqual([], collect_forbidden_keys(preview_payload, FORBIDDEN_STATUS_PREVIEW_KEYS))
        self.assertGreaterEqual(preview_payload["count"], 1)

        item = preview_payload["items"][0]
        self.assertEqual(expected_source_type, item["source_type"])
        source_id = item["source_id"]
        raw_response = self.client.get(
            f"/api/github/context/raw?source_id={source_id}&max_chars=1000"
        )
        self.assertEqual(raw_response.status_code, 200)
        raw_payload = raw_response.json()
        self.assertEqual(source_id, raw_payload["source_id"])
        self.assertEqual(expected_source_type, raw_payload["source_type"])
        self.assertLessEqual(raw_payload["returned_chars"], 1000)
        self.assertLessEqual(len(raw_payload["raw_text"]), 1000)
        self.assertEqual([], collect_forbidden_keys(raw_payload, FORBIDDEN_RAW_KEYS))

    def test_project_memory_preview_to_raw_roundtrip(self):
        api_server.agent.MEMORY_STORE = FakeMemoryStore()
        self.assert_preview_raw_roundtrip("pm-e2e", "project_memory")

    def test_project_compact_facts_preview_to_raw_roundtrip(self):
        api_server.agent.MEMORY_STORE = FakeMemoryStore()
        self.assert_preview_raw_roundtrip("compact-e2e", "project_compact_facts")

    def test_chroma_preview_to_raw_roundtrip_with_fake_store(self):
        api_server.agent.MEMORY_STORE = FakeMemoryStore()
        self.assert_preview_raw_roundtrip("chroma-e2e", "chroma_github_evidence")


if __name__ == "__main__":
    unittest.main()
