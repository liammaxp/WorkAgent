import json
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402


FORBIDDEN_PREVIEW_KEYS = {
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


class GitHubContextPhase1RawTests(unittest.TestCase):
    def setUp(self):
        self.original_flag = api_server.USE_GITHUB_CONTEXT_STATUS_V2
        self.original_compact_path = api_server.PROJECT_COMPACT_FACTS_PATH
        self.temp_dir = tempfile.TemporaryDirectory()
        self.compact_path = Path(self.temp_dir.name) / "project_compact_facts.json"
        self.record_key = "raw-test-record"
        self.project_id = "raw-test-project"
        self.repo = "owner/raw-test-repo"
        self.source_id = api_server.preview_source_id(
            "project_compact_facts",
            self.record_key,
            self.project_id,
            self.repo,
        )
        payload = {
            self.record_key: {
                "id": self.record_key,
                "project_id": self.project_id,
                "project_name": "Raw Test Project",
                "repo_name": self.repo,
                "source_hash": "raw-test-source-hash",
                "compact_facts_json": {
                    "summary": "Raw inspect test record",
                    "resumeRelevantClaims": ["bounded raw inspect"],
                    "largeField": "x" * (api_server.GITHUB_CONTEXT_RAW_MAX_CHARS + 5000),
                },
                "updated_at": "2026-07-07T00:00:00",
            }
        }
        self.compact_path.write_text(json.dumps(payload), encoding="utf-8")
        api_server.PROJECT_COMPACT_FACTS_PATH = self.compact_path

    def tearDown(self):
        api_server.USE_GITHUB_CONTEXT_STATUS_V2 = self.original_flag
        api_server.PROJECT_COMPACT_FACTS_PATH = self.original_compact_path
        self.temp_dir.cleanup()

    def test_raw_helper_returns_bounded_dict(self):
        result = api_server.get_github_context_raw_v2(self.source_id, max_chars=1000)
        self.assertIsInstance(result, dict)
        self.assertTrue(result["enabled"])
        self.assertEqual(self.source_id, result["source_id"])
        self.assertEqual(1000, result["max_chars"])
        self.assertLessEqual(result["returned_chars"], 1000)
        self.assertLessEqual(len(result["raw_text"]), 1000)
        self.assertIn("raw_text", result)
        self.assertEqual([], collect_forbidden_keys(result, FORBIDDEN_RAW_KEYS))

    def test_raw_route_disabled_response_is_safe(self):
        api_server.USE_GITHUB_CONTEXT_STATUS_V2 = False
        response = TestClient(api_server.app).get(f"/api/github/context/raw?source_id={self.source_id}")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["enabled"])
        self.assertEqual("", payload["raw_text"])
        self.assertEqual([], collect_forbidden_keys(payload, FORBIDDEN_RAW_KEYS))

    def test_raw_route_missing_source_id_does_not_crash(self):
        api_server.USE_GITHUB_CONTEXT_STATUS_V2 = True
        response = TestClient(api_server.app).get("/api/github/context/raw")
        self.assertEqual(response.status_code, 400)
        payload = response.json()["detail"]
        self.assertEqual("source_id is required", payload["message"])
        self.assertEqual([], collect_forbidden_keys(payload, FORBIDDEN_RAW_KEYS))

    def test_raw_route_unknown_source_id_does_not_crash(self):
        api_server.USE_GITHUB_CONTEXT_STATUS_V2 = True
        response = TestClient(api_server.app).get("/api/github/context/raw?source_id=unknown-source-id")
        self.assertEqual(response.status_code, 404)
        payload = response.json()["detail"]
        self.assertEqual("source_id not found", payload["message"])
        self.assertEqual([], collect_forbidden_keys(payload, FORBIDDEN_RAW_KEYS))

    def test_raw_route_caps_max_chars(self):
        api_server.USE_GITHUB_CONTEXT_STATUS_V2 = True
        response = TestClient(api_server.app).get(
            f"/api/github/context/raw?source_id={self.source_id}&max_chars=999999"
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(api_server.GITHUB_CONTEXT_RAW_MAX_CHARS, payload["max_chars"])
        self.assertLessEqual(payload["returned_chars"], api_server.GITHUB_CONTEXT_RAW_MAX_CHARS)
        self.assertLessEqual(len(payload["raw_text"]), api_server.GITHUB_CONTEXT_RAW_MAX_CHARS)
        self.assertTrue(payload["truncated"])
        self.assertEqual([], collect_forbidden_keys(payload, FORBIDDEN_RAW_KEYS))

    def test_raw_route_default_min_and_invalid_max_chars_are_safe(self):
        api_server.USE_GITHUB_CONTEXT_STATUS_V2 = True
        client = TestClient(api_server.app)
        default_response = client.get(f"/api/github/context/raw?source_id={self.source_id}")
        min_response = client.get(f"/api/github/context/raw?source_id={self.source_id}&max_chars=1")
        invalid_response = client.get(f"/api/github/context/raw?source_id={self.source_id}&max_chars=not-a-number")

        self.assertEqual(default_response.status_code, 200)
        default_payload = default_response.json()
        self.assertEqual(api_server.GITHUB_CONTEXT_RAW_DEFAULT_CHARS, default_payload["max_chars"])
        self.assertLessEqual(len(default_payload["raw_text"]), default_payload["max_chars"])

        self.assertEqual(min_response.status_code, 200)
        min_payload = min_response.json()
        self.assertEqual(api_server.GITHUB_CONTEXT_RAW_MIN_CHARS, min_payload["max_chars"])
        self.assertLessEqual(len(min_payload["raw_text"]), min_payload["max_chars"])

        self.assertEqual(invalid_response.status_code, 200)
        invalid_payload = invalid_response.json()
        self.assertEqual(api_server.GITHUB_CONTEXT_RAW_DEFAULT_CHARS, invalid_payload["max_chars"])
        self.assertLessEqual(len(invalid_payload["raw_text"]), invalid_payload["max_chars"])

        for payload in [default_payload, min_payload, invalid_payload]:
            self.assertEqual([], collect_forbidden_keys(payload, FORBIDDEN_RAW_KEYS))

    def test_preview_route_still_does_not_include_raw_text(self):
        api_server.USE_GITHUB_CONTEXT_STATUS_V2 = True
        response = TestClient(api_server.app).get("/api/github/context/preview?limit=5")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual([], collect_forbidden_keys(payload, FORBIDDEN_PREVIEW_KEYS))


if __name__ == "__main__":
    unittest.main()
