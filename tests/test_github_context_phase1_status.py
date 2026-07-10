import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402


FORBIDDEN_RAW_KEYS = {
    "raw_text",
    "patch",
    "content",
    "full_content",
    "document",
    "documents",
    "raw_json",
    "context",
}


def collect_forbidden_keys(value):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_RAW_KEYS:
                found.append(key)
            found.extend(collect_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(collect_forbidden_keys(item))
    return found


class GitHubContextPhase1StatusTests(unittest.TestCase):
    def test_status_helper_returns_safe_dict(self):
        status = api_server.get_github_context_status_v2()
        self.assertIsInstance(status, dict)
        for key in [
            "enabled",
            "saved",
            "last_sync_at",
            "repo_count",
            "record_count",
            "raw_chars",
            "indexed_count",
            "sources",
            "projects",
            "errors",
        ]:
            self.assertIn(key, status)
        self.assertIsInstance(status["sources"], dict)
        self.assertIsInstance(status["projects"], list)
        self.assertEqual([], collect_forbidden_keys(status))

    def test_missing_optional_files_do_not_crash_status_helper(self):
        original_project_memory_path = api_server.agent.PROJECT_MEMORY_PATH
        original_compact_facts_path = api_server.PROJECT_COMPACT_FACTS_PATH
        original_scan_state_path = api_server.agent.GITHUB_REPO_SCAN_STATE_PATH

        with tempfile.TemporaryDirectory() as temp_dir:
            missing_root = Path(temp_dir)
            try:
                api_server.agent.PROJECT_MEMORY_PATH = missing_root / "missing_project_memory.json"
                api_server.PROJECT_COMPACT_FACTS_PATH = missing_root / "missing_project_compact_facts.json"
                api_server.agent.GITHUB_REPO_SCAN_STATE_PATH = missing_root / "missing_scan_state.json"
                status = api_server.get_github_context_status_v2()
            finally:
                api_server.agent.PROJECT_MEMORY_PATH = original_project_memory_path
                api_server.PROJECT_COMPACT_FACTS_PATH = original_compact_facts_path
                api_server.agent.GITHUB_REPO_SCAN_STATE_PATH = original_scan_state_path

        self.assertIsInstance(status, dict)
        self.assertFalse(status["sources"]["project_memory"]["exists"])
        self.assertFalse(status["sources"]["project_compact_facts"]["exists"])
        self.assertFalse(status["sources"]["github_scan_state"]["exists"])
        self.assertEqual([], collect_forbidden_keys(status))

    def test_disabled_status_route_returns_safe_response(self):
        original_flag = api_server.USE_GITHUB_CONTEXT_STATUS_V2
        try:
            api_server.USE_GITHUB_CONTEXT_STATUS_V2 = False
            response = TestClient(api_server.app).get("/api/github/context/status")
        finally:
            api_server.USE_GITHUB_CONTEXT_STATUS_V2 = original_flag

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["enabled"])
        self.assertFalse(payload["saved"])
        self.assertIn("disabled", payload["message"])
        self.assertEqual([], collect_forbidden_keys(payload))

    def test_enabled_status_route_returns_summary_without_raw_content(self):
        original_flag = api_server.USE_GITHUB_CONTEXT_STATUS_V2
        try:
            api_server.USE_GITHUB_CONTEXT_STATUS_V2 = True
            response = TestClient(api_server.app).get("/api/github/context/status")
        finally:
            api_server.USE_GITHUB_CONTEXT_STATUS_V2 = original_flag

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["enabled"])
        self.assertIn("sources", payload)
        self.assertIn("projects", payload)
        self.assertEqual([], collect_forbidden_keys(payload))


if __name__ == "__main__":
    unittest.main()
