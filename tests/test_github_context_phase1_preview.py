import sys
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


class GitHubContextPhase1PreviewTests(unittest.TestCase):
    def test_preview_helper_returns_safe_dict(self):
        preview = api_server.get_github_context_preview_v2(limit=5)
        self.assertIsInstance(preview, dict)
        for key in ["enabled", "project_id", "limit", "count", "items", "errors"]:
            self.assertIn(key, preview)
        self.assertTrue(preview["enabled"])
        self.assertLessEqual(preview["limit"], api_server.GITHUB_CONTEXT_PREVIEW_MAX_LIMIT)
        self.assertEqual([], collect_forbidden_keys(preview))
        for item in preview["items"]:
            self.assertIn("source_id", item)
            self.assertLessEqual(len(item.get("summary") or ""), api_server.GITHUB_CONTEXT_PREVIEW_SUMMARY_CHARS)
            self.assertLessEqual(len(item.get("preview_text") or ""), api_server.GITHUB_CONTEXT_PREVIEW_TEXT_CHARS)

    def test_preview_route_disabled_response_is_safe(self):
        original_flag = api_server.USE_GITHUB_CONTEXT_STATUS_V2
        try:
            api_server.USE_GITHUB_CONTEXT_STATUS_V2 = False
            response = TestClient(api_server.app).get("/api/github/context/preview?limit=5")
        finally:
            api_server.USE_GITHUB_CONTEXT_STATUS_V2 = original_flag

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["enabled"])
        self.assertEqual([], payload["items"])
        self.assertIn("disabled", payload["message"])
        self.assertEqual([], collect_forbidden_keys(payload))

    def test_preview_route_enabled_response_is_safe_and_caps_limit(self):
        original_flag = api_server.USE_GITHUB_CONTEXT_STATUS_V2
        try:
            api_server.USE_GITHUB_CONTEXT_STATUS_V2 = True
            response = TestClient(api_server.app).get("/api/github/context/preview?limit=999")
        finally:
            api_server.USE_GITHUB_CONTEXT_STATUS_V2 = original_flag

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["enabled"])
        self.assertEqual(api_server.GITHUB_CONTEXT_PREVIEW_MAX_LIMIT, payload["limit"])
        self.assertLessEqual(payload["count"], api_server.GITHUB_CONTEXT_PREVIEW_MAX_LIMIT)
        self.assertEqual([], collect_forbidden_keys(payload))
        for item in payload["items"]:
            self.assertLessEqual(len(item.get("preview_text") or ""), api_server.GITHUB_CONTEXT_PREVIEW_TEXT_CHARS)

    def test_preview_route_default_and_invalid_limits_are_safe(self):
        original_flag = api_server.USE_GITHUB_CONTEXT_STATUS_V2
        try:
            api_server.USE_GITHUB_CONTEXT_STATUS_V2 = True
            default_response = TestClient(api_server.app).get("/api/github/context/preview")
            invalid_response = TestClient(api_server.app).get("/api/github/context/preview?limit=not-a-number")
            negative_response = TestClient(api_server.app).get("/api/github/context/preview?limit=-5")
        finally:
            api_server.USE_GITHUB_CONTEXT_STATUS_V2 = original_flag

        for response in [default_response, invalid_response, negative_response]:
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(api_server.GITHUB_CONTEXT_PREVIEW_DEFAULT_LIMIT, payload["limit"])
            self.assertLessEqual(payload["count"], api_server.GITHUB_CONTEXT_PREVIEW_DEFAULT_LIMIT)
            self.assertEqual([], collect_forbidden_keys(payload))

    def test_unknown_project_id_returns_empty_without_crashing(self):
        original_flag = api_server.USE_GITHUB_CONTEXT_STATUS_V2
        try:
            api_server.USE_GITHUB_CONTEXT_STATUS_V2 = True
            response = TestClient(api_server.app).get(
                "/api/github/context/preview?project_id=definitely-not-a-real-project-id-for-preview&limit=5"
            )
        finally:
            api_server.USE_GITHUB_CONTEXT_STATUS_V2 = original_flag

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["enabled"])
        self.assertEqual("definitely-not-a-real-project-id-for-preview", payload["project_id"])
        self.assertEqual([], payload["items"])
        self.assertEqual(0, payload["count"])
        self.assertEqual([], collect_forbidden_keys(payload))


if __name__ == "__main__":
    unittest.main()
