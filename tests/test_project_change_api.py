import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402
import project_change_memory as project_change  # noqa: E402
import project_change_pipeline  # noqa: E402


FORBIDDEN_RESPONSE_KEYS = {
    "patch_text",
    "hunk_text",
    "added_lines",
    "removed_lines",
    "raw_text",
    "content",
    "token",
    "credential",
}


def collect_forbidden_keys(value):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_RESPONSE_KEYS:
                found.append(key)
            found.extend(collect_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(collect_forbidden_keys(item))
    return found


def build_result(status="completed"):
    return project_change_pipeline.ProjectChangePipelineResult(
        enabled=True,
        status=status,
        source_context_count=2,
        raw_diff_input_count=3,
        diff_unit_count=4,
        raw_change_summary_count=4,
        evidence_card_candidate_count=4,
        qualified_evidence_card_count=2,
        capability_fact_count=1,
        persisted_project_count=1,
        skipped_source_count=1,
        skipped_sources=[
            {
                "project_id": "ProjectA",
                "repo": "owner/project-a",
                "file_path": "backend/example.py",
                "commit_sha": "abc123",
                "reason": "missing_patch_text",
            }
        ],
        project_summaries=[
            {
                "project_id": "ProjectA",
                "raw_diff_input_count": 3,
                "diff_unit_count": 4,
                "raw_change_summary_count": 4,
                "evidence_card_candidate_count": 4,
                "qualified_evidence_card_count": 2,
                "capability_fact_count": 1,
                "capability_types": ["validation_and_repair"],
            }
        ],
        memory_path="information/project_change_memory.json",
        errors=[],
    )


class ProjectChangeApiTests(unittest.TestCase):
    def setUp(self):
        self.flag_env = project_change.PROJECT_CHANGE_MEMORY_ENV
        self.original_flag = os.environ.get(self.flag_env)
        self.client = TestClient(api_server.app)

    def tearDown(self):
        if self.original_flag is None:
            os.environ.pop(self.flag_env, None)
        else:
            os.environ[self.flag_env] = self.original_flag

    def test_build_route_disabled_does_not_process_sources(self):
        os.environ[self.flag_env] = "0"

        with patch.object(
            api_server.project_change_pipeline,
            "load_saved_github_contexts_for_project_change_memory",
            side_effect=AssertionError("disabled route should not load saved GitHub context"),
        ):
            response = self.client.post("/api/github/change-memory/build")

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertFalse(payload["ok"])
        self.assertFalse(payload["enabled"])
        self.assertEqual("disabled", payload["status"])
        self.assertIn("disabled", payload["message"])
        self.assertEqual(0, payload["raw_diff_input_count"])
        self.assertEqual([], collect_forbidden_keys(payload))

    def test_build_route_success_returns_safe_summary(self):
        os.environ[self.flag_env] = "1"

        with patch.object(
            api_server.project_change_pipeline,
            "run_project_change_memory_pipeline",
            return_value=build_result(),
        ) as run_pipeline:
            response = self.client.post("/api/github/change-memory/build")

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["enabled"])
        self.assertEqual("completed", payload["status"])
        self.assertEqual(3, payload["raw_diff_input_count"])
        self.assertEqual(1, payload["capability_fact_count"])
        self.assertEqual("missing_patch_text", payload["skipped_sources"][0]["reason"])
        self.assertEqual([], collect_forbidden_keys(payload))
        run_pipeline.assert_called_once_with()

    def test_build_route_failed_result_is_not_reported_as_ok(self):
        os.environ[self.flag_env] = "1"

        with patch.object(
            api_server.project_change_pipeline,
            "run_project_change_memory_pipeline",
            return_value=build_result(status="failed"),
        ):
            response = self.client.post("/api/github/change-memory/build")

        payload = response.json()
        self.assertEqual(200, response.status_code)
        self.assertFalse(payload["ok"])
        self.assertEqual("failed", payload["status"])

    def test_inspect_route_passes_project_and_clamps_limits_through_helper(self):
        calls = []

        def fake_inspect(project_id=None, sample_limit=project_change_pipeline.DEFAULT_INSPECT_SAMPLE_LIMIT):
            calls.append((project_id, sample_limit))
            return {
                "enabled": True,
                "project_id": project_id,
                "sample_limit": project_change_pipeline.safe_sample_limit(sample_limit),
                "raw_change_summary_count": 0,
                "evidence_card_count": 0,
                "capability_fact_count": 0,
                "capability_types": [],
                "sample_raw_change_summaries": [],
                "sample_evidence_cards": [],
                "sample_capability_facts": [],
                "errors": [],
            }

        with patch.object(api_server.project_change_pipeline, "inspect_project_change_memory", side_effect=fake_inspect):
            negative = self.client.get("/api/github/change-memory/inspect?project_id=ProjectA&sample_limit=-3")
            invalid = self.client.get("/api/github/change-memory/inspect?sample_limit=bad")
            oversized = self.client.get("/api/github/change-memory/inspect?sample_limit=999")

        self.assertEqual(200, negative.status_code)
        self.assertEqual("ProjectA", calls[0][0])
        self.assertEqual("-3", calls[0][1])
        self.assertEqual(0, negative.json()["sample_limit"])
        self.assertEqual(project_change_pipeline.DEFAULT_INSPECT_SAMPLE_LIMIT, invalid.json()["sample_limit"])
        self.assertEqual(project_change_pipeline.MAX_INSPECT_SAMPLE_LIMIT, oversized.json()["sample_limit"])
        self.assertEqual([], collect_forbidden_keys(negative.json()))

    def test_health_route_returns_structured_safe_states(self):
        states = [
            {
                "enabled": False,
                "status": "disabled",
                "schema_version": None,
                "memory_exists": False,
                "memory_readable": False,
                "updated_at": None,
                "project_count": 0,
                "raw_change_summary_count": 0,
                "evidence_card_count": 0,
                "capability_fact_count": 0,
                "issues": [],
            },
            {
                "enabled": True,
                "status": "empty",
                "schema_version": "project_change_memory.v1",
                "memory_exists": False,
                "memory_readable": True,
                "updated_at": None,
                "project_count": 0,
                "raw_change_summary_count": 0,
                "evidence_card_count": 0,
                "capability_fact_count": 0,
                "issues": [],
            },
            {
                "enabled": True,
                "status": "ready",
                "schema_version": "project_change_memory.v1",
                "memory_exists": True,
                "memory_readable": True,
                "updated_at": "2026-07-10T00:00:00Z",
                "project_count": 1,
                "raw_change_summary_count": 2,
                "evidence_card_count": 1,
                "capability_fact_count": 1,
                "issues": [],
            },
            {
                "enabled": True,
                "status": "error",
                "schema_version": None,
                "memory_exists": True,
                "memory_readable": False,
                "updated_at": None,
                "project_count": 0,
                "raw_change_summary_count": 0,
                "evidence_card_count": 0,
                "capability_fact_count": 0,
                "issues": ["Invalid project change memory project memory JSON"],
            },
        ]

        for state in states:
            with self.subTest(status=state["status"]):
                with patch.object(api_server.project_change_pipeline, "get_project_change_health", return_value=state):
                    response = self.client.get("/api/github/change-memory/health")

                payload = response.json()
                self.assertEqual(200, response.status_code)
                self.assertEqual(state["status"], payload["status"])
                self.assertIn("enabled", payload)
                self.assertIn("issues", payload)
                self.assertEqual([], collect_forbidden_keys(payload))

    def test_project_change_routes_are_registered(self):
        paths = {route.path for route in api_server.app.routes}

        self.assertIn("/api/github/change-memory/build", paths)
        self.assertIn("/api/github/change-memory/inspect", paths)
        self.assertIn("/api/github/change-memory/health", paths)


if __name__ == "__main__":
    unittest.main()
