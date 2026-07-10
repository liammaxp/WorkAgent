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
import evidence_pipeline  # noqa: E402


UNSAFE_RESPONSE_KEYS = ["raw_text", "text", "chunk_text", "full_text", "raw", "patch", "content"]


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


class Phase2EvidencePipelineTests(unittest.TestCase):
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

    def pipeline_diff_text(self):
        return """diff --git a/backend/api_server.py b/backend/api_server.py
--- a/backend/api_server.py
+++ b/backend/api_server.py
@@ -10,6 +10,17 @@ def merge_staged_resume(final_bullets):
+    bullet_depth_profile = build_bullet_depth_profile(final_bullets)
+    final_bullets = merge_project_bullets(final_bullets)
+    latex_validation = run_latex_validation(final_bullets)
+    template_pollution_guard = validate_template_pollution(final_bullets)
+    # template pollution guard keeps generated content out of final_bullets
+    if template_pollution_guard.has_pollution:
+        raise ValueError("template pollution")
+    return latex_validation
"""

    def write_raw_source(
        self,
        *,
        project_id="WorkAgent",
        repo="owner/WorkAgent",
        source_type="commit_patch",
        path="backend/api_server.py",
        raw_text=None,
        commit_sha="abc123",
    ):
        return evidence_memory.upsert_github_raw_source(
            evidence_memory.make_github_raw_source(
                project_id=project_id,
                repo=repo,
                source_type=source_type,
                path=path,
                commit_sha=commit_sha,
                raw_text=raw_text if raw_text is not None else self.pipeline_diff_text(),
            )
        )

    def write_chunk(self, *, project_id="WorkAgent", chunk_id="chunk-workagent"):
        return evidence_memory.upsert_evidence_chunk(
            evidence_memory.make_evidence_chunk(
                chunk_id=chunk_id,
                source_id=f"source-{chunk_id}",
                project_id=project_id,
                repo=f"owner/{project_id}",
                path="backend/api_server.py",
                symbol="merge_staged_resume",
                chunk_type="diff_hunk",
                text="""@@ -1,3 +1,8 @@ def merge_staged_resume(final_bullets):
+    bullet_depth_profile = build_bullet_depth_profile(final_bullets)
+    final_bullets = merge_project_bullets(final_bullets)
+    latex_validation = run_latex_validation(final_bullets)
""",
                summary="Diff hunk touching merge and validation.",
                keywords=["bullet_depth_profile", "final_bullets", "LaTeX", "validation"],
                technical_tags=["validation", "latex", "resume_generation"],
            )
        )

    def write_evidence_card(self, *, project_id="WorkAgent", evidence_id="evidence-workagent"):
        return evidence_memory.upsert_evidence_card(
            evidence_memory.make_evidence_card(
                evidence_id=evidence_id,
                project_id=project_id,
                source_chunk_ids=[f"chunk-{evidence_id}"],
                problem="Phase 2 evidence needed source-traceable handling.",
                mechanism="Split raw GitHub context into bounded, source-traceable evidence chunks.",
                implementation_details=["Changed file: backend/evidence_chunker.py"],
                safe_impact="Improved source traceability by preserving evidence identifiers.",
                resume_angle="source_traceability",
                confidence="high",
                metric_support="none",
                allowed_claims=["converted raw GitHub context into source-traceable evidence chunks"],
                forbidden_claims=["do not claim ATS score improvement"],
            )
        )

    def test_disabled_mode_does_not_build(self):
        os.environ[self.flag_env] = "0"
        self.write_raw_source()

        result = evidence_pipeline.run_phase2_evidence_pipeline()

        self.assertFalse(result["enabled"])
        self.assertEqual([], result["ran_stages"])
        self.assertEqual({}, result["counts_before"])
        self.assertFalse(evidence_memory.get_record_path(evidence_memory.EVIDENCE_CHUNKS).exists())
        self.assertFalse(evidence_memory.get_record_path(evidence_memory.RAW_CHANGE_SUMMARIES).exists())
        self.assertFalse(evidence_memory.get_record_path(evidence_memory.EVIDENCE_CARDS).exists())
        self.assertFalse(evidence_memory.get_record_path(evidence_memory.CAPABILITY_FACTS).exists())

    def test_disabled_mutating_routes_do_not_write(self):
        os.environ[self.flag_env] = "0"
        self.write_raw_source()

        for route in [
            "/api/github/context/phase2/chunk",
            "/api/github/context/phase2/summarize-changes",
            "/api/github/context/phase2/build-evidence-cards",
            "/api/github/context/phase2/build-capability-facts",
            "/api/github/context/phase2/build",
        ]:
            with self.subTest(route=route):
                response = self.client.post(route)
                self.assertEqual(200, response.status_code)
                self.assertFalse(response.json()["enabled"])

        self.assertFalse(evidence_memory.get_record_path(evidence_memory.EVIDENCE_CHUNKS).exists())
        self.assertFalse(evidence_memory.get_record_path(evidence_memory.RAW_CHANGE_SUMMARIES).exists())
        self.assertFalse(evidence_memory.get_record_path(evidence_memory.EVIDENCE_CARDS).exists())
        self.assertFalse(evidence_memory.get_record_path(evidence_memory.CAPABILITY_FACTS).exists())

    def test_missing_raw_sources_safe(self):
        os.environ[self.flag_env] = "1"

        result = evidence_pipeline.run_phase2_evidence_pipeline()

        self.assertTrue(result["enabled"])
        self.assertEqual(0, result["counts_after"]["raw_sources_count"])
        self.assertEqual(0, result["counts_after"]["chunks_count"])
        self.assertEqual(0, result["counts_after"]["raw_change_summaries_count"])
        self.assertEqual(0, result["counts_after"]["evidence_cards_count"])
        self.assertEqual(0, result["counts_after"]["capability_facts_count"])
        self.assertEqual([], result["errors"])
        self.assertFalse(self.storage_dir.exists())

    def test_enabled_empty_routes_are_safe_and_read_only(self):
        os.environ[self.flag_env] = "1"

        status = self.client.get("/api/github/context/phase2/status")
        preview = self.client.get("/api/github/context/phase2/preview")
        health = self.client.get("/api/github/context/phase2/health")
        inspect = self.client.get("/api/github/context/phase2/inspect")
        build = self.client.post("/api/github/context/phase2/build")

        for response in [status, preview, health, inspect, build]:
            self.assertEqual(200, response.status_code)
        self.assertFalse(status.json()["available"])
        self.assertEqual([], preview.json()["items"])
        self.assertEqual("wait_for_raw_sources", health.json()["next_recommended_action"])
        self.assertTrue(all(not items for items in inspect.json()["samples"].values()))
        self.assertEqual(0, build.json()["counts_after"]["raw_sources_count"])
        self.assertEqual([], build.json()["errors"])
        self.assertFalse(self.storage_dir.exists())

    def test_full_pipeline_from_fake_raw_source(self):
        os.environ[self.flag_env] = "1"
        self.write_raw_source()

        result = evidence_pipeline.run_phase2_evidence_pipeline(project_id="WorkAgent")

        self.assertTrue(result["enabled"])
        self.assertEqual(evidence_pipeline.STAGE_ORDER, result["ran_stages"])
        self.assertGreater(result["counts_after"]["chunks_count"], 0)
        self.assertGreater(result["counts_after"]["raw_change_summaries_count"], 0)
        self.assertGreater(result["counts_after"]["evidence_cards_count"], 0)
        self.assertGreater(result["counts_after"]["capability_facts_count"], 0)
        self.assertEqual([], collect_keys(result, "raw_text"))
        self.assertEqual([], collect_keys(result, "text"))

        chunks = evidence_memory.read_records(evidence_memory.EVIDENCE_CHUNKS)
        evidence_cards = evidence_memory.read_records(evidence_memory.EVIDENCE_CARDS)
        capability_facts = evidence_memory.read_records(evidence_memory.CAPABILITY_FACTS)
        chunk_ids = {chunk["chunk_id"] for chunk in chunks}
        evidence_ids = {card["evidence_id"] for card in evidence_cards}
        self.assertTrue(all(card["source_chunk_ids"] for card in evidence_cards))
        self.assertTrue(all(set(card["source_chunk_ids"]).issubset(chunk_ids) for card in evidence_cards))
        self.assertTrue(all(fact["source_evidence_ids"] for fact in capability_facts))
        self.assertTrue(all(set(fact["source_evidence_ids"]).issubset(evidence_ids) for fact in capability_facts))

        safe_responses = [
            result,
            self.client.get("/api/github/context/phase2/status").json(),
            self.client.get("/api/github/context/phase2/preview").json(),
            self.client.get("/api/github/context/phase2/health").json(),
            self.client.get("/api/github/context/phase2/inspect").json(),
        ]
        for unsafe_key in UNSAFE_RESPONSE_KEYS:
            with self.subTest(unsafe_key=unsafe_key):
                self.assertTrue(all(not collect_keys(response, unsafe_key) for response in safe_responses))

        second = evidence_pipeline.run_phase2_evidence_pipeline(project_id="WorkAgent")
        self.assertEqual(result["counts_after"], second["counts_after"])

    def test_stage_subset(self):
        os.environ[self.flag_env] = "1"
        self.write_raw_source()

        chunk_result = evidence_pipeline.run_phase2_evidence_pipeline(stages=["chunk"])

        self.assertEqual(["chunk"], chunk_result["ran_stages"])
        self.assertGreater(chunk_result["counts_after"]["chunks_count"], 0)
        self.assertEqual(0, chunk_result["counts_after"]["raw_change_summaries_count"])

        summary_result = evidence_pipeline.run_phase2_evidence_pipeline(stages=["summarize_changes"])
        self.assertEqual(["summarize_changes"], summary_result["ran_stages"])
        self.assertGreater(summary_result["counts_after"]["raw_change_summaries_count"], 0)

    def test_idempotency(self):
        os.environ[self.flag_env] = "1"
        self.write_raw_source()

        first = evidence_pipeline.run_phase2_evidence_pipeline()
        second = evidence_pipeline.run_phase2_evidence_pipeline()

        self.assertEqual(first["counts_after"], second["counts_after"])

    def test_project_filtering(self):
        os.environ[self.flag_env] = "1"
        self.write_raw_source(project_id="WorkAgent", repo="owner/WorkAgent")
        self.write_raw_source(
            project_id="Event-Lottery-System",
            repo="owner/Event-Lottery-System",
            path="backend/lottery.py",
            commit_sha="def456",
        )

        result = evidence_pipeline.run_phase2_evidence_pipeline(project_id="WorkAgent")

        self.assertGreater(result["counts_after"]["chunks_count"], 0)
        self.assertEqual({"WorkAgent"}, {record["project_id"] for record in evidence_memory.read_records(evidence_memory.EVIDENCE_CHUNKS)})
        self.assertEqual({"WorkAgent"}, {record["project_id"] for record in evidence_memory.read_records(evidence_memory.RAW_CHANGE_SUMMARIES)})
        self.assertEqual({"WorkAgent"}, {record["project_id"] for record in evidence_memory.read_records(evidence_memory.EVIDENCE_CARDS)})
        self.assertEqual({"WorkAgent"}, {record["project_id"] for record in evidence_memory.read_records(evidence_memory.CAPABILITY_FACTS)})

    def test_inspect_route_and_helper(self):
        os.environ[self.flag_env] = "1"
        self.write_raw_source()
        evidence_pipeline.run_phase2_evidence_pipeline(project_id="WorkAgent")

        result = evidence_pipeline.get_phase2_project_inspect(project_id="WorkAgent", limit=1)

        self.assertTrue(result["enabled"])
        for sample_name in [
            "raw_sources",
            "chunks",
            "raw_change_summaries",
            "evidence_cards",
            "capability_facts",
        ]:
            self.assertEqual(1, len(result["samples"][sample_name]))
        self.assertEqual([], collect_keys(result, "raw_text"))
        self.assertEqual([], collect_keys(result, "text"))

        response = self.client.get("/api/github/context/phase2/inspect?project_id=WorkAgent&limit=1")
        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(1, len(payload["samples"]["raw_sources"]))
        self.assertEqual([], collect_keys(payload, "raw_text"))
        self.assertEqual([], collect_keys(payload, "text"))

        missing = self.client.get("/api/github/context/phase2/inspect?project_id=Missing&limit=10")
        self.assertEqual([], missing.json()["samples"]["raw_sources"])
        self.assertEqual([], missing.json()["samples"]["chunks"])
        self.assertEqual([], missing.json()["samples"]["raw_change_summaries"])
        self.assertEqual([], missing.json()["samples"]["evidence_cards"])
        self.assertEqual([], missing.json()["samples"]["capability_facts"])

        clamped = self.client.get("/api/github/context/phase2/inspect?limit=999")
        self.assertEqual(evidence_pipeline.MAX_LIMIT, clamped.json()["limit"])

    def test_health_route_and_helper(self):
        os.environ[self.flag_env] = "1"
        self.assertEqual("wait_for_raw_sources", evidence_pipeline.get_phase2_pipeline_health()["next_recommended_action"])

        self.write_raw_source()
        self.assertEqual("run_chunk", evidence_pipeline.get_phase2_pipeline_health(project_id="WorkAgent")["next_recommended_action"])

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ[self.storage_env] = str(Path(temp_dir) / "phase2_memory")
            self.write_chunk()
            self.assertEqual("summarize_changes", evidence_pipeline.get_phase2_pipeline_health()["next_recommended_action"])

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ[self.storage_env] = str(Path(temp_dir) / "phase2_memory")
            self.write_evidence_card()
            self.assertEqual("build_capability_facts", evidence_pipeline.get_phase2_pipeline_health()["next_recommended_action"])

        os.environ[self.storage_env] = str(self.storage_dir)
        evidence_pipeline.run_phase2_evidence_pipeline(project_id="WorkAgent")
        response = self.client.get("/api/github/context/phase2/health?project_id=WorkAgent")
        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertIn(payload["next_recommended_action"], {"inspect", "complete"})
        self.assertTrue(payload["pipeline_complete"])

    def test_invalid_stages(self):
        os.environ[self.flag_env] = "1"
        self.write_raw_source()

        result = evidence_pipeline.run_phase2_evidence_pipeline(stages=["not_a_stage"])

        self.assertFalse(result["ok"])
        self.assertEqual([], result["ran_stages"])
        self.assertEqual(0, result["counts_after"]["chunks_count"])
        self.assertFalse(evidence_memory.get_record_path(evidence_memory.EVIDENCE_CHUNKS).exists())

        response = self.client.post(
            "/api/github/context/phase2/build",
            json={"stages": ["not_a_stage"]},
        )
        self.assertEqual(400, response.status_code)
        self.assertEqual("invalid_stage", response.json()["detail"]["errors"][0]["type"])

    def test_build_route(self):
        os.environ[self.flag_env] = "1"
        self.write_raw_source()

        response = self.client.post(
            "/api/github/context/phase2/build",
            json={"project_id": "WorkAgent", "stages": ["build_capability_facts", "chunk", "summarize_changes", "build_evidence_cards"]},
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertTrue(payload["enabled"])
        self.assertEqual(evidence_pipeline.STAGE_ORDER, payload["ran_stages"])
        self.assertGreater(payload["counts_after"]["capability_facts_count"], 0)


if __name__ == "__main__":
    unittest.main()
