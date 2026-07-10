import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402
import evidence_change_summary  # noqa: E402
import evidence_memory  # noqa: E402


FORBIDDEN_PHRASES = [
    "reduced hallucinations",
    "eliminated hallucinations",
    "improved ATS",
    "guaranteed",
    "%",
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


class Phase2RawChangeSummaryTests(unittest.TestCase):
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

    def write_chunk(
        self,
        *,
        project_id="WorkAgent",
        chunk_id="chunk-workagent",
        path="backend/api_server.py",
        symbol="merge_staged_resume",
        chunk_type="diff_hunk",
        text=None,
        keywords=None,
        technical_tags=None,
    ):
        if text is None:
            text = """@@ -10,6 +10,14 @@ def merge_staged_resume():
+    bullet_depth_profile = build_bullet_depth_profile(final_bullets)
+    final_bullets = merge_project_bullets(final_bullets)
+    validate_latex_output(final_bullets)
+    template_pollution_guard = validate_template_pollution(final_bullets)
"""
        return evidence_memory.upsert_evidence_chunk(
            evidence_memory.make_evidence_chunk(
                chunk_id=chunk_id,
                source_id=f"source-{chunk_id}",
                project_id=project_id,
                repo=f"owner/{project_id}",
                path=path,
                symbol=symbol,
                chunk_type=chunk_type,
                text=text,
                summary="Diff hunk touching merge and validation.",
                keywords=keywords
                or ["bullet_depth_profile", "final_bullets", "LaTeX", "validation", "template pollution"],
                technical_tags=technical_tags or ["validation", "latex", "resume_generation"],
            )
        )

    def test_disabled_mode_does_not_extract(self):
        os.environ[self.flag_env] = "0"
        self.write_chunk()

        result = api_server.build_phase2_raw_change_summaries()

        self.assertFalse(result["enabled"])
        self.assertEqual(0, result["raw_change_summaries_count"])
        self.assertFalse(evidence_memory.get_record_path(evidence_memory.RAW_CHANGE_SUMMARIES).exists())

    def test_direct_builder_disabled_mode_does_not_extract(self):
        os.environ[self.flag_env] = "0"
        self.write_chunk()

        result = evidence_change_summary.build_phase2_raw_change_summaries()

        self.assertFalse(result["enabled"])
        self.assertEqual(0, result["raw_change_summaries_count"])
        self.assertFalse(evidence_memory.get_record_path(evidence_memory.RAW_CHANGE_SUMMARIES).exists())

    def test_missing_chunks_safe(self):
        os.environ[self.flag_env] = "1"

        result = api_server.build_phase2_raw_change_summaries()

        self.assertTrue(result["enabled"])
        self.assertEqual(0, result["processed_chunks"])
        self.assertEqual(0, result["raw_change_summaries_count"])
        self.assertFalse(self.storage_dir.exists())

    def test_diff_hunk_summary_extraction(self):
        os.environ[self.flag_env] = "1"
        chunk = self.write_chunk()

        result = api_server.build_phase2_raw_change_summaries()

        self.assertGreater(result["raw_change_summaries_count"], 0)
        summaries = evidence_memory.read_records(evidence_memory.RAW_CHANGE_SUMMARIES)
        self.assertEqual(1, len(summaries))
        summary = summaries[0]
        self.assertIn("backend/api_server.py", summary["files_changed"])
        self.assertIn(chunk["chunk_id"], summary["source_chunk_ids"])
        self.assertTrue(
            {"merge_logic_update", "validation_rule_update"}.intersection(summary["raw_change_type"])
        )
        self.assertIn("Updated", summary["what_changed"])
        self.assertNotIn("reduced hallucinations", summary["what_changed"].lower())
        self.assertTrue(summary["direct_code_evidence"])

    def test_no_high_level_hallucination_inference(self):
        os.environ[self.flag_env] = "1"
        self.write_chunk(
            chunk_id="chunk-unsupported",
            symbol="validate_retrieval",
            text="""@@ -1,3 +1,7 @@ def validate_retrieval():
+    fallback = validate_evidence_retrieval()
+    # improved ATS score by 40% and reduced hallucinations
+    guaranteed = False
""",
            keywords=["retrieval", "evidence", "validation", "fallback"],
            technical_tags=["retrieval", "validation"],
        )

        api_server.build_phase2_raw_change_summaries()

        summary = evidence_memory.read_records(evidence_memory.RAW_CHANGE_SUMMARIES)[0]
        combined = " ".join(
            [summary["what_changed"], *summary["direct_code_evidence"], *summary["uncertain_intent"]]
        ).lower()
        for phrase in FORBIDDEN_PHRASES:
            self.assertNotIn(phrase.lower(), combined)
        self.assertNotIn("%", combined)

    def test_symbol_and_file_preservation(self):
        os.environ[self.flag_env] = "1"
        self.write_chunk(path="backend/evidence_memory.py", symbol="upsert_evidence_chunk")

        api_server.build_phase2_raw_change_summaries()

        summary = evidence_memory.read_records(evidence_memory.RAW_CHANGE_SUMMARIES)[0]
        self.assertEqual(["backend/evidence_memory.py"], summary["files_changed"])
        self.assertEqual(["upsert_evidence_chunk"], summary["symbols_changed"])

    def test_idempotency(self):
        os.environ[self.flag_env] = "1"
        self.write_chunk()

        api_server.build_phase2_raw_change_summaries()
        first = evidence_memory.read_records(evidence_memory.RAW_CHANGE_SUMMARIES)
        api_server.build_phase2_raw_change_summaries()
        second = evidence_memory.read_records(evidence_memory.RAW_CHANGE_SUMMARIES)

        self.assertEqual(1, len(first))
        self.assertEqual(1, len(second))
        self.assertEqual(first[0]["change_id"], second[0]["change_id"])

    def test_project_filtering(self):
        os.environ[self.flag_env] = "1"
        self.write_chunk(project_id="WorkAgent", chunk_id="chunk-workagent")
        self.write_chunk(
            project_id="Event-Lottery-System",
            chunk_id="chunk-lottery",
            path="backend/lottery.py",
            symbol="draw_lottery",
        )

        result = api_server.build_phase2_raw_change_summaries(project_id="WorkAgent")

        self.assertEqual(1, result["raw_change_summaries_count"])
        summaries = evidence_memory.read_records(evidence_memory.RAW_CHANGE_SUMMARIES)
        self.assertEqual({"WorkAgent"}, {summary["project_id"] for summary in summaries})

    def test_preview_route(self):
        os.environ[self.flag_env] = "1"
        self.write_chunk(project_id="WorkAgent", chunk_id="chunk-workagent")
        self.write_chunk(
            project_id="Event-Lottery-System",
            chunk_id="chunk-lottery",
            path="backend/lottery.py",
        )
        self.client.post("/api/github/context/phase2/summarize-changes?project_id=WorkAgent")

        response = self.client.get("/api/github/context/phase2/change-summaries/preview?project_id=WorkAgent&limit=10")

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertTrue(payload["enabled"])
        self.assertEqual(1, payload["count"])
        self.assertEqual([], collect_keys(payload, "text"))
        item = payload["items"][0]
        for key in [
            "change_id",
            "project_id",
            "source_chunk_ids",
            "files_changed",
            "symbols_changed",
            "raw_change_type",
            "what_changed",
            "direct_code_evidence",
            "uncertain_intent",
            "metadata",
        ]:
            self.assertIn(key, item)
        self.assertEqual("WorkAgent", item["project_id"])

        missing = self.client.get("/api/github/context/phase2/change-summaries/preview?project_id=Missing")
        self.assertEqual([], missing.json()["items"])

        clamped = self.client.get("/api/github/context/phase2/change-summaries/preview?limit=999")
        self.assertEqual(api_server.GITHUB_CONTEXT_PHASE2_PREVIEW_MAX_LIMIT, clamped.json()["limit"])

    def test_status_count_integration(self):
        os.environ[self.flag_env] = "1"
        self.write_chunk()
        api_server.build_phase2_raw_change_summaries()

        status = self.client.get("/api/github/context/phase2/status").json()

        self.assertGreater(status["raw_change_summaries_count"], 0)
        self.assertEqual([], collect_keys(status, "direct_code_evidence"))
        self.assertGreater(status["projects"][0]["raw_change_summaries"], 0)


if __name__ == "__main__":
    unittest.main()
