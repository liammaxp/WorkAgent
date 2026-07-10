import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402
import evidence_memory  # noqa: E402


PHASE2_COUNT_KEYS = [
    "raw_sources_count",
    "chunks_count",
    "raw_change_summaries_count",
    "evidence_cards_count",
    "capability_facts_count",
]


class Phase2EvidenceMemorySchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "phase2_memory"
        self.original_env = {
            api_server.GITHUB_CONTEXT_PHASE2_ENV: os.environ.get(api_server.GITHUB_CONTEXT_PHASE2_ENV),
            evidence_memory.PHASE2_EVIDENCE_MEMORY_DIR_ENV: os.environ.get(
                evidence_memory.PHASE2_EVIDENCE_MEMORY_DIR_ENV
            ),
        }
        os.environ[evidence_memory.PHASE2_EVIDENCE_MEMORY_DIR_ENV] = str(self.storage_dir)

    def tearDown(self):
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temp_dir.cleanup()

    def assert_zero_counts(self, counts):
        for key in PHASE2_COUNT_KEYS:
            self.assertEqual(0, counts[key])

    def write_one_of_each(self, project_id="project-a"):
        raw = evidence_memory.upsert_github_raw_source(
            evidence_memory.make_github_raw_source(
                project_id=project_id,
                repo="owner/repo",
                source_type="commit_patch",
                path="backend/api.py",
                commit_sha="abc123",
                raw_text="diff --git a/backend/api.py b/backend/api.py",
            ),
            self.storage_dir,
        )
        chunk = evidence_memory.upsert_evidence_chunk(
            evidence_memory.make_evidence_chunk(
                source_id=raw["source_id"],
                project_id=project_id,
                repo="owner/repo",
                path="backend/api.py",
                symbol="github_context_phase2_status",
                chunk_type="endpoint",
                text="def github_context_phase2_status(): return status",
                summary="Read-only status endpoint.",
                keywords=["status"],
                technical_tags=["FastAPI"],
                start_line=10,
                end_line=12,
            ),
            self.storage_dir,
        )
        summary = evidence_memory.upsert_raw_change_summary(
            evidence_memory.make_raw_change_summary(
                project_id=project_id,
                source_chunk_ids=[chunk["chunk_id"]],
                files_changed=["backend/api.py"],
                symbols_changed=["github_context_phase2_status"],
                raw_change_type=["api"],
                what_changed="Added a read-only Phase 2 status endpoint.",
                direct_code_evidence=["GET /api/github/context/phase2/status"],
                uncertain_intent=["Future storage integration is not implemented yet."],
            ),
            self.storage_dir,
        )
        card = evidence_memory.upsert_evidence_card(
            evidence_memory.make_evidence_card(
                project_id=project_id,
                source_chunk_ids=[chunk["chunk_id"]],
                problem="Expose safe status before storage exists.",
                mechanism="Return placeholder counts from JSONL helpers.",
                implementation_details=["No GitHub sync", "No Chroma access"],
                safe_impact="Allows Phase 2 work to remain feature-gated.",
                resume_angle="Built safe feature-flagged backend foundations.",
                confidence="medium",
                metric_support="none",
                allowed_claims=["Implemented feature-gated status reporting."],
                forbidden_claims=["Implemented full evidence retrieval."],
            ),
            self.storage_dir,
        )
        capability = evidence_memory.upsert_capability_fact(
            evidence_memory.make_capability_fact(
                project_id=project_id,
                capability_type="safe_backend_foundation",
                present=True,
                confidence="medium",
                mechanisms=["feature flag", "JSONL storage"],
                source_evidence_ids=[card["evidence_id"]],
                allowed_resume_claims=["Designed guarded backend storage foundations."],
                forbidden_claims=["Designed hybrid retrieval."],
                metric_support="none",
            ),
            self.storage_dir,
        )
        return raw, chunk, summary, card, capability

    def test_missing_storage_files_are_safe(self):
        for record_type in evidence_memory.RECORD_FILES:
            self.assertEqual([], evidence_memory.read_records(record_type, self.storage_dir))
            self.assertEqual(0, evidence_memory.count_records(record_type, storage_dir=self.storage_dir))

        self.assert_zero_counts(evidence_memory.get_phase2_memory_counts(storage_dir=self.storage_dir))
        self.assertFalse(self.storage_dir.exists())

    def test_github_raw_source_write_read_upsert(self):
        raw = evidence_memory.make_github_raw_source(
            project_id="project-a",
            repo="owner/repo",
            source_type="commit_patch",
            path="src/app.py",
            commit_sha="abc123",
            raw_text="sample diff",
            metadata={"version": 1},
        )
        evidence_memory.upsert_github_raw_source(raw, self.storage_dir)
        records = evidence_memory.read_records(evidence_memory.GITHUB_RAW_SOURCES, self.storage_dir)

        self.assertEqual(1, len(records))
        for key in [
            "source_id",
            "project_id",
            "repo",
            "source_type",
            "path",
            "commit_sha",
            "raw_text",
            "raw_hash",
            "created_at",
            "metadata",
        ]:
            self.assertIn(key, records[0])
        self.assertEqual(evidence_memory.stable_hash("sample diff"), records[0]["raw_hash"])

        duplicate = dict(raw)
        duplicate["metadata"] = {"version": 2}
        evidence_memory.upsert_github_raw_source(duplicate, self.storage_dir)
        records = evidence_memory.read_records(evidence_memory.GITHUB_RAW_SOURCES, self.storage_dir)
        self.assertEqual(1, len(records))
        self.assertEqual({"version": 2}, records[0]["metadata"])

    def test_evidence_chunk_write_read_upsert(self):
        raw, chunk, _, _, _ = self.write_one_of_each()
        records = evidence_memory.read_records_by_project(
            evidence_memory.EVIDENCE_CHUNKS,
            "project-a",
            self.storage_dir,
        )

        self.assertEqual(1, len(records))
        self.assertEqual(raw["source_id"], records[0]["source_id"])
        self.assertEqual("project-a", records[0]["project_id"])
        self.assertEqual("backend/api.py", records[0]["path"])
        self.assertEqual("endpoint", records[0]["chunk_type"])
        self.assertTrue(records[0]["hash"])

        duplicate = dict(chunk)
        duplicate["summary"] = "Updated summary."
        evidence_memory.upsert_evidence_chunk(duplicate, self.storage_dir)
        records = evidence_memory.read_records(evidence_memory.EVIDENCE_CHUNKS, self.storage_dir)
        self.assertEqual(1, len(records))
        self.assertEqual("Updated summary.", records[0]["summary"])

    def test_raw_change_summary_write_read_upsert(self):
        _, chunk, summary, _, _ = self.write_one_of_each()
        records = evidence_memory.read_records(evidence_memory.RAW_CHANGE_SUMMARIES, self.storage_dir)

        self.assertEqual(1, len(records))
        self.assertEqual([chunk["chunk_id"]], records[0]["source_chunk_ids"])
        self.assertEqual(["GET /api/github/context/phase2/status"], records[0]["direct_code_evidence"])

        duplicate = dict(summary)
        duplicate["what_changed"] = "Updated wording only."
        evidence_memory.upsert_raw_change_summary(duplicate, self.storage_dir)
        records = evidence_memory.read_records(evidence_memory.RAW_CHANGE_SUMMARIES, self.storage_dir)
        self.assertEqual(1, len(records))
        self.assertEqual("Updated wording only.", records[0]["what_changed"])

    def test_evidence_card_write_read_upsert(self):
        _, _, _, card, _ = self.write_one_of_each()
        records = evidence_memory.read_records(evidence_memory.EVIDENCE_CARDS, self.storage_dir)

        self.assertEqual(1, len(records))
        for key in ["problem", "mechanism", "implementation_details", "allowed_claims", "forbidden_claims"]:
            self.assertIn(key, records[0])
        self.assertIsInstance(records[0]["allowed_claims"], list)
        self.assertIsInstance(records[0]["forbidden_claims"], list)

        duplicate = dict(card)
        duplicate["allowed_claims"] = ["Updated allowed claim."]
        evidence_memory.upsert_evidence_card(duplicate, self.storage_dir)
        records = evidence_memory.read_records(evidence_memory.EVIDENCE_CARDS, self.storage_dir)
        self.assertEqual(1, len(records))
        self.assertEqual(["Updated allowed claim."], records[0]["allowed_claims"])

    def test_capability_fact_write_read_upsert(self):
        _, _, _, _, capability = self.write_one_of_each()
        records = evidence_memory.read_records(evidence_memory.CAPABILITY_FACTS, self.storage_dir)

        self.assertEqual(1, len(records))
        for key in [
            "project_id",
            "capability_type",
            "mechanisms",
            "allowed_resume_claims",
            "forbidden_claims",
        ]:
            self.assertIn(key, records[0])

        duplicate = evidence_memory.make_capability_fact(
            project_id=capability["project_id"],
            capability_type=capability["capability_type"],
            capability_id="different-manual-id",
            mechanisms=["upsert merge"],
            allowed_resume_claims=["Updated claim."],
            forbidden_claims=["Still no retrieval claim."],
        )
        evidence_memory.upsert_capability_fact(duplicate, self.storage_dir)
        records = evidence_memory.read_records(evidence_memory.CAPABILITY_FACTS, self.storage_dir)
        self.assertEqual(1, len(records))
        self.assertIn("feature flag", records[0]["mechanisms"])
        self.assertIn("upsert merge", records[0]["mechanisms"])

    def test_project_filtering(self):
        self.write_one_of_each(project_id="project-a")
        evidence_memory.upsert_github_raw_source(
            evidence_memory.make_github_raw_source(
                project_id="project-b",
                repo="owner/other",
                source_type="readme",
                path="README.md",
                raw_text="other project",
            ),
            self.storage_dir,
        )

        records = evidence_memory.read_records_by_project(
            evidence_memory.GITHUB_RAW_SOURCES,
            "project-a",
            self.storage_dir,
        )
        self.assertEqual(1, len(records))
        self.assertEqual({"project-a"}, {record["project_id"] for record in records})

    def test_counts_global_and_by_project(self):
        self.write_one_of_each(project_id="project-a")
        evidence_memory.upsert_github_raw_source(
            evidence_memory.make_github_raw_source(
                project_id="project-b",
                repo="owner/other",
                source_type="readme",
                path="README.md",
                raw_text="other project",
            ),
            self.storage_dir,
        )

        global_counts = evidence_memory.get_phase2_memory_counts(storage_dir=self.storage_dir)
        self.assertEqual(2, global_counts["raw_sources_count"])
        self.assertEqual(1, global_counts["chunks_count"])
        self.assertEqual(1, global_counts["raw_change_summaries_count"])
        self.assertEqual(1, global_counts["evidence_cards_count"])
        self.assertEqual(1, global_counts["capability_facts_count"])

        project_counts = evidence_memory.get_phase2_memory_counts(
            project_id="project-a",
            storage_dir=self.storage_dir,
        )
        self.assertEqual(1, project_counts["raw_sources_count"])
        self.assertEqual(1, project_counts["chunks_count"])

        other_counts = evidence_memory.get_phase2_memory_counts(
            project_id="project-b",
            storage_dir=self.storage_dir,
        )
        self.assertEqual(1, other_counts["raw_sources_count"])
        self.assertEqual(0, other_counts["chunks_count"])

    def test_status_integration(self):
        os.environ[api_server.GITHUB_CONTEXT_PHASE2_ENV] = "1"
        empty_status = api_server.get_github_context_phase2_status()
        self.assertTrue(empty_status["enabled"])
        self.assertFalse(empty_status["available"])
        self.assert_zero_counts(empty_status)

        self.write_one_of_each(project_id="project-a")
        populated_status = api_server.get_github_context_phase2_status()
        self.assertTrue(populated_status["enabled"])
        self.assertTrue(populated_status["available"])
        self.assertEqual(1, populated_status["raw_sources_count"])
        self.assertEqual(1, populated_status["chunks_count"])
        self.assertEqual(1, populated_status["raw_change_summaries_count"])
        self.assertEqual(1, populated_status["evidence_cards_count"])
        self.assertEqual(1, populated_status["capability_facts_count"])

        os.environ[api_server.GITHUB_CONTEXT_PHASE2_ENV] = "0"
        original_counter = api_server.evidence_memory.get_phase2_memory_counts

        def fail_if_read(*args, **kwargs):
            raise AssertionError("disabled Phase 2 status must not read storage")

        try:
            api_server.evidence_memory.get_phase2_memory_counts = fail_if_read
            disabled_status = api_server.get_github_context_phase2_status()
        finally:
            api_server.evidence_memory.get_phase2_memory_counts = original_counter

        self.assertFalse(disabled_status["enabled"])
        self.assertFalse(disabled_status["available"])
        self.assert_zero_counts(disabled_status)


if __name__ == "__main__":
    unittest.main()
