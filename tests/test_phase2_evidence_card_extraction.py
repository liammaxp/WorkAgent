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


FORBIDDEN_TEXT = [
    "reduced hallucinations",
    "eliminated hallucinations",
    "improved ATS score",
    "interview success",
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


class Phase2EvidenceCardExtractionTests(unittest.TestCase):
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

    def write_chunk(self, chunk_id="chunk-workagent", project_id="WorkAgent", path="backend/api_server.py"):
        return evidence_memory.upsert_evidence_chunk(
            evidence_memory.make_evidence_chunk(
                chunk_id=chunk_id,
                source_id=f"source-{chunk_id}",
                project_id=project_id,
                repo=f"owner/{project_id}",
                path=path,
                symbol="merge_staged_resume",
                chunk_type="diff_hunk",
                text="bullet_depth_profile final_bullets validation JSONL upsert stable IDs",
                summary="Diff hunk touching merge and validation.",
                keywords=["bullet_depth_profile", "final_bullets", "validation", "JSONL", "upsert"],
                technical_tags=["resume_generation", "validation", "storage"],
            )
        )

    def write_summary(
        self,
        *,
        project_id="WorkAgent",
        change_id="change-workagent",
        source_chunk_ids=None,
        files_changed=None,
        symbols_changed=None,
        raw_change_type=None,
        what_changed="Updated merge_staged_resume in backend/api_server.py for merge_logic_update.",
        direct_code_evidence=None,
        metadata=None,
    ):
        if source_chunk_ids is None:
            source_chunk_ids = ["chunk-workagent"]
        if files_changed is None:
            files_changed = ["backend/api_server.py"]
        if symbols_changed is None:
            symbols_changed = ["merge_staged_resume"]
        if raw_change_type is None:
            raw_change_type = ["merge_logic_update"]
        if direct_code_evidence is None:
            direct_code_evidence = [
                "Preserved bullet_depth_profile before final_bullets.",
                "Updated merge_project_bullets ordering.",
            ]
        return evidence_memory.upsert_raw_change_summary(
            evidence_memory.make_raw_change_summary(
                change_id=change_id,
                project_id=project_id,
                source_chunk_ids=source_chunk_ids,
                files_changed=files_changed,
                symbols_changed=symbols_changed,
                raw_change_type=raw_change_type,
                what_changed=what_changed,
                direct_code_evidence=direct_code_evidence,
                uncertain_intent=[],
                metadata=metadata or {},
            )
        )

    def test_disabled_mode_does_not_build_cards(self):
        os.environ[self.flag_env] = "0"
        self.write_chunk()
        self.write_summary()

        result = api_server.build_phase2_evidence_cards()

        self.assertFalse(result["enabled"])
        self.assertEqual(0, result["evidence_cards_count"])
        self.assertFalse(evidence_memory.get_record_path(evidence_memory.EVIDENCE_CARDS).exists())

    def test_missing_summaries_safe(self):
        os.environ[self.flag_env] = "1"

        result = api_server.build_phase2_evidence_cards()

        self.assertTrue(result["enabled"])
        self.assertEqual(0, result["processed_summaries"])
        self.assertEqual(0, result["evidence_cards_count"])
        self.assertFalse(self.storage_dir.exists())

    def test_merge_logic_evidence_card(self):
        os.environ[self.flag_env] = "1"
        self.write_chunk()
        self.write_summary()

        result = api_server.build_phase2_evidence_cards()

        self.assertGreater(result["evidence_cards_count"], 0)
        card = evidence_memory.read_records(evidence_memory.EVIDENCE_CARDS)[0]
        self.assertIn("displace", card["problem"])
        self.assertTrue("merge logic" in card["mechanism"] or "mechanism-rich evidence" in card["mechanism"])
        self.assertIn("Improved", card["safe_impact"])
        self.assertEqual("none", card["metric_support"])
        self.assertTrue(card["allowed_claims"])
        self.assertTrue(all("%" not in claim for claim in card["allowed_claims"]))

    def test_validation_evidence_card(self):
        os.environ[self.flag_env] = "1"
        self.write_chunk()
        self.write_summary(
            change_id="change-validation",
            raw_change_type=["validation_rule_update"],
            what_changed="Updated validate_template_pollution in backend/api_server.py for validation_rule_update.",
            direct_code_evidence=["Added template pollution guard around final_bullets."],
        )

        api_server.build_phase2_evidence_cards()

        card = evidence_memory.read_records(evidence_memory.EVIDENCE_CARDS)[0]
        self.assertEqual("validation_and_guardrails", card["resume_angle"])
        self.assertTrue(any("hallucination" in claim for claim in card["forbidden_claims"]))
        self.assertTrue(any("percentages" in claim for claim in card["forbidden_claims"]))

    def test_storage_schema_evidence_card(self):
        os.environ[self.flag_env] = "1"
        self.write_chunk(path="backend/evidence_memory.py")
        self.write_summary(
            change_id="change-storage",
            files_changed=["backend/evidence_memory.py"],
            symbols_changed=["upsert_evidence_chunk"],
            raw_change_type=["storage_update", "schema_update"],
            what_changed="Updated upsert_evidence_chunk in backend/evidence_memory.py for storage_update.",
            direct_code_evidence=["Added JSONL upsert helper with stable IDs and hashes."],
        )

        api_server.build_phase2_evidence_cards()

        card = evidence_memory.read_records(evidence_memory.EVIDENCE_CARDS)[0]
        combined = " ".join([card["problem"], card["mechanism"], card["safe_impact"]])
        self.assertRegex(combined, r"traceability|JSONL|upsert|stable IDs|schema")
        self.assertNotIn("ATS", combined)
        self.assertNotIn("hallucination", card["safe_impact"].lower())

    def test_unsupported_claim_blocker(self):
        os.environ[self.flag_env] = "1"
        self.write_chunk()
        self.write_summary(
            change_id="change-unsupported",
            raw_change_type=["validation_rule_update"],
            what_changed="Reduced hallucinations by 40% and improved ATS score.",
            direct_code_evidence=[
                "reduced hallucinations by 40%",
                "improved ATS score",
                "Added template pollution guard.",
            ],
            metadata={"source_claim_text": ["reduced hallucinations by 40%"]},
        )

        api_server.build_phase2_evidence_cards()

        card = evidence_memory.read_records(evidence_memory.EVIDENCE_CARDS)[0]
        factual = " ".join(
            [
                card["problem"],
                card["mechanism"],
                card["safe_impact"],
                *card["allowed_claims"],
                *card["implementation_details"],
            ]
        )
        for phrase in FORBIDDEN_TEXT:
            self.assertNotIn(phrase.lower(), factual.lower())
        self.assertEqual("none", card["metric_support"])
        self.assertIn("unsupported_source_claims", card["metadata"])

    def test_idempotency(self):
        os.environ[self.flag_env] = "1"
        self.write_chunk()
        self.write_summary()

        api_server.build_phase2_evidence_cards()
        first = evidence_memory.read_records(evidence_memory.EVIDENCE_CARDS)
        api_server.build_phase2_evidence_cards()
        second = evidence_memory.read_records(evidence_memory.EVIDENCE_CARDS)

        self.assertEqual(1, len(first))
        self.assertEqual(1, len(second))
        self.assertEqual(first[0]["evidence_id"], second[0]["evidence_id"])

    def test_project_filtering(self):
        os.environ[self.flag_env] = "1"
        self.write_chunk(project_id="WorkAgent", chunk_id="chunk-workagent")
        self.write_summary(project_id="WorkAgent", change_id="change-workagent", source_chunk_ids=["chunk-workagent"])
        self.write_chunk(project_id="Event-Lottery-System", chunk_id="chunk-lottery", path="backend/lottery.py")
        self.write_summary(
            project_id="Event-Lottery-System",
            change_id="change-lottery",
            source_chunk_ids=["chunk-lottery"],
            files_changed=["backend/lottery.py"],
        )

        result = api_server.build_phase2_evidence_cards(project_id="WorkAgent")

        self.assertEqual(1, result["evidence_cards_count"])
        cards = evidence_memory.read_records(evidence_memory.EVIDENCE_CARDS)
        self.assertEqual({"WorkAgent"}, {card["project_id"] for card in cards})

    def test_weak_summary_skipped(self):
        os.environ[self.flag_env] = "1"
        self.write_summary(
            change_id="weak-change",
            source_chunk_ids=[],
            files_changed=[],
            symbols_changed=[],
            raw_change_type=["unknown_update"],
            what_changed="Updated code.",
            direct_code_evidence=[],
        )

        result = api_server.build_phase2_evidence_cards()

        self.assertEqual(0, result["evidence_cards_count"])
        self.assertEqual(1, result["skipped_summaries"])
        self.assertFalse(evidence_memory.get_record_path(evidence_memory.EVIDENCE_CARDS).exists())

    def test_preview_route(self):
        os.environ[self.flag_env] = "1"
        self.write_chunk()
        self.write_summary()
        self.client.post("/api/github/context/phase2/build-evidence-cards?project_id=WorkAgent")

        response = self.client.get("/api/github/context/phase2/evidence-cards/preview?project_id=WorkAgent&limit=10")

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertTrue(payload["enabled"])
        self.assertEqual(1, payload["count"])
        self.assertEqual([], collect_keys(payload, "raw_text"))
        self.assertEqual([], collect_keys(payload, "text"))
        item = payload["items"][0]
        for key in [
            "evidence_id",
            "project_id",
            "source_chunk_ids",
            "problem",
            "mechanism",
            "implementation_details",
            "safe_impact",
            "resume_angle",
            "confidence",
            "metric_support",
            "allowed_claims",
            "forbidden_claims",
        ]:
            self.assertIn(key, item)

        missing = self.client.get("/api/github/context/phase2/evidence-cards/preview?project_id=Missing")
        self.assertEqual([], missing.json()["items"])

        clamped = self.client.get("/api/github/context/phase2/evidence-cards/preview?limit=999")
        self.assertEqual(api_server.GITHUB_CONTEXT_PHASE2_PREVIEW_MAX_LIMIT, clamped.json()["limit"])

    def test_status_count_integration(self):
        os.environ[self.flag_env] = "1"
        self.write_chunk()
        self.write_summary()
        api_server.build_phase2_evidence_cards()

        status = self.client.get("/api/github/context/phase2/status").json()

        self.assertGreater(status["evidence_cards_count"], 0)
        self.assertEqual([], collect_keys(status, "allowed_claims"))
        self.assertGreater(status["projects"][0]["evidence_cards"], 0)


if __name__ == "__main__":
    unittest.main()
