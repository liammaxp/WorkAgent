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


class Phase2CapabilityFactsTests(unittest.TestCase):
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

    def write_card(
        self,
        *,
        project_id="WorkAgent",
        evidence_id="evidence-workagent",
        resume_angle="source_traceability",
        confidence="high",
        mechanism="Converted raw GitHub context into bounded, source-traceable evidence chunks.",
        implementation_details=None,
        safe_impact="Improved source traceability by preserving evidence identifiers.",
        allowed_claims=None,
        forbidden_claims=None,
        metric_support="none",
    ):
        return evidence_memory.upsert_evidence_card(
            evidence_memory.make_evidence_card(
                evidence_id=evidence_id,
                project_id=project_id,
                source_chunk_ids=[f"chunk-{evidence_id}"],
                problem="Phase 2 evidence needed source-traceable handling.",
                mechanism=mechanism,
                implementation_details=implementation_details
                if implementation_details is not None
                else ["Changed file: backend/evidence_chunker.py", "Technical tags: storage, validation"],
                safe_impact=safe_impact,
                resume_angle=resume_angle,
                confidence=confidence,
                metric_support=metric_support,
                allowed_claims=allowed_claims
                if allowed_claims is not None
                else ["converted raw GitHub context into source-traceable evidence chunks"],
                forbidden_claims=forbidden_claims
                if forbidden_claims is not None
                else [
                    "do not claim ATS score improvement",
                    "do not claim hallucinations were eliminated",
                ],
            )
        )

    def test_disabled_mode_does_not_build_capabilities(self):
        os.environ[self.flag_env] = "0"
        self.write_card()

        result = api_server.build_phase2_capability_facts()

        self.assertFalse(result["enabled"])
        self.assertEqual(0, result["capability_facts_count"])
        self.assertFalse(evidence_memory.get_record_path(evidence_memory.CAPABILITY_FACTS).exists())

    def test_missing_evidence_cards_safe(self):
        os.environ[self.flag_env] = "1"

        result = api_server.build_phase2_capability_facts()

        self.assertTrue(result["enabled"])
        self.assertEqual(0, result["processed_evidence_cards"])
        self.assertEqual(0, result["capability_facts_count"])
        self.assertFalse(self.storage_dir.exists())

    def test_source_traceability_capability(self):
        os.environ[self.flag_env] = "1"
        card = self.write_card()

        api_server.build_phase2_capability_facts()

        facts = evidence_memory.read_records(evidence_memory.CAPABILITY_FACTS)
        fact = next(item for item in facts if item["capability_type"] == "source_traceability")
        self.assertIn(card["evidence_id"], fact["source_evidence_ids"])
        self.assertTrue(fact["mechanisms"])
        self.assertEqual("none", fact["metric_support"])
        self.assertEqual("source_traceability", fact["capability_type"])

    def test_validation_and_unsupported_claim_boundary_capability(self):
        os.environ[self.flag_env] = "1"
        self.write_card(
            evidence_id="evidence-validation",
            resume_angle="validation_and_guardrails",
            mechanism="Added deterministic validation guard around allowed and forbidden claims.",
            safe_impact="Improved deterministic guarding of generated or stored evidence before downstream use.",
            allowed_claims=["added deterministic guards around unsupported or exaggerated claims"],
            forbidden_claims=[
                "do not claim hallucination reduction unless explicit evaluation evidence exists",
                "do not use percentages or before/after numeric improvements",
            ],
        )

        api_server.build_phase2_capability_facts()

        capability_types = {
            fact["capability_type"]
            for fact in evidence_memory.read_records(evidence_memory.CAPABILITY_FACTS)
        }
        self.assertIn("validation_and_guardrails", capability_types)
        self.assertIn("unsupported_claim_boundary", capability_types)
        for fact in evidence_memory.read_records(evidence_memory.CAPABILITY_FACTS):
            if fact["capability_type"] in {"validation_and_guardrails", "unsupported_claim_boundary"}:
                self.assertTrue(any("hallucination" in claim for claim in fact["forbidden_claims"]))
                self.assertFalse(any("hallucination reduction" in claim for claim in fact["allowed_resume_claims"]))

    def test_schema_storage_capability(self):
        os.environ[self.flag_env] = "1"
        self.write_card(
            evidence_id="evidence-storage",
            resume_angle="schema_and_storage_design",
            mechanism="Added JSONL-backed storage helpers with stable IDs, hashes, and upsert behavior.",
            implementation_details=["Changed symbol: upsert_evidence_chunk", "Related keywords: JSONL, stable IDs, upsert"],
            safe_impact="Improved traceability and repeatability of Phase 2 evidence memory.",
            allowed_claims=["implemented JSONL-backed Phase 2 evidence memory with stable IDs and upsert behavior"],
        )

        api_server.build_phase2_capability_facts()

        facts = evidence_memory.read_records(evidence_memory.CAPABILITY_FACTS)
        capability_types = {fact["capability_type"] for fact in facts}
        self.assertTrue({"schema_and_storage_design", "storage_idempotency"}.intersection(capability_types))
        combined = " ".join(" ".join(fact["mechanisms"]) for fact in facts)
        self.assertRegex(combined, r"JSONL|stable IDs|upsert")
        self.assertNotIn("ATS", combined)
        self.assertNotIn("cost reduction", combined.lower())

    def test_merge_quality_control_capability(self):
        os.environ[self.flag_env] = "1"
        self.write_card(
            evidence_id="evidence-merge",
            resume_angle="merge_quality_control",
            mechanism="Updated merge logic to preserve mechanism-rich evidence before generic generated content.",
            safe_impact="Improved technical specificity and ordering reliability of generated project evidence before final output.",
            allowed_claims=["preserved mechanism-rich project evidence during deterministic merge ordering"],
        )

        api_server.build_phase2_capability_facts()

        fact = next(
            item
            for item in evidence_memory.read_records(evidence_memory.CAPABILITY_FACTS)
            if item["capability_type"] == "merge_quality_control"
        )
        self.assertRegex(" ".join(fact["mechanisms"]), r"merge|mechanism-rich")
        self.assertEqual("none", fact["metric_support"])

    def test_aggregation_and_dedupe(self):
        os.environ[self.flag_env] = "1"
        self.write_card(evidence_id="evidence-one", resume_angle="source_traceability")
        self.write_card(
            evidence_id="evidence-two",
            resume_angle="source_traceability",
            mechanism="Preserved source evidence identifiers across Phase 2 evidence records.",
            allowed_claims=["preserved source evidence identifiers across Phase 2 evidence records"],
        )

        api_server.build_phase2_capability_facts()

        facts = [
            fact
            for fact in evidence_memory.read_records(evidence_memory.CAPABILITY_FACTS)
            if fact["capability_type"] == "source_traceability"
        ]
        self.assertEqual(1, len(facts))
        self.assertEqual({"evidence-one", "evidence-two"}, set(facts[0]["source_evidence_ids"]))
        self.assertEqual(len(facts[0]["mechanisms"]), len(set(facts[0]["mechanisms"])))

    def test_idempotency(self):
        os.environ[self.flag_env] = "1"
        self.write_card()

        api_server.build_phase2_capability_facts()
        first = evidence_memory.read_records(evidence_memory.CAPABILITY_FACTS)
        api_server.build_phase2_capability_facts()
        second = evidence_memory.read_records(evidence_memory.CAPABILITY_FACTS)

        self.assertEqual(len(first), len(second))
        self.assertEqual(
            {fact["capability_id"] for fact in first},
            {fact["capability_id"] for fact in second},
        )

    def test_project_filtering(self):
        os.environ[self.flag_env] = "1"
        self.write_card(project_id="WorkAgent", evidence_id="evidence-workagent")
        self.write_card(
            project_id="Event-Lottery-System",
            evidence_id="evidence-lottery",
            resume_angle="api_inspection",
        )

        result = api_server.build_phase2_capability_facts(project_id="WorkAgent")

        self.assertGreater(result["capability_facts_count"], 0)
        facts = evidence_memory.read_records(evidence_memory.CAPABILITY_FACTS)
        self.assertEqual({"WorkAgent"}, {fact["project_id"] for fact in facts})

    def test_unsupported_claim_blocker(self):
        os.environ[self.flag_env] = "1"
        self.write_card(
            evidence_id="evidence-unsupported",
            resume_angle="validation_and_guardrails",
            mechanism="Added deterministic validation guard around generated claims.",
            allowed_claims=["reduced hallucinations by 40%", "improved ATS score"],
            forbidden_claims=["do not claim ATS score improvement", "do not claim hallucinations were eliminated"],
        )

        api_server.build_phase2_capability_facts()

        facts = evidence_memory.read_records(evidence_memory.CAPABILITY_FACTS)
        self.assertTrue(facts)
        for fact in facts:
            factual = " ".join([*fact["mechanisms"], *fact["allowed_resume_claims"]]).lower()
            self.assertNotIn("reduced hallucinations", factual)
            self.assertNotIn("improved ats", factual)
            self.assertNotIn("%", factual)
            self.assertTrue(any("ATS score" in claim for claim in fact["forbidden_claims"]))
            self.assertIn("unsupported_source_claims", fact["metadata"])

    def test_weak_card_group_skipped(self):
        os.environ[self.flag_env] = "1"
        self.write_card(
            evidence_id="weak-evidence",
            resume_angle="unknown",
            confidence="low",
            mechanism="updated code",
            implementation_details=[],
            safe_impact="updated code",
            allowed_claims=[],
            forbidden_claims=[],
        )

        result = api_server.build_phase2_capability_facts()

        self.assertEqual(0, result["capability_facts_count"])
        self.assertEqual(1, result["skipped_groups"])
        self.assertFalse(evidence_memory.get_record_path(evidence_memory.CAPABILITY_FACTS).exists())

    def test_preview_route(self):
        os.environ[self.flag_env] = "1"
        self.write_card(project_id="WorkAgent", evidence_id="evidence-workagent")
        self.client.post("/api/github/context/phase2/build-capability-facts?project_id=WorkAgent")

        response = self.client.get("/api/github/context/phase2/capability-facts/preview?project_id=WorkAgent&limit=10")

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertTrue(payload["enabled"])
        self.assertGreater(payload["count"], 0)
        self.assertEqual([], collect_keys(payload, "raw_text"))
        self.assertEqual([], collect_keys(payload, "text"))
        item = payload["items"][0]
        for key in [
            "capability_id",
            "project_id",
            "capability_type",
            "present",
            "confidence",
            "mechanisms",
            "source_evidence_ids",
            "allowed_resume_claims",
            "forbidden_claims",
            "metric_support",
        ]:
            self.assertIn(key, item)

        missing = self.client.get("/api/github/context/phase2/capability-facts/preview?project_id=Missing")
        self.assertEqual([], missing.json()["items"])

        clamped = self.client.get("/api/github/context/phase2/capability-facts/preview?limit=999")
        self.assertEqual(api_server.GITHUB_CONTEXT_PHASE2_PREVIEW_MAX_LIMIT, clamped.json()["limit"])

    def test_status_count_integration(self):
        os.environ[self.flag_env] = "1"
        self.write_card()
        api_server.build_phase2_capability_facts()

        status = self.client.get("/api/github/context/phase2/status").json()

        self.assertGreater(status["capability_facts_count"], 0)
        self.assertEqual([], collect_keys(status, "allowed_resume_claims"))
        self.assertGreater(status["projects"][0]["capability_facts"], 0)


if __name__ == "__main__":
    unittest.main()
