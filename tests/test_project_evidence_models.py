import json
import unittest

from backend.project_evidence_models import (
    MAX_SUMMARY_LENGTH,
    ClaimSubjectType,
    Confidence,
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    ProjectCapabilityFact,
    ProjectClaimBoundary,
    ProjectEvidenceFact,
    ProjectEvidenceInput,
    ProjectEvidenceBuildResult,
    ProjectEvidencePipelineWarning,
    ProjectEvidenceMemory,
    EvidenceSourceRef,
    PipelineStatus,
    build_project_evidence_stable_id,
)


DIGEST = "a" * 64


def source(**changes):
    values = dict(source_type="project_change_evidence_card", source_id="ev-1", project_id="workagent", content_hash=DIGEST)
    values.update(changes)
    return EvidenceSourceRef(**values)


def fact(**changes):
    values = dict(project_id="workagent", mechanism="Added strict validation.", implementation=["Validated model fields."], source_refs=[source()])
    values.update(changes)
    return ProjectEvidenceFact(**values)


class ProjectEvidenceModelTests(unittest.TestCase):
    def test_valid_model_construction_and_normalization(self):
        ref = source(file_path=r"backend\model.py", start_line=1, end_line=2)
        item = ProjectEvidenceInput(project_id="workagent", input_type="project_change_evidence_card", title=" A title ", summary="A summary", source_refs=[ref], content_hash=DIGEST, technical_tags=["Python", "python", "API"])
        self.assertEqual("backend/model.py", ref.file_path)
        self.assertEqual(["API", "Python"], item.technical_tags)
        self.assertTrue(item.input_id.startswith("pei_"))

    def test_missing_required_fields(self):
        with self.assertRaises(TypeError):
            EvidenceSourceRef(source_type="x", source_id="x", content_hash=DIGEST)  # type: ignore[call-arg]
        with self.assertRaises(ValueError):
            source(project_id=" ")

    def test_invalid_enums(self):
        for name, value in (("confidence", "certain"), ("metric_support", "some"), ("status", "maybe")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                fact(**{name: value})

    def test_invalid_line_ranges(self):
        for changes in ({"start_line": 0}, {"end_line": -1}, {"start_line": 10, "end_line": 9}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                source(**changes)

    def test_stable_ids_ignore_mapping_tag_order_and_whitespace(self):
        first = build_project_evidence_stable_id("pef_", "workagent", {"technical_tags": ["b", "a"], "text": "a  b"})
        second = build_project_evidence_stable_id("pef_", "workagent", {"text": "a b", "technical_tags": ["a", "b"]})
        changed = build_project_evidence_stable_id("pef_", "workagent", {"text": "different", "technical_tags": ["a", "b"]})
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_order_sensitive_implementation_changes_stable_id(self):
        first = build_project_evidence_stable_id("pef_", "workagent", {"implementation": ["retrieve", "validate"]})
        second = build_project_evidence_stable_id("pef_", "workagent", {"implementation": ["validate", "retrieve"]})
        self.assertNotEqual(first, second)

    def test_order_insensitive_tags_do_not_change_stable_id(self):
        first = build_project_evidence_stable_id("pef_", "workagent", {"technical_tags": ["retrieval", "validation"]})
        second = build_project_evidence_stable_id("pef_", "workagent", {"technical_tags": ["validation", "retrieval"]})
        self.assertEqual(first, second)

    def test_order_sensitive_model_lists_preserve_input_order(self):
        refs = [source(source_id="second"), source(source_id="first")]
        evidence = fact(source_refs=refs, implementation=["retrieve", "validate"])
        self.assertEqual(["second", "first"], [item.source_id for item in evidence.source_refs])
        self.assertEqual(["retrieve", "validate"], evidence.implementation)

    def test_stable_json_and_round_trip(self):
        boundary = ProjectClaimBoundary(project_id="workagent", subject_type=ClaimSubjectType.EVIDENCE_FACT, subject_id=fact().evidence_fact_id, allowed_claims=["Implemented validation"])
        capability = ProjectCapabilityFact(project_id="workagent", capability_type="output_quality_control", present=True, source_evidence_fact_ids=[fact().evidence_fact_id])
        warning = ProjectEvidencePipelineWarning(code="limited_evidence", message="Only code evidence was available.", project_id="workagent")
        memory = ProjectEvidenceMemory(project_id="workagent", project_name="WorkAgent", evidence_facts=[fact()], capability_facts=[capability], claim_boundaries=[boundary], warnings=[warning])
        serialized = memory.to_json()
        self.assertEqual(serialized, memory.to_json())
        self.assertEqual(memory.to_dict(), ProjectEvidenceMemory.from_dict(json.loads(serialized)).to_dict())

    def test_mutable_defaults_are_not_shared(self):
        one = ProjectEvidenceBuildResult(status=PipelineStatus.EMPTY, enabled=True)
        two = ProjectEvidenceBuildResult(status=PipelineStatus.EMPTY, enabled=True)
        self.assertIsNot(one.project_ids, two.project_ids)
        self.assertIsNot(one.warnings, two.warnings)

    def test_duplicate_lists_are_normalized(self):
        capability = ProjectCapabilityFact(project_id="workagent", capability_type="retrieval", present=True, source_evidence_fact_ids=["pef_b", "pef_a", "pef_a"], technical_tags=["Python", "python"])
        self.assertEqual(["pef_a", "pef_b"], capability.source_evidence_fact_ids)
        self.assertEqual(["Python"], capability.technical_tags)

    def test_project_id_consistency(self):
        with self.assertRaises(ValueError):
            ProjectEvidenceMemory(project_id="other", project_name="Other", evidence_facts=[fact()])

    def test_duplicate_evidence_ids_rejected(self):
        duplicate = fact(evidence_fact_id="pef_same")
        with self.assertRaises(ValueError):
            ProjectEvidenceMemory(project_id="workagent", project_name="WorkAgent", evidence_facts=[duplicate, duplicate])

    def test_present_capability_requires_evidence_binding(self):
        with self.assertRaises(ValueError):
            ProjectCapabilityFact(project_id="workagent", capability_type="retrieval", present=True, source_evidence_fact_ids=[])

    def test_claim_conflict_rejected_after_normalization(self):
        with self.assertRaises(ValueError):
            ProjectClaimBoundary(project_id="workagent", subject_type="evidence_fact", subject_id="pef_x", allowed_claims=["Added validation"], forbidden_claims=[" added   validation "])

    def test_forbidden_top_level_raw_field_is_rejected(self):
        payload = source().to_dict()
        payload["raw_text"] = "complete source"
        with self.assertRaisesRegex(ValueError, "unknown"):
            EvidenceSourceRef.from_dict(payload)

    def test_forbidden_nested_secret_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "forbidden"):
            source(metadata={"nested": {"api_key": "do-not-store"}})

    def test_oversized_content_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "maximum length"):
            ProjectEvidenceInput(project_id="workagent", input_type="project_change", title="title", summary="x" * (MAX_SUMMARY_LENGTH + 1), source_refs=[source()], content_hash=DIGEST)

    def test_pipeline_result_rejects_negative_counters(self):
        with self.assertRaises(ValueError):
            ProjectEvidenceBuildResult(status="complete", enabled=True, items_skipped=-1)

    def test_all_declared_enum_values(self):
        self.assertEqual({"low", "medium", "high"}, {item.value for item in Confidence})
        self.assertEqual({"none", "approximate", "explicit"}, {item.value for item in MetricSupport})
        self.assertEqual({"accepted", "supporting", "weak", "rejected"}, {item.value for item in EvidenceStatus})
        self.assertIn(EvidenceType.FAILURE_RECOVERY, EvidenceType)

    def test_source_refs_required_and_accepted_fact_needs_detail(self):
        with self.assertRaises(ValueError):
            ProjectEvidenceInput(project_id="workagent", input_type="project_change", title="title", summary="summary", source_refs=[], content_hash=DIGEST)
        with self.assertRaises(ValueError):
            fact(implementation=[])
        rejected = fact(implementation=[], status=EvidenceStatus.REJECTED)
        self.assertEqual(EvidenceStatus.REJECTED, rejected.status)

    def test_unknown_round_trip_field_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            ProjectEvidenceBuildResult.from_dict({"status": "empty", "enabled": True, "surprise": True})


if __name__ == "__main__":
    unittest.main()
