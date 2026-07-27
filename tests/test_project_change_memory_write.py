import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import project_change_memory as project_change  # noqa: E402


def summary(project_id: str = "ProjectA", change_id: str = "change_1", what_changed: str = "Added validation.") -> project_change.RawChangeSummary:
    return project_change.RawChangeSummary(
        change_id=change_id,
        project_id=project_id,
        repo="owner/repo",
        commit_sha="abc123",
        file_path="backend/example.py",
        symbols_changed=["validate_resume_quality"],
        raw_change_types=["validation_logic_update"],
        what_changed=what_changed,
        direct_code_evidence=["Changed symbol: validate_resume_quality."],
        uncertain_intent=[],
        confidence="high",
    )


def evidence_card(
    project_id: str = "ProjectA",
    evidence_id: str = "evidence_1",
    *,
    confidence: str = "high",
    resume_angle: str = "validation_and_safety",
    mechanism: str = "Added rule-based validation for unsupported claims or metrics.",
    details: list[str] | None = None,
    source_change_ids: list[str] | None = None,
) -> project_change.EvidenceCard:
    return project_change.EvidenceCard(
        evidence_id=evidence_id,
        project_id=project_id,
        source_change_ids=source_change_ids or ["change_1"],
        problem="Generated output lacked a validation rule for unsupported claims or metrics.",
        mechanism=mechanism,
        implementation_details=details or [
            "Changed symbol: validate_resume_quality.",
            "Added a conditional referencing unsupported_metric.",
        ],
        safe_impact="Added an explicit safeguard against unsupported generated claims.",
        resume_angle=resume_angle,
        confidence=confidence,
        metric_support="none",
        allowed_claims=["added validation for unsupported generated claims"],
        forbidden_claims=["guaranteed factual correctness"],
    )


def capability(
    project_id: str = "ProjectA",
    capability_id: str = "capability_1",
    evidence_ids: list[str] | None = None,
) -> project_change.CapabilityFact:
    return project_change.CapabilityFact(
        capability_id=capability_id,
        project_id=project_id,
        capability_type="validation_and_repair",
        present=True,
        confidence="high",
        mechanisms=["Added rule-based validation for unsupported claims or metrics."],
        source_evidence_ids=evidence_ids or ["evidence_1"],
        allowed_resume_claims=["implemented explicit validation behavior"],
        forbidden_claims=["guaranteed factual correctness"],
        metric_support="none",
    )


def strip_updated_at(memory: dict) -> dict:
    clone = json.loads(json.dumps(memory))
    clone["updated_at"] = "<timestamp>"
    return clone


class ProjectChangeProjectMemoryWriteTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "project_change" / "project_change_memory.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_missing_file_loads_empty_without_creating_file(self):
        memory = project_change.load_project_change_memory(self.path)

        self.assertEqual("project_change_memory.v1", memory["schema_version"])
        self.assertIsNone(memory["updated_at"])
        self.assertEqual({}, memory["projects"])
        self.assertFalse(self.path.exists())

    def test_initial_write_creates_valid_schema_and_project_counts(self):
        memory = project_change.write_project_change_memory(
            "ProjectA",
            [summary()],
            [evidence_card()],
            [capability()],
            self.path,
        )

        self.assertTrue(self.path.exists())
        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual("project_change_memory.v1", loaded["schema_version"])
        self.assertIsInstance(loaded["updated_at"], str)
        project = memory["projects"]["ProjectA"]
        self.assertEqual(1, len(project["raw_change_summaries"]))
        self.assertEqual(1, len(project["evidence_cards"]))
        self.assertEqual(1, len(project["capability_facts"]))

    def test_duplicate_write_does_not_increase_counts(self):
        artifacts = ([summary()], [evidence_card()], [capability()])

        project_change.write_project_change_memory("ProjectA", *artifacts, self.path)
        memory = project_change.write_project_change_memory("ProjectA", *artifacts, self.path)
        project = memory["projects"]["ProjectA"]

        self.assertEqual(1, len(project["raw_change_summaries"]))
        self.assertEqual(1, len(project["evidence_cards"]))
        self.assertEqual(1, len(project["capability_facts"]))

    def test_stable_id_replacement_uses_newer_representation(self):
        project_change.write_project_change_memory(
            "ProjectA",
            [summary(what_changed="Old wording.")],
            [evidence_card(details=["Old detail."])],
            [capability()],
            self.path,
        )
        memory = project_change.write_project_change_memory(
            "ProjectA",
            [summary(what_changed="Corrected wording.")],
            [evidence_card(details=["Corrected detail.", "Changed symbol: validate_resume_quality."])],
            [capability()],
            self.path,
        )
        project = memory["projects"]["ProjectA"]

        self.assertEqual(1, len(project["raw_change_summaries"]))
        self.assertEqual("Corrected wording.", project["raw_change_summaries"][0]["what_changed"])
        self.assertEqual(1, len(project["evidence_cards"]))
        self.assertIn("Corrected detail.", project["evidence_cards"][0]["implementation_details"])

    def test_multiple_projects_remain_isolated(self):
        project_change.write_project_change_memory("ProjectA", [summary("ProjectA")], [evidence_card("ProjectA")], [capability("ProjectA")], self.path)
        memory = project_change.write_project_change_memory(
            "ProjectB",
            [summary("ProjectB", "change_b")],
            [evidence_card("ProjectB", "evidence_b", source_change_ids=["change_b"])],
            [capability("ProjectB", "capability_b", ["evidence_b"])],
            self.path,
        )

        self.assertEqual(["ProjectA", "ProjectB"], sorted(memory["projects"]))
        self.assertEqual("ProjectA", memory["projects"]["ProjectA"]["project_id"])
        self.assertEqual("ProjectB", memory["projects"]["ProjectB"]["project_id"])

    def test_project_mismatch_raises_for_all_artifact_types(self):
        with self.assertRaisesRegex(ValueError, "expected ProjectA, got ProjectB"):
            project_change.write_project_change_memory("ProjectA", [summary("ProjectB")], [], [], self.path)
        with self.assertRaisesRegex(ValueError, "expected ProjectA, got ProjectB"):
            project_change.write_project_change_memory("ProjectA", [], [evidence_card("ProjectB")], [], self.path)
        with self.assertRaisesRegex(ValueError, "expected ProjectA, got ProjectB"):
            project_change.write_project_change_memory("ProjectA", [], [], [capability("ProjectB")], self.path)

    def test_weak_evidence_card_filtering(self):
        strong = evidence_card(evidence_id="strong")
        weak = evidence_card(
            evidence_id="weak",
            confidence="low",
            resume_angle="implementation_change",
            mechanism="Modified implementation logic in the affected source file.",
            details=[],
        )

        memory = project_change.write_project_change_memory(
            "ProjectA",
            [summary()],
            [strong, weak],
            [],
            self.path,
        )

        evidence_ids = [card["evidence_id"] for card in memory["projects"]["ProjectA"]["evidence_cards"]]
        self.assertEqual(["strong"], evidence_ids)

    def test_orphan_capability_filtering(self):
        memory = project_change.write_project_change_memory(
            "ProjectA",
            [summary()],
            [evidence_card()],
            [capability(evidence_ids=["missing_evidence"])],
            self.path,
        )

        self.assertEqual([], memory["projects"]["ProjectA"]["capability_facts"])

    def test_existing_capability_preserved_when_new_input_empty(self):
        project_change.write_project_change_memory("ProjectA", [summary()], [evidence_card()], [capability()], self.path)
        memory = project_change.write_project_change_memory("ProjectA", [], [], [], self.path)

        self.assertEqual(1, len(memory["projects"]["ProjectA"]["capability_facts"]))

    def test_missing_project_accessor_returns_empty_without_disk_write(self):
        entry = project_change.get_project_change_memory("MissingProject", self.path)

        self.assertEqual("MissingProject", entry["project_id"])
        self.assertEqual([], entry["raw_change_summaries"])
        self.assertEqual([], entry["evidence_cards"])
        self.assertEqual([], entry["capability_facts"])
        self.assertFalse(self.path.exists())

    def test_summary_helper_returns_counts_only(self):
        project_change.write_project_change_memory("ProjectA", [summary()], [evidence_card()], [capability()], self.path)

        summary_payload = project_change.summarize_project_change_memory("ProjectA", self.path)

        self.assertEqual(
            {
                "project_id": "ProjectA",
                "raw_change_summary_count": 1,
                "evidence_card_count": 1,
                "capability_fact_count": 1,
                "capability_types": ["validation_and_repair"],
            },
            summary_payload,
        )
        self.assertNotIn("raw_change_summaries", summary_payload)

    def test_malformed_json_raises_and_does_not_overwrite(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{not valid", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Invalid project change memory project memory JSON"):
            project_change.load_project_change_memory(self.path)
        self.assertEqual("{not valid", self.path.read_text(encoding="utf-8"))

    def test_empty_file_loads_empty_memory(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("   ", encoding="utf-8")

        memory = project_change.load_project_change_memory(self.path)

        self.assertEqual(project_change.create_empty_project_change_memory(), memory)

    def test_unsupported_schema_raises(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"schema_version": "project_change_memory.v99", "projects": {}}), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Unsupported project change memory project memory schema"):
            project_change.load_project_change_memory(self.path)

    def test_atomic_write_leaves_valid_json_and_no_temp_files(self):
        project_change.write_project_change_memory("ProjectA", [summary()], [evidence_card()], [capability()], self.path)

        json.loads(self.path.read_text(encoding="utf-8"))
        temp_files = list(self.path.parent.glob(f".{self.path.name}.*.tmp"))
        self.assertEqual([], temp_files)

    def test_atomic_write_failure_preserves_existing_destination(self):
        project_change.atomic_write_json(self.path, project_change.create_empty_project_change_memory())
        original_text = self.path.read_text(encoding="utf-8")

        with patch.object(project_change.os, "replace", side_effect=RuntimeError("replace failed")):
            with self.assertRaises(RuntimeError):
                project_change.atomic_write_json(
                    self.path,
                    {
                        "schema_version": "project_change_memory.v1",
                        "updated_at": "later",
                        "projects": {"ProjectA": project_change.normalize_project_change_entry("ProjectA", None)},
                    },
                )

        self.assertEqual(original_text, self.path.read_text(encoding="utf-8"))
        self.assertEqual([], list(self.path.parent.glob(f".{self.path.name}.*.tmp")))

    def test_deterministic_ordering_ignoring_updated_at(self):
        artifacts_a = (
            [summary("ProjectB", "change_b"), summary("ProjectA", "change_a")],
            [
                evidence_card("ProjectB", "evidence_b", source_change_ids=["change_b"]),
                evidence_card("ProjectA", "evidence_a", source_change_ids=["change_a"]),
            ],
            [
                capability("ProjectB", "capability_b", ["evidence_b"]),
                capability("ProjectA", "capability_a", ["evidence_a"]),
            ],
        )
        artifacts_b = (
            list(reversed(artifacts_a[0])),
            list(reversed(artifacts_a[1])),
            list(reversed(artifacts_a[2])),
        )
        path_a = Path(self.temp_dir.name) / "a" / "memory.json"
        path_b = Path(self.temp_dir.name) / "b" / "memory.json"

        memory_a = project_change.persist_project_change_artifacts(*artifacts_a, path=path_a)
        memory_b = project_change.persist_project_change_artifacts(*artifacts_b, path=path_b)

        self.assertEqual(strip_updated_at(memory_a), strip_updated_at(memory_b))

    def test_persist_project_change_artifacts_groups_multiple_projects(self):
        memory = project_change.persist_project_change_artifacts(
            [summary("ProjectA", "change_a"), summary("ProjectB", "change_b")],
            [
                evidence_card("ProjectA", "evidence_a", source_change_ids=["change_a"]),
                evidence_card("ProjectB", "evidence_b", source_change_ids=["change_b"]),
            ],
            [
                capability("ProjectA", "capability_a", ["evidence_a"]),
                capability("ProjectB", "capability_b", ["evidence_b"]),
            ],
            path=self.path,
        )

        self.assertEqual(["ProjectA", "ProjectB"], sorted(memory["projects"]))
        self.assertEqual(["evidence_a"], [card["evidence_id"] for card in memory["projects"]["ProjectA"]["evidence_cards"]])
        self.assertEqual(["evidence_b"], [card["evidence_id"] for card in memory["projects"]["ProjectB"]["evidence_cards"]])

    def test_no_raw_patch_or_diff_payload_is_persisted(self):
        project_change.write_project_change_memory("ProjectA", [summary()], [evidence_card()], [capability()], self.path)

        serialized = self.path.read_text(encoding="utf-8")

        for forbidden_field in ["patch_text", "hunk_text", "added_lines", "removed_lines"]:
            self.assertNotIn(forbidden_field, serialized)


if __name__ == "__main__":
    unittest.main()
