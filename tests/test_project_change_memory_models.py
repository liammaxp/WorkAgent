import importlib
import json
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROJECT_CHANGE_MODULE = "backend.project_change_memory"
PROJECT_CHANGE_ENV = "USE_PROJECT_CHANGE_MEMORY"
OPTIONAL_RUNTIME_MODULES = ("fastapi", "chromadb", "openai")


def import_project_change_module():
    return importlib.import_module(PROJECT_CHANGE_MODULE)


def sample_models(project_change):
    raw_input = project_change.RawDiffInput(
        project_id="agent-develop",
        repo="owner/agent-develop",
        commit_sha="abc123",
        file_path="backend/example.py",
        patch_text="+def validate_resume():\n+    return True",
        commit_message="Add validation helper",
    )
    diff_unit = project_change.DiffUnit(
        unit_id="unit_1",
        project_id=raw_input.project_id,
        repo=raw_input.repo,
        commit_sha=raw_input.commit_sha,
        file_path=raw_input.file_path,
        hunk_text="@@ -1 +1 @@\n+def validate_resume():",
        added_lines=["def validate_resume():"],
        removed_lines=[],
        symbols_changed=["validate_resume"],
        change_hints=["validation_logic_update"],
    )
    change_summary = project_change.RawChangeSummary(
        change_id="change_1",
        project_id=raw_input.project_id,
        repo=raw_input.repo,
        commit_sha=raw_input.commit_sha,
        file_path=raw_input.file_path,
        symbols_changed=["validate_resume"],
        raw_change_types=["validation_logic_update"],
        what_changed="Added resume validation helper.",
        direct_code_evidence=["def validate_resume():"],
        uncertain_intent=["May support stricter resume quality gates."],
        confidence="medium",
    )
    evidence_card = project_change.EvidenceCard(
        evidence_id="evidence_1",
        project_id=raw_input.project_id,
        source_change_ids=[change_summary.change_id],
        problem="Resume output needs validation before claims are made.",
        mechanism="Added a validation helper.",
        implementation_details=["Introduced validate_resume function."],
        safe_impact="Supports safer resume output checks.",
        resume_angle="Built validation foundations for evidence-grounded resumes.",
        confidence="medium",
        metric_support="none",
        allowed_claims=["Built validation foundations for resume generation."],
        forbidden_claims=["Improved resume success rate by 50%."],
    )
    capability_fact = project_change.CapabilityFact(
        capability_id="capability_1",
        project_id=raw_input.project_id,
        capability_type="validation_and_repair",
        present=True,
        confidence="medium",
        mechanisms=["validation helper"],
        source_evidence_ids=[evidence_card.evidence_id],
        allowed_resume_claims=["Built validation foundations for resume generation."],
        forbidden_claims=["Guaranteed ATS compatibility."],
        metric_support="none",
    )
    return [raw_input, diff_unit, change_summary, evidence_card, capability_fact]


class ProjectChangeDiffMemoryModelTests(unittest.TestCase):
    def setUp(self):
        self.original_present = PROJECT_CHANGE_ENV in os.environ
        self.original_value = os.environ.get(PROJECT_CHANGE_ENV)

    def tearDown(self):
        if self.original_present:
            os.environ[PROJECT_CHANGE_ENV] = self.original_value or ""
        else:
            os.environ.pop(PROJECT_CHANGE_ENV, None)

    def test_feature_flag_disabled_by_default(self):
        project_change = import_project_change_module()
        os.environ.pop(PROJECT_CHANGE_ENV, None)

        self.assertFalse(project_change.is_project_change_memory_enabled())

    def test_feature_flag_enabled_when_set_to_one(self):
        project_change = import_project_change_module()
        os.environ[PROJECT_CHANGE_ENV] = "1"

        self.assertTrue(project_change.is_project_change_memory_enabled())

    def test_models_can_be_instantiated_and_serialized(self):
        project_change = import_project_change_module()

        for model in sample_models(project_change):
            with self.subTest(model=type(model).__name__):
                payload = project_change.model_to_dict(model)
                self.assertIsInstance(payload, dict)
                self.assertEqual("agent-develop", payload["project_id"])
                json.dumps(payload, sort_keys=True)

    def test_stable_hash_text_is_deterministic(self):
        project_change = import_project_change_module()

        self.assertEqual(project_change.stable_hash_text("same"), project_change.stable_hash_text("same"))
        self.assertNotEqual(project_change.stable_hash_text("same"), project_change.stable_hash_text("different"))

    def test_import_does_not_load_optional_runtime_dependencies(self):
        sys.modules.pop(PROJECT_CHANGE_MODULE, None)
        previously_loaded = {
            name for name in OPTIONAL_RUNTIME_MODULES if name in sys.modules
        }

        import_project_change_module()

        for module_name in OPTIONAL_RUNTIME_MODULES:
            if module_name not in previously_loaded:
                self.assertNotIn(module_name, sys.modules)


if __name__ == "__main__":
    unittest.main()
