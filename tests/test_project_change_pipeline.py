import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import project_change_memory as project_change  # noqa: E402
import project_change_pipeline  # noqa: E402


FORBIDDEN_INSPECT_KEYS = {
    "patch_text",
    "hunk_text",
    "added_lines",
    "removed_lines",
    "raw_text",
    "content",
    "token",
    "credential",
}


VALIDATION_PATCH = """@@ -1,2 +1,7 @@
 def validate_resume_quality(payload):
-    return ok()
+    if unsupported_metric(payload):
+        return fail("unsupported metric")
+    if unsupported_claim(payload):
+        return fail("unsupported claim")
+    return ok()
"""


WEAK_PATCH = """@@ -1 +1 @@
-value = 1
+value = 2
"""


def collect_forbidden_keys(value):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_INSPECT_KEYS:
                found.append(key)
            found.extend(collect_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(collect_forbidden_keys(item))
    return found


def github_context(
    *,
    project_id="ProjectA",
    repo="owner/project-a",
    commit_sha="abc123",
    patch_text=VALIDATION_PATCH,
    file_path="backend/validators.py",
    file_changes=None,
):
    if file_changes is None:
        file_changes = [{"filename": file_path, "patch": patch_text}]
    return {
        "project_id": project_id,
        "repository": repo,
        "latest_commit_sha": commit_sha,
        "contribution_evidence": [
            {
                "commits": [
                    {
                        "sha": commit_sha,
                        "message": "Add validation",
                        "file_changes": file_changes,
                    }
                ],
                "compare_file_changes": [],
            }
        ],
    }


def summary(project_id="ProjectA", change_id="change_1"):
    return project_change.RawChangeSummary(
        change_id=change_id,
        project_id=project_id,
        repo=f"owner/{project_id}",
        commit_sha="abc123",
        file_path="backend/validators.py",
        symbols_changed=["validate_resume_quality"],
        raw_change_types=["validation_logic_update"],
        what_changed="Updated validation logic in validate_resume_quality.",
        direct_code_evidence=["Changed symbol: validate_resume_quality."],
        uncertain_intent=[],
        confidence="high",
    )


def evidence_card(project_id="ProjectA", evidence_id="evidence_1", source_change_ids=None):
    return project_change.EvidenceCard(
        evidence_id=evidence_id,
        project_id=project_id,
        source_change_ids=source_change_ids or ["change_1"],
        problem="Generated output lacked a validation rule for unsupported claims or metrics.",
        mechanism="Added rule-based validation for unsupported claims or metrics.",
        implementation_details=[
            "Changed symbol: validate_resume_quality.",
            "Added a conditional referencing unsupported_metric.",
        ],
        safe_impact="Added an explicit safeguard against unsupported generated claims.",
        resume_angle="validation_and_safety",
        confidence="high",
        metric_support="none",
        allowed_claims=["added validation for unsupported generated claims"],
        forbidden_claims=["guaranteed factual correctness"],
    )


def capability(project_id="ProjectA", capability_id="capability_1", evidence_ids=None):
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


class ProjectChangePipelineTests(unittest.TestCase):
    def setUp(self):
        self.flag_env = project_change.PROJECT_CHANGE_MEMORY_ENV
        self.original_flag = os.environ.get(self.flag_env)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.memory_path = Path(self.temp_dir.name) / "project_change" / "project_change_memory.json"

    def tearDown(self):
        if self.original_flag is None:
            os.environ.pop(self.flag_env, None)
        else:
            os.environ[self.flag_env] = self.original_flag
        self.temp_dir.cleanup()

    def enable_project_change_memory(self):
        os.environ[self.flag_env] = "1"

    def disable_project_change_memory(self):
        os.environ.pop(self.flag_env, None)

    def test_disabled_pipeline_does_not_load_or_write(self):
        self.disable_project_change_memory()

        with patch.object(
            project_change_pipeline,
            "load_saved_github_contexts_for_project_change_memory",
            side_effect=AssertionError("disabled pipeline should not load sources"),
        ):
            result = project_change_pipeline.run_project_change_memory_pipeline(
                github_contexts=None,
                memory_path=self.memory_path,
            )

        self.assertFalse(result.enabled)
        self.assertEqual("disabled", result.status)
        self.assertFalse(self.memory_path.exists())

    def test_enabled_empty_source_is_safe_and_does_not_write(self):
        self.enable_project_change_memory()

        result = project_change_pipeline.run_project_change_memory_pipeline([], self.memory_path)

        self.assertTrue(result.enabled)
        self.assertEqual("no_source", result.status)
        self.assertEqual(0, result.raw_diff_input_count)
        self.assertFalse(self.memory_path.exists())

    def test_one_valid_patch_builds_and_persists_project_change_memory(self):
        self.enable_project_change_memory()

        result = project_change_pipeline.run_project_change_memory_pipeline(
            [github_context()],
            self.memory_path,
        )
        payload = project_change_pipeline.pipeline_result_to_dict(result)

        self.assertEqual("completed", result.status)
        self.assertEqual(1, result.source_context_count)
        self.assertEqual(1, result.raw_diff_input_count)
        self.assertEqual(1, result.diff_unit_count)
        self.assertEqual(1, result.raw_change_summary_count)
        self.assertEqual(1, result.evidence_card_candidate_count)
        self.assertEqual(1, result.qualified_evidence_card_count)
        self.assertGreaterEqual(result.capability_fact_count, 1)
        self.assertTrue(self.memory_path.exists())
        self.assertEqual([], collect_forbidden_keys(payload))

    def test_multiple_projects_remain_isolated(self):
        self.enable_project_change_memory()

        result = project_change_pipeline.run_project_change_memory_pipeline(
            [
                github_context(project_id="ProjectB", repo="owner/project-b", commit_sha="bbb222"),
                github_context(project_id="ProjectA", repo="owner/project-a", commit_sha="aaa111"),
            ],
            self.memory_path,
        )
        memory = project_change.load_project_change_memory(self.memory_path)

        self.assertEqual(["ProjectA", "ProjectB"], [item["project_id"] for item in result.project_summaries])
        self.assertEqual(["ProjectA", "ProjectB"], sorted(memory["projects"]))
        for project_id, entry in memory["projects"].items():
            for collection_name in ["raw_change_summaries", "evidence_cards", "capability_facts"]:
                self.assertTrue(all(item["project_id"] == project_id for item in entry[collection_name]))

    def test_duplicate_patches_are_deduped_and_repeated_build_does_not_grow_memory(self):
        self.enable_project_change_memory()
        duplicate_contexts = [github_context(), github_context()]

        first = project_change_pipeline.run_project_change_memory_pipeline(duplicate_contexts, self.memory_path)
        first_memory = project_change.load_project_change_memory(self.memory_path)
        first_counts = project_change_pipeline.project_change_counts(first_memory["projects"]["ProjectA"])
        second = project_change_pipeline.run_project_change_memory_pipeline(duplicate_contexts, self.memory_path)
        second_memory = project_change.load_project_change_memory(self.memory_path)
        second_counts = project_change_pipeline.project_change_counts(second_memory["projects"]["ProjectA"])

        self.assertEqual(2, first.source_context_count)
        self.assertEqual(1, first.raw_diff_input_count)
        self.assertEqual(1, second.raw_diff_input_count)
        self.assertEqual(first_counts, second_counts)

    def test_missing_patch_is_skipped_and_valid_sources_continue(self):
        self.enable_project_change_memory()
        mixed = github_context(
            file_changes=[
                {"filename": "backend/missing.py"},
                {"filename": "backend/validators.py", "patch": VALIDATION_PATCH},
            ]
        )

        result = project_change_pipeline.run_project_change_memory_pipeline([mixed], self.memory_path)

        self.assertEqual("completed_with_skips", result.status)
        self.assertEqual(1, result.raw_diff_input_count)
        self.assertIn("missing_patch_text", {item["reason"] for item in result.skipped_sources})
        self.assertTrue(self.memory_path.exists())

    def test_malformed_source_entry_does_not_stop_valid_sources(self):
        self.enable_project_change_memory()
        malformed = github_context(
            project_id="BadProject",
            repo="owner/bad",
            commit_sha="bad123",
            file_changes=[None],
        )

        result = project_change_pipeline.run_project_change_memory_pipeline(
            [malformed, github_context()],
            self.memory_path,
        )

        self.assertEqual("completed_with_skips", result.status)
        self.assertEqual(1, result.raw_diff_input_count)
        self.assertIn("malformed_file_change", {item["reason"] for item in result.skipped_sources})
        self.assertEqual([], collect_forbidden_keys(project_change_pipeline.pipeline_result_to_dict(result)))

    def test_weak_evidence_is_not_promoted_to_formal_evidence_or_capability(self):
        self.enable_project_change_memory()

        result = project_change_pipeline.run_project_change_memory_pipeline(
            [
                github_context(
                    project_id="WeakProject",
                    repo="owner/weak",
                    patch_text=WEAK_PATCH,
                    file_path="misc/config.txt",
                )
            ],
            self.memory_path,
        )
        entry = project_change.load_project_change_memory(self.memory_path)["projects"]["WeakProject"]

        self.assertEqual(1, result.evidence_card_candidate_count)
        self.assertEqual(0, result.qualified_evidence_card_count)
        self.assertEqual(0, result.capability_fact_count)
        self.assertEqual(1, len(entry["raw_change_summaries"]))
        self.assertEqual(0, len(entry["evidence_cards"]))
        self.assertEqual(0, len(entry["capability_facts"]))

    def test_persistence_failure_returns_failed_without_corrupting_existing_memory(self):
        self.enable_project_change_memory()
        project_change.write_project_change_memory(
            "ProjectA",
            [summary()],
            [evidence_card()],
            [capability()],
            self.memory_path,
        )
        before_text = self.memory_path.read_text(encoding="utf-8")

        with patch.object(
            project_change_pipeline,
            "persist_project_change_artifacts",
            side_effect=RuntimeError("disk unavailable"),
        ):
            result = project_change_pipeline.run_project_change_memory_pipeline([github_context()], self.memory_path)

        self.assertEqual("failed", result.status)
        self.assertIn("disk unavailable", result.errors)
        self.assertEqual(before_text, self.memory_path.read_text(encoding="utf-8"))
        self.assertIn("ProjectA", project_change.load_project_change_memory(self.memory_path)["projects"])

    def test_load_saved_github_contexts_uses_existing_scan_state_shape(self):
        scan_state_path = Path(self.temp_dir.name) / "github_repo_scan_state.json"
        scan_state_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "repositories": {
                        "owner/project-a": {
                            "repository": "owner/project-a",
                            "context": github_context(project_id="ProjectA", repo="owner/project-a"),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        contexts = project_change_pipeline.load_saved_github_contexts_for_project_change_memory(scan_state_path)

        self.assertEqual(1, len(contexts))
        self.assertEqual("ProjectA", contexts[0]["project_id"])
        self.assertEqual("owner/project-a", contexts[0]["repository"])

    def test_inspect_all_projects_is_ordered_and_redacted(self):
        self.enable_project_change_memory()
        project_change_pipeline.run_project_change_memory_pipeline(
            [
                github_context(project_id="ProjectB", repo="owner/project-b", commit_sha="bbb222"),
                github_context(project_id="ProjectA", repo="owner/project-a", commit_sha="aaa111"),
            ],
            self.memory_path,
        )

        inspect = project_change_pipeline.inspect_project_change_memory(memory_path=self.memory_path)

        self.assertEqual(2, inspect["project_count"])
        self.assertEqual(["ProjectA", "ProjectB"], [item["project_id"] for item in inspect["projects"]])
        self.assertEqual([], collect_forbidden_keys(inspect))

    def test_inspect_one_project_samples_are_safe_limited_and_read_only(self):
        self.enable_project_change_memory()
        project_change_pipeline.run_project_change_memory_pipeline([github_context()], self.memory_path)

        inspect = project_change_pipeline.inspect_project_change_memory(
            project_id="ProjectA",
            memory_path=self.memory_path,
            sample_limit="1",
        )
        missing_path = Path(self.temp_dir.name) / "missing" / "project_change_memory.json"
        missing = project_change_pipeline.inspect_project_change_memory(
            project_id="MissingProject",
            memory_path=missing_path,
            sample_limit="-4",
        )

        self.assertEqual("ProjectA", inspect["project_id"])
        self.assertLessEqual(len(inspect["sample_raw_change_summaries"]), 1)
        self.assertLessEqual(len(inspect["sample_evidence_cards"]), 1)
        self.assertLessEqual(len(inspect["sample_capability_facts"]), 1)
        self.assertEqual([], collect_forbidden_keys(inspect))
        self.assertEqual(0, missing["raw_change_summary_count"])
        self.assertEqual(0, missing["sample_limit"])
        self.assertFalse(missing_path.exists())

    def test_inspect_sample_limit_is_clamped(self):
        self.enable_project_change_memory()
        self.assertEqual(project_change_pipeline.DEFAULT_INSPECT_SAMPLE_LIMIT, project_change_pipeline.safe_sample_limit("bad"))
        self.assertEqual(0, project_change_pipeline.safe_sample_limit("-10"))
        self.assertEqual(project_change_pipeline.MAX_INSPECT_SAMPLE_LIMIT, project_change_pipeline.safe_sample_limit("999"))

    def test_health_states(self):
        self.disable_project_change_memory()
        disabled = project_change_pipeline.get_project_change_health(self.memory_path)
        self.assertEqual("disabled", disabled["status"])

        self.enable_project_change_memory()
        empty = project_change_pipeline.get_project_change_health(self.memory_path)
        self.assertEqual("empty", empty["status"])
        self.assertFalse(empty["memory_exists"])

        ready_path = Path(self.temp_dir.name) / "ready.json"
        project_change.write_project_change_memory(
            "ReadyProject",
            [summary("ReadyProject")],
            [evidence_card("ReadyProject")],
            [capability("ReadyProject")],
            ready_path,
        )
        ready = project_change_pipeline.get_project_change_health(ready_path)
        self.assertEqual("ready", ready["status"])
        self.assertEqual(1, ready["project_count"])

        degraded_path = Path(self.temp_dir.name) / "degraded.json"
        project_change.write_project_change_memory("DegradedProject", [summary("DegradedProject")], [], [], degraded_path)
        degraded = project_change_pipeline.get_project_change_health(degraded_path)
        self.assertEqual("degraded", degraded["status"])
        self.assertIn("no_evidence_cards", degraded["issues"])
        self.assertIn("no_capability_facts", degraded["issues"])

        malformed_path = Path(self.temp_dir.name) / "malformed.json"
        malformed_path.write_text("{ invalid json", encoding="utf-8")
        malformed = project_change_pipeline.get_project_change_health(malformed_path)
        self.assertEqual("error", malformed["status"])
        self.assertFalse(malformed["memory_readable"])

        unsupported_path = Path(self.temp_dir.name) / "unsupported.json"
        unsupported_path.write_text(json.dumps({"schema_version": "project_change_memory.v0", "projects": {}}), encoding="utf-8")
        unsupported = project_change_pipeline.get_project_change_health(unsupported_path)
        self.assertEqual("error", unsupported["status"])

    def test_pipeline_stays_isolated_from_resume_generation_and_remote_fetches(self):
        self.enable_project_change_memory()

        with patch.object(
            project_change_pipeline,
            "load_saved_github_contexts_for_project_change_memory",
            side_effect=AssertionError("explicit contexts should avoid saved-source loading"),
        ):
            result = project_change_pipeline.run_project_change_memory_pipeline(
                [github_context()],
                self.memory_path,
            )

        self.assertEqual("completed", result.status)
        self.assertFalse(hasattr(project_change_pipeline, "agent"))
        self.assertFalse(hasattr(project_change_pipeline, "MEMORY_STORE"))


if __name__ == "__main__":
    unittest.main()
