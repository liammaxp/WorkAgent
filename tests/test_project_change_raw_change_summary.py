import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import project_change_memory as project_change  # noqa: E402


FORBIDDEN_LANGUAGE = [
    "reduced hallucinations",
    "eliminated hallucinations",
    "guaranteed",
    "improved accuracy",
    "improved performance",
    "by 40%",
    "ATS success",
]


def diff_unit(patch_text: str, file_path: str = "backend/example.py") -> project_change.DiffUnit:
    raw = project_change.RawDiffInput(
        project_id="agent-develop",
        repo="owner/agent-develop",
        commit_sha="abc123",
        file_path=file_path,
        patch_text=patch_text,
    )
    return project_change.extract_diff_units(raw)[0]


def summary_for(patch_text: str, file_path: str = "backend/example.py") -> project_change.RawChangeSummary:
    return project_change.build_raw_change_summary(diff_unit(patch_text, file_path))


class ProjectChangeRawChangeSummaryTests(unittest.TestCase):
    def assert_no_forbidden_language(self, summary: project_change.RawChangeSummary) -> None:
        generated_text = " ".join(
            [
                summary.what_changed,
                " ".join(summary.uncertain_intent),
            ]
        ).lower()
        for forbidden in FORBIDDEN_LANGUAGE:
            self.assertNotIn(forbidden.lower(), generated_text)

    def test_validation_change_is_conservative(self):
        summary = summary_for(
            """@@ -1,2 +1,4 @@ def validate_resume_quality(payload):
+    if unsupported_metric:
+        return fail("unsupported metric")
     return ok()
""",
            "backend/validators.py",
        )

        self.assertIn("validation_logic_update", summary.raw_change_types)
        self.assertIn("validation logic", summary.what_changed)
        self.assertTrue(any("unsupported_metric" in item for item in summary.direct_code_evidence))
        self.assertNotIn("%", summary.what_changed)
        self.assertIn(summary.confidence, {"medium", "high"})
        self.assert_no_forbidden_language(summary)

    def test_merge_change_records_symbol_and_uncertain_intent(self):
        summary = summary_for(
            """@@ -8,5 +8,7 @@ def merge_staged_resume(candidate):
+    final_bullets = select_final_bullets(candidate)
+    bullet_depth_profile = candidate.get("bullet_depth_profile")
     return final_bullets
""",
            "backend/merge.py",
        )

        self.assertIn("merge_logic_update", summary.raw_change_types)
        self.assertTrue(any("merge_staged_resume" in item for item in summary.direct_code_evidence))
        self.assertTrue(any("may be intended to preserve stronger project evidence" in item for item in summary.uncertain_intent))
        self.assertNotIn("improved", summary.what_changed.lower())

    def test_retrieval_change_uses_retrieval_type_without_recall_claims(self):
        summary = summary_for(
            """@@ -1,3 +1,5 @@ def retrieve_evidence_for_project(project_id, query):
+    candidates = query(project_id)
+    return rerank(candidates, evidence=True)
""",
            "backend/retrieval.py",
        )

        self.assertIn("retrieval_logic_update", summary.raw_change_types)
        self.assertIn("retrieval logic", summary.what_changed)
        self.assertNotIn("recall", summary.what_changed.lower())

    def test_memory_cache_change_is_grounded(self):
        summary = summary_for(
            """@@ -1,3 +1,5 @@ def load_project_memory(project_id):
+    cache_key = f"project_memory:{project_id}"
+    return sqlite_cache.get(cache_key)
""",
            "backend/memory_store.py",
        )

        self.assertIn("memory_storage_update", summary.raw_change_types)
        self.assertTrue(any("project_memory" in item or "sqlite" in item or "cache" in item for item in summary.direct_code_evidence))
        generated = " ".join([summary.what_changed, *summary.uncertain_intent]).lower()
        self.assertNotIn("cost", generated)
        self.assertNotIn("token reduction", generated)

    def test_fallback_retry_change_is_conservative(self):
        summary = summary_for(
            """@@ -1,3 +1,6 @@ def call_model():
+    for retry in range(2):
+        if fallback_enabled:
+            return repair(last_result)
""",
            "backend/retry.py",
        )

        self.assertIn("fallback_update", summary.raw_change_types)
        self.assertNotIn("eliminated", " ".join([summary.what_changed, *summary.uncertain_intent]).lower())

    def test_prompt_only_change_stays_prompt_constraint(self):
        summary = summary_for(
            """@@ -1,2 +1,4 @@ def build_system_message():
+    prompt = "Follow the instruction and constraint boundaries."
+    return prompt
""",
            "backend/prompts.py",
        )

        self.assertEqual(["prompt_constraint_update"], summary.raw_change_types)
        generated = " ".join([summary.what_changed, *summary.uncertain_intent]).lower()
        self.assertNotIn("llm_reliability", generated)
        self.assertNotIn("hallucination", generated)
        self.assertNotIn("factuality improvement", generated)

    def test_test_only_change(self):
        summary = summary_for(
            """@@ -1,2 +1,4 @@ def test_project_change_diff_units():
+    result = extract_diff_units(raw)
+    assert result[0].unit_id
""",
            "tests/test_project_change_diff_unit_extraction.py",
        )

        self.assertIn("test_update", summary.raw_change_types)
        self.assertIn("tests", summary.what_changed.lower())

    def test_unknown_change_is_low_confidence(self):
        summary = summary_for(
            """@@ -1 +1 @@
-value = 1
+value = 2
""",
            "backend/plain.py",
        )

        self.assertEqual(["unknown"], summary.raw_change_types)
        self.assertEqual("low", summary.confidence)
        self.assertIn("Modified implementation logic", summary.what_changed)

    def test_change_id_is_deterministic_and_hunk_sensitive(self):
        first = summary_for("@@ -1 +1 @@\n-value = 1\n+value = 2\n", "backend/plain.py")
        same = summary_for("@@ -1 +1 @@\n-value = 1\n+value = 2\n", "backend/plain.py")
        different = summary_for("@@ -1 +1 @@\n-value = 1\n+value = 3\n", "backend/plain.py")

        self.assertEqual(first.change_id, same.change_id)
        self.assertNotEqual(first.change_id, different.change_id)

    def test_forbidden_language_is_not_adopted_as_confirmed_result(self):
        summary = summary_for(
            """@@ -1,2 +1,4 @@ def validate_claim_text(text):
+    unsupported_claim = "reduced hallucinations by 40%"
+    if unsupported_claim:
+        return fail("unsupported claim")
""",
            "backend/validators.py",
        )

        self.assertIn("validation_logic_update", summary.raw_change_types)
        self.assert_no_forbidden_language(summary)


if __name__ == "__main__":
    unittest.main()
