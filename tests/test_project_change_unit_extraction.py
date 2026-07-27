import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import project_change_memory as project_change  # noqa: E402


def raw_input(patch_text: str, file_path: str = "backend/example.py") -> project_change.RawDiffInput:
    return project_change.RawDiffInput(
        project_id="agent-develop",
        repo="owner/agent-develop",
        commit_sha="abc123",
        file_path=file_path,
        patch_text=patch_text,
        commit_message="Update backend behavior",
    )


class ProjectChangeDiffUnitExtractionTests(unittest.TestCase):
    def test_standard_unified_diff_one_hunk(self):
        patch_text = """diff --git a/backend/validators.py b/backend/validators.py
index 1111111..2222222 100644
--- a/backend/validators.py
+++ b/backend/validators.py
@@ -10,7 +10,12 @@ def validate_resume_quality(payload):
-    return old_validator(payload)
+    if not payload:
+        return False
+    return validate(payload)
"""

        units = project_change.extract_diff_units(raw_input(patch_text, "backend/validators.py"))

        self.assertEqual(1, len(units))
        unit = units[0]
        self.assertEqual(["    if not payload:", "        return False", "    return validate(payload)"], unit.added_lines)
        self.assertEqual(["    return old_validator(payload)"], unit.removed_lines)
        self.assertIn("validate_resume_quality", unit.symbols_changed)
        self.assertIn("validation_logic", unit.change_hints)

    def test_unified_diff_multiple_hunks(self):
        patch_text = """@@ -1,5 +1,5 @@ def first_handler():
-    return "old"
+    return "new"

@@ -20,5 +20,5 @@ def second_handler():
-    enabled = False
+    enabled = True
"""
        raw = raw_input(patch_text)

        first_units = project_change.extract_diff_units(raw)
        second_units = project_change.extract_diff_units(raw)

        self.assertEqual(2, len(first_units))
        self.assertEqual([unit.unit_id for unit in first_units], [unit.unit_id for unit in second_units])
        self.assertEqual(['    return "new"'], first_units[0].added_lines)
        self.assertEqual(['    return "old"'], first_units[0].removed_lines)
        self.assertEqual(["    enabled = True"], first_units[1].added_lines)
        self.assertEqual(["    enabled = False"], first_units[1].removed_lines)
        self.assertIn("first_handler", first_units[0].symbols_changed)
        self.assertIn("second_handler", first_units[1].symbols_changed)

    def test_patch_without_hunk_header_returns_one_unit(self):
        patch_text = """+def no_header_patch():
+    return True
"""

        units = project_change.extract_diff_units(raw_input(patch_text))

        self.assertEqual(1, len(units))
        self.assertEqual(patch_text.strip("\n"), units[0].hunk_text)
        self.assertEqual(["def no_header_patch():", "    return True"], units[0].added_lines)

    def test_symbol_extraction_detects_python_and_fastapi_symbols(self):
        patch_text = """@@ -1,0 +1,12 @@
+@app.get("/api/github/change-memory/inspect")
+def inspect_project_change():
+    pass
+@router.post('/api/github/change-memory/build')
+async def build_project_change():
+    pass
+class ProjectChangeBuilder:
+    pass
"""

        symbols = project_change.extract_diff_units(raw_input(patch_text, "backend/api_routes.py"))[0].symbols_changed

        self.assertIn("GET /api/github/change-memory/inspect", symbols)
        self.assertIn("inspect_project_change", symbols)
        self.assertIn("POST /api/github/change-memory/build", symbols)
        self.assertIn("build_project_change", symbols)
        self.assertIn("ProjectChangeBuilder", symbols)

    def test_hint_extraction(self):
        cases = [
            (
                "validation_logic",
                "backend/validators.py",
                "@@ -1 +1 @@ def validate_claims():\n+allowed_claims = []\n+forbidden_claims = []",
            ),
            (
                "merge_logic",
                "backend/merge.py",
                "@@ -1 +1 @@\n+final_bullets = merge_staged_resume(candidate)",
            ),
            (
                "retrieval_logic",
                "backend/search.py",
                "@@ -1 +1 @@\n+rerank_evidence(query, retrieve(project_id))",
            ),
            (
                "local_memory_or_cache",
                "backend/memory_store.py",
                "@@ -1 +1 @@\n+project_memory = sqlite_cache.load()",
            ),
            (
                "latex_pipeline",
                "backend/export.py",
                "@@ -1 +1 @@\n+compile_latex_to_pdf(tex_source)",
            ),
            (
                "test_update",
                "tests/test_project_change.py",
                "@@ -1 +1 @@\n+assert unit.unit_id",
            ),
            (
                "ui_debug",
                "frontend/src/pages/GitHubContext.jsx",
                "@@ -1 +1 @@\n+const debugInspect = preview.status",
            ),
            (
                "model_or_schema",
                "backend/project_change_memory.py",
                "@@ -1 +1 @@\n+@dataclass\n+class DiffUnit:",
            ),
        ]

        for expected_hint, file_path, patch_text in cases:
            with self.subTest(expected_hint=expected_hint):
                hints = project_change.extract_diff_units(raw_input(patch_text, file_path))[0].change_hints
                self.assertIn(expected_hint, hints)

        unknown_hints = project_change.extract_diff_units(
            raw_input("@@ -1 +1 @@\n-value = 1\n+value = 2", "src/plain.py")
        )[0].change_hints
        self.assertEqual(["unknown"], unknown_hints)

    def test_unit_id_is_deterministic_and_hunk_sensitive(self):
        first = raw_input("@@ -1 +1 @@\n-value = 1\n+value = 2", "src/plain.py")
        same = raw_input("@@ -1 +1 @@\n-value = 1\n+value = 2", "src/plain.py")
        different = raw_input("@@ -1 +1 @@\n-value = 1\n+value = 3", "src/plain.py")

        first_unit_id = project_change.extract_diff_units(first)[0].unit_id
        same_unit_id = project_change.extract_diff_units(same)[0].unit_id
        different_unit_id = project_change.extract_diff_units(different)[0].unit_id

        self.assertEqual(first_unit_id, same_unit_id)
        self.assertNotEqual(first_unit_id, different_unit_id)


if __name__ == "__main__":
    unittest.main()
