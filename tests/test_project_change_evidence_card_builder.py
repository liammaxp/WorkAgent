import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import project_change_memory as project_change  # noqa: E402


UNSAFE_GENERATED_LANGUAGE = [
    "reduced hallucinations",
    "guaranteed factual correctness",
    "improved performance",
    "improved recall",
    "ATS success",
]


def summary(
    raw_change_type: str,
    *,
    change_id: str = "change_1",
    file_path: str = "backend/example.py",
    symbol: str = "changed_symbol",
    evidence: list[str] | None = None,
    confidence: str = "high",
) -> project_change.RawChangeSummary:
    return project_change.RawChangeSummary(
        change_id=change_id,
        project_id="agent-develop",
        repo="owner/agent-develop",
        commit_sha="abc123",
        file_path=file_path,
        symbols_changed=[symbol] if symbol else [],
        raw_change_types=[raw_change_type],
        what_changed=f"Modified {raw_change_type} behavior in {symbol or file_path}.",
        direct_code_evidence=evidence or [f"Changed symbol: {symbol}.", f"Detected {raw_change_type}."],
        uncertain_intent=[],
        confidence=confidence,
    )


class ProjectChangeEvidenceCardBuilderTests(unittest.TestCase):
    def assert_allowed_fields_are_safe(self, card: project_change.EvidenceCard) -> None:
        generated = " ".join([card.problem, card.mechanism, card.safe_impact, *card.allowed_claims]).lower()
        for unsafe in UNSAFE_GENERATED_LANGUAGE:
            self.assertNotIn(unsafe.lower(), generated)

    def test_validation_summary_builds_safe_card(self):
        card = project_change.build_evidence_card(
            summary(
                "validation_logic_update",
                symbol="validate_resume_quality",
                evidence=[
                    "Changed symbol: validate_resume_quality.",
                    "Added a conditional referencing unsupported_metric.",
                    "Added a failure return branch.",
                ],
            )
        )

        self.assertIn("validation", card.problem)
        self.assertIn("rule-based validation", card.mechanism)
        self.assertEqual("validation_and_safety", card.resume_angle)
        self.assertEqual("none", card.metric_support)
        self.assertTrue(any("unsupported" in claim for claim in card.allowed_claims))
        self.assertTrue(any("hallucinations" in claim for claim in card.forbidden_claims))
        self.assertTrue(any("guaranteed" in claim for claim in card.forbidden_claims))
        self.assert_allowed_fields_are_safe(card)

    def test_category_mappings_are_conservative(self):
        cases = [
            (
                "merge_logic_update",
                "deterministic_merge",
                "deterministic merge",
                "eliminated generic bullets",
            ),
            (
                "retrieval_logic_update",
                "evidence_retrieval",
                "retrieval",
                "improved retrieval recall without measured evidence",
            ),
            (
                "memory_storage_update",
                "project_memory",
                "project-memory",
                "reduced token usage without measured evidence",
            ),
            (
                "chunking_update",
                "content_chunking",
                "segmentation",
                "",
            ),
            (
                "fallback_update",
                "fallback_and_repair",
                "fallback",
                "eliminated pipeline failures",
            ),
            (
                "prompt_constraint_update",
                "generation_constraints",
                "prompt constraints",
                "improved LLM reliability",
            ),
            (
                "quality_gate_update",
                "output_quality_control",
                "quality-gate",
                "guaranteed all generated output is correct",
            ),
            (
                "test_update",
                "testing_and_regression",
                "tests",
                "",
            ),
            (
                "ui_debug_update",
                "developer_observability",
                "debug",
                "",
            ),
        ]

        for raw_change_type, expected_angle, mechanism_term, forbidden in cases:
            with self.subTest(raw_change_type=raw_change_type):
                card = project_change.build_evidence_card(summary(raw_change_type))
                self.assertEqual(expected_angle, card.resume_angle)
                self.assertIn(mechanism_term, card.mechanism.lower())
                if forbidden:
                    self.assertIn(forbidden, card.forbidden_claims)
                self.assert_allowed_fields_are_safe(card)

        merge_card = project_change.build_evidence_card(summary("merge_logic_update"))
        self.assertNotIn("improved", merge_card.safe_impact.lower())

        retrieval_card = project_change.build_evidence_card(summary("retrieval_logic_update"))
        self.assertNotIn("improved recall", " ".join(retrieval_card.allowed_claims).lower())

        memory_card = project_change.build_evidence_card(summary("memory_storage_update"))
        self.assertNotIn("cost reduction", " ".join(memory_card.allowed_claims).lower())

        chunk_card = project_change.build_evidence_card(summary("chunking_update"))
        self.assertNotIn("performance", chunk_card.safe_impact.lower())

        prompt_card = project_change.build_evidence_card(summary("prompt_constraint_update"))
        generated = " ".join([prompt_card.safe_impact, *prompt_card.allowed_claims]).lower()
        self.assertNotIn("llm_reliability", generated)
        self.assertNotIn("hallucination", generated)

        test_card = project_change.build_evidence_card(summary("test_update"))
        self.assertIn("coverage", test_card.safe_impact.lower())

        ui_card = project_change.build_evidence_card(summary("ui_debug_update"))
        self.assertIn("inspectable", ui_card.safe_impact.lower())

    def test_unknown_summary_builds_conservative_fallback(self):
        card = project_change.build_evidence_card(
            summary(
                "unknown",
                symbol="",
                evidence=["The diff contains source-code changes."],
                confidence="low",
            )
        )

        self.assertEqual("implementation_change", card.resume_angle)
        self.assertEqual("low", card.confidence)
        self.assertEqual("The source implementation required a localized code change.", card.problem)
        self.assertEqual("Modified implementation logic in the affected source file.", card.mechanism)
        self.assertLessEqual(project_change.score_evidence_card(card), 6)
        self.assert_allowed_fields_are_safe(card)

    def test_implementation_details_do_not_use_uncertain_intent(self):
        source_summary = summary("retrieval_logic_update")
        source_summary = project_change.RawChangeSummary(
            **{
                **project_change.model_to_dict(source_summary),
                "uncertain_intent": ["The retrieval change may be intended to adjust evidence selection."],
            }
        )

        card = project_change.build_evidence_card(source_summary)

        self.assertFalse(any("may be intended" in detail for detail in card.implementation_details))

    def test_forbidden_language_guard_replaces_unsafe_allowed_fields(self):
        self.assertTrue(project_change.contains_unsupported_evidence_claim("guaranteed factual correctness"))
        guarded = project_change.guard_allowed_evidence_text(
            "guaranteed factual correctness",
            "modified implementation logic in the affected module",
        )

        self.assertEqual("modified implementation logic in the affected module", guarded)


if __name__ == "__main__":
    unittest.main()
