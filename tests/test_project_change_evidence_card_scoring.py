import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import project_change_memory as project_change  # noqa: E402


def summary(
    raw_change_type: str,
    *,
    change_id: str = "change_1",
    evidence: list[str] | None = None,
    confidence: str = "high",
) -> project_change.RawChangeSummary:
    return project_change.RawChangeSummary(
        change_id=change_id,
        project_id="agent-develop",
        repo="owner/agent-develop",
        commit_sha="abc123",
        file_path="backend/example.py",
        symbols_changed=["validate_resume_quality"],
        raw_change_types=[raw_change_type],
        what_changed="Added or modified validation logic in validate_resume_quality.",
        direct_code_evidence=evidence or [
            "Changed symbol: validate_resume_quality.",
            "Added a conditional referencing unsupported_metric.",
        ],
        uncertain_intent=[],
        confidence=confidence,
    )


class ProjectChangeEvidenceCardScoringTests(unittest.TestCase):
    def test_metric_support_distinguishes_explicit_ambiguous_and_none(self):
        ambiguous = project_change.build_evidence_card(
            summary(
                "chunking_update",
                evidence=[
                    "Configured chunk size 18000.",
                    "Configured top_k 20.",
                    "Configured threshold 0.7.",
                    "Configured retry count 3.",
                ],
            )
        )
        explicit = project_change.build_evidence_card(
            summary(
                "retrieval_logic_update",
                evidence=["Reduced processing time from 10 seconds to 6 seconds."],
            )
        )
        none = project_change.build_evidence_card(summary("validation_logic_update"))

        self.assertEqual("ambiguous", ambiguous.metric_support)
        self.assertEqual("explicit", explicit.metric_support)
        self.assertEqual("none", none.metric_support)

    def test_evidence_id_is_deterministic_and_source_sensitive(self):
        first = project_change.build_evidence_card(summary("validation_logic_update", change_id="change_same"))
        same = project_change.build_evidence_card(summary("validation_logic_update", change_id="change_same"))
        different = project_change.build_evidence_card(summary("merge_logic_update", change_id="change_different"))

        self.assertEqual(first.evidence_id, same.evidence_id)
        self.assertNotEqual(first.evidence_id, different.evidence_id)

    def test_strong_card_scores_at_least_formal_memory_threshold(self):
        card = project_change.build_evidence_card(summary("validation_logic_update"))

        self.assertGreaterEqual(project_change.score_evidence_card(card), 6)

    def test_weak_unknown_card_scores_lower(self):
        card = project_change.build_evidence_card(
            project_change.RawChangeSummary(
                change_id="weak_change",
                project_id="agent-develop",
                repo="owner/agent-develop",
                commit_sha=None,
                file_path="backend/plain.py",
                symbols_changed=[],
                raw_change_types=["unknown"],
                what_changed="Modified implementation logic in backend/plain.py.",
                direct_code_evidence=["The diff contains source-code changes."],
                uncertain_intent=[],
                confidence="low",
            )
        )

        self.assertLess(project_change.score_evidence_card(card), 6)

    def test_empty_source_ids_and_details_are_penalized(self):
        card = project_change.EvidenceCard(
            evidence_id="manual",
            project_id="agent-develop",
            source_change_ids=[],
            problem="Generated output lacked a validation rule for unsupported claims or metrics.",
            mechanism="Added rule-based validation for unsupported claims or metrics.",
            implementation_details=[],
            safe_impact="Added an explicit safeguard against unsupported generated claims.",
            resume_angle="validation_and_safety",
            confidence="medium",
            metric_support="none",
            allowed_claims=["added validation for unsupported generated claims"],
            forbidden_claims=["guaranteed factual correctness"],
        )

        self.assertLess(project_change.score_evidence_card(card), 6)

    def test_unsupported_claims_are_heavily_penalized(self):
        card = project_change.EvidenceCard(
            evidence_id="unsafe",
            project_id="agent-develop",
            source_change_ids=["change_1"],
            problem="Generated output lacked a validation rule.",
            mechanism="Added rule-based validation.",
            implementation_details=["Changed symbol: validate_resume_quality."],
            safe_impact="Guaranteed factual correctness and improved performance.",
            resume_angle="validation_and_safety",
            confidence="high",
            metric_support="none",
            allowed_claims=["guaranteed factual correctness"],
            forbidden_claims=["guaranteed factual correctness"],
        )

        self.assertTrue(project_change.contains_unsupported_evidence_claim(card.safe_impact))
        self.assertLess(project_change.score_evidence_card(card), 6)


if __name__ == "__main__":
    unittest.main()
