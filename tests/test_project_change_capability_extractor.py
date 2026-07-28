import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import project_change_memory as project_change  # noqa: E402


def card(
    evidence_id: str,
    resume_angle: str,
    mechanism: str,
    *,
    project_id: str = "ProjectA",
    details: list[str] | None = None,
    allowed: list[str] | None = None,
    forbidden: list[str] | None = None,
    confidence: str = "high",
    metric_support: str = "none",
    source_change_ids: list[str] | None = None,
    problem: str = "The changed behavior required explicit implementation support.",
    safe_impact: str = "Recorded source-backed implementation behavior.",
) -> project_change.EvidenceCard:
    return project_change.EvidenceCard(
        evidence_id=evidence_id,
        project_id=project_id,
        source_change_ids=source_change_ids or [f"change_{evidence_id}"],
        problem=problem,
        mechanism=mechanism,
        implementation_details=details or [f"Changed mechanism: {mechanism}"],
        safe_impact=safe_impact,
        resume_angle=resume_angle,
        confidence=confidence,
        metric_support=metric_support,
        allowed_claims=allowed or [mechanism.lower()],
        forbidden_claims=forbidden or ["guaranteed factual correctness"],
    )


def facts_by_type(cards: list[project_change.EvidenceCard], min_score: int = 6) -> dict[str, project_change.CapabilityFact]:
    facts = project_change.extract_capability_facts(cards, min_evidence_score=min_score)
    return {fact.capability_type: fact for fact in facts}


class ProjectChangeCapabilityExtractorTests(unittest.TestCase):
    def test_retrieval_capability_does_not_claim_recall(self):
        retrieval = card(
            "retrieval_1",
            "evidence_retrieval",
            "Updated evidence query, retrieval, ranking, and filtering logic.",
            details=["Added query and rerank rules for evidence selection."],
            allowed=["implemented project evidence retrieval and ranking logic"],
        )

        fact = facts_by_type([retrieval])["retrieval_and_reranking"]

        self.assertEqual("retrieval_and_reranking", fact.capability_type)
        self.assertNotIn("improved recall", " ".join(fact.allowed_resume_claims).lower())

    def test_local_memory_does_not_create_token_or_cost_reduction(self):
        memory = card(
            "memory_1",
            "project_memory",
            "Added structured local project-memory persistence with SQLite cache reuse.",
            details=["Persisted structured project-analysis state in SQLite cache."],
            allowed=["implemented structured local project-memory persistence"],
        )

        facts = facts_by_type([memory])

        self.assertIn("local_project_memory", facts)
        self.assertNotIn("token_or_cost_reduction", facts)

    def test_deterministic_latex_validation_requires_latex_terms(self):
        latex = card(
            "latex_1",
            "output_quality_control",
            "Added compile validation for LaTeX and PDF generation.",
            details=["Validated LaTeX structure before PDF compilation."],
            allowed=["implemented deterministic checks for LaTeX or PDF generation"],
        )
        generic_validator = card(
            "validator_1",
            "validation_and_safety",
            "Added rule-based validation for unsupported claims.",
            details=["Added unsupported-claim validation."],
        )

        self.assertIn("deterministic_latex_validation", facts_by_type([latex]))
        self.assertNotIn("deterministic_latex_validation", facts_by_type([generic_validator]))

    def test_template_pollution_requires_direct_contamination_evidence(self):
        pollution = card(
            "pollution_1",
            "output_quality_control",
            "Added a pollution blocker for template pollution and placeholder contamination.",
            details=["Blocked forbidden template terms and cross-project technology contamination."],
            allowed=["blocked explicitly identified template contamination patterns"],
        )
        generic_quality = card(
            "quality_1",
            "output_quality_control",
            "Added post-generation quality-gate checks.",
            details=["Added generated output checks."],
        )

        self.assertIn("template_pollution_blocking", facts_by_type([pollution]))
        self.assertNotIn("template_pollution_blocking", facts_by_type([generic_quality]))

    def test_output_quality_control_includes_correctness_boundaries(self):
        quality = card(
            "quality_2",
            "output_quality_control",
            "Added post-generation quality-gate checks.",
            details=["Checked final output acceptance conditions."],
            allowed=["implemented post-generation quality checks"],
        )

        fact = facts_by_type([quality])["output_quality_control"]

        self.assertIn("guaranteed factual correctness", fact.forbidden_claims)

    def test_validation_and_repair_preserves_present_mechanisms(self):
        validation = card(
            "validation_1",
            "validation_and_safety",
            "Added rule-based validation for unsupported claim detection.",
            details=["Added validator branch for unsupported claims."],
        )
        fallback = card(
            "fallback_1",
            "fallback_and_repair",
            "Added explicit fallback and repair path for failed-state handling.",
            details=["Added retry path and repair handling."],
        )

        fact = facts_by_type([validation, fallback])["validation_and_repair"]
        mechanisms = " ".join(fact.mechanisms).lower()

        self.assertIn("validation", mechanisms)
        self.assertIn("fallback", mechanisms)

    def test_evidence_grounded_generation_requires_evidence_plus_guard(self):
        retrieval = card(
            "retrieval_2",
            "evidence_retrieval",
            "Updated evidence retrieval and ranking logic.",
            details=["Retrieved project evidence for generated output."],
        )
        validation = card(
            "validation_2",
            "validation_and_safety",
            "Added rule-based validation for unsupported generated claims.",
            details=["Checked generated claims against supported evidence."],
        )
        prompt = card(
            "prompt_1",
            "generation_constraints",
            "Added explicit prompt constraints for generated output.",
            details=["Added structured output instructions."],
        )

        self.assertNotIn("evidence_grounded_generation", facts_by_type([retrieval]))
        self.assertNotIn("evidence_grounded_generation", facts_by_type([prompt]))
        self.assertIn("evidence_grounded_generation", facts_by_type([retrieval, validation]))

    def test_llm_reliability_strict_gate(self):
        prompt = card(
            "prompt_2",
            "generation_constraints",
            "Added explicit prompt constraints for generated output.",
            details=["Added structured output instructions."],
        )
        retrieval = card(
            "retrieval_3",
            "evidence_retrieval",
            "Updated retrieval grounding with evidence query and ranking logic.",
            details=["Retrieved evidence for generated output."],
        )
        validation = card(
            "validation_3",
            "validation_and_safety",
            "Added unsupported-claim validation for generated output.",
            details=["Added unsupported claim quality gate."],
        )

        self.assertNotIn("llm_reliability", facts_by_type([prompt]))
        self.assertNotIn("llm_reliability", facts_by_type([retrieval]))

        fact = facts_by_type([retrieval, validation])["llm_reliability"]
        allowed = " ".join(fact.allowed_resume_claims).lower()

        self.assertNotIn("reduced hallucinations", allowed)
        self.assertNotIn("improved factuality", allowed)
        self.assertNotIn("guaranteed reliability", allowed)
        self.assertTrue(any("retrieval" in claim or "evidence" in claim for claim in fact.allowed_resume_claims))

    def test_token_or_cost_reduction_strict_gate(self):
        cache_only = card(
            "cache_1",
            "project_memory",
            "Added SQLite cache for local project memory.",
            details=["Stored project-analysis state in cache."],
        )
        incremental = card(
            "incremental_1",
            "content_chunking",
            "Added diff-based incremental analysis replaced repeated full-repository processing.",
            details=["Used diff-based incremental analysis instead of repeated full-repository processing."],
            allowed=["implemented diff-based incremental analysis instead of repeated full-repository processing"],
        )
        measured = card(
            "token_metric_1",
            "content_chunking",
            "Added measured token reduction support.",
            details=["Reduced API token usage by 32%."],
            allowed=["implemented measured token usage tracking"],
            metric_support="explicit",
        )

        cache_facts = facts_by_type([cache_only])
        self.assertIn("local_project_memory", cache_facts)
        self.assertNotIn("token_or_cost_reduction", cache_facts)

        incremental_fact = facts_by_type([incremental])["token_or_cost_reduction"]
        self.assertEqual("none", incremental_fact.metric_support)
        self.assertNotIn("by 32%", " ".join(incremental_fact.allowed_resume_claims))

        measured_fact = facts_by_type([measured])["token_or_cost_reduction"]
        self.assertEqual("explicit", measured_fact.metric_support)

    def test_weak_card_filtering_and_min_score(self):
        weak = card(
            "weak_1",
            "implementation_change",
            "Modified implementation logic in the affected source file.",
            details=[],
            confidence="low",
        )

        self.assertEqual([], project_change.extract_capability_facts([weak]))
        self.assertEqual([], project_change.extract_capability_facts([weak], min_evidence_score=0))

    def test_multi_project_separation(self):
        project_a = card(
            "retrieval_a",
            "evidence_retrieval",
            "Updated evidence retrieval and ranking logic.",
            project_id="ProjectA",
            details=["ProjectA query and ranking logic."],
        )
        project_b = card(
            "retrieval_b",
            "evidence_retrieval",
            "Updated evidence retrieval and ranking logic.",
            project_id="ProjectB",
            details=["ProjectB query and ranking logic."],
        )

        facts = project_change.extract_capability_facts([project_b, project_a])
        retrieval_facts = [fact for fact in facts if fact.capability_type == "retrieval_and_reranking"]

        self.assertEqual(["ProjectA", "ProjectB"], [fact.project_id for fact in retrieval_facts])
        self.assertEqual(["retrieval_a"], retrieval_facts[0].source_evidence_ids)
        self.assertEqual(["retrieval_b"], retrieval_facts[1].source_evidence_ids)
        self.assertNotIn("ProjectB", " ".join(retrieval_facts[0].mechanisms))
        self.assertNotIn("ProjectA", " ".join(retrieval_facts[1].mechanisms))

    def test_deterministic_ids_and_order(self):
        retrieval = card(
            "retrieval_det",
            "evidence_retrieval",
            "Updated evidence retrieval and ranking logic.",
        )
        validation = card(
            "validation_det",
            "validation_and_safety",
            "Added unsupported-claim validation for generated output.",
            details=["Added unsupported claim quality gate."],
        )
        changed = card(
            "validation_changed",
            "validation_and_safety",
            "Added unsupported-claim validation for generated output.",
            details=["Added unsupported claim quality gate."],
        )

        first = [project_change.model_to_dict(fact) for fact in project_change.extract_capability_facts([retrieval, validation])]
        second = [project_change.model_to_dict(fact) for fact in project_change.extract_capability_facts([validation, retrieval])]
        changed_facts = {
            fact.capability_type: fact
            for fact in project_change.extract_capability_facts([retrieval, changed])
        }
        original_facts = {
            fact.capability_type: fact
            for fact in project_change.extract_capability_facts([retrieval, validation])
        }

        self.assertEqual(first, second)
        self.assertNotEqual(
            original_facts["evidence_grounded_generation"].capability_id,
            changed_facts["evidence_grounded_generation"].capability_id,
        )

    def test_metric_aggregation_and_no_unrelated_leakage(self):
        token_none = card(
            "token_none",
            "content_chunking",
            "Added diff-based incremental analysis replaced repeated full-repository processing.",
            details=["Used diff-based incremental analysis instead of repeated full-repository processing."],
        )
        token_ambiguous = card(
            "token_ambiguous",
            "content_chunking",
            "Added diff-based incremental analysis replaced repeated full-repository processing.",
            details=["Configured chunk_size = 18000."],
            metric_support="ambiguous",
        )
        token_explicit = card(
            "token_explicit",
            "content_chunking",
            "Added reduced API token usage evidence.",
            details=["Reduced API token usage by 32%."],
            metric_support="explicit",
        )
        retrieval_explicit = card(
            "retrieval_metric",
            "evidence_retrieval",
            "Updated evidence retrieval and ranking logic.",
            details=["Reduced processing time from 10 seconds to 6 seconds."],
            metric_support="explicit",
        )

        self.assertEqual("none", facts_by_type([token_none])["token_or_cost_reduction"].metric_support)
        self.assertEqual("ambiguous", facts_by_type([token_ambiguous])["token_or_cost_reduction"].metric_support)
        self.assertEqual("explicit", facts_by_type([token_explicit])["token_or_cost_reduction"].metric_support)
        self.assertEqual(
            "none",
            facts_by_type([token_none, retrieval_explicit])["token_or_cost_reduction"].metric_support,
        )

    def test_claim_boundary_inheritance_and_unsafe_allowed_removal(self):
        retrieval = card(
            "retrieval_boundary",
            "evidence_retrieval",
            "Updated evidence retrieval and ranking logic.",
            allowed=["improved recall", "implemented project evidence retrieval and ranking logic"],
            forbidden=["source-specific forbidden boundary"],
        )

        fact = facts_by_type([retrieval])["retrieval_and_reranking"]
        allowed = " ".join(fact.allowed_resume_claims).lower()

        self.assertIn("source-specific forbidden boundary", fact.forbidden_claims)
        self.assertNotIn("improved recall", allowed)
        self.assertFalse(set(fact.allowed_resume_claims) & set(fact.forbidden_claims))

    def test_duplicate_cards_do_not_duplicate_fields(self):
        retrieval = card(
            "retrieval_dup",
            "evidence_retrieval",
            "Updated evidence retrieval and ranking logic.",
            details=["Added query and ranking logic."],
            allowed=["implemented project evidence retrieval and ranking logic"],
            forbidden=["guaranteed all relevant evidence was found"],
        )

        fact = facts_by_type([retrieval, retrieval])["retrieval_and_reranking"]

        self.assertEqual(["retrieval_dup"], fact.source_evidence_ids)
        self.assertEqual(len(fact.mechanisms), len(set(fact.mechanisms)))
        self.assertEqual(len(fact.allowed_resume_claims), len(set(fact.allowed_resume_claims)))
        self.assertEqual(len(fact.forbidden_claims), len(set(fact.forbidden_claims)))

    def test_unsupported_cards_create_no_capabilities(self):
        testing = card(
            "test_only",
            "testing_and_regression",
            "Added deterministic tests for changed behavior.",
        )
        ui = card(
            "ui_only",
            "developer_observability",
            "Added inspect or debug visibility for internal processing state.",
        )
        unknown = card(
            "unknown_only",
            "implementation_change",
            "Modified implementation logic in the affected source file.",
            confidence="low",
        )

        self.assertEqual([], project_change.extract_capability_facts([testing, ui, unknown]))


if __name__ == "__main__":
    unittest.main()
