"""Strict, immutable project evidence capability taxonomy.

This module defines capability vocabulary and evidence requirements only.  It
does not inspect Evidence Facts, extract signals, assign capabilities, persist
data, or call external services.

Required signal groups use AND-of-OR semantics: at least one signal from every
group is required.  Group order is meaningful and preserved; alternatives
inside each group and set-like evidence collections are stored alphabetically.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Iterable, Mapping

from backend.project_evidence_models import EvidenceType


MAX_VALIDATION_MESSAGES = 100
_IDENTIFIER_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_UNSAFE_TEMPLATE_RE = re.compile(
    r"(?:\b(?:eliminated|guaranteed|production[- ]scale|enterprise[- ]grade)\b|"
    r"\bhallucinations?\s+reduction\b|\b\d+(?:\.\d+)?\s*%|\bx\s*%)",
    re.IGNORECASE,
)
_PROJECT_TECHNOLOGY_RE = re.compile(
    r"\b(?:python|react|fastapi|sqlite|chroma|latex|openai|anthropic|firebase|mongodb)\b",
    re.IGNORECASE,
)
_VALID_EVIDENCE_TYPES = frozenset(item.value for item in EvidenceType)
_VALID_SOURCE_CATEGORIES = frozenset({
    "capability_context",
    "direct_evidence",
    "other",
    "project_context",
})


@dataclass(frozen=True)
class ProjectCapabilityDefinition:
    capability_type: str
    display_name: str
    description: str
    required_signal_groups: tuple[tuple[str, ...], ...]
    supporting_signals: tuple[str, ...]
    accepted_evidence_types: tuple[str, ...]
    contextual_evidence_types: tuple[str, ...]
    unsupported_evidence_types: tuple[str, ...]
    accepted_source_categories: tuple[str, ...]
    contextual_source_categories: tuple[str, ...]
    unsupported_source_categories: tuple[str, ...]
    minimum_quality_score: int
    minimum_direct_fact_count: int
    minimum_total_fact_count: int
    requires_direct_provenance: bool
    allows_contextual_support: bool
    high_risk: bool
    forbidden_inferences: tuple[str, ...]
    safe_claim_templates: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectCapabilityOverlapRule:
    left_capability: str
    right_capability: str
    shared_signals: tuple[str, ...]
    left_distinguishing_signals: tuple[str, ...]
    right_distinguishing_signals: tuple[str, ...]
    both_allowed_when: str
    non_implication_rule: str


@dataclass(frozen=True)
class ProjectCapabilityTaxonomyValidationReport:
    valid: bool
    capability_count: int
    signal_count: int
    alias_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


SIGNAL_REGISTRY = tuple(sorted({
    "allowed_forbidden_claim_handling",
    "atomic_persistence",
    "backend_route",
    "blocked_term_check",
    "branching_gate",
    "cache_reuse",
    "candidate_filtering",
    "canonical_serialization",
    "change_summary_generation",
    "changed_file_detection",
    "claim_validation",
    "compatibility_check",
    "context_compression",
    "database_storage",
    "deterministic_ordering",
    "diff_only_analysis",
    "diff_processing",
    "document_merge",
    "error_state_handling",
    "evidence_card_generation",
    "evidence_selection",
    "exact_deduplication",
    "factuality_evaluation",
    "failure_case_coverage",
    "failure_detection",
    "failure_propagation",
    "fallback",
    "field_normalization",
    "frontend_api_call",
    "frontend_state_handling",
    "generic_content_blocking",
    "integrity_conflict_detection",
    "integrity_hash",
    "incremental_update",
    "invalid_section_detection",
    "invariant_check",
    "latex_compile_check",
    "latex_repair",
    "latex_validation",
    "load_validate_write_lifecycle",
    "low_evidence_refusal",
    "measured_input_reduction",
    "metric_support_validation",
    "output_evidence_validation",
    "persistent_storage",
    "prior_state_comparison",
    "project_memory_read",
    "project_memory_write",
    "prompt_constraint",
    "quality_dimensions",
    "query_operation",
    "rag_terminology",
    "regression_testing",
    "repeated_input_avoidance",
    "repair",
    "reranking",
    "response_schema",
    "retrieval",
    "retry",
    "safe_degradation",
    "schema_validation",
    "schema_versioning",
    "section_ordering",
    "selective_context_expansion",
    "source_attribution",
    "source_bound_transformation",
    "source_grounding",
    "source_linkage",
    "source_ownership_validation",
    "stable_identity",
    "stable_project_identity",
    "stage_io_contract",
    "state_transition",
    "status_band_decision",
    "structured_document_assembly",
    "structured_extraction",
    "structured_output_validation",
    "targeted_testing",
    "template_constraint",
    "template_pollution_detection",
    "unsupported_claim_blocking",
    "unsupported_template_filtering",
    "user_triggered_workflow",
    "validation_gate",
    "workflow_staging",
}))


FORBIDDEN_INFERENCE_REGISTRY = tuple(sorted({
    "api_contract_implies_deployment",
    "backend_route_alone_proves_integration",
    "cache_alone_proves_efficiency",
    "calling_functions_proves_orchestration",
    "citations_alone_prove_llm_reliability",
    "claim_validation_implies_llm_reliability",
    "complete_test_coverage",
    "context_in_prompt_proves_grounding",
    "data_persistence_implies_project_memory",
    "diff_alone_proves_efficiency",
    "diff_file_alone_proves_incremental_processing",
    "document_generation_implies_latex_repair",
    "durability_guarantee",
    "eliminated_hallucinations",
    "eliminated_template_pollution",
    "enterprise_grade_output",
    "escaping_character_proves_latex_repair",
    "factual_output_guarantee",
    "frontend_component_alone_proves_integration",
    "generic_parsing_proves_structured_extraction",
    "generic_validation_proves_quality_control",
    "grammar_checking_proves_claim_validation",
    "high_availability",
    "high_precision_retrieval",
    "high_recall_retrieval",
    "improved_latency",
    "improved_retrieval_quality",
    "in_memory_dictionary_proves_project_memory",
    "json_object_proves_project_memory",
    "json_output_alone_proves_llm_reliability",
    "json_use_proves_structured_extraction",
    "keyword_filtering_proves_template_blocking",
    "latex_export_proves_validation_repair",
    "latex_template_proves_validation_repair",
    "llm_call_proves_document_generation",
    "llm_use_proves_grounding",
    "logging_alone_proves_recovery",
    "manual_review_proves_quality_control",
    "multi_agent_orchestration",
    "one_happy_path_test_proves_hardening",
    "one_json_object_proves_persistence",
    "one_time_patch_parse_proves_incremental_processing",
    "parallel_execution",
    "persistence_implies_scalability",
    "printing_json_proves_persistence",
    "production_deployment",
    "production_ready",
    "project_memory_alone_proves_grounding",
    "prompt_constraints_alone_prove_llm_reliability",
    "prompt_presence_proves_structured_extraction",
    "pytest_mention_proves_hardening",
    "rag_term_proves_grounding",
    "rag_term_proves_llm_reliability",
    "reading_files_proves_structured_extraction",
    "reduced_api_cost",
    "reduced_hallucinations",
    "reduced_token_usage",
    "responsive_user_experience",
    "retrieval_alone_proves_llm_reliability",
    "retrieval_implies_hallucination_reduction",
    "returning_none_proves_recovery",
    "returning_object_proves_persistence",
    "schema_validation_alone_proves_claim_validation",
    "search_word_proves_retrieval_reranking",
    "sha_field_proves_incremental_processing",
    "sha_use_proves_normalization",
    "simple_lookup_proves_retrieval_reranking",
    "single_conditional_proves_quality_control",
    "sorting_alone_proves_normalization",
    "string_creation_proves_document_generation",
    "structured_extraction_implies_normalization",
    "structured_output_alone_proves_llm_reliability",
    "summaries_alone_prove_efficiency",
    "template_alone_proves_document_generation",
    "test_count_proves_hardening",
    "tests_imply_bug_free",
    "tests_imply_high_reliability",
    "temporary_context_proves_project_memory",
    "temporary_file_proves_persistence",
    "try_except_alone_proves_recovery",
    "validation_implies_production_reliability",
    "whitespace_removal_proves_normalization",
    "workflow_implies_distributed_execution",
}))


def _evidence_policy(
    *accepted: str,
    contextual: tuple[str, ...] = (EvidenceType.FEATURE.value, EvidenceType.UNKNOWN.value),
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    accepted_values = tuple(sorted(set(accepted)))
    contextual_values = tuple(sorted(set(contextual) - set(accepted_values)))
    unsupported_values = tuple(sorted(
        _VALID_EVIDENCE_TYPES - set(accepted_values) - set(contextual_values)
    ))
    return accepted_values, contextual_values, unsupported_values


def _definition(
    capability_type: str,
    display_name: str,
    description: str,
    required: tuple[tuple[str, ...], ...],
    supporting: tuple[str, ...],
    accepted_types: tuple[str, ...],
    forbidden: tuple[str, ...],
    template: str,
    *,
    minimum_quality_score: int = 60,
    minimum_direct_fact_count: int = 1,
    minimum_total_fact_count: int = 1,
    allows_contextual_support: bool = True,
    high_risk: bool = False,
    notes: tuple[str, ...] = (),
) -> ProjectCapabilityDefinition:
    accepted, contextual, unsupported = _evidence_policy(*accepted_types)
    return ProjectCapabilityDefinition(
        capability_type=capability_type,
        display_name=display_name,
        description=description,
        required_signal_groups=tuple(tuple(sorted(group)) for group in required),
        supporting_signals=tuple(sorted(supporting)),
        accepted_evidence_types=accepted,
        contextual_evidence_types=contextual,
        unsupported_evidence_types=unsupported,
        accepted_source_categories=("direct_evidence",),
        contextual_source_categories=("project_context",),
        unsupported_source_categories=("capability_context", "other"),
        minimum_quality_score=minimum_quality_score,
        minimum_direct_fact_count=minimum_direct_fact_count,
        minimum_total_fact_count=minimum_total_fact_count,
        requires_direct_provenance=True,
        allows_contextual_support=allows_contextual_support,
        high_risk=high_risk,
        forbidden_inferences=forbidden,
        safe_claim_templates=(template,),
        notes=notes,
    )


_DEFINITIONS = (
    _definition(
        "claim_validation",
        "Claim Validation",
        "Checks whether claims remain within explicit evidence and metric-support boundaries.",
        (("allowed_forbidden_claim_handling", "claim_validation", "metric_support_validation", "unsupported_claim_blocking"),
         ("output_evidence_validation", "source_attribution", "source_grounding")),
        ("validation_gate",),
        (EvidenceType.VALIDATION.value, EvidenceType.WORKFLOW.value),
        ("grammar_checking_proves_claim_validation", "schema_validation_alone_proves_claim_validation",
         "factual_output_guarantee", "claim_validation_implies_llm_reliability"),
        "Validated {bounded_claim} against {evidence_boundary}.",
        minimum_quality_score=65,
    ),
    _definition(
        "data_persistence",
        "Data Persistence",
        "Stores structured or versioned artifacts through an integrity-aware persistence lifecycle.",
        (("atomic_persistence", "database_storage", "persistent_storage"),
         ("integrity_hash", "load_validate_write_lifecycle", "schema_versioning")),
        ("source_linkage", "stable_identity"),
        (EvidenceType.ARCHITECTURE.value, EvidenceType.DATA_PERSISTENCE.value, EvidenceType.WORKFLOW.value),
        ("temporary_file_proves_persistence", "printing_json_proves_persistence", "returning_object_proves_persistence",
         "persistence_implies_scalability", "high_availability", "durability_guarantee", "improved_latency",
         "data_persistence_implies_project_memory"),
        "Stored {bounded_artifact} through {integrity_control}.",
    ),
    _definition(
        "deterministic_document_generation",
        "Deterministic Document Generation",
        "Produces structured documents through explicit composition, ordering, and template constraints.",
        (("document_merge", "structured_document_assembly"),
         ("deterministic_ordering", "section_ordering", "template_constraint")),
        ("schema_validation",),
        (EvidenceType.ARCHITECTURE.value, EvidenceType.VALIDATION.value, EvidenceType.WORKFLOW.value),
        ("string_creation_proves_document_generation", "llm_call_proves_document_generation",
         "template_alone_proves_document_generation", "document_generation_implies_latex_repair"),
        "Composed {bounded_document} using {ordering_rule}.",
    ),
    _definition(
        "deterministic_evidence_normalization",
        "Deterministic Evidence Normalization",
        "Normalizes and deduplicates evidence using stable identity and canonicalization rules.",
        (("canonical_serialization", "stable_identity"),
         ("deterministic_ordering", "exact_deduplication", "field_normalization")),
        ("integrity_conflict_detection", "schema_validation"),
        (EvidenceType.ARCHITECTURE.value, EvidenceType.VALIDATION.value, EvidenceType.WORKFLOW.value),
        ("sorting_alone_proves_normalization", "sha_use_proves_normalization",
         "whitespace_removal_proves_normalization", "structured_extraction_implies_normalization"),
        "Normalized {bounded_record} using {identity_rule}.",
    ),
    _definition(
        "evidence_grounded_generation",
        "Evidence-Grounded Generation",
        "Constrains generated output using explicit, source-bound project evidence and validation controls.",
        (("evidence_selection", "retrieval", "source_attribution", "source_grounding"),
         ("claim_validation", "output_evidence_validation", "unsupported_claim_blocking")),
        ("low_evidence_refusal", "structured_output_validation"),
        (EvidenceType.ARCHITECTURE.value, EvidenceType.RETRIEVAL.value, EvidenceType.VALIDATION.value, EvidenceType.WORKFLOW.value),
        ("llm_use_proves_grounding", "project_memory_alone_proves_grounding", "context_in_prompt_proves_grounding",
         "rag_term_proves_grounding", "factual_output_guarantee", "reduced_hallucinations"),
        "Applied {grounding_control} to constrain {bounded_output}.",
        minimum_quality_score=65,
    ),
    _definition(
        "failure_recovery",
        "Failure Recovery",
        "Detects supported failure states and applies retry, fallback, repair, or safe degradation.",
        (("error_state_handling", "failure_detection"),
         ("fallback", "repair", "retry", "safe_degradation")),
        ("failure_propagation", "validation_gate"),
        (EvidenceType.BUG_FIX.value, EvidenceType.FAILURE_RECOVERY.value, EvidenceType.VALIDATION.value, EvidenceType.WORKFLOW.value),
        ("try_except_alone_proves_recovery", "logging_alone_proves_recovery",
         "returning_none_proves_recovery", "factual_output_guarantee"),
        "Handled {bounded_failure} through {recovery_path}.",
    ),
    _definition(
        "frontend_backend_integration",
        "Frontend-Backend Integration",
        "Connects a frontend workflow to backend APIs through explicit request, response, and state contracts.",
        (("backend_route", "response_schema"),
         ("frontend_api_call", "frontend_state_handling", "user_triggered_workflow")),
        ("stage_io_contract",),
        (EvidenceType.ARCHITECTURE.value, EvidenceType.INTEGRATION.value, EvidenceType.WORKFLOW.value),
        ("backend_route_alone_proves_integration", "frontend_component_alone_proves_integration",
         "api_contract_implies_deployment", "production_deployment", "responsive_user_experience"),
        "Connected {bounded_frontend_flow} to {bounded_backend_contract}.",
    ),
    _definition(
        "incremental_change_processing",
        "Incremental Change Processing",
        "Processes commits, diffs, changed files, or prior state through an incremental lifecycle.",
        (("changed_file_detection", "diff_processing", "prior_state_comparison"),
         ("exact_deduplication", "incremental_update", "repeated_input_avoidance")),
        ("cache_reuse", "stable_identity"),
        (EvidenceType.ARCHITECTURE.value, EvidenceType.OPTIMIZATION.value, EvidenceType.WORKFLOW.value),
        ("diff_file_alone_proves_incremental_processing", "one_time_patch_parse_proves_incremental_processing",
         "sha_field_proves_incremental_processing", "reduced_token_usage", "reduced_api_cost", "improved_latency"),
        "Processed {bounded_change_set} through {incremental_rule}.",
    ),
    _definition(
        "latex_validation_and_repair",
        "LaTeX Validation and Repair",
        "Validates document structure or compile results and repairs explicitly supported failures.",
        (("latex_compile_check", "latex_validation"),
         ("invalid_section_detection", "latex_repair")),
        ("failure_detection", "structured_document_assembly"),
        (EvidenceType.FAILURE_RECOVERY.value, EvidenceType.TESTING.value, EvidenceType.VALIDATION.value),
        ("latex_export_proves_validation_repair", "latex_template_proves_validation_repair",
         "escaping_character_proves_latex_repair", "factual_output_guarantee"),
        "Validated {bounded_document_output} and repaired {supported_failure}.",
    ),
    _definition(
        "llm_reliability",
        "LLM Reliability Controls",
        "Combines multiple independent controls for unsupported or structurally invalid model output.",
        (("source_attribution", "source_grounding"),
         ("claim_validation", "unsupported_claim_blocking"),
         ("factuality_evaluation", "low_evidence_refusal", "structured_output_validation")),
        ("prompt_constraint", "rag_terminology", "validation_gate"),
        (EvidenceType.ARCHITECTURE.value, EvidenceType.RETRIEVAL.value, EvidenceType.TESTING.value, EvidenceType.VALIDATION.value, EvidenceType.WORKFLOW.value),
        ("prompt_constraints_alone_prove_llm_reliability", "json_output_alone_proves_llm_reliability",
         "rag_term_proves_llm_reliability", "citations_alone_prove_llm_reliability",
         "structured_output_alone_proves_llm_reliability", "retrieval_alone_proves_llm_reliability",
         "eliminated_hallucinations", "factual_output_guarantee"),
        "Combined {grounding_control} with {validation_control} for {bounded_output}.",
        minimum_quality_score=70,
        minimum_direct_fact_count=2,
        minimum_total_fact_count=2,
        allows_contextual_support=False,
        high_risk=True,
    ),
    _definition(
        "output_quality_control",
        "Output Quality Control",
        "Evaluates or filters synthesized output through explicit quality dimensions and gates.",
        (("generic_content_blocking", "quality_dimensions", "status_band_decision", "validation_gate"),),
        ("failure_detection", "structured_output_validation"),
        (EvidenceType.TESTING.value, EvidenceType.VALIDATION.value, EvidenceType.WORKFLOW.value),
        ("manual_review_proves_quality_control", "single_conditional_proves_quality_control",
         "generic_validation_proves_quality_control", "factual_output_guarantee",
         "validation_implies_production_reliability"),
        "Applied {quality_rule} before accepting {bounded_output}.",
    ),
    _definition(
        "project_memory_management",
        "Project Memory Management",
        "Creates, reads, updates, and validates persistent structured project memory with stable identity.",
        (("persistent_storage", "project_memory_write"),
         ("load_validate_write_lifecycle", "project_memory_read"),
         ("schema_versioning", "source_linkage", "stable_project_identity")),
        ("atomic_persistence", "incremental_update"),
        (EvidenceType.ARCHITECTURE.value, EvidenceType.DATA_PERSISTENCE.value, EvidenceType.WORKFLOW.value),
        ("in_memory_dictionary_proves_project_memory", "temporary_context_proves_project_memory",
         "json_object_proves_project_memory", "reduced_token_usage", "reduced_api_cost"),
        "Managed {bounded_project_record} through {memory_lifecycle}.",
    ),
    _definition(
        "retrieval_and_reranking",
        "Retrieval and Reranking",
        "Retrieves candidate evidence and applies ranking, filtering, reranking, or selection criteria.",
        (("query_operation", "retrieval"),
         ("candidate_filtering", "evidence_selection", "reranking")),
        ("source_attribution",),
        (EvidenceType.ARCHITECTURE.value, EvidenceType.OPTIMIZATION.value, EvidenceType.RETRIEVAL.value, EvidenceType.WORKFLOW.value),
        ("simple_lookup_proves_retrieval_reranking", "search_word_proves_retrieval_reranking",
         "high_recall_retrieval", "high_precision_retrieval", "improved_retrieval_quality",
         "retrieval_implies_hallucination_reduction"),
        "Selected {bounded_candidates} using {ranking_rule}.",
    ),
    _definition(
        "structured_evidence_extraction",
        "Structured Evidence Extraction",
        "Transforms raw or semi-structured change information into validated, source-linked evidence records.",
        (("change_summary_generation", "evidence_card_generation", "source_bound_transformation", "structured_extraction"),
         ("schema_validation", "source_attribution")),
        ("field_normalization", "structured_output_validation"),
        (EvidenceType.ARCHITECTURE.value, EvidenceType.VALIDATION.value, EvidenceType.WORKFLOW.value),
        ("generic_parsing_proves_structured_extraction", "reading_files_proves_structured_extraction",
         "json_use_proves_structured_extraction", "prompt_presence_proves_structured_extraction",
         "structured_extraction_implies_normalization"),
        "Transformed {bounded_input} into {validated_record}.",
    ),
    _definition(
        "template_pollution_blocking",
        "Template Pollution Blocking",
        "Detects and blocks unrelated template-derived or unsupported content from entering output.",
        (("blocked_term_check", "source_ownership_validation", "template_pollution_detection"),
         ("unsupported_template_filtering", "validation_gate")),
        ("source_attribution",),
        (EvidenceType.BUG_FIX.value, EvidenceType.TESTING.value, EvidenceType.VALIDATION.value),
        ("keyword_filtering_proves_template_blocking", "eliminated_template_pollution",
         "factual_output_guarantee"),
        "Blocked {unsupported_template_content} using {ownership_rule}.",
    ),
    _definition(
        "test_and_regression_hardening",
        "Test and Regression Hardening",
        "Adds deterministic tests and regression checks that protect established behavior and invariants.",
        (("compatibility_check", "invariant_check", "regression_testing"),
         ("failure_case_coverage", "targeted_testing")),
        ("schema_validation",),
        (EvidenceType.TESTING.value, EvidenceType.VALIDATION.value),
        ("one_happy_path_test_proves_hardening", "test_count_proves_hardening",
         "pytest_mention_proves_hardening", "tests_imply_bug_free", "production_ready",
         "tests_imply_high_reliability", "complete_test_coverage"),
        "Protected {bounded_behavior} with {regression_check}.",
    ),
    _definition(
        "token_or_context_efficiency",
        "Token or Context Efficiency",
        "Avoids repeated or unnecessary model input through explicit context-management mechanisms.",
        (("cache_reuse", "context_compression", "diff_only_analysis", "selective_context_expansion"),
         ("measured_input_reduction", "repeated_input_avoidance")),
        ("stable_identity",),
        (EvidenceType.ARCHITECTURE.value, EvidenceType.OPTIMIZATION.value, EvidenceType.WORKFLOW.value),
        ("diff_alone_proves_efficiency", "cache_alone_proves_efficiency", "summaries_alone_prove_efficiency",
         "retrieval_alone_proves_llm_reliability", "reduced_token_usage", "reduced_api_cost", "improved_latency"),
        "Avoided repeated {bounded_input} through {reuse_mechanism}.",
        minimum_quality_score=70,
        allows_contextual_support=False,
        high_risk=True,
    ),
    _definition(
        "workflow_orchestration",
        "Workflow Orchestration",
        "Coordinates explicit processing stages through controlled state transitions and failure propagation.",
        (("state_transition", "workflow_staging"),
         ("branching_gate", "failure_propagation", "stage_io_contract")),
        ("error_state_handling",),
        (EvidenceType.ARCHITECTURE.value, EvidenceType.INTEGRATION.value, EvidenceType.WORKFLOW.value),
        ("calling_functions_proves_orchestration", "multi_agent_orchestration",
         "workflow_implies_distributed_execution", "parallel_execution"),
        "Coordinated {bounded_stages} through {state_rule}.",
    ),
)


CAPABILITY_DEFINITIONS = tuple(sorted(_DEFINITIONS, key=lambda item: item.capability_type))
CAPABILITY_TAXONOMY = MappingProxyType({
    definition.capability_type: definition for definition in CAPABILITY_DEFINITIONS
})

# Each alias has a concrete legacy source in the repository.  The GitHub evidence
# aliases occur in capability_facts.jsonl; the project change memory aliases occur in the
# CAPABILITY_TYPES producer registry.  Broad legacy labels are intentionally
# omitted when they do not have one unambiguous canonical meaning.
CAPABILITY_ALIAS_ITEMS = (
    ("deterministic_latex_validation", "latex_validation_and_repair"),
    ("local_project_memory", "project_memory_management"),
    ("testing_and_regression_safety", "test_and_regression_hardening"),
    ("token_or_cost_reduction", "token_or_context_efficiency"),
    ("unsupported_claim_boundary", "claim_validation"),
)
CAPABILITY_ALIASES = MappingProxyType(dict(CAPABILITY_ALIAS_ITEMS))


CAPABILITY_OVERLAP_RULES = (
    ProjectCapabilityOverlapRule(
        "data_persistence", "project_memory_management",
        ("load_validate_write_lifecycle", "persistent_storage", "schema_versioning"),
        ("atomic_persistence", "database_storage"),
        ("project_memory_read", "project_memory_write", "stable_project_identity"),
        "Both may be present when a persistent store has an explicit project-memory lifecycle.",
        "Storage alone must not imply project-memory management.",
    ),
    ProjectCapabilityOverlapRule(
        "deterministic_document_generation", "latex_validation_and_repair",
        ("structured_document_assembly",),
        ("document_merge", "section_ordering"),
        ("latex_repair", "latex_validation"),
        "Both may be present when deterministic assembly is followed by explicit validation and repair.",
        "Document assembly must not imply validation or repair.",
    ),
    ProjectCapabilityOverlapRule(
        "evidence_grounded_generation", "llm_reliability",
        ("claim_validation", "source_grounding", "unsupported_claim_blocking"),
        ("evidence_selection", "output_evidence_validation"),
        ("factuality_evaluation", "low_evidence_refusal", "structured_output_validation"),
        "Both may be present only when grounding and multiple independent reliability controls are evidenced.",
        "Grounded generation alone must not imply broad LLM reliability.",
    ),
    ProjectCapabilityOverlapRule(
        "output_quality_control", "claim_validation",
        ("validation_gate",),
        ("generic_content_blocking", "quality_dimensions", "status_band_decision"),
        ("metric_support_validation", "unsupported_claim_blocking"),
        "Both may be present when general quality gates also enforce evidence-bound claim rules.",
        "General output quality checks must not imply claim validation.",
    ),
    ProjectCapabilityOverlapRule(
        "structured_evidence_extraction", "deterministic_evidence_normalization",
        ("field_normalization", "schema_validation"),
        ("evidence_card_generation", "structured_extraction"),
        ("canonical_serialization", "exact_deduplication", "stable_identity"),
        "Both may be present when structured extraction feeds a separate canonicalization lifecycle.",
        "Extraction alone must not imply stable identity or deterministic deduplication.",
    ),
)


def _alias_items(
    aliases: Mapping[str, str] | Iterable[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    if aliases is None:
        return CAPABILITY_ALIAS_ITEMS
    if isinstance(aliases, Mapping):
        return tuple(aliases.items())
    return tuple(aliases)


def validate_project_capability_taxonomy(
    definitions: Iterable[ProjectCapabilityDefinition] | None = None,
    *,
    signal_registry: Iterable[str] | None = None,
    aliases: Mapping[str, str] | Iterable[tuple[str, str]] | None = None,
    overlap_rules: Iterable[ProjectCapabilityOverlapRule] | None = None,
) -> ProjectCapabilityTaxonomyValidationReport:
    """Validate the canonical taxonomy or deterministic in-memory fixtures."""

    items = tuple(CAPABILITY_DEFINITIONS if definitions is None else definitions)
    signals = tuple(SIGNAL_REGISTRY if signal_registry is None else signal_registry)
    alias_items = _alias_items(aliases)
    overlaps = tuple(CAPABILITY_OVERLAP_RULES if overlap_rules is None else overlap_rules)
    errors: list[str] = []
    warnings: list[str] = []

    def error(code: str) -> None:
        if len(errors) < MAX_VALIDATION_MESSAGES:
            errors.append(code)

    def warning(code: str) -> None:
        if len(warnings) < MAX_VALIDATION_MESSAGES:
            warnings.append(code)

    if len(signals) != len(set(signals)):
        error("signal_registry:duplicate_identifier")
    if tuple(signals) != tuple(sorted(signals)):
        error("signal_registry:unstable_order")
    for index, signal in enumerate(signals):
        if not isinstance(signal, str) or not _IDENTIFIER_RE.fullmatch(signal):
            error(f"signal_registry[{index}]:invalid_identifier")
    signal_set = set(signals)

    identifiers = [item.capability_type for item in items]
    if len(identifiers) != len(set(identifiers)):
        error("taxonomy:duplicate_capability_id")
    if identifiers != sorted(identifiers):
        error("taxonomy:unstable_order")
    display_names = [item.display_name.casefold() for item in items if item.display_name]
    if len(display_names) != len(set(display_names)):
        error("taxonomy:duplicate_display_name")

    for index, item in enumerate(items):
        prefix = f"definition[{index}]"
        if not isinstance(item.capability_type, str) or not _IDENTIFIER_RE.fullmatch(item.capability_type):
            error(f"{prefix}:invalid_capability_id")
        if not item.display_name.strip():
            error(f"{prefix}:empty_display_name")
        if not item.description.strip():
            error(f"{prefix}:empty_description")
        if not item.required_signal_groups:
            error(f"{prefix}:empty_required_signal_groups")
        definition_signals: list[str] = []
        for group_index, group in enumerate(item.required_signal_groups):
            if not group:
                error(f"{prefix}:empty_required_group[{group_index}]")
            if tuple(group) != tuple(sorted(group)):
                error(f"{prefix}:unstable_required_group[{group_index}]")
            definition_signals.extend(group)
        definition_signals.extend(item.supporting_signals)
        if len(definition_signals) != len(set(definition_signals)):
            error(f"{prefix}:duplicate_signal")
        if any(signal not in signal_set for signal in definition_signals):
            error(f"{prefix}:unknown_signal")
        if item.supporting_signals != tuple(sorted(item.supporting_signals)):
            error(f"{prefix}:unstable_supporting_signals")

        evidence_sets = (
            item.accepted_evidence_types,
            item.contextual_evidence_types,
            item.unsupported_evidence_types,
        )
        flattened_evidence = [value for values in evidence_sets for value in values]
        if any(value not in _VALID_EVIDENCE_TYPES for value in flattened_evidence):
            error(f"{prefix}:invalid_evidence_type")
        if len(flattened_evidence) != len(set(flattened_evidence)):
            error(f"{prefix}:overlapping_evidence_type_policy")
        if set(flattened_evidence) != _VALID_EVIDENCE_TYPES:
            error(f"{prefix}:incomplete_evidence_type_policy")
        if any(values != tuple(sorted(values)) for values in evidence_sets):
            error(f"{prefix}:unstable_evidence_type_order")

        source_sets = (
            item.accepted_source_categories,
            item.contextual_source_categories,
            item.unsupported_source_categories,
        )
        flattened_sources = [value for values in source_sets for value in values]
        if any(value not in _VALID_SOURCE_CATEGORIES for value in flattened_sources):
            error(f"{prefix}:invalid_source_category")
        if len(flattened_sources) != len(set(flattened_sources)):
            error(f"{prefix}:overlapping_source_policy")
        if set(flattened_sources) != _VALID_SOURCE_CATEGORIES:
            error(f"{prefix}:incomplete_source_policy")

        numeric_values = (
            item.minimum_quality_score,
            item.minimum_direct_fact_count,
            item.minimum_total_fact_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in numeric_values):
            error(f"{prefix}:invalid_numeric_policy")
        elif not 0 <= item.minimum_quality_score <= 100:
            error(f"{prefix}:invalid_quality_threshold")
        elif item.minimum_direct_fact_count < 0 or item.minimum_total_fact_count < 0:
            error(f"{prefix}:negative_fact_count")
        elif item.minimum_direct_fact_count > item.minimum_total_fact_count:
            error(f"{prefix}:direct_count_exceeds_total")
        if item.requires_direct_provenance and item.minimum_direct_fact_count < 1:
            error(f"{prefix}:direct_provenance_without_direct_fact")
        if item.high_risk and item.minimum_quality_score < 70:
            error(f"{prefix}:weak_high_risk_threshold")

        if not item.forbidden_inferences:
            error(f"{prefix}:empty_forbidden_inferences")
        if any(value not in FORBIDDEN_INFERENCE_REGISTRY for value in item.forbidden_inferences):
            error(f"{prefix}:unsupported_forbidden_inference")
        if len(item.forbidden_inferences) != len(set(item.forbidden_inferences)):
            error(f"{prefix}:duplicate_forbidden_inference")
        if not item.safe_claim_templates:
            warning(f"{prefix}:no_safe_claim_template")
        for template_index, template in enumerate(item.safe_claim_templates):
            if (
                not isinstance(template, str)
                or not template.strip()
                or _UNSAFE_TEMPLATE_RE.search(template)
                or _PROJECT_TECHNOLOGY_RE.search(template)
                or "{" not in template
                or "}" not in template
            ):
                error(f"{prefix}:unsafe_template[{template_index}]")

    canonical_ids = set(identifiers)
    alias_names = [alias for alias, _target in alias_items]
    if len(alias_names) != len(set(alias_names)):
        error("aliases:duplicate_alias")
    alias_map = dict(alias_items)
    for index, (alias, target) in enumerate(alias_items):
        prefix = f"alias[{index}]"
        if not isinstance(alias, str) or not _IDENTIFIER_RE.fullmatch(alias):
            error(f"{prefix}:invalid_alias")
        if alias in canonical_ids:
            error(f"{prefix}:canonical_name_used_as_alias")
        if target not in canonical_ids:
            error(f"{prefix}:missing_target")
        if target in alias_map:
            error(f"{prefix}:alias_chain")
        visited: set[str] = set()
        current = alias
        while current in alias_map:
            if current in visited:
                error(f"{prefix}:alias_cycle")
                break
            visited.add(current)
            current = alias_map[current]

    seen_pairs: set[tuple[str, str]] = set()
    for index, rule in enumerate(overlaps):
        prefix = f"overlap[{index}]"
        pair = tuple(sorted((rule.left_capability, rule.right_capability)))
        if pair in seen_pairs:
            error(f"{prefix}:duplicate_pair")
        seen_pairs.add(pair)
        if rule.left_capability == rule.right_capability:
            error(f"{prefix}:self_overlap")
        if rule.left_capability not in canonical_ids or rule.right_capability not in canonical_ids:
            error(f"{prefix}:unknown_capability")
        overlap_signals = (
            *rule.shared_signals,
            *rule.left_distinguishing_signals,
            *rule.right_distinguishing_signals,
        )
        if any(signal not in signal_set for signal in overlap_signals):
            error(f"{prefix}:unknown_signal")
        if not rule.left_distinguishing_signals or not rule.right_distinguishing_signals:
            error(f"{prefix}:missing_distinction")
        if not rule.both_allowed_when.strip() or not rule.non_implication_rule.strip():
            error(f"{prefix}:empty_policy")

    bounded_errors = tuple(sorted(set(errors))[:MAX_VALIDATION_MESSAGES])
    bounded_warnings = tuple(sorted(set(warnings))[:MAX_VALIDATION_MESSAGES])
    return ProjectCapabilityTaxonomyValidationReport(
        valid=not bounded_errors,
        capability_count=len(items),
        signal_count=len(signals),
        alias_count=len(alias_items),
        errors=bounded_errors,
        warnings=bounded_warnings,
    )


def validate_project_capability_type(capability_type: str) -> str:
    """Return the canonical identifier or raise a clear validation error."""

    if not isinstance(capability_type, str):
        raise TypeError("capability_type must be a string")
    if capability_type != capability_type.strip() or not _IDENTIFIER_RE.fullmatch(capability_type):
        raise ValueError("invalid project evidence capability type")
    canonical = CAPABILITY_ALIASES.get(capability_type, capability_type)
    if canonical not in CAPABILITY_TAXONOMY:
        raise ValueError("unsupported project evidence capability type")
    return canonical


def is_project_capability_supported(capability_type: str) -> bool:
    """Return whether a canonical identifier or declared alias is supported."""

    try:
        validate_project_capability_type(capability_type)
    except (TypeError, ValueError):
        return False
    return True


def get_project_capability_definition(capability_type: str) -> ProjectCapabilityDefinition:
    """Return one immutable definition, resolving declared aliases."""

    return CAPABILITY_TAXONOMY[validate_project_capability_type(capability_type)]


def list_project_capability_definitions() -> tuple[ProjectCapabilityDefinition, ...]:
    """Return definitions in stable alphabetical canonical order."""

    return CAPABILITY_DEFINITIONS


def get_project_capability_alias_target(capability_type: str) -> str | None:
    """Return an explicitly declared alias target without speculative normalization."""

    if not isinstance(capability_type, str) or not _IDENTIFIER_RE.fullmatch(capability_type):
        return None
    return CAPABILITY_ALIASES.get(capability_type)


def list_project_evidence_signal_identifiers() -> tuple[str, ...]:
    return SIGNAL_REGISTRY


def list_project_capability_overlap_rules() -> tuple[ProjectCapabilityOverlapRule, ...]:
    return CAPABILITY_OVERLAP_RULES


__all__ = [
    "CAPABILITY_ALIASES",
    "CAPABILITY_DEFINITIONS",
    "CAPABILITY_OVERLAP_RULES",
    "CAPABILITY_TAXONOMY",
    "FORBIDDEN_INFERENCE_REGISTRY",
    "SIGNAL_REGISTRY",
    "ProjectCapabilityDefinition",
    "ProjectCapabilityOverlapRule",
    "ProjectCapabilityTaxonomyValidationReport",
    "get_project_capability_alias_target",
    "get_project_capability_definition",
    "is_project_capability_supported",
    "list_project_capability_definitions",
    "list_project_capability_overlap_rules",
    "list_project_evidence_signal_identifiers",
    "validate_project_capability_taxonomy",
    "validate_project_capability_type",
]
