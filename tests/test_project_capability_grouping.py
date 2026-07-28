from dataclasses import FrozenInstanceError, replace
import hashlib
from pathlib import Path

import pytest

from backend.project_capability_grouping import group_project_evidence_facts
from backend.project_capability_memory import CapabilityCandidate
from backend.project_evidence_memory import load_project_evidence_memory
from backend.project_evidence_models import (
    Confidence,
    EvidenceSourceRef,
    EvidenceType,
    ProjectCapabilityFact,
    ProjectEvidenceFact,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "information" / "project_evidence_memory.json"


def _fact(
    evidence_id: str,
    *,
    project_id: str = "project-a",
    technical_tags: list[str] | None = None,
    evidence_type: EvidenceType = EvidenceType.FEATURE,
    mechanism: str = "A bounded implementation mechanism.",
) -> ProjectEvidenceFact:
    return ProjectEvidenceFact(
        project_id=project_id,
        mechanism=mechanism,
        implementation=["Implemented a bounded project-local control."],
        source_refs=[EvidenceSourceRef(
            source_type="github_evidence_card",
            source_id=f"source-{evidence_id}",
            project_id=project_id,
            content_hash=hashlib.sha256(evidence_id.encode()).hexdigest(),
        )],
        evidence_type=evidence_type,
        confidence=Confidence.HIGH,
        technical_tags=list(technical_tags or []),
        quality_score=90,
        evidence_fact_id=evidence_id,
    )


def _candidate(result, capability_type: str) -> CapabilityCandidate:
    return next(item for item in result.candidates if item.capability_type == capability_type)


def test_groups_evidence_facts_into_capability_candidates():
    fact = _fact("pef_retrieval", technical_tags=["retrieval", "reranking"])
    result = group_project_evidence_facts(project_id="project-a", evidence_facts=[fact])

    candidate = _candidate(result, "retrieval_and_reranking")
    assert candidate.supporting_evidence_ids == ("pef_retrieval",)
    assert set(candidate.supporting_signals) >= {"retrieval", "reranking"}
    assert candidate.candidate_score == 0.0
    assert candidate.metadata["evaluation_state"] == "unscored"


def test_aliases_group_into_one_canonical_candidate():
    alias = _fact("pef_alias", technical_tags=["local_project_memory", "project_memory_read"])
    canonical = _fact("pef_canonical", technical_tags=["project_memory_management", "project_memory_write"])
    result = group_project_evidence_facts(project_id="project-a", evidence_facts=[alias, canonical])

    matches = [item for item in result.candidates if item.capability_type == "project_memory_management"]
    assert len(matches) == 1
    assert matches[0].supporting_evidence_ids == ("pef_alias", "pef_canonical")


def test_one_evidence_fact_may_support_multiple_capability_candidates():
    fact = _fact("pef_grounding", technical_tags=["source_grounding"])
    result = group_project_evidence_facts(project_id="project-a", evidence_facts=[fact])

    supported = {item.capability_type for item in result.candidates}
    assert {"evidence_grounded_generation", "llm_reliability"} <= supported
    assert all(item.supporting_evidence_ids == (fact.evidence_fact_id,) for item in result.candidates)


def test_grouping_rejects_cross_project_evidence():
    with pytest.raises(ValueError, match="requested project_id"):
        group_project_evidence_facts(
            project_id="project-a",
            evidence_facts=[_fact("pef_a"), _fact("pef_b", project_id="project-b")],
        )


def test_grouping_accounts_for_every_input_evidence_fact():
    matched = _fact("pef_matched", technical_tags=["retrieval"])
    unmatched = _fact("pef_unmatched", technical_tags=["unrecognized_precise_tag"])
    ambiguous = _fact("pef_ambiguous", technical_tags=["validation_and_repair"])
    result = group_project_evidence_facts(
        project_id="project-a", evidence_facts=[matched, unmatched, ambiguous]
    )

    categories = (
        set(result.matched_evidence_ids),
        set(result.unmatched_evidence_ids),
        set(result.ambiguous_evidence_ids),
        set(result.skipped_evidence_ids),
    )
    assert set(result.input_evidence_ids) == set().union(*categories)
    assert sum(map(len, categories)) == len(set(result.input_evidence_ids))
    assert result.matched_evidence_ids == ("pef_matched",)
    assert result.unmatched_evidence_ids == ("pef_unmatched",)
    assert result.ambiguous_evidence_ids == ("pef_ambiguous",)


def test_unknown_signals_do_not_create_candidates():
    result = group_project_evidence_facts(
        project_id="project-a",
        evidence_facts=[_fact("pef_unknown", technical_tags=["maybe_magic_reliability"])],
    )
    assert result.candidates == ()
    assert result.unmatched_evidence_ids == ("pef_unknown",)


def test_ambiguous_capability_labels_do_not_create_candidates():
    result = group_project_evidence_facts(
        project_id="project-a",
        evidence_facts=[_fact("pef_ambiguous", technical_tags=["validation_and_repair"])],
    )
    assert result.candidates == ()
    assert result.ambiguous_evidence_ids == ("pef_ambiguous",)


def test_duplicate_evidence_ids_are_deduplicated_or_rejected_safely():
    fact = _fact("pef_duplicate", technical_tags=["retrieval"])
    result = group_project_evidence_facts(project_id="project-a", evidence_facts=[fact, fact])
    assert result.input_evidence_ids == ("pef_duplicate",)
    assert result.diagnostics["duplicate_input_count"] == 1

    conflicting = replace(fact, mechanism="A different semantic mechanism.")
    with pytest.raises(ValueError, match="conflicting semantic content"):
        group_project_evidence_facts(project_id="project-a", evidence_facts=[fact, conflicting])


def test_grouping_is_deterministic_and_input_order_independent():
    facts = [
        _fact("pef_z", technical_tags=["reranking", "retrieval", "retrieval"]),
        _fact("pef_a", technical_tags=["source_grounding"]),
    ]
    first = group_project_evidence_facts(project_id="project-a", evidence_facts=facts)
    second = group_project_evidence_facts(project_id="project-a", evidence_facts=list(reversed(facts)))
    assert first == second
    assert first.to_json() == second.to_json()


def test_grouped_candidates_do_not_share_mutable_input_state():
    tags = ["retrieval"]
    fact = _fact("pef_immutable", technical_tags=tags)
    result = group_project_evidence_facts(project_id="project-a", evidence_facts=[fact])
    tags.append("reranking")

    candidate = _candidate(result, "retrieval_and_reranking")
    assert candidate.supporting_signals == ("retrieval",)
    with pytest.raises(FrozenInstanceError):
        candidate.candidate_score = 1.0
    with pytest.raises(TypeError):
        candidate.metadata["evaluation_state"] = "evaluated"


def test_grouping_does_not_apply_candidate_scoring():
    result = group_project_evidence_facts(
        project_id="project-a",
        evidence_facts=[_fact("pef_low_support", technical_tags=["retrieval"])],
    )
    assert result.candidates
    assert all(item.candidate_score == 0.0 for item in result.candidates)
    assert all(item.metadata["evaluation_state"] == "unscored" for item in result.candidates)
    assert all("confidence" not in item.to_dict() and "status" not in item.to_dict() for item in result.candidates)


def test_grouping_does_not_generate_project_capability_facts():
    result = group_project_evidence_facts(
        project_id="project-a",
        evidence_facts=[_fact("pef_candidate", technical_tags=["retrieval"])],
    )
    assert all(isinstance(item, CapabilityCandidate) for item in result.candidates)
    assert not any(isinstance(item, ProjectCapabilityFact) for item in result.candidates)


def test_grouping_result_does_not_expose_raw_evidence_content():
    sentinel = "PRIVATE_RAW_GITHUB_PATCH_SENTINEL"
    fact = _fact("pef_private", technical_tags=["retrieval"], mechanism=sentinel)
    result = group_project_evidence_facts(project_id="project-a", evidence_facts=[fact])
    serialized = result.to_json().casefold()
    assert sentinel.casefold() not in serialized
    assert not {"raw", "patch", "diff", "source_code", "github_context"} & set(result.to_dict())


def test_real_project_evidence_memory_grouping_is_read_only_and_project_isolated():
    before = ARTIFACT.read_bytes()
    before_mtime = ARTIFACT.stat().st_mtime_ns
    loaded = load_project_evidence_memory(ARTIFACT)
    assert loaded.status == "ready" and loaded.snapshot is not None

    results = [
        group_project_evidence_facts(project_id=project.project_id, evidence_facts=project.evidence_facts)
        for project in loaded.snapshot.projects
    ]
    assert len(results) == 11
    assert sum(len(result.input_evidence_ids) for result in results) == 283
    assert all(result.project_id == candidate.project_id for result in results for candidate in result.candidates)
    assert sum(len(project.capability_facts) for project in loaded.snapshot.projects) == 0
    assert ARTIFACT.read_bytes() == before
    assert ARTIFACT.stat().st_mtime_ns == before_mtime


def test_project_capability_grouping_uses_semantic_naming():
    source = (ROOT / "backend" / "project_capability_grouping.py").read_text(encoding="utf-8").casefold()
    forbidden = ("phase" + "5", "phase_" + "5", "project_memory_" + "phase" + "5", "use_" + "phase" + "5")
    assert not any(token in source for token in forbidden)
