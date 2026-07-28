from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from pathlib import Path

import pytest

from backend.project_capability_memory import (
    CONFIDENCE_LEVELS,
    METRIC_SUPPORT_LEVELS,
    CapabilityCandidate,
    ProjectCapabilityFact,
    normalize_capability_candidate,
    normalize_project_capability_fact,
    validate_capability_candidate,
    validate_project_capability_fact,
)
from backend.project_evidence_models import (
    Confidence,
    MetricSupport,
    ProjectCapabilityFact as EvidenceMemoryCapabilityFact,
)


def candidate(**overrides):
    values = {
        "project_id": "  project-a  ",
        "capability_type": " Retrieval And Reranking ",
        "supporting_evidence_ids": ["pef_b", " pef_a ", "pef_b"],
        "supporting_signals": [" query ranking ", "cache reuse", "Query Ranking"],
        "conflicting_signals": [],
        "candidate_score": 0.65,
        "metadata": {"source": [" synthetic "]},
    }
    values.update(overrides)
    return CapabilityCandidate(**values)


def upstream_fact(**overrides):
    values = {
        "project_id": "project-a",
        "capability_type": "retrieval_and_reranking",
        "present": True,
        "source_evidence_fact_ids": ["pef_b", "pef_a", "pef_b"],
        "confidence": Confidence.HIGH,
        "mechanisms": ["deterministic query ranking"],
        "allowed_resume_claims": ["Ranks project evidence"],
        "forbidden_claims": ["Guaranteed perfect retrieval"],
        "metric_support": MetricSupport.NONE,
        "technical_tags": ["retrieval"],
    }
    values.update(overrides)
    return ProjectCapabilityFact(**values)


def test_capability_candidate_normalizes_and_validates():
    model = candidate()
    assert model.project_id == "project-a"
    assert model.capability_type == "retrieval_and_reranking"
    assert model.supporting_evidence_ids == ("pef_a", "pef_b")
    assert model.supporting_signals == ("cache reuse", "query ranking")
    assert model.candidate_score == 0.65
    assert model.metadata["source"] == ("synthetic",)
    assert validate_capability_candidate(model) == model
    assert normalize_capability_candidate(model.to_dict()) == model


@pytest.mark.parametrize(
    "overrides",
    [
        {"project_id": " "},
        {"capability_type": " "},
        {"capability_type": "unsafe/type"},
        {"candidate_score": -0.1},
        {"candidate_score": 1.1},
        {"candidate_score": math.nan},
        {"candidate_score": math.inf},
        {"supporting_evidence_ids": []},
        {"metadata": {"raw_text": "private"}},
        {"metadata": {"object": object()}},
    ],
)
def test_capability_candidate_rejects_invalid_inputs(overrides):
    with pytest.raises((TypeError, ValueError)):
        candidate(**overrides)


def test_project_capability_fact_is_the_single_authoritative_fact_model():
    assert ProjectCapabilityFact is EvidenceMemoryCapabilityFact
    model = upstream_fact()
    assert model.capability_id.startswith("pcf_")
    assert model.source_evidence_fact_ids == ["pef_a", "pef_b"]
    assert validate_project_capability_fact(model) == model
    assert normalize_project_capability_fact(model.to_dict()) == model


def test_project_capability_fact_id_uses_upstream_identity_rule():
    first = upstream_fact()
    reordered = upstream_fact(source_evidence_fact_ids=["pef_a", "pef_b", "pef_a"])
    assert first.capability_id == reordered.capability_id
    assert first.capability_id != upstream_fact(project_id="project-b").capability_id
    assert first.capability_id != upstream_fact(capability_type="workflow_orchestration").capability_id
    assert first.capability_id != upstream_fact(present=False).capability_id


def test_project_capability_fact_round_trips_with_upstream_serializer():
    model = upstream_fact()
    assert ProjectCapabilityFact.from_dict(model.to_dict()) == model
    assert normalize_project_capability_fact(model) == model
    with pytest.raises(ValueError, match="unknown"):
        ProjectCapabilityFact.from_dict({**model.to_dict(), "unexpected": True})


def test_metric_support_contract_matches_project_evidence_memory():
    assert CONFIDENCE_LEVELS == {item.value for item in Confidence}
    assert METRIC_SUPPORT_LEVELS == {item.value for item in MetricSupport}
    assert METRIC_SUPPORT_LEVELS == {"none", "approximate", "explicit"}


@pytest.mark.parametrize(
    "key",
    ["raw_text", "Raw_Text", "PATCH", "rawDiff", "authorization", "access_token", "credential"],
)
def test_capability_candidate_rejects_raw_or_sensitive_metadata(key):
    with pytest.raises(ValueError, match="forbidden"):
        candidate(metadata={"nested": {key: "not retained"}})


def test_capability_candidate_does_not_share_mutable_input_state():
    metadata = {"source": ["evidence"]}
    evidence_ids = ["pef_a"]
    model = candidate(metadata=metadata, supporting_evidence_ids=evidence_ids)
    metadata["source"].append("changed")
    evidence_ids.append("pef_b")
    assert model.metadata["source"] == ("evidence",)
    assert model.supporting_evidence_ids == ("pef_a",)
    with pytest.raises(TypeError):
        model.metadata["new"] = True
    with pytest.raises(FrozenInstanceError):
        model.project_id = "changed"


def test_project_capability_models_use_semantic_naming_and_no_duplicate_fact():
    module_path = Path(__file__).resolve().parents[1] / "backend" / "project_capability_memory.py"
    source = module_path.read_text(encoding="utf-8").casefold()
    forbidden = set()
    for number in (4, 5):
        label = "phase" + str(number)
        forbidden.update({label, "project_memory_" + label, "use_" + label, label + ".v1"})
    assert not {token for token in forbidden if token in source}
    assert "class capabilityfact" not in source

