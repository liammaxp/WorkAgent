from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from hashlib import sha256
import inspect
import json
from pathlib import Path
import random

import pytest

from backend.engineering_story_evidence import (
    CapabilityLineageState,
    MAX_STORY_EVIDENCE_INPUTS,
    SourceLineageState,
    StoryEventAnchorKind,
    StoryEvidenceBundle,
    StoryEvidenceLineageState,
    StoryEvidenceRelationStrength,
    StoryEvidenceRelationType,
    StoryEvidenceResolutionCode,
    StoryEvidenceResolutionError,
    resolve_story_evidence_bundle,
    resolve_story_evidence_bundle_from_memory,
)
from backend.engineering_story_models import EngineeringStory
from backend.project_evidence_models import (
    ClaimSubjectType,
    Confidence,
    EvidenceSourceRef,
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    ProjectCapabilityFact,
    ProjectClaimBoundary,
    ProjectEvidenceFact,
    ProjectEvidenceMemory,
)


PROJECT_ID = "workagent"
OTHER_PROJECT_ID = "event-lottery"


def _source_ref(
    source_id: str = "chunk_alpha",
    *,
    project_id: str = PROJECT_ID,
    source_type: str = "github_evidence_chunk",
    content: str = "alpha",
    repo: str | None = "owner/workagent",
    commit_sha: str | None = "abcdef1",
    file_path: str | None = "backend/service.py",
    symbol: str | None = "resolve",
    metadata: dict[str, object] | None = None,
) -> EvidenceSourceRef:
    return EvidenceSourceRef(
        source_type=source_type,
        source_id=source_id,
        project_id=project_id,
        content_hash=sha256(content.encode("utf-8")).hexdigest(),
        repo=repo,
        commit_sha=commit_sha,
        file_path=file_path,
        symbol=symbol,
        metadata={"upstream_source_id": "raw_alpha"}
        if metadata is None
        else metadata,
    )


def _fact(
    evidence_fact_id: str = "pef_alpha",
    *,
    project_id: str = PROJECT_ID,
    source_refs: list[EvidenceSourceRef] | None = None,
    mechanism: str = "Resolved evidence through a bounded authority map",
    implementation: list[str] | None = None,
    status: EvidenceStatus = EvidenceStatus.ACCEPTED,
    confidence: Confidence = Confidence.HIGH,
    metric_support: MetricSupport = MetricSupport.NONE,
    evidence_type: EvidenceType = EvidenceType.ARCHITECTURE,
    technical_tags: list[str] | None = None,
) -> ProjectEvidenceFact:
    return ProjectEvidenceFact(
        project_id=project_id,
        evidence_fact_id=evidence_fact_id,
        mechanism=mechanism,
        implementation=["Used exact IDs"] if implementation is None else implementation,
        source_refs=[_source_ref(project_id=project_id)]
        if source_refs is None
        else source_refs,
        status=status,
        confidence=confidence,
        metric_support=metric_support,
        evidence_type=evidence_type,
        technical_tags=["lineage", "validation"]
        if technical_tags is None
        else technical_tags,
    )


def _capability(
    capability_id: str = "pcf_alpha",
    *,
    project_id: str = PROJECT_ID,
    evidence_ids: list[str] | None = None,
    present: bool = True,
    confidence: Confidence = Confidence.LOW,
    metric_support: MetricSupport = MetricSupport.NONE,
) -> ProjectCapabilityFact:
    return ProjectCapabilityFact(
        project_id=project_id,
        capability_id=capability_id,
        capability_type="retrieval_and_reranking",
        present=present,
        source_evidence_fact_ids=["pef_alpha"] if evidence_ids is None else evidence_ids,
        confidence=confidence,
        metric_support=metric_support,
    )


def _boundary(
    boundary_id: str,
    *,
    subject_type: ClaimSubjectType = ClaimSubjectType.EVIDENCE_FACT,
    subject_id: str = "pef_alpha",
    project_id: str = PROJECT_ID,
    forbidden_claims: list[str] | None = None,
) -> ProjectClaimBoundary:
    return ProjectClaimBoundary(
        project_id=project_id,
        boundary_id=boundary_id,
        subject_type=subject_type,
        subject_id=subject_id,
        forbidden_claims=["metric:unsupported"]
        if forbidden_claims is None
        else forbidden_claims,
        metric_support=MetricSupport.NONE,
    )


def _resolve(
    *,
    facts: tuple[ProjectEvidenceFact, ...] | None = None,
    evidence_ids: tuple[str, ...] = ("pef_alpha",),
    capabilities: tuple[ProjectCapabilityFact, ...] = (),
    capability_ids: tuple[str, ...] = (),
    boundaries: tuple[ProjectClaimBoundary, ...] = (),
    boundary_ids: tuple[str, ...] = (),
    project_id: str = PROJECT_ID,
) -> StoryEvidenceBundle:
    return resolve_story_evidence_bundle(
        project_id=project_id,
        evidence_fact_ids=evidence_ids,
        evidence_facts=(_fact(),) if facts is None else facts,
        capability_ids=capability_ids,
        capability_facts=capabilities,
        claim_boundary_ids=boundary_ids,
        claim_boundaries=boundaries,
    )


def _assert_error(
    expected: StoryEvidenceResolutionCode,
    function,
) -> StoryEvidenceResolutionError:
    with pytest.raises(StoryEvidenceResolutionError) as raised:
        function()
    assert raised.value.code is expected
    assert len(str(raised.value)) <= 201
    return raised.value


def test_valid_same_project_authority_resolves_to_bounded_derived_view() -> None:
    fact = _fact()
    capability = _capability()
    evidence_boundary = _boundary("pcb_evidence")
    capability_boundary = _boundary(
        "pcb_capability",
        subject_type=ClaimSubjectType.CAPABILITY_FACT,
        subject_id=capability.capability_id,
    )
    project_boundary = _boundary(
        "pcb_project",
        subject_type=ClaimSubjectType.PROJECT,
        subject_id=PROJECT_ID,
    )

    bundle = _resolve(
        facts=(fact,),
        evidence_ids=(),
        capabilities=(capability,),
        capability_ids=(capability.capability_id,),
        boundaries=(evidence_boundary, capability_boundary, project_boundary),
    )

    assert bundle.project_id == PROJECT_ID
    assert tuple(item.evidence_fact_id for item in bundle.evidence_inputs) == (
        fact.evidence_fact_id,
    )
    assert bundle.capability_lineages[0].source_evidence_fact_ids == (
        fact.evidence_fact_id,
    )
    assert bundle.claim_boundary_ids == (
        "pcb_capability",
        "pcb_evidence",
        "pcb_project",
    )
    assert bundle.evidence_inputs[0].claim_boundary_ids == bundle.claim_boundary_ids
    assert bundle.lineage_state is StoryEvidenceLineageState.COMPLETE


def test_loaded_memory_adapter_reuses_authoritative_memory_without_io() -> None:
    fact = _fact()
    capability = _capability()
    boundary = _boundary("pcb_evidence")
    memory = ProjectEvidenceMemory(
        project_id=PROJECT_ID,
        project_name="WorkAgent",
        evidence_facts=[fact],
        capability_facts=[capability],
        claim_boundaries=[boundary],
    )

    direct = _resolve(
        facts=(fact,),
        capabilities=(capability,),
        capability_ids=(capability.capability_id,),
        boundaries=(boundary,),
    )
    adapted = resolve_story_evidence_bundle_from_memory(
        project_memory=memory,
        evidence_fact_ids=(fact.evidence_fact_id,),
        capability_ids=(capability.capability_id,),
    )

    assert adapted == direct


@pytest.mark.parametrize(
    ("project_id", "facts", "capabilities", "boundaries"),
    [
        (OTHER_PROJECT_ID, (_fact(),), (), ()),
        (
            PROJECT_ID,
            (_fact(),),
            (_capability(project_id=OTHER_PROJECT_ID, evidence_ids=[], present=False),),
            (),
        ),
        (
            PROJECT_ID,
            (_fact(),),
            (),
            (_boundary("pcb_foreign", project_id=OTHER_PROJECT_ID),),
        ),
        (
            PROJECT_ID,
            (_fact(), _fact("pef_foreign", project_id=OTHER_PROJECT_ID)),
            (),
            (),
        ),
    ],
)
def test_cross_project_authority_fails_before_clustering(
    project_id: str,
    facts: tuple[ProjectEvidenceFact, ...],
    capabilities: tuple[ProjectCapabilityFact, ...],
    boundaries: tuple[ProjectClaimBoundary, ...],
) -> None:
    _assert_error(
        StoryEvidenceResolutionCode.CROSS_PROJECT_AUTHORITY,
        lambda: _resolve(
            project_id=project_id,
            facts=facts,
            capabilities=capabilities,
            boundaries=boundaries,
        ),
    )


def test_capability_resolves_only_its_authoritative_supporting_facts() -> None:
    first = _fact("pef_alpha")
    second = _fact(
        "pef_beta",
        source_refs=[_source_ref("chunk_beta", content="beta")],
    )
    capability = _capability(evidence_ids=[second.evidence_fact_id])

    bundle = _resolve(
        facts=(first, second),
        evidence_ids=(),
        capabilities=(capability,),
        capability_ids=(capability.capability_id,),
    )

    assert tuple(item.evidence_fact_id for item in bundle.evidence_inputs) == ("pef_beta",)
    assert bundle.evidence_inputs[0].capability_ids == (capability.capability_id,)


@pytest.mark.parametrize(
    "status",
    [EvidenceStatus.SUPPORTING, EvidenceStatus.WEAK, EvidenceStatus.REJECTED],
)
def test_capability_and_fact_strength_semantics_are_preserved_not_upgraded(
    status: EvidenceStatus,
) -> None:
    fact = _fact(
        status=status,
        confidence=Confidence.LOW,
        metric_support=MetricSupport.APPROXIMATE,
    )
    capability = _capability(
        confidence=Confidence.LOW,
        metric_support=MetricSupport.APPROXIMATE,
    )

    bundle = _resolve(
        facts=(fact,),
        capabilities=(capability,),
        capability_ids=(capability.capability_id,),
    )

    item = bundle.evidence_inputs[0]
    lineage = bundle.capability_lineages[0]
    assert item.evidence_status is status
    assert item.confidence is Confidence.LOW
    assert item.metric_support is MetricSupport.APPROXIMATE
    assert lineage.confidence is Confidence.LOW
    assert lineage.metric_support is MetricSupport.APPROXIMATE
    assert lineage.state is CapabilityLineageState.RESOLVED


def test_unresolved_capability_lineage_fails_closed() -> None:
    capability = _capability(evidence_ids=["pef_missing"])
    _assert_error(
        StoryEvidenceResolutionCode.INVALID_CAPABILITY_LINEAGE,
        lambda: _resolve(
            capabilities=(capability,),
            capability_ids=(capability.capability_id,),
        ),
    )


def test_not_present_capability_does_not_establish_positive_evidence() -> None:
    capability = _capability(present=False, evidence_ids=[])
    _assert_error(
        StoryEvidenceResolutionCode.NO_EVIDENCE_INPUTS,
        lambda: _resolve(
            facts=(),
            evidence_ids=(),
            capabilities=(capability,),
            capability_ids=(capability.capability_id,),
        ),
    )
    with_evidence = _resolve(
        capabilities=(capability,),
        capability_ids=(capability.capability_id,),
    )
    assert with_evidence.capability_lineages[0].state is CapabilityLineageState.NOT_PRESENT
    assert with_evidence.evidence_inputs[0].capability_ids == ()


def test_claim_boundary_alone_never_establishes_positive_evidence() -> None:
    boundary = _boundary("pcb_evidence")
    _assert_error(
        StoryEvidenceResolutionCode.NO_EVIDENCE_INPUTS,
        lambda: _resolve(
            facts=(_fact(),),
            evidence_ids=(),
            boundaries=(boundary,),
            boundary_ids=(boundary.boundary_id,),
        ),
    )


def test_irrelevant_requested_boundary_fails_instead_of_becoming_evidence() -> None:
    second = _fact(
        "pef_beta",
        source_refs=[_source_ref("chunk_beta", content="beta")],
    )
    boundary = _boundary("pcb_beta", subject_id=second.evidence_fact_id)
    _assert_error(
        StoryEvidenceResolutionCode.IRRELEVANT_CLAIM_BOUNDARY,
        lambda: _resolve(
            facts=(_fact(), second),
            boundaries=(boundary,),
            boundary_ids=(boundary.boundary_id,),
        ),
    )


def test_invalid_boundary_subject_fails_closed() -> None:
    boundary = _boundary("pcb_missing", subject_id="pef_missing")
    _assert_error(
        StoryEvidenceResolutionCode.INVALID_CLAIM_BOUNDARY,
        lambda: _resolve(boundaries=(boundary,)),
    )


def test_source_lineage_normalizes_structural_fields_and_anchors() -> None:
    ref = _source_ref(
        source_type="project_change_raw_change_summary",
        source_id="change_7",
        commit_sha="ABCDEF1",
        file_path="backend\\service.py",
        metadata={"change_id": "change_7", "parent_change_id": "change_6"},
    )
    bundle = _resolve(facts=(_fact(source_refs=[ref]),))
    lineage = bundle.evidence_inputs[0].source_lineages[0]

    assert lineage.state is SourceLineageState.AVAILABLE
    assert lineage.commit_sha == "abcdef1"
    assert lineage.file_path == "backend/service.py"
    assert lineage.explicit_change_id == "change_7"
    assert lineage.parent_change_id == "change_6"
    assert {item.anchor_kind for item in bundle.event_anchors} >= {
        StoryEventAnchorKind.EXPLICIT_CHANGE,
        StoryEventAnchorKind.PARENT_CHANGE,
        StoryEventAnchorKind.COMMIT_AND_SYMBOL,
        StoryEventAnchorKind.COMMIT_AND_PATH,
        StoryEventAnchorKind.SOURCE_IDENTITY,
    }


def test_missing_structural_lineage_is_explicit_and_still_source_scoped() -> None:
    ref = _source_ref(
        repo=None,
        commit_sha=None,
        file_path=None,
        symbol=None,
        metadata={},
    )
    bundle = _resolve(facts=(_fact(source_refs=[ref]),))
    item = bundle.evidence_inputs[0]

    assert item.source_lineages[0].state is SourceLineageState.MISSING_STRUCTURAL_CONTEXT
    assert bundle.lineage_state is StoryEvidenceLineageState.MISSING_STRUCTURAL_CONTEXT
    assert tuple(anchor.anchor_kind for anchor in item.event_anchors) == (
        StoryEventAnchorKind.SOURCE_IDENTITY,
    )


def test_conflicting_source_lineage_fails_closed() -> None:
    first = _fact("pef_alpha", source_refs=[_source_ref(content="one")])
    second = _fact("pef_beta", source_refs=[_source_ref(content="two")])
    _assert_error(
        StoryEvidenceResolutionCode.CONFLICTING_SOURCE_LINEAGE,
        lambda: _resolve(facts=(first, second), evidence_ids=("pef_alpha", "pef_beta")),
    )


def test_mutated_nested_source_ref_project_is_rechecked_at_consumption() -> None:
    fact = _fact()
    fact.source_refs.append(_source_ref("chunk_foreign", project_id=OTHER_PROJECT_ID))
    _assert_error(
        StoryEvidenceResolutionCode.CROSS_PROJECT_AUTHORITY,
        lambda: _resolve(facts=(fact,)),
    )


def test_event_anchor_and_bundle_ignore_mutable_evidence_prose() -> None:
    first = _fact(
        mechanism="First generated explanation",
        implementation=["First mutable summary"],
    )
    second = _fact(
        mechanism="Completely rewritten explanation",
        implementation=["Different mutable summary"],
    )

    before = _resolve(facts=(first,))
    after = _resolve(facts=(second,))

    assert before == after
    serialized = after.to_json()
    assert "rewritten" not in serialized
    assert "mutable summary" not in serialized


def test_event_anchor_contract_has_no_anchor_or_story_id() -> None:
    anchor = _resolve().event_anchors[0]
    assert "anchor_id" not in anchor.to_dict()
    assert "story_id" not in anchor.to_dict()
    assert not isinstance(_resolve(), EngineeringStory)


def test_jd_company_and_employer_context_cannot_enter_anchor_api() -> None:
    parameters = inspect.signature(resolve_story_evidence_bundle).parameters
    assert not ({"jd", "jd_text", "company", "employer"} & set(parameters))
    serialized = _resolve().to_json().lower()
    assert "jd_text" not in serialized
    assert "employer" not in serialized
    assert "company" not in serialized


def test_exact_duplicates_dedupe_but_conflicting_fact_definitions_fail() -> None:
    fact = _fact()
    deduped = _resolve(
        facts=(fact, fact),
        evidence_ids=(fact.evidence_fact_id, fact.evidence_fact_id),
    )
    assert len(deduped.evidence_inputs) == 1

    conflicting = _fact(mechanism="A divergent authoritative definition")
    _assert_error(
        StoryEvidenceResolutionCode.CONFLICTING_EVIDENCE_FACT,
        lambda: _resolve(facts=(fact, conflicting)),
    )


def test_conflicting_capability_and_boundary_definitions_fail_closed() -> None:
    capability = _capability()
    conflicting_capability = _capability(confidence=Confidence.HIGH)
    _assert_error(
        StoryEvidenceResolutionCode.CONFLICTING_CAPABILITY,
        lambda: _resolve(
            capabilities=(capability, conflicting_capability),
            capability_ids=(capability.capability_id,),
        ),
    )

    boundary = _boundary("pcb_evidence")
    conflicting_boundary = _boundary(
        "pcb_evidence", forbidden_claims=["impact:unsupported"]
    )
    _assert_error(
        StoryEvidenceResolutionCode.CONFLICTING_CLAIM_BOUNDARY,
        lambda: _resolve(boundaries=(boundary, conflicting_boundary)),
    )


def test_input_permutations_produce_identical_canonical_serialization() -> None:
    first = _fact(
        "pef_alpha",
        source_refs=[
            _source_ref("chunk_alpha", content="alpha"),
            _source_ref("chunk_alpha_2", content="alpha-2", symbol="validate"),
        ],
    )
    second = _fact(
        "pef_beta",
        source_refs=[
            _source_ref(
                "chunk_beta",
                content="beta",
                metadata={"upstream_source_id": "raw_alpha"},
            )
        ],
    )
    capability = _capability(evidence_ids=[first.evidence_fact_id, second.evidence_fact_id])
    boundaries = (
        _boundary("pcb_alpha", subject_id=first.evidence_fact_id),
        _boundary("pcb_beta", subject_id=second.evidence_fact_id),
    )
    baseline = _resolve(
        facts=(first, second),
        evidence_ids=(first.evidence_fact_id, second.evidence_fact_id),
        capabilities=(capability,),
        capability_ids=(capability.capability_id,),
        boundaries=boundaries,
        boundary_ids=tuple(item.boundary_id for item in boundaries),
    )
    randomizer = random.Random(7)

    for _ in range(8):
        facts = [first, second]
        ids = [first.evidence_fact_id, second.evidence_fact_id]
        shuffled_boundaries = list(boundaries)
        randomizer.shuffle(facts)
        randomizer.shuffle(ids)
        randomizer.shuffle(shuffled_boundaries)
        candidate = _resolve(
            facts=tuple(facts),
            evidence_ids=tuple(ids),
            capabilities=(capability,),
            capability_ids=(capability.capability_id,),
            boundaries=tuple(shuffled_boundaries),
            boundary_ids=tuple(item.boundary_id for item in shuffled_boundaries),
        )
        assert candidate.to_json() == baseline.to_json()


def test_relations_are_structural_explainable_and_fixed_strength() -> None:
    first = _fact("pef_alpha")
    second = _fact(
        "pef_beta",
        source_refs=[_source_ref("chunk_beta", content="beta")],
    )
    capability = _capability(evidence_ids=[first.evidence_fact_id, second.evidence_fact_id])
    bundle = _resolve(
        facts=(first, second),
        evidence_ids=(first.evidence_fact_id, second.evidence_fact_id),
        capabilities=(capability,),
        capability_ids=(capability.capability_id,),
    )
    by_type = {item.relation_type: item for item in bundle.relations}

    assert by_type[StoryEvidenceRelationType.SAME_COMMIT].strength is (
        StoryEvidenceRelationStrength.STRONG
    )
    assert by_type[StoryEvidenceRelationType.SAME_PATH].basis_ids == (
        "backend/service.py",
        "owner/workagent",
    )
    assert by_type[
        StoryEvidenceRelationType.CAPABILITY_SUPPORT_RELATION
    ].basis_ids == (capability.capability_id,)
    assert all(not isinstance(item.strength.value, float) for item in bundle.relations)


def test_shared_technology_tag_alone_does_not_create_a_relation() -> None:
    first = _fact(
        "pef_alpha",
        source_refs=[
            _source_ref(
                "chunk_alpha",
                repo="owner/one",
                commit_sha="aaaaaaa",
                file_path="one.py",
                symbol="one",
                metadata={},
            )
        ],
        technical_tags=["Python"],
    )
    second = _fact(
        "pef_beta",
        source_refs=[
            _source_ref(
                "chunk_beta",
                repo="owner/two",
                commit_sha="bbbbbbb",
                file_path="two.py",
                symbol="two",
                content="beta",
                metadata={},
            )
        ],
        technical_tags=["Python"],
    )
    bundle = _resolve(
        facts=(first, second),
        evidence_ids=(first.evidence_fact_id, second.evidence_fact_id),
    )
    assert bundle.relations == ()


@pytest.mark.parametrize(
    "ref",
    [
        _source_ref(file_path="../private.py"),
        _source_ref(file_path="diff --git a/private.py b/private.py"),
        _source_ref(repo="authorization: Bearer abcdefghijklmnop"),
    ],
)
def test_raw_patch_secret_and_unsafe_lineage_values_fail_closed(
    ref: EvidenceSourceRef,
) -> None:
    _assert_error(
        StoryEvidenceResolutionCode.MALFORMED_SOURCE_LINEAGE,
        lambda: _resolve(facts=(_fact(source_refs=[ref]),)),
    )


def test_unrecognized_source_metadata_is_not_copied_to_output() -> None:
    secret = "unrelated_metadata_never_copy_72c5"
    ref = _source_ref(metadata={"unrelated_note": secret})
    serialized = _resolve(facts=(_fact(source_refs=[ref]),)).to_json()
    assert secret not in serialized
    assert "unrelated_note" not in serialized


@pytest.mark.parametrize(
    ("identifier", "code"),
    [
        ("pef_", StoryEvidenceResolutionCode.MALFORMED_AUTHORITY_ID),
        ("pef_bad/path", StoryEvidenceResolutionCode.MALFORMED_AUTHORITY_ID),
        ("pef_missing", StoryEvidenceResolutionCode.UNKNOWN_EVIDENCE_FACT),
    ],
)
def test_malformed_and_unknown_authority_ids_have_structured_failures(
    identifier: str,
    code: StoryEvidenceResolutionCode,
) -> None:
    _assert_error(code, lambda: _resolve(evidence_ids=(identifier,)))


def test_bounds_are_checked_before_lookup_or_partial_resolution() -> None:
    evidence_ids = tuple(
        f"pef_item_{index}" for index in range(MAX_STORY_EVIDENCE_INPUTS + 1)
    )
    _assert_error(
        StoryEvidenceResolutionCode.BOUND_EXCEEDED,
        lambda: _resolve(evidence_ids=evidence_ids),
    )


def test_contract_round_trip_is_strict_immutable_and_mutation_isolated() -> None:
    fact = _fact()
    fact_before = fact.to_json()
    bundle = _resolve(facts=(fact,))
    payload = bundle.to_dict()
    rebuilt = StoryEvidenceBundle.from_dict(payload)

    assert rebuilt == bundle
    assert rebuilt.to_json() == bundle.to_json()
    payload["claim_boundary_ids"].append("pcb_external_mutation")
    assert rebuilt.claim_boundary_ids == ()
    assert fact.to_json() == fact_before
    assert not hasattr(bundle, "__dict__")
    with pytest.raises(FrozenInstanceError):
        bundle.project_id = OTHER_PROJECT_ID  # type: ignore[misc]


def test_direct_bundle_constructor_fails_cleanly_on_wrong_nested_types() -> None:
    bundle = _resolve()
    with pytest.raises(TypeError, match="StoryEvidenceInput"):
        StoryEvidenceBundle(
            project_id=PROJECT_ID,
            evidence_inputs=(object(),),  # type: ignore[arg-type]
            capability_lineages=bundle.capability_lineages,
            claim_boundary_ids=bundle.claim_boundary_ids,
            event_anchors=bundle.event_anchors,
            relations=bundle.relations,
            lineage_state=bundle.lineage_state,
        )


def test_module_is_pure_and_has_no_runtime_or_persistence_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = Path("backend/engineering_story_evidence.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {
        "backend.api_server",
        "backend.memory_store",
        "backend.project_retrieval_v2",
        "backend.evidence_hybrid_retrieval",
        "backend.github_raw_storage",
        "chromadb",
        "os",
    }
    assert not any(
        name == blocked or name.startswith(f"{blocked}.")
        for name in imported
        for blocked in forbidden
    )
    source = module_path.read_text(encoding="utf-8")
    assert "PersistentClient" not in source
    assert "EngineeringStory(" not in source
    assert "schema_version" not in source
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: pytest.fail("I/O"))
    assert _resolve().project_id == PROJECT_ID


def test_serialized_contract_contains_only_bounded_structural_derived_data() -> None:
    payload = json.loads(_resolve().to_json())
    forbidden_keys = {
        "raw_patch",
        "raw_text",
        "document",
        "embedding",
        "query",
        "score",
        "jd_text",
        "company",
        "employer",
        "mechanism",
        "implementation",
        "safe_impact",
        "story_id",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    assert forbidden_keys.isdisjoint(keys(payload))
