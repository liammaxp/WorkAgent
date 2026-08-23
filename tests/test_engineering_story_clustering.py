from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import inspect
import json
from pathlib import Path
import random

import pytest

from backend.engineering_story_clustering import (
    MAX_EVIDENCE_MEMBERS_PER_CLUSTER,
    StoryClusterCoreKind,
    StoryClusterDecisionOutcome,
    StoryClusterIdentityState,
    StoryClusterLineageState,
    StoryClusterMembershipBasis,
    StoryClusterQuality,
    StoryClusterReasonCode,
    StoryClusterRelationRole,
    StoryClusteringError,
    StoryClusteringErrorCode,
    StoryClusteringResult,
    cluster_story_evidence_bundle,
    story_cluster_relation_role,
)
from backend.engineering_story_evidence import (
    StoryEvidenceBundle,
    StoryEvidenceRelationType,
    StoryEvidenceResolutionError,
    resolve_story_evidence_bundle,
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
)


PROJECT_ID = "workagent"
OTHER_PROJECT_ID = "event-lottery"


def _ref(
    source_id: str,
    *,
    project_id: str = PROJECT_ID,
    source_type: str = "github_evidence_chunk",
    repo: str | None = "owner/workagent",
    commit_sha: str | None = "aaaaaaa",
    file_path: str | None = "backend/service.py",
    symbol: str | None = "resolve",
    metadata: dict[str, object] | None = None,
    content: str | None = None,
) -> EvidenceSourceRef:
    material = content or "|".join((
        source_id,
        repo or "",
        commit_sha or "",
        file_path or "",
        symbol or "",
    ))
    return EvidenceSourceRef(
        source_type=source_type,
        source_id=source_id,
        project_id=project_id,
        content_hash=sha256(material.encode("utf-8")).hexdigest(),
        repo=repo,
        commit_sha=commit_sha,
        file_path=file_path,
        symbol=symbol,
        metadata={} if metadata is None else metadata,
    )


def _fact(
    evidence_fact_id: str,
    refs: list[EvidenceSourceRef],
    *,
    project_id: str = PROJECT_ID,
    mechanism: str = "Bounded structural evidence",
    status: EvidenceStatus = EvidenceStatus.ACCEPTED,
    confidence: Confidence = Confidence.HIGH,
) -> ProjectEvidenceFact:
    return ProjectEvidenceFact(
        project_id=project_id,
        evidence_fact_id=evidence_fact_id,
        mechanism=mechanism,
        implementation=["Validated exact authority"],
        source_refs=refs,
        evidence_type=EvidenceType.ARCHITECTURE,
        status=status,
        confidence=confidence,
        metric_support=MetricSupport.NONE,
        technical_tags=["Python", "validation"],
    )


def _capability(
    evidence_ids: list[str],
    *,
    capability_id: str = "pcf_shared_validation",
    confidence: Confidence = Confidence.LOW,
) -> ProjectCapabilityFact:
    return ProjectCapabilityFact(
        project_id=PROJECT_ID,
        capability_id=capability_id,
        capability_type="validation_and_repair",
        present=True,
        source_evidence_fact_ids=evidence_ids,
        confidence=confidence,
        metric_support=MetricSupport.NONE,
    )


def _project_boundary(boundary_id: str = "pcb_project") -> ProjectClaimBoundary:
    return ProjectClaimBoundary(
        project_id=PROJECT_ID,
        boundary_id=boundary_id,
        subject_type=ClaimSubjectType.PROJECT,
        subject_id=PROJECT_ID,
        forbidden_claims=["metric:unsupported"],
        metric_support=MetricSupport.NONE,
    )


def _bundle(
    facts: list[ProjectEvidenceFact],
    *,
    capabilities: list[ProjectCapabilityFact] | None = None,
    boundaries: list[ProjectClaimBoundary] | None = None,
) -> StoryEvidenceBundle:
    capabilities = [] if capabilities is None else capabilities
    boundaries = [] if boundaries is None else boundaries
    return resolve_story_evidence_bundle(
        project_id=PROJECT_ID,
        evidence_fact_ids=tuple(item.evidence_fact_id for item in facts),
        evidence_facts=tuple(facts),
        capability_ids=tuple(item.capability_id for item in capabilities),
        capability_facts=tuple(capabilities),
        claim_boundary_ids=tuple(item.boundary_id for item in boundaries),
        claim_boundaries=tuple(boundaries),
    )


def _cluster(facts: list[ProjectEvidenceFact], **kwargs) -> StoryClusteringResult:
    return cluster_story_evidence_bundle(_bundle(facts, **kwargs))


def _explicit_fact(
    evidence_fact_id: str,
    source_id: str,
    change_id: str,
    *,
    mechanism: str = "Bounded structural evidence",
) -> ProjectEvidenceFact:
    return _fact(
        evidence_fact_id,
        [_ref(source_id, metadata={"change_id": change_id})],
        mechanism=mechanism,
    )


def _members(result: StoryClusteringResult) -> set[tuple[str, ...]]:
    return {cluster.member_evidence_fact_ids for cluster in result.clusters}


def test_relation_precedence_is_explicit_and_not_equal_to_raw_strength() -> None:
    assert story_cluster_relation_role(
        StoryEvidenceRelationType.SAME_EXPLICIT_CHANGE
    ) is StoryClusterRelationRole.ESTABLISHES_MEMBERSHIP
    assert story_cluster_relation_role(
        StoryEvidenceRelationType.SAME_SOURCE
    ) is StoryClusterRelationRole.ESTABLISHES_MEMBERSHIP
    assert story_cluster_relation_role(
        StoryEvidenceRelationType.SAME_COMMIT
    ) is StoryClusterRelationRole.SUPPORTS_MEMBERSHIP
    assert story_cluster_relation_role(
        StoryEvidenceRelationType.CAPABILITY_SUPPORT_RELATION
    ) is StoryClusterRelationRole.SUPPORTS_MEMBERSHIP
    assert story_cluster_relation_role(
        StoryEvidenceRelationType.SAME_PARENT_CHANGE
    ) is StoryClusterRelationRole.CONTEXT_ONLY


def test_same_explicit_event_establishes_one_strong_cluster() -> None:
    result = _cluster([
        _explicit_fact("pef_alpha", "chunk_alpha", "change_1"),
        _explicit_fact("pef_beta", "chunk_beta", "change_1"),
    ])

    assert _members(result) == {("pef_alpha", "pef_beta")}
    cluster = result.clusters[0]
    assert cluster.event_core.core_kind is StoryClusterCoreKind.EXPLICIT_CHANGE
    assert cluster.quality is StoryClusterQuality.STRONG
    assert cluster.identity_state is StoryClusterIdentityState.STABLE_EVENT_CORE
    assert cluster.membership_links[0].basis is (
        StoryClusterMembershipBasis.SAME_EXPLICIT_CHANGE
    )
    assert {item.reason_code for item in result.decisions} == {
        StoryClusterReasonCode.SAME_EXPLICIT_CHANGE
    }


def test_direct_parent_child_change_establishes_membership() -> None:
    parent = _fact(
        "pef_parent",
        [_ref(
            "change_parent",
            source_type="project_change_raw_change_summary",
            metadata={},
        )],
    )
    child = _fact(
        "pef_child",
        [_ref(
            "change_child",
            source_type="project_change_raw_change_summary",
            metadata={"parent_change_id": "change_parent"},
        )],
    )

    result = _cluster([parent, child])

    assert _members(result) == {("pef_child", "pef_parent")}
    cluster = result.clusters[0]
    assert cluster.event_core.core_kind is StoryClusterCoreKind.PARENT_CHILD_CHANGE
    assert cluster.membership_links[0].basis is (
        StoryClusterMembershipBasis.DIRECT_PARENT_CHILD_CHANGE
    )
    assert cluster.quality is StoryClusterQuality.STRONG


def test_parent_event_keeps_existing_explicit_peers_and_child_together() -> None:
    parent_one = _explicit_fact("pef_parent_a", "chunk_parent_a", "change_parent")
    parent_two = _explicit_fact("pef_parent_b", "chunk_parent_b", "change_parent")
    child = _fact(
        "pef_child",
        [_ref(
            "change_child",
            source_type="project_change_raw_change_summary",
            metadata={"parent_change_id": "change_parent"},
        )],
    )
    result = _cluster([parent_one, parent_two, child])
    assert _members(result) == {("pef_child", "pef_parent_a", "pef_parent_b")}
    assert {item.basis for item in result.clusters[0].membership_links} == {
        StoryClusterMembershipBasis.SAME_EXPLICIT_CHANGE,
        StoryClusterMembershipBasis.DIRECT_PARENT_CHILD_CHANGE,
    }


def test_shared_parent_without_present_parent_event_is_context_only() -> None:
    children = [
        _fact(
            f"pef_child_{index}",
            [_ref(
                f"change_child_{index}",
                source_type="project_change_raw_change_summary",
                metadata={"parent_change_id": "change_parent"},
            )],
        )
        for index in range(2)
    ]
    result = _cluster(children)
    assert len(result.clusters) == 2


def test_same_commit_and_symbol_can_establish_membership() -> None:
    first = _fact(
        "pef_alpha",
        [_ref("chunk_alpha", file_path="backend/a.py", symbol="resolve")],
    )
    second = _fact(
        "pef_beta",
        [_ref("chunk_beta", file_path="backend/b.py", symbol="resolve")],
    )

    result = _cluster([first, second])

    assert _members(result) == {("pef_alpha", "pef_beta")}
    assert result.clusters[0].event_core.core_kind is (
        StoryClusterCoreKind.COMMIT_AND_SYMBOL
    )
    assert result.clusters[0].membership_links[0].basis is (
        StoryClusterMembershipBasis.SAME_COMMIT_AND_SYMBOL
    )


def test_same_commit_different_subsystems_are_split() -> None:
    result = _cluster([
        _fact(
            "pef_retrieval",
            [_ref("chunk_retrieval", file_path="backend/retrieval.py", symbol="search")],
        ),
        _fact(
            "pef_frontend",
            [_ref("chunk_frontend", file_path="frontend/styles.css", symbol="theme")],
        ),
    ])
    assert _members(result) == {("pef_frontend",), ("pef_retrieval",)}


def test_same_path_alone_across_history_does_not_merge() -> None:
    result = _cluster([
        _fact(
            "pef_old",
            [_ref("chunk_old", commit_sha="aaaaaaa", file_path="backend/store.py")],
        ),
        _fact(
            "pef_new",
            [_ref("chunk_new", commit_sha="bbbbbbb", file_path="backend/store.py")],
        ),
    ])
    assert _members(result) == {("pef_new",), ("pef_old",)}


def test_same_symbol_across_unrelated_changes_does_not_merge() -> None:
    result = _cluster([
        _fact(
            "pef_old",
            [_ref("chunk_old", commit_sha="aaaaaaa", symbol="validate")],
        ),
        _fact(
            "pef_new",
            [_ref("chunk_new", commit_sha="bbbbbbb", symbol="validate")],
        ),
    ])
    assert len(result.clusters) == 2


def test_same_capability_alone_is_context_not_event_identity() -> None:
    facts = [
        _fact(
            "pef_alpha",
            [_ref(
                "chunk_alpha",
                repo="owner/one",
                commit_sha="aaaaaaa",
                file_path="one.py",
                symbol="one",
            )],
        ),
        _fact(
            "pef_beta",
            [_ref(
                "chunk_beta",
                repo="owner/two",
                commit_sha="bbbbbbb",
                file_path="two.py",
                symbol="two",
            )],
        ),
    ]
    capability = _capability([item.evidence_fact_id for item in facts])
    result = _cluster(facts, capabilities=[capability])

    assert len(result.clusters) == 2
    assert result.capability_lineages[0].confidence is Confidence.LOW
    assert all(cluster.member_capability_ids == (capability.capability_id,) for cluster in result.clusters)


def test_commit_path_needs_extra_support_and_never_becomes_strong() -> None:
    facts = [
        _fact(
            "pef_alpha",
            [_ref("chunk_alpha", symbol=None)],
        ),
        _fact(
            "pef_beta",
            [_ref("chunk_beta", symbol=None)],
        ),
    ]
    capability = _capability([item.evidence_fact_id for item in facts])
    result = _cluster(facts, capabilities=[capability])

    assert _members(result) == {("pef_alpha", "pef_beta")}
    cluster = result.clusters[0]
    assert cluster.event_core.core_kind is StoryClusterCoreKind.COMMIT_AND_PATH_SUPPORT
    assert cluster.quality is StoryClusterQuality.MODERATE
    assert cluster.identity_state is StoryClusterIdentityState.CANDIDATE
    assert cluster.capability_lineages[0].confidence is Confidence.LOW


def test_commit_path_support_cannot_override_distinct_symbol_cores() -> None:
    facts = [
        _fact("pef_alpha", [_ref("chunk_alpha", symbol="alpha")]),
        _fact("pef_beta", [_ref("chunk_beta", symbol="beta")]),
    ]
    capability = _capability([item.evidence_fact_id for item in facts])
    assert len(_cluster(facts, capabilities=[capability]).clusters) == 2


def test_cluster_contract_rejects_quality_or_identity_upgrades() -> None:
    facts = [
        _fact("pef_alpha", [_ref("chunk_alpha", symbol=None)]),
        _fact("pef_beta", [_ref("chunk_beta", symbol=None)]),
    ]
    capability = _capability([item.evidence_fact_id for item in facts])
    cluster = _cluster(facts, capabilities=[capability]).clusters[0]
    assert cluster.quality is StoryClusterQuality.MODERATE
    assert cluster.identity_state is StoryClusterIdentityState.CANDIDATE
    with pytest.raises(ValueError, match="quality"):
        replace(cluster, quality=StoryClusterQuality.STRONG)
    with pytest.raises(ValueError, match="identity state"):
        replace(cluster, identity_state=StoryClusterIdentityState.STABLE_EVENT_CORE)


def test_weak_bridge_cannot_merge_two_strong_event_clusters() -> None:
    first = [
        _explicit_fact("pef_a", "chunk_a", "change_a"),
        _explicit_fact("pef_b", "chunk_b", "change_a"),
    ]
    second = [
        _explicit_fact("pef_c", "chunk_c", "change_c"),
        _explicit_fact("pef_d", "chunk_d", "change_c"),
    ]
    bridge = _fact(
        "pef_bridge",
        [
            _ref("chunk_bridge_a", file_path="backend/service.py", symbol="resolve"),
            _ref(
                "chunk_bridge_c",
                commit_sha="bbbbbbb",
                file_path="backend/other.py",
                symbol="other",
            ),
        ],
    )
    second = [
        replace(
            item,
            source_refs=[
                _ref(
                    f"chunk_{item.evidence_fact_id}_c",
                    commit_sha="bbbbbbb",
                    file_path="backend/other.py",
                    symbol="other",
                    metadata={"change_id": "change_c"},
                )
            ],
        )
        for item in second
    ]
    result = _cluster([*first, *second, bridge])

    assert ("pef_a", "pef_b") in _members(result)
    assert ("pef_c", "pef_d") in _members(result)
    assert ("pef_bridge",) in _members(result)
    assert len(result.clusters) == 3


def test_strong_then_weak_chain_has_no_transitive_leakage() -> None:
    first = _explicit_fact("pef_a", "chunk_a", "change_a")
    second = _explicit_fact("pef_b", "chunk_b", "change_a")
    third = _fact(
        "pef_c",
        [_ref(
            "chunk_c",
            commit_sha="bbbbbbb",
            file_path="backend/service.py",
            symbol="different",
        )],
    )
    result = _cluster([first, second, third])
    assert _members(result) == {("pef_a", "pef_b"), ("pef_c",)}


def test_equal_strength_multi_core_membership_isolated_as_ambiguous() -> None:
    shared_one = _ref("shared_one", content="same-one")
    shared_two = _ref(
        "shared_two",
        commit_sha="bbbbbbb",
        file_path="backend/two.py",
        symbol="two",
        content="same-two",
    )
    first = _fact("pef_a", [shared_one])
    bridge = _fact("pef_bridge", [shared_one, shared_two])
    second = _fact("pef_c", [shared_two])

    result = _cluster([first, bridge, second])

    assert len(result.clusters) == 3
    decision = next(
        item for item in result.decisions if item.evidence_fact_id == "pef_bridge"
    )
    assert decision.outcome is StoryClusterDecisionOutcome.ISOLATED_AMBIGUOUS
    assert decision.reason_code is StoryClusterReasonCode.AMBIGUOUS_EVENT_CORES
    assert decision.competing_core_count == 2
    ambiguous_cluster = next(
        item for item in result.clusters if item.member_evidence_fact_ids == ("pef_bridge",)
    )
    assert ambiguous_cluster.lineage_state is StoryClusterLineageState.AMBIGUOUS
    assert ambiguous_cluster.event_core.core_kind is StoryClusterCoreKind.EVIDENCE_SINGLETON


def test_clear_explicit_singleton_is_retained() -> None:
    result = _cluster([_explicit_fact("pef_alpha", "chunk_alpha", "change_1")])
    assert len(result.clusters) == 1
    assert result.clusters[0].quality is StoryClusterQuality.STRONG
    assert result.decisions[0].outcome is StoryClusterDecisionOutcome.SINGLETON


def test_missing_lineage_remains_weak_singleton_without_semantic_rescue() -> None:
    fact = _fact(
        "pef_alpha",
        [_ref(
            "chunk_alpha",
            repo=None,
            commit_sha=None,
            file_path=None,
            symbol=None,
            metadata={},
        )],
    )
    result = _cluster([fact])
    cluster = result.clusters[0]
    assert cluster.quality is StoryClusterQuality.WEAK
    assert cluster.lineage_state is StoryClusterLineageState.INCOMPLETE
    assert cluster.identity_state is StoryClusterIdentityState.CANDIDATE
    assert result.decisions[0].reason_code is (
        StoryClusterReasonCode.SINGLETON_INCOMPLETE_LINEAGE
    )


def test_cross_project_override_remains_impossible_after_bundle_validation() -> None:
    bundle = _bundle([_explicit_fact("pef_alpha", "chunk_alpha", "change_1")])
    with pytest.raises(StoryEvidenceResolutionError):
        replace(bundle, project_id=OTHER_PROJECT_ID)
    with pytest.raises(TypeError, match="StoryEvidenceBundle"):
        cluster_story_evidence_bundle(object())  # type: ignore[arg-type]
    assert set(inspect.signature(cluster_story_evidence_bundle).parameters) == {"bundle"}


def test_large_mixed_commit_splits_by_stronger_subanchors() -> None:
    areas = (
        ("retrieval", "backend/retrieval.py", "search"),
        ("latex", "backend/latex.py", "render"),
        ("frontend", "frontend/styles.css", "theme"),
        ("tests", "tests/config.py", "configure"),
    )
    facts = [
        _fact(
            f"pef_{name}",
            [_ref(
                f"chunk_{name}",
                commit_sha="abcdef1",
                file_path=path,
                symbol=symbol,
            )],
        )
        for name, path, symbol in areas
    ]
    result = _cluster(facts)
    assert len(result.clusters) == 4
    assert all(len(item.evidence_inputs) == 1 for item in result.clusters)


def test_repeated_subsystem_history_without_parent_relation_stays_split() -> None:
    commits = ("aaaaaaa", "bbbbbbb", "ccccccc")
    facts = [
        _fact(
            f"pef_history_{index}",
            [_ref(
                f"chunk_history_{index}",
                commit_sha=commit,
                file_path="backend/chroma_lifecycle.py",
                symbol="start",
            )],
        )
        for index, commit in enumerate(commits)
    ]
    assert len(_cluster(facts).clusters) == 3


def test_five_unrelated_fixes_sharing_capability_stay_separate() -> None:
    facts = [
        _fact(
            f"pef_fix_{index}",
            [_ref(
                f"chunk_fix_{index}",
                repo=f"owner/repo{index}",
                commit_sha=f"{index + 1:07x}",
                file_path=f"module_{index}.py",
                symbol=f"fix_{index}",
            )],
        )
        for index in range(5)
    ]
    capability = _capability([item.evidence_fact_id for item in facts])
    result = _cluster(facts, capabilities=[capability])
    assert len(result.clusters) == 5


def test_claim_boundaries_are_preserved_but_never_create_membership() -> None:
    facts = [
        _fact(
            "pef_alpha",
            [_ref("chunk_alpha", repo="owner/one", commit_sha="aaaaaaa")],
        ),
        _fact(
            "pef_beta",
            [_ref("chunk_beta", repo="owner/two", commit_sha="bbbbbbb")],
        ),
    ]
    boundary = _project_boundary()
    result = _cluster(facts, boundaries=[boundary])

    assert len(result.clusters) == 2
    assert result.claim_boundary_ids == (boundary.boundary_id,)
    assert all(cluster.claim_boundary_ids == (boundary.boundary_id,) for cluster in result.clusters)


def test_cluster_id_is_stable_when_supporting_evidence_is_added_to_event_core() -> None:
    first = _explicit_fact("pef_alpha", "chunk_alpha", "change_1")
    second = _explicit_fact("pef_beta", "chunk_beta", "change_1")
    baseline = _cluster([first, second]).clusters[0]
    expanded = _cluster([
        first,
        second,
        _explicit_fact("pef_gamma", "chunk_gamma", "change_1"),
    ]).clusters[0]
    assert baseline.cluster_id == expanded.cluster_id
    assert baseline.event_core == expanded.event_core


def test_cluster_identity_and_output_ignore_mutable_authoritative_prose() -> None:
    before = _cluster([
        _explicit_fact(
            "pef_alpha", "chunk_alpha", "change_1", mechanism="First summary"
        ),
        _explicit_fact(
            "pef_beta", "chunk_beta", "change_1", mechanism="Second summary"
        ),
    ])
    after = _cluster([
        _explicit_fact(
            "pef_alpha", "chunk_alpha", "change_1", mechanism="Rewritten prose"
        ),
        _explicit_fact(
            "pef_beta", "chunk_beta", "change_1", mechanism="Different prose"
        ),
    ])
    assert before == after
    assert "Rewritten prose" not in after.to_json()


def test_jd_company_and_hiring_context_are_absent_from_clustering_contract() -> None:
    parameters = inspect.signature(cluster_story_evidence_bundle).parameters
    assert not ({"jd", "jd_text", "company", "hiring_context"} & set(parameters))
    source = Path("backend/engineering_story_clustering.py").read_text(encoding="utf-8")
    assert "jd_text" not in source.lower()
    assert "hiring_context" not in source.lower()


def test_input_permutations_produce_identical_clusters_ids_and_decisions() -> None:
    facts = [
        _explicit_fact("pef_alpha", "chunk_alpha", "change_1"),
        _explicit_fact("pef_beta", "chunk_beta", "change_1"),
        _fact(
            "pef_single",
            [_ref(
                "chunk_single",
                repo="owner/other",
                commit_sha="bbbbbbb",
                file_path="other.py",
                symbol="other",
            )],
        ),
    ]
    baseline = _cluster(facts)
    randomizer = random.Random(11)
    for _ in range(10):
        shuffled = list(facts)
        randomizer.shuffle(shuffled)
        bundle = _bundle(shuffled)
        shuffled_bundle = StoryEvidenceBundle(
            project_id=bundle.project_id,
            evidence_inputs=tuple(reversed(bundle.evidence_inputs)),
            capability_lineages=tuple(reversed(bundle.capability_lineages)),
            claim_boundary_ids=tuple(reversed(bundle.claim_boundary_ids)),
            event_anchors=tuple(reversed(bundle.event_anchors)),
            relations=tuple(reversed(bundle.relations)),
            lineage_state=bundle.lineage_state,
        )
        assert cluster_story_evidence_bundle(shuffled_bundle).to_json() == baseline.to_json()


def test_structurally_valid_oversized_event_fails_instead_of_truncating() -> None:
    facts = [
        _fact(
            f"pef_bulk_{index}",
            [_ref(
                f"chunk_bulk_{index}",
                repo=f"owner/repo{index}",
                commit_sha=f"{index + 1:07x}",
                file_path=f"module_{index}.py",
                symbol=f"symbol_{index}",
                metadata={"change_id": "change_bulk"},
            )],
        )
        for index in range(MAX_EVIDENCE_MEMBERS_PER_CLUSTER + 1)
    ]
    bundle = _bundle(facts)
    with pytest.raises(StoryClusteringError) as raised:
        cluster_story_evidence_bundle(bundle)
    assert raised.value.code is StoryClusteringErrorCode.CLUSTER_BOUND_EXCEEDED


def test_clustering_result_round_trip_is_strict_frozen_and_mutation_isolated() -> None:
    result = _cluster([
        _explicit_fact("pef_alpha", "chunk_alpha", "change_1"),
        _explicit_fact("pef_beta", "chunk_beta", "change_1"),
    ])
    payload = result.to_dict()
    rebuilt = StoryClusteringResult.from_dict(payload)
    assert rebuilt == result
    assert rebuilt.to_json() == result.to_json()
    payload["claim_boundary_ids"].append("pcb_external")
    assert rebuilt.claim_boundary_ids == ()
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.project_id = OTHER_PROJECT_ID  # type: ignore[misc]


def test_clustering_never_constructs_engineering_story_or_story_fields() -> None:
    result = _cluster([_explicit_fact("pef_alpha", "chunk_alpha", "change_1")])
    assert not isinstance(result, EngineeringStory)
    serialized = result.to_json()
    for field in (
        "problem_context",
        "decision",
        "mechanism",
        "observable_outcome",
        "resume_bullet",
        "story_id",
    ):
        assert f'"{field}"' not in serialized


def test_diagnostics_are_bounded_ids_and_reason_codes_without_paths() -> None:
    result = _cluster([
        _explicit_fact("pef_alpha", "chunk_alpha", "change_1"),
        _explicit_fact("pef_beta", "chunk_beta", "change_1"),
    ])
    for decision in result.decisions:
        assert set(decision.to_dict()) == {
            "project_id",
            "evidence_fact_id",
            "cluster_id",
            "outcome",
            "reason_code",
            "related_evidence_fact_ids",
            "competing_core_count",
        }
        assert "path" not in decision.to_json().lower()
        assert "query" not in decision.to_json().lower()


def test_module_is_pure_and_has_no_graph_runtime_or_model_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = Path("backend/engineering_story_clustering.py")
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
        "backend.evidence_memory",
        "chromadb",
        "networkx",
        "os",
    }
    assert not any(
        name == blocked or name.startswith(f"{blocked}.")
        for name in imported
        for blocked in forbidden
    )
    source = module_path.read_text(encoding="utf-8")
    assert "connected_components" not in source
    assert "PersistentClient" not in source
    assert "EngineeringStory(" not in source
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: pytest.fail("I/O"))
    assert _cluster([
        _explicit_fact("pef_alpha", "chunk_alpha", "change_1")
    ]).project_id == PROJECT_ID


def test_serialized_cluster_contains_no_raw_or_semantic_fallback_fields() -> None:
    payload = json.loads(_cluster([
        _explicit_fact("pef_alpha", "chunk_alpha", "change_1")
    ]).to_json())
    forbidden = {
        "raw_patch",
        "raw_text",
        "document",
        "embedding",
        "query",
        "score",
        "jd",
        "company",
        "problem",
        "decision",
        "observable_outcome",
        "resume_bullet",
    }

    def all_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(all_keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(all_keys(item) for item in value), set())
        return set()

    assert forbidden.isdisjoint(all_keys(payload))
