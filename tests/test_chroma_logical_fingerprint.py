from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from backend import chroma_logical_fingerprint as logical
from backend.chroma_collection_registry import get_collection_definition
from backend.chroma_http_transport import ChromaTransportRecords
from backend.chroma_logical_fingerprint_models import (
    CHROMA_LOGICAL_FINGERPRINT_SCHEMA,
    CHROMA_LOGICAL_GATE_SCHEMA,
    ChromaLogicalFingerprint,
    ChromaLogicalModelError,
)
from backend.project_repository_identity import build_project_repository_identity_authority


def repository_authority(*pairs: tuple[str, str]):
    projects = sorted({project for project, _ in pairs})
    return build_project_repository_identity_authority(
        project_memory={"projects": [{"project_id": project} for project in projects]},
        user_confirmed_links=[
            {"project_id": project, "repository": repository, "confirmed": True}
            for project, repository in pairs
        ],
    )


AUTHORITY = repository_authority(("project-a", "owner/repo"))


def github_metadata(**changes):
    value = {
        "chunk_type": "code",
        "commit_sha": "abc123",
        "project_id": "project-a",
        "repository": "owner/repo",
        "repository_project_id": "project-a",
        "source_id": "source-1",
        "source_type": "file",
    }
    value.update(changes)
    return value


def github_fingerprint(records=None, **kwargs):
    return logical.build_logical_fingerprint(
        "github_evidence",
        records if records is not None else [("id-1", github_metadata())],
        repository_authority=kwargs.pop("repository_authority", AUTHORITY),
        **kwargs,
    )


def profile_fingerprint(records=None, **kwargs):
    return logical.build_logical_fingerprint(
        "profile_facts",
        records if records is not None else [("profile-1", {"index": 1, "is_list": False, "section": "skills"})],
        **kwargs,
    )


class FakeHandle:
    def __init__(self, pages, counts):
        self.pages = pages
        self.counts = list(counts)
        self.page_calls = []

    def safe_count(self):
        return self.counts.pop(0)

    def safe_get_page(self, *, limit, offset):
        self.page_calls.append((limit, offset))
        ids, metadatas = self.pages[offset]
        return ChromaTransportRecords(
            semantic_collection_id="profile_facts",
            ids=tuple(ids),
            metadatas=tuple(metadatas),
            included=("metadatas",),
        )


def test_model_schema_is_strict_immutable_and_privacy_safe():
    value = profile_fingerprint()
    assert value.schema == CHROMA_LOGICAL_FINGERPRINT_SCHEMA
    assert len(value.aggregate_digest) == 64
    with pytest.raises(FrozenInstanceError):
        value.record_count = 99
    payload = value.to_dict()
    assert ChromaLogicalFingerprint.from_mapping(payload) == value
    with pytest.raises(ChromaLogicalModelError, match="invalid_logical_fingerprint_shape"):
        ChromaLogicalFingerprint.from_mapping({**payload, "unknown": True})
    serialized = json.dumps(value.safe_summary()).casefold()
    for forbidden in ("document", "embedding", "absolute_path", "raw_metadata", "uri"):
        assert forbidden not in serialized


def test_unknown_collection_and_invalid_schema_fail_closed(monkeypatch):
    with pytest.raises(logical.ChromaLogicalSchemaMismatch, match="unknown_logical_collection"):
        logical.build_logical_fingerprint("unknown", [])
    definition = replace(get_collection_definition("profile_facts"), schema_version="invalid")
    monkeypatch.setattr(logical, "_definition", lambda _value: definition)
    with pytest.raises(logical.ChromaLogicalSchemaMismatch, match="logical_collection_schema_mismatch"):
        logical.build_logical_fingerprint("profile_facts", [])


def test_record_and_metadata_order_do_not_affect_digests():
    first = github_fingerprint(
        [("b", github_metadata(source_id="two")), ("a", github_metadata(source_id="one"))]
    )
    reordered_metadata = dict(reversed(list(github_metadata(source_id="two").items())))
    second = github_fingerprint(
        [("a", github_metadata(source_id="one")), ("b", reordered_metadata)]
    )
    assert first == second


def test_record_add_remove_and_rename_change_identity_digest():
    one = github_fingerprint([("a", github_metadata())])
    added = github_fingerprint([("a", github_metadata()), ("b", github_metadata())])
    renamed = github_fingerprint([("renamed", github_metadata())])
    assert one.record_id_digest != added.record_id_digest
    assert one.record_id_digest != renamed.record_id_digest
    assert added.record_id_digest != renamed.record_id_digest


@pytest.mark.parametrize("record_id", ["", None])
def test_blank_or_invalid_record_id_is_rejected(record_id):
    with pytest.raises(logical.ChromaLogicalSnapshotUnstable, match="invalid_logical_record_id"):
        github_fingerprint([(record_id, github_metadata())])


def test_duplicate_record_id_is_rejected():
    with pytest.raises(logical.ChromaLogicalSnapshotUnstable, match="duplicate_logical_record_id"):
        github_fingerprint([("same", github_metadata()), ("same", github_metadata())])


def test_count_consistency_and_unstable_snapshot_fail_closed():
    with pytest.raises(logical.ChromaLogicalSnapshotUnstable, match="logical_record_count_mismatch"):
        github_fingerprint(count_before=2, count_after=2)
    with pytest.raises(logical.ChromaLogicalSnapshotUnstable, match="logical_snapshot_count_changed"):
        github_fingerprint(count_before=1, count_after=2)


def test_bounded_pagination_collects_pages_and_ignores_page_order():
    metadata = {"index": 1, "is_list": False, "section": "skills"}
    handle = FakeHandle(
        {0: (("b", "a"), (metadata, metadata)), 2: (("c",), (metadata,))},
        counts=(3, 3),
    )
    records, before, after = logical.retrieve_logical_collection_records(handle, page_size=2)
    assert (before, after) == (3, 3)
    assert handle.page_calls == [(2, 0), (1, 2)]
    assert profile_fingerprint(records) == profile_fingerprint(list(reversed(records)))


def test_pagination_duplicate_missing_overrun_and_count_change_fail():
    metadata = {"section": "skills"}
    duplicate = FakeHandle(
        {0: (("a", "b"), (metadata, metadata)), 2: (("b",), (metadata,))},
        counts=(3, 3),
    )
    with pytest.raises(logical.ChromaLogicalSnapshotUnstable, match="duplicate_logical_record_id"):
        logical.retrieve_logical_collection_records(duplicate, page_size=2)
    missing = FakeHandle({0: (("a",), (metadata,))}, counts=(2, 2))
    with pytest.raises(logical.ChromaLogicalSnapshotUnstable, match="logical_pagination_incomplete"):
        logical.retrieve_logical_collection_records(missing, page_size=2)
    overrun = FakeHandle({0: (("a", "b"), (metadata, metadata))}, counts=(1, 1))
    with pytest.raises(logical.ChromaLogicalSnapshotUnstable, match="logical_pagination_incomplete"):
        logical.retrieve_logical_collection_records(overrun, page_size=1)
    changing = FakeHandle({0: (("a",), (metadata,))}, counts=(1, 2))
    with pytest.raises(logical.ChromaLogicalSnapshotUnstable, match="logical_snapshot_count_changed"):
        logical.retrieve_logical_collection_records(changing)


@pytest.mark.parametrize("page_size", [0, logical.DEFAULT_PAGE_SIZE + 1, True])
def test_page_size_is_bounded(page_size):
    with pytest.raises(logical.ChromaLogicalRecordLimitExceeded, match="invalid_logical_page_size"):
        logical.retrieve_logical_collection_records(FakeHandle({}, (0, 0)), page_size=page_size)


def test_collection_and_metadata_size_bounds_fail_safely(monkeypatch):
    monkeypatch.setattr(logical, "MAX_LOGICAL_RECORDS", 1)
    with pytest.raises(logical.ChromaLogicalRecordLimitExceeded, match="logical_record_limit_exceeded"):
        profile_fingerprint([("a", {}), ("b", {})])
    monkeypatch.setattr(logical, "MAX_METADATA_BYTES_PER_RECORD", 20)
    with pytest.raises(logical.ChromaLogicalMetadataUnsafe, match="logical_metadata_record_limit_exceeded"):
        profile_fingerprint([("a", {"section": "x" * 40})])


def test_allowed_metadata_changes_digest_but_non_allowlisted_data_does_not():
    first = github_fingerprint([("a", {**github_metadata(), "ignored": "one"})])
    ignored = github_fingerprint([("a", {**github_metadata(), "ignored": "two"})])
    allowed = github_fingerprint([("a", github_metadata(source_type="commit"))])
    assert first == ignored
    assert first.metadata_digest != allowed.metadata_digest
    assert first.aggregate_digest != allowed.aggregate_digest


def test_absent_null_and_empty_metadata_states_are_distinct():
    absent = profile_fingerprint([("a", {})])
    null = profile_fingerprint([("a", {"section": None})])
    empty = profile_fingerprint([("a", {"section": ""})])
    assert len({absent.metadata_digest, null.metadata_digest, empty.metadata_digest}) == 3


@pytest.mark.parametrize(
    "metadata,code",
    [
        ({"section": float("nan")}, "non_finite_logical_metadata"),
        ({"section": {"nested": "value"}}, "unsupported_logical_metadata_value"),
        ({"section": "C:\\private\\file"}, "unsafe_logical_metadata_string"),
        ({"section": "https://example.invalid/private"}, "unsafe_logical_metadata_string"),
        ({"raw_metadata": "value"}, "forbidden_logical_metadata_field"),
        ({"documents": "value"}, "forbidden_logical_metadata_field"),
        ({"embeddings": "value"}, "forbidden_logical_metadata_field"),
    ],
)
def test_unsafe_metadata_is_rejected_without_exposure(metadata, code):
    with pytest.raises(logical.ChromaLogicalMetadataUnsafe, match=code) as error:
        profile_fingerprint([("a", metadata)])
    assert "private" not in str(error.value)


def test_valid_github_repository_project_authority_passes():
    result = github_fingerprint()
    assert result.integrity_state == "valid"


@pytest.mark.parametrize(
    "metadata,code",
    [
        (github_metadata(repository="owner/unknown"), "unknown_github_repository"),
        (github_metadata(project_id="other"), "github_repository_project_mismatch"),
        (github_metadata(repository_project_id="other"), "github_cross_project_record"),
        ({}, "missing_github_project_authority"),
        ({"project_id": "project-a"}, "missing_github_project_authority"),
    ],
)
def test_github_authority_violations_fail_closed(metadata, code):
    with pytest.raises(logical.ChromaLogicalAuthorityViolation, match=code):
        github_fingerprint([("a", metadata)])


def test_missing_or_conflicted_mapping_authority_fails_closed():
    with pytest.raises(logical.ChromaLogicalAuthorityViolation, match="repository_mapping_authority_unavailable"):
        github_fingerprint(repository_authority=None)
    conflicted = repository_authority(
        ("project-a", "owner/repo"),
        ("project-b", "owner/repo"),
    )
    with pytest.raises(logical.ChromaLogicalAuthorityViolation):
        github_fingerprint(repository_authority=conflicted)


def test_repository_only_legacy_metadata_uses_unique_authority_without_guessing():
    result = github_fingerprint([("legacy", {"repository": "owner/repo"})])
    assert result.integrity_state == "valid"
    with pytest.raises(logical.ChromaLogicalAuthorityViolation, match="unknown_github_repository"):
        github_fingerprint([("legacy", {"repository": "owner/not-authorized"})])


def test_mapping_authority_change_changes_authority_digest():
    first = github_fingerprint()
    expanded = repository_authority(
        ("project-a", "owner/repo"),
        ("project-b", "owner/other"),
    )
    second = github_fingerprint(repository_authority=expanded)
    assert first.metadata_digest == second.metadata_digest
    assert first.authority_digest != second.authority_digest


def test_profile_uses_existing_single_profile_scope_without_invented_identity():
    empty = profile_fingerprint([])
    existing = profile_fingerprint()
    assert empty.integrity_state == existing.integrity_state == "valid"
    assert "user" not in json.dumps(existing.safe_summary()).casefold()


def test_registry_identity_and_allowlist_participate_in_authority(monkeypatch):
    original = get_collection_definition("profile_facts")
    first = profile_fingerprint()
    changed = replace(
        original,
        collection_name="profile_facts_rekeyed",
        logical_integrity_metadata_allowlist=(*original.logical_integrity_metadata_allowlist, "stable_tag"),
    )
    monkeypatch.setattr(logical, "_definition", lambda _value: changed)
    second = profile_fingerprint()
    assert first.authority_digest != second.authority_digest
    assert first.aggregate_digest != second.aggregate_digest


def test_automatic_creation_is_rejected(monkeypatch):
    changed = replace(get_collection_definition("profile_facts"), automatic_creation=True)
    monkeypatch.setattr(logical, "_definition", lambda _value: changed)
    with pytest.raises(logical.ChromaLogicalSchemaMismatch, match="unsafe_logical_collection_creation_policy"):
        profile_fingerprint([])


@pytest.mark.parametrize(
    "field,state",
    [
        ("record_count", "record_count_mismatch"),
        ("record_id_digest", "record_identity_mismatch"),
        ("metadata_digest", "metadata_mismatch"),
        ("authority_digest", "authority_mismatch"),
        ("collection_schema_version", "schema_mismatch"),
    ],
)
def test_comparison_classifies_semantic_mismatch(field, state):
    expected = profile_fingerprint()
    value = expected.record_count + 1 if field == "record_count" else (
        "profile_facts.v2" if field == "collection_schema_version" else "f" * 64
    )
    actual = replace(expected, **{field: value})
    assert logical.compare_logical_fingerprints(expected, actual).state == state


def test_comparison_match_missing_and_invalid_are_bounded():
    expected = profile_fingerprint()
    assert logical.compare_logical_fingerprints(expected, expected).state == "match"
    assert logical.compare_logical_fingerprints(expected, None).state == "collection_missing"
    assert logical.compare_logical_fingerprints(expected, object()).state == "invalid"


def test_self_check_proves_mutation_sensitivity_and_safe_exclusion():
    assert logical.run_logical_fingerprint_self_check() is True


def test_gate_requires_complete_deterministic_real_evidence():
    baseline = logical.ChromaLogicalBaseline(
        schema=logical.CHROMA_LOGICAL_BASELINE_SCHEMA,
        backup_id="20260810T053709Z-a47ec430cbf1",
        backup_aggregate_sha256="a" * 64,
        chroma_version="1.5.9",
        registry_schema=logical.CHROMA_COLLECTION_REGISTRY_SCHEMA,
        fingerprint_schema=logical.CHROMA_LOGICAL_FINGERPRINT_SCHEMA,
        created_at="2026-08-10T00:00:00Z",
        fingerprints=tuple(sorted((github_fingerprint(), profile_fingerprint()), key=lambda item: item.collection_semantic_id)),
        repeated_run_deterministic=True,
        restart_deterministic=True,
        workspace_byte_fingerprint_before="b" * 64,
        workspace_byte_fingerprint_after="c" * 64,
        workspace_byte_mutated=True,
        logical_fingerprints_stable=True,
        immutable_backup_unchanged=True,
        production_persistence_unchanged=True,
        synthetic_validation_passed=True,
        privacy_safe=True,
    )
    ready = logical.evaluate_logical_integrity_gate(baseline)
    assert ready.schema == CHROMA_LOGICAL_GATE_SCHEMA
    assert ready.production_logical_integrity_gate == "logical_integrity_ready"
    blocked = logical.evaluate_logical_integrity_gate(replace(baseline, restart_deterministic=False))
    assert blocked.production_logical_integrity_gate == "blocked"


def test_source_uses_factory_transport_and_no_direct_chroma_clients():
    source = Path(logical.__file__).read_text(encoding="utf-8")
    assert "Persistent" + "Client(" not in source
    assert "Http" + "Client(" not in source
    assert "create_" + "collection" not in source
    assert "get_or_create_" + "collection" not in source
    assert "information/chroma" not in source
    assert "openai" not in source.casefold()
