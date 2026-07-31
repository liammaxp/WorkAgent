from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402
import evidence_memory  # noqa: E402
import evidence_change_summary  # noqa: E402
import evidence_pipeline  # noqa: E402


def test_concurrent_jsonl_upserts_preserve_every_record(tmp_path):
    path = tmp_path / "records.jsonl"

    def upsert(index: int) -> None:
        evidence_memory.upsert_jsonl_record(path, {"id": str(index)}, "id")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(upsert, range(40)))

    assert {record["id"] for record in evidence_memory.read_jsonl(path)} == {
        str(index) for index in range(40)
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_jsonl_upsert_reports_created_updated_and_unchanged(tmp_path):
    path = tmp_path / "records.jsonl"

    _, created = evidence_memory.upsert_jsonl_record_with_status(path, {"id": "one", "value": 1}, "id")
    first_mtime = path.stat().st_mtime_ns
    _, unchanged = evidence_memory.upsert_jsonl_record_with_status(path, {"id": "one", "value": 1}, "id")
    unchanged_mtime = path.stat().st_mtime_ns
    _, updated = evidence_memory.upsert_jsonl_record_with_status(path, {"id": "one", "value": 2}, "id")

    assert (created, unchanged, updated) == ("created", "unchanged", "updated")
    assert unchanged_mtime == first_mtime
    assert evidence_memory.read_jsonl(path) == [{"id": "one", "value": 2}]


@pytest.mark.parametrize("value", [False, "false", "0", "no", "off", 0])
def test_capability_present_false_values_remain_false(value):
    normalized = evidence_memory.normalize_capability_fact(
        {"project_id": "p", "capability_type": "atomic_persistence", "present": value}
    )
    assert normalized["present"] is False


def test_capability_present_rejects_ambiguous_values():
    with pytest.raises(ValueError, match="present must be a boolean"):
        evidence_memory.normalize_capability_fact(
            {"project_id": "p", "capability_type": "atomic_persistence", "present": "disabled-ish"}
        )


def test_chunk_line_numbers_still_normalize_to_integers():
    normalized = evidence_memory.normalize_evidence_chunk(
        {"source_id": "s", "project_id": "p", "start_line": "12", "end_line": 18}
    )
    assert (normalized["start_line"], normalized["end_line"]) == (12, 18)


@pytest.mark.parametrize("max_chars", [0, -1, True, 1.5])
def test_split_text_rejects_invalid_chunk_size(max_chars):
    with pytest.raises(ValueError, match="positive integer"):
        api_server.split_text_into_chunks("content", max_chars)


def test_output_text_preview_paginates_oversized_file(tmp_path, monkeypatch):
    output_file = tmp_path / "large.txt"
    output_file.write_bytes(b"x" * (api_server.MAX_OUTPUT_TEXT_BYTES + 1))
    monkeypatch.setattr(api_server.agent, "OUTPUT_DIR", tmp_path)

    first = api_server.get_output_file(str(output_file), 0, api_server.MAX_OUTPUT_TEXT_BYTES)
    second = api_server.get_output_file(str(output_file), first["next_offset"], api_server.MAX_OUTPUT_TEXT_BYTES)

    assert len(first["content"]) == api_server.MAX_OUTPUT_TEXT_BYTES
    assert first["truncated"] is True
    assert first["total_bytes"] == api_server.MAX_OUTPUT_TEXT_BYTES + 1
    assert second["content"] == "x"
    assert second["truncated"] is False


def test_output_text_preview_keeps_utf8_characters_whole(tmp_path, monkeypatch):
    output_file = tmp_path / "chinese.txt"
    output_file.write_text("a" * 1023 + "求职", encoding="utf-8")
    monkeypatch.setattr(api_server.agent, "OUTPUT_DIR", tmp_path)

    first = api_server.get_output_file(str(output_file), 0, 1024)
    second = api_server.get_output_file(str(output_file), first["next_offset"], 1024)

    assert first["content"] == "a" * 1023
    assert second["content"] == "求职"


def test_change_summary_counts_nonqualifying_chunk_as_skipped(monkeypatch):
    chunk = {"chunk_id": "c1", "project_id": "p1"}
    monkeypatch.setattr(evidence_memory, "read_records", lambda *_: [chunk])
    monkeypatch.setattr(evidence_change_summary, "extract_raw_change_summary_from_chunk", lambda _: None)
    monkeypatch.setattr(
        evidence_memory,
        "get_github_evidence_memory_counts",
        lambda **_: {"raw_change_summaries_count": 0},
    )

    result = evidence_change_summary.build_github_evidence_raw_change_summaries()

    assert result["processed_chunks"] == 1
    assert result["skipped_chunks"] == 1
    assert result["created_or_updated_summaries"] == 0


def test_structural_diff_does_not_trust_reported_additions():
    counts = api_server.structural_json_diff_counts(
        {"project": {"name": "old", "skills": ["Python"]}},
        {"project": {"name": "new", "skills": ["Python", "FastAPI"]}},
    )
    assert counts == {"added": 1, "updated": 1, "removed": 0}

    summary = api_server.build_project_memory_status_summary(
        {
            "updated": True,
            "change_counts": counts,
            "reported_additions": ["model note one", "model note two"],
        },
        was_reanalyzed=True,
        scan_results=[{}],
        before_mtime=1,
        after_mtime=2,
    )
    assert summary["additions_count"] == 1
    assert summary["reported_additions_count"] == 2
    assert "1 added, 1 changed, 0 removed" in summary["label_en"]


def test_pipeline_manifest_becomes_stale_when_project_records_change(tmp_path, monkeypatch):
    monkeypatch.setenv(evidence_memory.GITHUB_EVIDENCE_MEMORY_DIR_ENV, str(tmp_path))
    for record_type in evidence_memory.RECORD_FILES:
        evidence_memory.write_records(record_type, [{"project_id": "p1", "value": record_type}])

    evidence_pipeline.save_pipeline_run_manifest("p1")
    assert evidence_pipeline.pipeline_run_manifest_is_current("p1") is True

    evidence_memory.write_records(
        evidence_memory.EVIDENCE_CHUNKS,
        [{"project_id": "p1", "value": "changed-after-run"}],
    )
    assert evidence_pipeline.pipeline_run_manifest_is_current("p1") is False


def test_application_same_value_is_not_reported_as_updated(tmp_path, monkeypatch):
    monkeypatch.setattr(api_server.agent, "APPLICATION_DB_PATH", tmp_path / "applications.db")
    created = json.loads(api_server.agent.add_application_record("Acme", "Engineer"))

    result = json.loads(
        api_server.agent.update_application_record(created["id"], company="Acme", role="Engineer")
    )

    assert result["updated"] is False
    assert result["reason"] == "Values unchanged."
