import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402
import evidence_memory  # noqa: E402


def fake_repo_context(project_id="WorkAgent", repo="owner/WorkAgent"):
    return {
        "project_id": project_id,
        "repository": repo,
        "url": f"https://github.com/{repo}",
        "description": "Fake repository context for Phase 2 raw persistence.",
        "languages": ["Python"],
        "root_files": ["README.md", "backend/api_server.py"],
    }


class GitHubContextPhase2RawPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.flag_env = api_server.GITHUB_CONTEXT_PHASE2_ENV
        self.storage_env = evidence_memory.PHASE2_EVIDENCE_MEMORY_DIR_ENV
        self.original_flag = os.environ.get(self.flag_env)
        self.original_storage = os.environ.get(self.storage_env)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "phase2_memory"
        os.environ[self.storage_env] = str(self.storage_dir)

    def tearDown(self):
        if self.original_flag is None:
            os.environ.pop(self.flag_env, None)
        else:
            os.environ[self.flag_env] = self.original_flag
        if self.original_storage is None:
            os.environ.pop(self.storage_env, None)
        else:
            os.environ[self.storage_env] = self.original_storage
        self.temp_dir.cleanup()

    def test_disabled_mode_does_not_write(self):
        os.environ[self.flag_env] = "0"

        result = api_server.persist_github_context_phase2_raw_sources(fake_repo_context())

        self.assertFalse(result["enabled"])
        self.assertEqual(0, result["raw_sources_count"])
        self.assertFalse(self.storage_dir.exists())

    def test_enabled_mode_writes_raw_source(self):
        os.environ[self.flag_env] = "1"

        result = api_server.persist_github_context_phase2_raw_sources(fake_repo_context())

        self.assertTrue(result["enabled"])
        self.assertEqual(1, result["raw_sources_count"])
        self.assertGreater(result["raw_chars"], 0)
        records = evidence_memory.read_records(evidence_memory.GITHUB_RAW_SOURCES, self.storage_dir)
        self.assertEqual(1, len(records))
        for key in ["project_id", "repo", "raw_text", "raw_hash", "source_id"]:
            self.assertIn(key, records[0])
            self.assertTrue(records[0][key])

    def test_upsert_prevents_duplicates(self):
        os.environ[self.flag_env] = "1"
        context = fake_repo_context()

        api_server.persist_github_context_phase2_raw_sources(context)
        api_server.persist_github_context_phase2_raw_sources(context)

        counts = evidence_memory.get_phase2_memory_counts(storage_dir=self.storage_dir)
        self.assertEqual(1, counts["raw_sources_count"])

    def test_multiple_projects_filtering(self):
        os.environ[self.flag_env] = "1"

        api_server.persist_github_context_phase2_raw_sources(
            fake_repo_context(project_id="WorkAgent", repo="owner/WorkAgent")
        )
        api_server.persist_github_context_phase2_raw_sources(
            fake_repo_context(project_id="Event-Lottery-System", repo="owner/Event-Lottery-System")
        )

        global_counts = evidence_memory.get_phase2_memory_counts(storage_dir=self.storage_dir)
        self.assertEqual(2, global_counts["raw_sources_count"])
        workagent_records = evidence_memory.read_records_by_project(
            evidence_memory.GITHUB_RAW_SOURCES,
            "WorkAgent",
            self.storage_dir,
        )
        lottery_records = evidence_memory.read_records_by_project(
            evidence_memory.GITHUB_RAW_SOURCES,
            "Event-Lottery-System",
            self.storage_dir,
        )
        self.assertEqual(1, len(workagent_records))
        self.assertEqual(1, len(lottery_records))
        self.assertEqual({"WorkAgent"}, {record["project_id"] for record in workagent_records})


if __name__ == "__main__":
    unittest.main()
