import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402


class GitHubContextPhase2FlagTests(unittest.TestCase):
    def setUp(self):
        self.env_name = api_server.GITHUB_CONTEXT_PHASE2_ENV
        self.original_present = self.env_name in os.environ
        self.original_value = os.environ.get(self.env_name)

    def tearDown(self):
        if self.original_present:
            os.environ[self.env_name] = self.original_value or ""
        else:
            os.environ.pop(self.env_name, None)

    def test_default_disabled(self):
        os.environ.pop(self.env_name, None)
        self.assertFalse(api_server.is_github_context_phase2_enabled())

    def test_disabled_values(self):
        for value in ["0", "false", "False", "FALSE", "", "no"]:
            with self.subTest(value=value):
                os.environ[self.env_name] = value
                self.assertFalse(api_server.is_github_context_phase2_enabled())

    def test_enabled_values(self):
        for value in ["1", "true", "True", "TRUE"]:
            with self.subTest(value=value):
                os.environ[self.env_name] = value
                self.assertTrue(api_server.is_github_context_phase2_enabled())


if __name__ == "__main__":
    unittest.main()
