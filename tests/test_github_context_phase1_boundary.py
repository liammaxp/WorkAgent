import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = str(ROOT / "backend")


class GitHubContextPhase1BoundaryTests(unittest.TestCase):
    def run_import_smoke(self, flag_value: str, expected_enabled: bool) -> None:
        env = os.environ.copy()
        env["USE_GITHUB_CONTEXT_STATUS_V2"] = flag_value
        script = (
            "import sys; "
            f"sys.path.insert(0, {BACKEND_PATH!r}); "
            "import api_server; "
            f"assert api_server.github_context_status_v2_enabled() is {expected_enabled!r}; "
            "status = api_server.get_github_context_status_v2(); "
            "assert isinstance(status, dict); "
            "assert status.get('enabled') is True"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_api_server_imports_with_phase1_flag_disabled(self):
        self.run_import_smoke("0", False)

    def test_api_server_imports_with_phase1_flag_enabled(self):
        self.run_import_smoke("1", True)


if __name__ == "__main__":
    unittest.main()
