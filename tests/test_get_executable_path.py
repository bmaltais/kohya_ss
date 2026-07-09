"""Regression for get_executable_path venv fallback.

When the venv Python is run without activation, PATH does not include
Scripts/bin, so shutil.which alone misses console scripts (e.g. accelerate)
that are installed in the environment. get_executable_path must still find
them under sys.prefix.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from kohya_gui.common_gui import get_executable_path


class TestGetExecutablePath(unittest.TestCase):
    def test_empty_name_returns_empty(self):
        self.assertEqual(get_executable_path(None), "")
        self.assertEqual(get_executable_path(""), "")

    def test_which_hit_is_preferred(self):
        with patch("kohya_gui.common_gui.shutil.which", return_value="/usr/bin/fake"):
            self.assertEqual(get_executable_path("fake"), "/usr/bin/fake")

    def test_venv_scripts_fallback_when_not_on_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_name = "Scripts" if os.name == "nt" else "bin"
            scripts_dir = os.path.join(tmpdir, scripts_name)
            os.makedirs(scripts_dir)
            exe_name = "acceltool.exe" if os.name == "nt" else "acceltool"
            tool_path = os.path.join(scripts_dir, exe_name)
            with open(tool_path, "w", encoding="utf-8") as f:
                f.write("")

            with patch("kohya_gui.common_gui.shutil.which", return_value=None):
                with patch.object(sys, "prefix", tmpdir):
                    # Callers pass the bare name without extension.
                    found = get_executable_path("acceltool")
            self.assertEqual(found, tool_path)

    def test_missing_executable_returns_empty(self):
        with patch("kohya_gui.common_gui.shutil.which", return_value=None):
            with patch.object(sys, "prefix", tempfile.gettempdir()):
                self.assertEqual(
                    get_executable_path("definitely-not-installed-xyz"), ""
                )


if __name__ == "__main__":
    unittest.main()
