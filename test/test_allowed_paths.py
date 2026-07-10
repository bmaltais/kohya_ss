import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# The launcher script "kohya_gui.py" shares its name with the "kohya_gui/"
# package, so it must be loaded explicitly by file path rather than via
# `import kohya_gui` (which would resolve to the package).
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_spec = importlib.util.spec_from_file_location(
    "kohya_gui_launcher", os.path.join(_repo_root, "kohya_gui.py")
)
kohya_gui_launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kohya_gui_launcher)


class TestAllowedPaths(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)

        self.allowed_path = self.temp_dir.name.replace("\\", "/")
        self.config_file = self.temp_dir_path / "config.toml"
        with open(self.config_file, "w") as f:
            f.write(f"""
[server]
allowed_paths = ["{self.allowed_path}"]
""")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_allowed_paths_forwarded_to_launch(self):
        mock_ui_interface = MagicMock()
        # Gradio 6: initialize_ui_interface returns (Blocks, shell_params).
        shell_params = {"css": "", "head": "", "theme": MagicMock()}
        with (
            patch.object(
                kohya_gui_launcher,
                "initialize_ui_interface",
                return_value=(mock_ui_interface, shell_params),
            ),
            patch.object(kohya_gui_launcher, "log", MagicMock(), create=True),
        ):
            kohya_gui_launcher.UI(config=str(self.config_file), headless=True)

        mock_ui_interface.launch.assert_called_once()
        launch_kwargs = mock_ui_interface.launch.call_args.kwargs
        self.assertIn("allowed_paths", launch_kwargs)
        self.assertEqual(launch_kwargs["allowed_paths"], [self.allowed_path])
        # Shell params (css/head/theme) must also reach launch() on Gradio 6.
        self.assertIn("css", launch_kwargs)
        self.assertIn("head", launch_kwargs)
        self.assertIn("theme", launch_kwargs)


if __name__ == "__main__":
    unittest.main()
