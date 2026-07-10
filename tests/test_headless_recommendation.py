"""Regression for remote/SSH headless guidance (#3570).

--headless already skips easygui overwrite confirms in check_if_model_exist.
These tests lock that contract and the startup heuristic that warns when a
native display is unlikely to be available.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from kohya_gui.common_gui import check_if_model_exist, should_recommend_headless


class TestShouldRecommendHeadless(unittest.TestCase):
    def test_headless_true_never_recommends(self):
        self.assertFalse(
            should_recommend_headless(
                True,
                environ={"SSH_CONNECTION": "1.2.3.4 5 6.7.8.9 10"},
                platform="linux",
            )
        )
        self.assertFalse(should_recommend_headless(True, environ={}, platform="linux"))

    def test_ssh_session_recommends_even_with_display(self):
        env = {
            "SSH_CONNECTION": "10.0.0.1 12345 10.0.0.2 22",
            "DISPLAY": ":0",
        }
        self.assertTrue(should_recommend_headless(False, environ=env, platform="linux"))
        self.assertTrue(should_recommend_headless(False, environ=env, platform="win32"))

    def test_ssh_client_marker_recommends(self):
        self.assertTrue(
            should_recommend_headless(
                False,
                environ={"SSH_CLIENT": "10.0.0.1 12345 22"},
                platform="win32",
            )
        )

    def test_unix_without_display_recommends(self):
        self.assertTrue(should_recommend_headless(False, environ={}, platform="linux"))
        self.assertTrue(should_recommend_headless(False, environ={}, platform="darwin"))

    def test_unix_with_display_local_does_not_recommend(self):
        self.assertFalse(
            should_recommend_headless(
                False, environ={"DISPLAY": ":1"}, platform="linux"
            )
        )
        self.assertFalse(
            should_recommend_headless(
                False,
                environ={"WAYLAND_DISPLAY": "wayland-0"},
                platform="linux",
            )
        )

    def test_windows_local_without_ssh_does_not_recommend(self):
        self.assertFalse(should_recommend_headless(False, environ={}, platform="win32"))


class TestCheckIfModelExistHeadless(unittest.TestCase):
    def test_headless_skips_ynbox_and_allows_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "existing.safetensors")
            with open(model_path, "w", encoding="utf-8") as f:
                f.write("")

            with patch("kohya_gui.common_gui.ynbox") as mock_ynbox:
                result = check_if_model_exist(
                    "existing", tmp, "safetensors", headless=True
                )

            self.assertFalse(result)
            mock_ynbox.assert_not_called()

    def test_non_headless_calls_ynbox_when_model_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "existing.safetensors")
            with open(model_path, "w", encoding="utf-8") as f:
                f.write("")

            with patch("kohya_gui.common_gui.ynbox", return_value=False) as mock_ynbox:
                result = check_if_model_exist(
                    "existing", tmp, "safetensors", headless=False
                )

            self.assertTrue(result)
            mock_ynbox.assert_called_once()


if __name__ == "__main__":
    unittest.main()
