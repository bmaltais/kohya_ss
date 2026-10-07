"""Optional path fields must accept ``None`` as "nothing to check".

Gradio passes an optional widget's untouched value through as ``None``
(``Textbox.preprocess`` returns ``None`` for a ``None`` payload), not as an
empty string. The path validators only special-cased ``""``, so a ``None``
fell through to ``os.path.isfile(None)`` / ``os.path.isdir(None)`` and raised
``TypeError`` -- an unreadable traceback instead of letting the run proceed.
"""

import os
import unittest

from kohya_gui import common_gui


class TestValidateFilePath(unittest.TestCase):
    def test_none_is_nothing_to_check(self):
        self.assertTrue(common_gui.validate_file_path(None))

    def test_empty_string_is_nothing_to_check(self):
        self.assertTrue(common_gui.validate_file_path(""))

    def test_existing_file_passes(self):
        self.assertTrue(common_gui.validate_file_path(__file__))

    def test_missing_file_fails(self):
        self.assertFalse(common_gui.validate_file_path("no/such/file.toml"))

    def test_directory_is_not_a_file(self):
        self.assertFalse(common_gui.validate_file_path(os.path.dirname(__file__)))


class TestValidateFolderPath(unittest.TestCase):
    def test_none_is_nothing_to_check(self):
        self.assertTrue(common_gui.validate_folder_path(None))

    def test_empty_string_is_nothing_to_check(self):
        self.assertTrue(common_gui.validate_folder_path(""))

    def test_existing_folder_passes(self):
        self.assertTrue(common_gui.validate_folder_path(os.path.dirname(__file__)))

    def test_missing_folder_fails(self):
        self.assertFalse(common_gui.validate_folder_path("no/such/folder"))

    def test_file_is_not_a_folder(self):
        self.assertFalse(common_gui.validate_folder_path(__file__))

    def test_unwritable_folder_fails_when_writability_required(self):
        # / is readable but not writable for a normal user; skip when running
        # as root, where the check would legitimately pass.
        if os.access("/", os.W_OK):
            self.skipTest("running as a user that can write to /")
        self.assertFalse(
            common_gui.validate_folder_path("/", can_be_written_to=True)
        )


if __name__ == "__main__":
    unittest.main()
