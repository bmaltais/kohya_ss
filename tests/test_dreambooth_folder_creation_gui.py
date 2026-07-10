"""Regression tests for GH issue #3436: Dataset Preparation must not destroy
source images (same-path rmtree, nested copytree, missing source).

Covers dreambooth_folder_preparation path validation and safe copy/replace.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from kohya_gui.dreambooth_folder_creation_gui import dreambooth_folder_preparation


def _write_image(folder: Path, name: str = "photo.png") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"fake-image")
    return path


class TestDreamboothFolderPreparation(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="kohya_prep_test_"))
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_happy_path_copies_without_removing_source(self):
        src = self.root / "raw_images"
        dest = self.root / "training"
        _write_image(src)
        _write_image(src, "photo.txt")  # caption sibling

        status = dreambooth_folder_preparation(
            str(src),
            40,
            "asd",
            "",
            1,
            "person",
            str(dest),
        )

        target = dest / "img" / "40_asd person"
        self.assertTrue(os.path.isdir(target), msg=status)
        self.assertTrue((target / "photo.png").is_file())
        self.assertTrue((src / "photo.png").is_file(), "source must remain")
        self.assertTrue((dest / "log").is_dir())
        self.assertTrue((dest / "model").is_dir())
        self.assertIn("Done", status)

    def test_same_path_refuses_and_deletes_nothing(self):
        dest = self.root / "training"
        # Already-prepared layout matching computed target
        src = dest / "img" / "40_asd person"
        img = _write_image(src)

        status = dreambooth_folder_preparation(
            str(src),
            40,
            "asd",
            "",
            1,
            "person",
            str(dest),
        )

        self.assertTrue(img.is_file(), "source must not be deleted")
        self.assertTrue(
            any(k in status.lower() for k in ("same", "refus", "error", "identical")),
            msg=status,
        )

    def test_nested_dest_under_source_refuses(self):
        # #1761 layout: source=.../img, dest=parent → target under source
        dest = self.root / "kohya_input"
        src = dest / "img"
        img = _write_image(src)

        status = dreambooth_folder_preparation(
            str(src),
            40,
            "asd",
            "",
            1,
            "person",
            str(dest),
        )

        self.assertTrue(img.is_file())
        # Must not create nested bomb under src
        nested = list(src.rglob("40_asd person"))
        self.assertEqual(nested, [], msg=f"unexpected nested dirs: {nested}; {status}")
        self.assertTrue(
            any(
                k in status.lower()
                for k in ("nested", "inside", "under", "refus", "error")
            ),
            msg=status,
        )

    def test_source_under_dest_refuses(self):
        # Source lives inside the computed target folder
        dest = self.root / "training"
        target = dest / "img" / "40_asd person"
        src = target / "nested_source"
        img = _write_image(src)

        status = dreambooth_folder_preparation(
            str(src),
            40,
            "asd",
            "",
            1,
            "person",
            str(dest),
        )

        self.assertTrue(img.is_file(), "source under dest must not be deleted")
        self.assertTrue(
            any(k in status.lower() for k in ("inside", "under", "refus", "error")),
            msg=status,
        )

    def test_regularization_happy_path_copies(self):
        src = self.root / "raw_images"
        reg_src = self.root / "raw_reg"
        dest = self.root / "training"
        _write_image(src)
        _write_image(reg_src, "reg.png")

        status = dreambooth_folder_preparation(
            str(src),
            40,
            "asd",
            str(reg_src),
            1,
            "person",
            str(dest),
        )

        self.assertTrue(
            (dest / "img" / "40_asd person" / "photo.png").is_file(), msg=status
        )
        self.assertTrue((dest / "reg" / "1_person" / "reg.png").is_file(), msg=status)
        self.assertTrue((reg_src / "reg.png").is_file(), "reg source must remain")
        self.assertIn("Done", status)

    def test_missing_source_refuses_before_mutation(self):
        dest = self.root / "training"
        dest.mkdir()
        missing = self.root / "does_not_exist"

        status = dreambooth_folder_preparation(
            str(missing),
            40,
            "asd",
            "",
            1,
            "person",
            str(dest),
        )

        self.assertFalse(
            (dest / "img").exists()
            or any((dest / "img").glob("*") if (dest / "img").exists() else [])
        )
        # log/model may still be created on partial runs — only require no img copy
        target = dest / "img" / "40_asd person"
        self.assertFalse(target.exists())
        self.assertTrue(
            any(
                k in status.lower()
                for k in ("missing", "not found", "does not exist", "error")
            ),
            msg=status,
        )

    def test_destination_already_exists_safe_replace(self):
        src = self.root / "raw_images"
        dest = self.root / "training"
        _write_image(src, "new.png")
        old_target = dest / "img" / "40_asd person"
        _write_image(old_target, "old.png")

        status = dreambooth_folder_preparation(
            str(src),
            40,
            "asd",
            "",
            1,
            "person",
            str(dest),
        )

        self.assertTrue((old_target / "new.png").is_file(), msg=status)
        self.assertFalse((old_target / "old.png").exists())
        self.assertTrue((src / "new.png").is_file())
        self.assertIn("Done", status)

    def test_regularization_same_path_refuses(self):
        src = self.root / "raw_images"
        dest = self.root / "training"
        _write_image(src)
        reg_src = dest / "reg" / "1_person"
        reg_img = _write_image(reg_src)

        status = dreambooth_folder_preparation(
            str(src),
            40,
            "asd",
            str(reg_src),
            1,
            "person",
            str(dest),
        )

        self.assertTrue(reg_img.is_file(), "reg source must not be deleted")
        self.assertTrue(
            any(
                k in status.lower()
                for k in ("same", "refus", "error", "identical", "regular")
            ),
            msg=status,
        )

    def test_missing_destination_returns_status(self):
        src = self.root / "raw_images"
        _write_image(src)

        status = dreambooth_folder_preparation(
            str(src),
            40,
            "asd",
            "",
            1,
            "person",
            "",
        )

        self.assertTrue(status)
        self.assertTrue(
            any(k in status.lower() for k in ("destination", "missing", "error")),
            msg=status,
        )

    def test_strips_prompt_whitespace(self):
        src = self.root / "raw_images"
        dest = self.root / "training"
        _write_image(src)

        status = dreambooth_folder_preparation(
            str(src),
            40,
            "asd\n",
            "",
            1,
            " person ",
            str(dest),
        )

        target = dest / "img" / "40_asd person"
        self.assertTrue(target.is_dir(), msg=status)
        self.assertTrue((target / "photo.png").is_file())


if __name__ == "__main__":
    unittest.main()
