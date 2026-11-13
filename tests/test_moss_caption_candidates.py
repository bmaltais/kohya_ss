"""Tests for MOSS-Image multi-caption candidates (offline tag dropout).

`--caption_tag_dropout_rate` is rejected by sd-scripts when
`--cache_text_encoder_outputs` is enabled, so the GUI pre-generates tag-dropout
variants as caption candidates (`enable_multiline_caption`), which the trainer
picks from at random every step. These tests cover the generator, the derived
dataset TOML and the train_model wiring. Originals must never be modified.
"""

import os
import random
import tempfile
import unittest

import toml

from kohya_gui import lora_gui
from kohya_gui.moss_caption_candidates import (
    CANDIDATE_EXTENSION,
    DEFAULT_SEPARATOR,
    build_candidates,
    drop_tags,
    prepare_dataset_config,
    render_candidate_file,
)
from conftest import build_train_model_kwargs, run_train_model_and_load_toml
from test_moss_image_lora_gui import (
    MOSS_OVERRIDES,
    NUMERIC_FIXUPS,
    STRING_OVERRIDES,
)

FIXTURE = "test/config/locon-AdamW8bit-toml.json"
CAPTION = "1girl, trigger word, red hair, long hair, smile, outdoors, sky"


def split_candidates(text, separator=DEFAULT_SEPARATOR):
    """Mirror sd-scripts' library.dataset.split_caption_candidates."""
    candidates, current = [], []
    for line in text.split("\n"):
        if line.strip().startswith(separator):
            candidates.append("\n".join(current))
            current = []
        else:
            current.append(line)
    candidates.append("\n".join(current))
    return [c.strip() for c in candidates if c.strip()]


class TestDropTags(unittest.TestCase):
    def test_leading_tags_are_protected(self):
        rng = random.Random(0)
        for _ in range(20):
            result = drop_tags(CAPTION, 0.9, 2, rng)
            self.assertTrue(result.startswith("1girl, trigger word"))

    def test_rate_one_keeps_at_least_one_flexible_tag(self):
        result = drop_tags(CAPTION, 1.0, 2, random.Random(1))
        self.assertEqual(result, "1girl, trigger word, red hair")

    def test_rate_zero_keeps_everything(self):
        self.assertEqual(drop_tags(CAPTION, 0.0, 0, random.Random(1)), CAPTION)

    def test_empty_caption(self):
        self.assertEqual(drop_tags("   ", 0.5, 2, random.Random(1)), "")

    def test_keep_tokens_larger_than_tag_count(self):
        self.assertEqual(drop_tags("a, b", 1.0, 10, random.Random(1)), "a, b")


class TestCandidateFile(unittest.TestCase):
    def test_first_candidate_is_the_original(self):
        candidates = build_candidates(CAPTION, 3, 0.15, 2, random.Random(7))
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0], CAPTION)

    def test_rendered_file_round_trips(self):
        candidates = build_candidates(CAPTION, 4, 0.3, 2, random.Random(7))
        text = render_candidate_file(candidates, 0.3)
        parsed = split_candidates(text)
        self.assertEqual(parsed, candidates)
        self.assertEqual(text.count(DEFAULT_SEPARATOR), 3)

    def test_variants_differ_from_the_original(self):
        candidates = build_candidates(CAPTION, 3, 0.5, 1, random.Random(3))
        self.assertTrue(any(c != CAPTION for c in candidates[1:]))


class TestPrepareDatasetConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.image_dir = os.path.join(self.root, "img", "5_concept")
        os.makedirs(self.image_dir)
        self.captions = {}
        for index in range(3):
            stem = f"image{index}"
            with open(os.path.join(self.image_dir, stem + ".png"), "wb") as handle:
                handle.write(b"not-a-real-png")
            caption = CAPTION if index else "1girl, trigger word, solo"
            self.captions[stem] = caption
            with open(
                os.path.join(self.image_dir, stem + ".txt"), "w", encoding="utf-8"
            ) as handle:
                handle.write(caption + "\n")
        self.dataset_config = os.path.join(self.root, "dataset.toml")
        self.original_config = {
            "general": {"caption_extension": ".txt", "enable_bucket": True},
            "datasets": [
                {
                    "resolution": 1024,
                    "subsets": [
                        {
                            "image_dir": self.image_dir,
                            "num_repeats": 2,
                            "class_tokens": "trigger word",
                        }
                    ],
                }
            ],
        }
        with open(self.dataset_config, "w", encoding="utf-8") as handle:
            toml.dump(self.original_config, handle)
        self.original_bytes = open(self.dataset_config, "rb").read()

    def _prepare(
        self, candidate_count=3, dropout_rate=0.15, keep_tokens=2, regenerate=False
    ):
        return prepare_dataset_config(
            self.dataset_config,
            candidate_count=candidate_count,
            dropout_rate=dropout_rate,
            keep_tokens=keep_tokens,
            regenerate=regenerate,
        )

    def test_originals_are_untouched(self):
        derived = self._prepare()
        self.assertEqual(open(self.dataset_config, "rb").read(), self.original_bytes)
        for stem, caption in self.captions.items():
            path = os.path.join(self.image_dir, stem + ".txt")
            self.assertEqual(open(path, encoding="utf-8").read(), caption + "\n")
        self.assertTrue(os.path.isfile(derived))
        self.assertNotEqual(
            os.path.abspath(derived), os.path.abspath(self.dataset_config)
        )

    def test_candidate_files_are_written(self):
        self._prepare()
        for stem in self.captions:
            path = os.path.join(self.image_dir, stem + CANDIDATE_EXTENSION)
            self.assertTrue(os.path.isfile(path), path)
            candidates = split_candidates(open(path, encoding="utf-8").read())
            self.assertEqual(len(candidates), 3)
            self.assertEqual(candidates[0], self.captions[stem])

    def test_derived_config_enables_multiline_candidates(self):
        derived = self._prepare()
        config = toml.load(derived)
        subset = config["datasets"][0]["subsets"][0]
        self.assertEqual(subset["caption_extension"], CANDIDATE_EXTENSION)
        self.assertTrue(subset["enable_multiline_caption"])
        self.assertEqual(subset["multiline_caption_separator"], DEFAULT_SEPARATOR)
        # the source extension in [general] must not shadow the subset override
        self.assertEqual(config["general"]["caption_extension"], ".txt")

    def test_existing_candidates_are_reused(self):
        self._prepare()
        path = os.path.join(self.image_dir, "image0" + CANDIDATE_EXTENSION)
        before = os.path.getmtime(path)
        os.utime(path, (before - 100, before - 100))
        self._prepare()
        self.assertEqual(os.path.getmtime(path), before - 100)

    def test_regenerate_rewrites_candidates(self):
        self._prepare()
        path = os.path.join(self.image_dir, "image0" + CANDIDATE_EXTENSION)
        before = os.path.getmtime(path)
        os.utime(path, (before - 100, before - 100))
        self._prepare(regenerate=True)
        self.assertGreater(os.path.getmtime(path), before - 100)

    def test_missing_caption_file_is_skipped(self):
        os.remove(os.path.join(self.image_dir, "image1.txt"))
        self._prepare()
        self.assertFalse(
            os.path.exists(os.path.join(self.image_dir, "image1" + CANDIDATE_EXTENSION))
        )

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            self._prepare(candidate_count=1)
        with self.assertRaises(ValueError):
            self._prepare(dropout_rate=0)
        with self.assertRaises(ValueError):
            prepare_dataset_config(os.path.join(self.root, "nope.toml"))

    def test_finetune_only_config_raises(self):
        config = {"datasets": [{"subsets": [{"caption_metadata": "meta.json"}]}]}
        path = os.path.join(self.root, "finetune.toml")
        with open(path, "w", encoding="utf-8") as handle:
            toml.dump(config, handle)
        with self.assertRaises(ValueError):
            prepare_dataset_config(path)


class TestTrainModelWiring(unittest.TestCase):
    """train_model must hand the derived TOML to the trainer, and only when the
    checkbox is on."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.image_dir = os.path.join(self.tmp.name, "img", "5_concept")
        os.makedirs(self.image_dir)
        with open(os.path.join(self.image_dir, "a.png"), "wb") as handle:
            handle.write(b"not-a-real-png")
        with open(
            os.path.join(self.image_dir, "a.txt"), "w", encoding="utf-8"
        ) as handle:
            handle.write(CAPTION + "\n")
        self.dataset_config = os.path.join(self.tmp.name, "dataset.toml")
        with open(self.dataset_config, "w", encoding="utf-8") as handle:
            toml.dump(
                {
                    "general": {"caption_extension": ".txt"},
                    "datasets": [
                        {"resolution": 1024, "subsets": [{"image_dir": self.image_dir}]}
                    ],
                },
                handle,
            )

    def _kwargs(self, **overrides):
        base = {
            **MOSS_OVERRIDES,
            "dataset_config": self.dataset_config,
            "keep_tokens": 2,
        }
        base.update(overrides)
        return build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides=base,
        )

    def test_enabled_uses_derived_dataset_config(self):
        config = run_train_model_and_load_toml(
            lora_gui,
            self._kwargs(
                moss_image_caption_candidates=True,
                moss_image_caption_candidate_count=3,
                moss_image_caption_tag_dropout_rate=0.15,
            ),
        )
        self.assertTrue(config["dataset_config"].endswith(".moss-multi.toml"))
        self.assertTrue(
            os.path.isfile(os.path.join(self.image_dir, "a" + CANDIDATE_EXTENSION))
        )

    def test_disabled_keeps_original_dataset_config(self):
        config = run_train_model_and_load_toml(
            lora_gui, self._kwargs(moss_image_caption_candidates=False)
        )
        self.assertEqual(
            os.path.abspath(config["dataset_config"]),
            os.path.abspath(self.dataset_config),
        )
        self.assertFalse(
            os.path.exists(os.path.join(self.image_dir, "a" + CANDIDATE_EXTENSION))
        )

    def test_invalid_rate_aborts_without_candidate_files(self):
        result = lora_gui.train_model(
            **self._kwargs(
                headless=True,
                print_only=True,
                moss_image_caption_candidates=True,
                moss_image_caption_tag_dropout_rate=0,
            )
        )
        self.assertIsNotNone(result)
        self.assertFalse(
            os.path.exists(os.path.join(self.image_dir, "a" + CANDIDATE_EXTENSION))
        )
        self.assertFalse(
            os.path.exists(os.path.join(self.tmp.name, "dataset.moss-multi.toml"))
        )


if __name__ == "__main__":
    unittest.main()
