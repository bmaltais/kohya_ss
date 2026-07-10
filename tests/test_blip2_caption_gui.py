"""BLIP2 caption GUI (#3367, #2992, #3037): device selection and HF load kwargs.

Exercises shipped helpers in kohya_gui.blip2_caption_gui — no live HF download,
no real XPU/CUDA hardware.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kohya_gui import blip2_caption_gui


def _mock_torch(*, cuda=False, xpu=False, mps=False):
    """Minimal torch stand-in with controllable backend availability."""
    t = SimpleNamespace()
    t.cuda = SimpleNamespace(is_available=lambda: bool(cuda))
    t.xpu = SimpleNamespace(is_available=lambda: bool(xpu))
    t.mps = SimpleNamespace(is_available=lambda: bool(mps))
    t.float16 = "float16-sentinel"
    return t


class TestGetCaptionDevice(unittest.TestCase):
    def test_prefers_cuda_over_xpu_mps_cpu(self):
        t = _mock_torch(cuda=True, xpu=True, mps=True)
        self.assertEqual(blip2_caption_gui.get_caption_device(t), "cuda")

    def test_prefers_xpu_when_no_cuda(self):
        t = _mock_torch(cuda=False, xpu=True, mps=True)
        self.assertEqual(blip2_caption_gui.get_caption_device(t), "xpu")

    def test_prefers_mps_when_no_cuda_or_xpu(self):
        t = _mock_torch(cuda=False, xpu=False, mps=True)
        self.assertEqual(blip2_caption_gui.get_caption_device(t), "mps")

    def test_falls_back_to_cpu(self):
        t = _mock_torch(cuda=False, xpu=False, mps=False)
        self.assertEqual(blip2_caption_gui.get_caption_device(t), "cpu")

    def test_missing_xpu_attr_does_not_raise(self):
        t = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            mps=SimpleNamespace(is_available=lambda: False),
        )
        self.assertEqual(blip2_caption_gui.get_caption_device(t), "cpu")

    def test_xpu_is_available_raising_falls_through(self):
        def _boom():
            raise RuntimeError("xpu driver broken")

        t = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            xpu=SimpleNamespace(is_available=_boom),
            mps=SimpleNamespace(is_available=lambda: True),
        )
        self.assertEqual(blip2_caption_gui.get_caption_device(t), "mps")


class TestBlip2LoadKwargs(unittest.TestCase):
    def test_processor_kwargs_include_revision_and_use_fast_false(self):
        kwargs = blip2_caption_gui.get_blip2_processor_load_kwargs()
        self.assertEqual(kwargs["revision"], blip2_caption_gui.BLIP2_HF_REVISION)
        self.assertIs(kwargs["use_fast"], False)
        self.assertEqual(
            blip2_caption_gui.BLIP2_HF_REVISION,
            "51572668da0eb669e01a189dc22abe6088589a24",
        )

    def test_model_kwargs_include_revision_and_optional_dtype(self):
        bare = blip2_caption_gui.get_blip2_model_load_kwargs()
        self.assertEqual(bare["revision"], blip2_caption_gui.BLIP2_HF_REVISION)
        self.assertNotIn("torch_dtype", bare)
        self.assertNotIn("use_fast", bare)

        with_dtype = blip2_caption_gui.get_blip2_model_load_kwargs(
            torch_dtype="float16-sentinel"
        )
        self.assertEqual(with_dtype["revision"], blip2_caption_gui.BLIP2_HF_REVISION)
        self.assertEqual(with_dtype["torch_dtype"], "float16-sentinel")

    def test_model_id_constant(self):
        self.assertEqual(blip2_caption_gui.BLIP2_MODEL_ID, "Salesforce/blip2-opt-2.7b")


class TestLoadModelUsesShippedHelpers(unittest.TestCase):
    """Drive load_model() with mocked transformers — asserts real call kwargs."""

    def test_load_model_passes_revision_use_fast_and_moves_to_device(self):
        mock_processor_cls = MagicMock()
        mock_model_cls = MagicMock()
        mock_processor = MagicMock(name="processor")
        mock_model = MagicMock(name="model")
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = mock_model

        fake_transformers = SimpleNamespace(
            Blip2Processor=mock_processor_cls,
            Blip2ForConditionalGeneration=mock_model_cls,
        )

        with (
            patch.dict("sys.modules", {"transformers": fake_transformers}),
            patch.object(
                blip2_caption_gui, "get_caption_device", return_value="xpu"
            ) as mock_dev,
            patch.object(blip2_caption_gui.torch, "float16", "float16-sentinel"),
        ):
            processor, model, device = blip2_caption_gui.load_model()

        mock_dev.assert_called_once_with()
        self.assertIs(processor, mock_processor)
        self.assertIs(model, mock_model)
        self.assertEqual(device, "xpu")

        proc_args, proc_kwargs = mock_processor_cls.from_pretrained.call_args
        self.assertEqual(proc_args[0], blip2_caption_gui.BLIP2_MODEL_ID)
        self.assertEqual(proc_kwargs["revision"], blip2_caption_gui.BLIP2_HF_REVISION)
        self.assertIs(proc_kwargs["use_fast"], False)

        model_args, model_kwargs = mock_model_cls.from_pretrained.call_args
        self.assertEqual(model_args[0], blip2_caption_gui.BLIP2_MODEL_ID)
        self.assertEqual(model_kwargs["revision"], blip2_caption_gui.BLIP2_HF_REVISION)
        self.assertEqual(model_kwargs["torch_dtype"], "float16-sentinel")
        self.assertNotIn("use_fast", model_kwargs)

        mock_model.to.assert_called_once_with("xpu")


if __name__ == "__main__":
    unittest.main()
