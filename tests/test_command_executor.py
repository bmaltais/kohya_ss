"""GH #3435: surface exit code + short log tail when training subprocess ends.

Covers pure helpers and CommandExecutor.wait_for_training_to_end behavior
with mocked process handles (no real training).
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import gradio as gr

from kohya_gui import class_command_executor as ce


class TestTailLines(unittest.TestCase):
    def test_returns_last_n_lines(self):
        text = "\n".join(f"line{i}" for i in range(10))
        self.assertEqual(ce.tail_lines(text, max_lines=3), "line7\nline8\nline9")

    def test_short_text_unchanged(self):
        self.assertEqual(ce.tail_lines("a\nb", max_lines=10), "a\nb")

    def test_empty(self):
        self.assertEqual(ce.tail_lines("", max_lines=5), "")

    def test_none_safe(self):
        self.assertEqual(ce.tail_lines(None, max_lines=5), "")


class TestDrainStreamToBuffer(unittest.TestCase):
    def test_drains_and_bounds_buffer(self):
        import collections
        import io

        stream = io.StringIO("a\nb\nc\nd\n")
        buf = collections.deque(maxlen=2)
        dest = io.StringIO()
        ce.drain_stream_to_buffer(stream, buf, dest=dest)
        self.assertEqual(list(buf), ["c", "d"])
        self.assertEqual(dest.getvalue(), "a\nb\nc\nd\n")


class TestReadLogTail(unittest.TestCase):
    def test_reads_last_lines_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "setup.log")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(f"L{i}" for i in range(20)) + "\n")
            tail = ce.read_log_tail(path, max_lines=3)
            self.assertEqual(tail, "L17\nL18\nL19")

    def test_missing_file_returns_empty(self):
        self.assertEqual(ce.read_log_tail("/no/such/setup.log"), "")


class TestSummarizeTrainingEnd(unittest.TestCase):
    def test_success_exit_zero(self):
        level, msg = ce.summarize_training_end(0, user_stopped=False, log_tail="")
        self.assertEqual(level, "info")
        self.assertIn("success", msg.lower())
        self.assertNotIn("failed", msg.lower())

    def test_failure_includes_exit_code(self):
        level, msg = ce.summarize_training_end(1, user_stopped=False, log_tail="")
        self.assertEqual(level, "error")
        self.assertIn("1", msg)
        self.assertIn("fail", msg.lower())

    def test_failure_includes_log_tail(self):
        level, msg = ce.summarize_training_end(
            2, user_stopped=False, log_tail="RuntimeError: boom"
        )
        self.assertEqual(level, "error")
        self.assertIn("2", msg)
        self.assertIn("RuntimeError: boom", msg)

    def test_user_stop_is_cancelled_not_failed(self):
        # Kill often yields non-zero; must not look like a training failure.
        level, msg = ce.summarize_training_end(
            1, user_stopped=True, log_tail="whatever"
        )
        self.assertEqual(level, "info")
        self.assertRegex(msg.lower(), r"stop|cancel")
        self.assertNotIn("failed", msg.lower())

    def test_none_returncode_generic_end(self):
        level, msg = ce.summarize_training_end(None, user_stopped=False, log_tail="")
        self.assertEqual(level, "info")
        self.assertIn("ended", msg.lower())


class TestWaitForTrainingToEnd(unittest.TestCase):
    def _make_executor(self, headless=True):
        with gr.Blocks():
            return ce.CommandExecutor(headless=headless)

    def test_nonzero_exit_logs_failure_with_code_and_idle_buttons(self):
        executor = self._make_executor(headless=True)
        proc = MagicMock()
        proc.poll.return_value = 7  # already finished
        proc.returncode = 7
        executor.process = proc

        with (
            patch.object(ce, "log") as mock_log,
            patch.object(ce, "output_message") as mock_msg,
            patch.object(ce, "read_log_tail", return_value="Traceback: OOM"),
            patch.object(ce.time, "sleep"),
        ):
            start_btn, stop_btn = executor.wait_for_training_to_end()

        error_msgs = [c.args[0] for c in mock_log.error.call_args_list if c.args]
        self.assertTrue(error_msgs, msg="expected log.error for failed training")
        self.assertTrue(
            any("7" in m and "fail" in m.lower() for m in error_msgs),
            msg=error_msgs,
        )
        self.assertTrue(
            any("OOM" in m or "Traceback" in m for m in error_msgs),
            msg=error_msgs,
        )
        mock_msg.assert_called()
        msg_arg = mock_msg.call_args.kwargs.get("msg") or mock_msg.call_args[0][0]
        self.assertIn("7", msg_arg)

        # Idle: Start visible, Stop hidden (headless keeps Stop visible per existing)
        self.assertTrue(
            start_btn["visible"] if isinstance(start_btn, dict) else start_btn.visible
        )
        # headless=True → stop remains visible (False or True → True)
        stop_visible = (
            stop_btn["visible"] if isinstance(stop_btn, dict) else stop_btn.visible
        )
        self.assertTrue(stop_visible)

    def test_zero_exit_does_not_report_failure(self):
        executor = self._make_executor(headless=True)
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.returncode = 0
        executor.process = proc

        with (
            patch.object(ce, "log") as mock_log,
            patch.object(ce, "output_message") as mock_msg,
            patch.object(ce.time, "sleep"),
        ):
            executor.wait_for_training_to_end()

        mock_log.error.assert_not_called()
        mock_msg.assert_not_called()
        info_msgs = [c.args[0] for c in mock_log.info.call_args_list if c.args]
        self.assertTrue(
            any("success" in m.lower() or "ended" in m.lower() for m in info_msgs)
        )

    def test_user_stop_not_reported_as_failure(self):
        executor = self._make_executor(headless=True)
        proc = MagicMock()
        proc.poll.return_value = 1
        proc.returncode = 1
        executor.process = proc
        executor._stopped_by_user = True

        with (
            patch.object(ce, "log") as mock_log,
            patch.object(ce, "output_message") as mock_msg,
            patch.object(ce.time, "sleep"),
        ):
            executor.wait_for_training_to_end()

        mock_log.error.assert_not_called()
        info_msgs = [c.args[0] for c in mock_log.info.call_args_list if c.args]
        self.assertTrue(
            any("stop" in m.lower() or "cancel" in m.lower() for m in info_msgs),
            msg=info_msgs,
        )

    def test_kill_command_sets_user_stopped_flag(self):
        executor = self._make_executor(headless=True)
        proc = MagicMock()
        proc.poll.return_value = None  # still running
        proc.pid = 12345
        executor.process = proc

        parent = MagicMock()
        parent.children.return_value = []
        with patch.object(ce.psutil, "Process", return_value=parent):
            executor.kill_command()

        self.assertTrue(executor._stopped_by_user)

    def test_headed_idle_buttons_after_end(self):
        executor = self._make_executor(headless=False)
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.returncode = 0
        executor.process = proc

        with patch.object(ce, "log"), patch.object(ce.time, "sleep"):
            start_btn, stop_btn = executor.wait_for_training_to_end()

        self.assertTrue(start_btn.visible)
        self.assertFalse(stop_btn.visible)

    def test_collect_log_tail_prefers_process_buffer(self):
        executor = self._make_executor(headless=True)
        executor._output_lines.append("CUDA out of memory")
        with patch.object(ce, "read_log_tail") as mock_read:
            tail = executor._collect_log_tail()
        self.assertEqual(tail, "CUDA out of memory")
        mock_read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
