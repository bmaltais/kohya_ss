import unittest
from unittest.mock import patch
import importlib

# Since we are modifying an existing file, we need to reload it
import kohya_gui.class_tensorboard
importlib.reload(kohya_gui.class_tensorboard)

class TestTensorboardVisibility(unittest.TestCase):

    @patch('shutil.which', return_value='/usr/bin/tensorboard')
    @patch('cpuinfo.get_cpu_info', return_value={'flags': ['avx']})
    def test_tensorboard_visibility_when_tensorboard_and_avx_are_present(self, mock_cpuinfo, mock_which):
        importlib.reload(kohya_gui.class_tensorboard)
        self.assertTrue(kohya_gui.class_tensorboard.visibility)

    @patch('shutil.which', return_value=None)
    @patch('cpuinfo.get_cpu_info', return_value={'flags': ['avx']})
    def test_tensorboard_visibility_when_tensorboard_is_absent(self, mock_cpuinfo, mock_which):
        importlib.reload(kohya_gui.class_tensorboard)
        self.assertFalse(kohya_gui.class_tensorboard.visibility)

    @patch('shutil.which', return_value='/usr/bin/tensorboard')
    @patch('cpuinfo.get_cpu_info', return_value={'flags': ['sse']})
    def test_tensorboard_visibility_when_avx_is_absent(self, mock_cpuinfo, mock_which):
        importlib.reload(kohya_gui.class_tensorboard)
        self.assertFalse(kohya_gui.class_tensorboard.visibility)

    @patch('cpuinfo.get_cpu_info', side_effect=Exception)
    def test_check_avx_support_exception(self, mock_cpuinfo):
        self.assertFalse(kohya_gui.class_tensorboard.check_avx_support())

if __name__ == '__main__':
    unittest.main()
