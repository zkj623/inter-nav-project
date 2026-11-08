import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from research.rsi.evaluation import sdk_launch_guard


class SDKLaunchTests(unittest.TestCase):
    def test_existing_ipc_name_skips_sdk_without_removing_foreign_files(self):
        fake_os = SimpleNamespace(name='posix', getpid=lambda: 12345, execv=Mock())
        with patch.object(sys, 'argv', ['guard', 'simulation.py']), patch.object(sdk_launch_guard, 'os', fake_os), \
                patch.object(sdk_launch_guard.Path, 'exists', return_value=True):
            self.assertEqual(sdk_launch_guard.main(), 75)
        fake_os.execv.assert_not_called()

    def test_clean_pid_executes_original_script_with_identical_arguments(self):
        with patch.object(sys, 'argv', ['guard', 'simulation.py', '--gpu', '7']), \
                patch.object(sdk_launch_guard.Path, 'exists', return_value=False), \
                patch.object(sdk_launch_guard.os, 'execv') as execute:
            sdk_launch_guard.main()
        execute.assert_called_once_with(sys.executable, [sys.executable, 'simulation.py', '--gpu', '7'])


if __name__ == '__main__':
    unittest.main()
