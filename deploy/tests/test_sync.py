"""Exercise the real rsync file selection against a disposable local device."""
import json
import os
import shutil
import subprocess

import pytest

from deploy.app.config import REPO


def test_code_sync_preserves_device_configuration_and_seeds_missing_files(tmp_path):
    rsync = shutil.which('rsync')
    if rsync is None:
        pytest.skip('rsync is required to exercise deployment sync')
    source, target, executables = (tmp_path/name for name in ('source', 'robot', 'bin'))
    for root in (source, target):
        (root/'deploy/configs').mkdir(parents=True)
    (source/'deploy/scripts').mkdir()
    shutil.copyfile(REPO/'deploy/scripts/sync.sh', source/'deploy/scripts/sync.sh')
    (source/'deploy/run.py').write_text('new code\n')
    (target/'deploy/run.py').write_text('old code\n')
    os.utime(target/'deploy/run.py', (1000, 1000))
    (source/'deploy/configs/robot.yaml').write_text('ground_z: 0\n')
    (source/'deploy/configs/interaction.yaml').write_text('tracking_enabled: false\n')
    measured = 'ground_z: -0.012\ntakeover_button_mask: 4096\n'
    (target/'deploy/configs/robot.yaml').write_text(measured)
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Sync Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'fixture'], check=True)
    executables.mkdir()
    wrapper = executables/'rsync'
    wrapper.write_text('''#!/usr/bin/env python3
import os, subprocess, sys
args = sys.argv[1:]
assert args[-1].startswith('fake:')
args[-1] = args[-1].split(':', 1)[1]
raise SystemExit(subprocess.call([os.environ['TEST_RSYNC'], *args]))
''')
    wrapper.chmod(0o755)
    env = {**os.environ, 'PATH': str(executables)+os.pathsep+os.environ['PATH'], 'TEST_RSYNC': rsync}
    command = ['bash', str(source/'deploy/scripts/sync.sh'), 'fake', str(target)]
    subprocess.run(command, env=env, check=True, capture_output=True)
    assert (target/'deploy/run.py').read_text() == 'old code\n'
    assert not (target/'deploy/configs/interaction.yaml').exists()
    subprocess.run([*command, '--apply'], env=env, check=True, capture_output=True)
    assert (target/'deploy/run.py').read_text() == 'new code\n'
    assert (target/'deploy/configs/robot.yaml').read_text() == measured
    assert (target/'deploy/configs/interaction.yaml').read_text() == 'tracking_enabled: false\n'
    assert json.loads((target/'.deploy-source.json').read_text())['dirty_worktree'] is False
