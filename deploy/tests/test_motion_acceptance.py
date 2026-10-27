"""Offline stop-path checks; all motion output goes to a fake SDK."""
import multiprocessing as mp
import math
import os
import queue
import shutil
import time
from types import SimpleNamespace

import pytest
import yaml

from deploy.app.config import load, ROOT, check_motion_configuration
from deploy.app.keyboard import Keyboard
from deploy.robot.motion import UnitreeSport, _worker
from deploy.tests.test_deploy import FakeSport


def telemetry_only_sport(code=100):
    """No constructor: never creates a real SDK client or subscriber."""
    config = load()
    sdk = UnitreeSport.__new__(UnitreeSport)
    sdk.robot, sdk.control = config['robot'], config['runtime']['control']
    sdk.state_at = sdk.remote_at = 10.
    sdk.state = SimpleNamespace(error_code=code, imu_state=SimpleNamespace(quaternion=[1., 0., 0., 0.]),
                                velocity=[0., 0., 0.], yaw_speed=0.)
    return sdk


@pytest.mark.parametrize('code,allowed', [(100, True), (0, False), (1001, False), (2012, False), (0xffffffff, False)])
def test_only_confirmed_sport_state_passes_telemetry_check(code, allowed):
    sdk = telemetry_only_sport(code)
    assert sdk.healthy(10.1) is allowed
    assert sdk.stopped(10.1) is allowed


@pytest.mark.parametrize('case', ['missing_codes', 'stale_state', 'stale_remote', 'invalid_quaternion',
                                 'tilted', 'invalid_velocity', 'invalid_yaw_speed'])
def test_allowed_sport_state_keeps_other_health_checks(case):
    sdk = telemetry_only_sport()
    if case == 'missing_codes':
        sdk.robot.pop('sport_state_motion_codes')
    elif case == 'stale_state':
        sdk.state_at = 9.
    elif case == 'stale_remote':
        sdk.remote_at = 9.
    elif case == 'invalid_quaternion':
        sdk.state.imu_state.quaternion = [0., 0., 0., 0.]
    elif case == 'tilted':
        angle = math.radians(30)/2
        sdk.state.imu_state.quaternion = [math.cos(angle), math.sin(angle), 0., 0.]
    elif case == 'invalid_velocity':
        sdk.state.velocity[0] = float('nan')
    else:
        sdk.state.yaw_speed = float('inf')
    assert not sdk.healthy(10.1)
    assert not sdk.stopped(10.1)


def test_allowed_state_is_not_itself_stationary_feedback():
    sdk = telemetry_only_sport()
    sdk.state.velocity[0] = .1
    assert sdk.healthy(10.1)
    assert not sdk.stopped(10.1)
    assert sdk.stationary_feedback_time(10.1) is None


@pytest.mark.parametrize('codes', [None, [], '100', [True], [100.0], [-1], [0x100000000]])
def test_invalid_sport_state_configuration_is_rejected(tmp_path, codes):
    for name in ('robot', 'navigation', 'runtime'):
        shutil.copyfile(ROOT/'configs'/f'{name}.yaml', tmp_path/f'{name}.yaml')
    path = tmp_path/'robot.yaml'
    robot = yaml.safe_load(path.read_text())
    robot['sport_state_motion_codes'] = codes
    path.write_text(yaml.safe_dump(robot))
    with pytest.raises(ValueError, match='sport_state_motion_codes'):
        load(tmp_path)


def test_old_config_can_observe_but_missing_state_codes_prevent_motion(tmp_path):
    for name in ('robot', 'navigation', 'runtime'):
        shutil.copyfile(ROOT/'configs'/f'{name}.yaml', tmp_path/f'{name}.yaml')
    path = tmp_path/'robot.yaml'
    robot = yaml.safe_load(path.read_text())
    robot.pop('sport_state_motion_codes')
    for key in ('calibration_verified', 'deskew_and_self_filter_verified', 'communication_verified', 'remote_arbitration_verified'):
        robot[key] = True
    robot['takeover_button_mask'] = 2
    path.write_text(yaml.safe_dump(robot))
    config = load(tmp_path)
    with pytest.raises(ValueError, match='sport_state_motion_codes'):
        check_motion_configuration(config)


class TakeoverSport(FakeSport):
    def __init__(self, robot, control):
        super().__init__(robot, control)
        self.signal = robot['signal']

    @property
    def takeover(self):
        return self.signal.is_set() if hasattr(self, 'signal') else False

    @takeover.setter
    def takeover(self, value):
        if hasattr(self, 'signal') and not value:
            self.signal.clear()


def test_remote_bit_latches_after_release_without_initializing_sdk():
    sdk = UnitreeSport.__new__(UnitreeSport)
    sdk.robot = {'takeover_button_mask': 2}
    sdk.takeover = False
    sdk._remote(SimpleNamespace(wireless_remote=b'\0\0\x01\0'))
    assert not sdk.takeover
    sdk._remote(SimpleNamespace(wireless_remote=b'\0\0\x02\0'))
    assert sdk.takeover
    sdk._remote(SimpleNamespace(wireless_remote=b'\0\0\0\0'))
    assert sdk.takeover


@pytest.mark.parametrize('reason', ['remote_takeover', 'parent_disconnected'])
def test_writer_stops_and_does_not_resume_after_takeover_or_disconnect(reason):
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    log, signal = context.Queue(), context.Event()
    process = context.Process(target=_worker, args=(child, {'log':log,'signal':signal},
                              load()['runtime']['control'], TakeoverSport))
    process.start()
    child.close()
    try:
        assert parent.poll(5) and parent.recv()['state'] == 'manual'
        parent.send({'op':'arm'})
        assert parent.poll(1) and parent.recv()['state'] == 'active'
        now = time.monotonic()
        parent.send({'op':'velocity','velocity':[.1,0,0],'generated_at':now,'expires_at':now+.3})
        assert log.get(timeout=1) == 'move'
        if reason == 'parent_disconnected':
            parent.close()
            process.join(3)
            assert process.exitcode == 0
        else:
            signal.set()
            events = []
            deadline = time.monotonic()+2
            while time.monotonic() < deadline:
                if parent.poll(.05):
                    events.append(parent.recv())
                    if events[-1]['state'] == 'stopped':
                        break
            assert any(e['state']=='stopping' and e['detail']==reason for e in events)
            assert events[-1]['state'] == 'stopped'
            signal.clear()
            now = time.monotonic()
            parent.send({'op':'velocity','velocity':[.1,0,0],'generated_at':now,'expires_at':now+.2})
            time.sleep(.25)
            parent.send({'op':'close'})
            process.join(2)
            assert process.exitcode == 0
        output = []
        while True:
            try:
                output.append(log.get(timeout=.1))
            except queue.Empty:
                break
        assert 'stop' in output
        assert 'move' not in output[output.index('stop')+1:]
    finally:
        if process.is_alive():
            process.terminate()
            process.join(1)
        parent.close()
        log.close()


@pytest.mark.parametrize('key,operation', [(' ', 'stop'),('p','p'),('m','m'),('o','o'),('q','q'),('\x03','q')])
def test_keyboard_stop_operations(key, operation):
    read, write = os.pipe()
    keyboard = Keyboard.__new__(Keyboard)
    keyboard.fd, keyboard.editing, keyboard.text = read, None, ''
    try:
        os.write(write, key.encode())
        assert keyboard.poll() == (operation, None)
    finally:
        os.close(read)
        os.close(write)


def test_keyboard_space_cancels_while_entering_goal():
    read, write = os.pipe()
    keyboard = Keyboard.__new__(Keyboard)
    keyboard.fd, keyboard.editing, keyboard.text = read, 'navigate', '2,'
    try:
        os.write(write, b' ')
        assert keyboard.poll() == ('stop', None)
        assert keyboard.editing is None and keyboard.text == ''
    finally:
        os.close(read)
        os.close(write)
