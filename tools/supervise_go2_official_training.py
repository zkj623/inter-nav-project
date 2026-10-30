"""Run one isolated training stage under the existing serial resource contract."""
from pathlib import Path
import argparse
import csv
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/data20t/embodied/zkj/inter-nav-results/go2-online-smooth-20261027')
PY = '/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python'
sys.path.insert(0, str(ROOT / 'tools'))
from go2_resource_pressure import MemoryPressureGuard

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-id', required=True)
p.add_argument('--num-envs', type=int, required=True)
p.add_argument('--iterations', type=int, required=True)
p.add_argument('--cap-gib', type=int, choices=[8, 24], default=8)
p.add_argument('--resume', type=Path)
a = p.parse_args()
assert re.fullmatch(r'[a-z0-9-]+', a.run_id)
OUT = BASE / a.run_id
OUT.mkdir(exist_ok=False)
STATUS = BASE / (a.run_id + '.status.json')

def emit(state, **values):
    record = {'state': state, 'supervisor_pid': os.getpid(), 'time_unix': time.time(),
              'training_only': True, 'original_task_accepted': False, **values}
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(STATUS)
    print(json.dumps(record), flush=True)

def memory():
    return next(int(s.split()[1]) / 1024**2 for s in Path('/proc/meminfo').read_text().splitlines()
                if s.startswith('MemAvailable:'))

required_ram = 40 if a.cap_gib == 8 else 56
with (BASE / 'v89-initialization-admission.lock').open('a') as lock:
    emit('waiting_serial_admission')
    fcntl.flock(lock, fcntl.LOCK_EX)
    stable = None
    while True:
        ram = memory()
        q = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                           text=True, capture_output=True)
        choices = [] if q.returncode else [(int(free), int(index)) for index, free in csv.reader(q.stdout.splitlines())
                                           if int(free) >= 14336]
        if ram < required_ram or not choices:
            stable = None
        elif stable is None:
            stable = time.monotonic()
        elapsed = 0 if stable is None else time.monotonic() - stable
        emit('waiting_resources', available_ram_gib=ram, stable_seconds=elapsed, required_ram_gib=required_ram)
        if elapsed >= 60:
            gpu = max(choices)[1]
            break
        time.sleep(15)
    helper = OUT / 'scope_exec.py'
    helper.write_text((BASE / 'scoped_exec_v94.py').read_text().replace('24GiB', f'{a.cap_gib}GiB')
                      .replace('maximum==24*1024**3', f'maximum=={a.cap_gib}*1024**3'))
    env = os.environ.copy()
    env['XDG_RUNTIME_DIR'] = f'/run/user/{os.getuid()}'
    env['DBUS_SESSION_BUS_ADDRESS'] = 'unix:path=' + env['XDG_RUNTIME_DIR'] + '/bus'
    env['MALLOC_ARENA_MAX'] = '2'
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    command = ['systemd-run', '--user', '--scope', '--quiet', '--collect', '--unit=codex-go2-' + a.run_id,
               '--property=MemoryMax=' + str(a.cap_gib) + 'G', '--property=MemorySwapMax=0', '--',
               PY, str(helper), str(OUT), PY, '-u', str(ROOT / 'tools/train_go2_official_baseline.py'),
               '--output', str(OUT / 'data'), '--gpu', str(gpu), '--num-envs', str(a.num_envs),
               '--iterations', str(a.iterations), '--native-lab', str(BASE / 'native-reference/IsaacLab-v1.4.0'),
               '--robot-asset', str(BASE / 'official-policy-review-20261030/nvidia_isaaclab_4_5/go2_training_flattened.usdc')]
    if a.resume:
        command.extend(['--resume', str(a.resume)])
    guard = MemoryPressureGuard(resume_at_gib=float(required_ram), abort_grace_seconds=30.)
    peak = 0.
    verified = False
    resource_stop = False
    with (OUT / 'run.log').open('w') as log, (OUT / 'resources.jsonl').open('w', buffering=1) as telemetry:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        emit('running', pid=child.pid, gpu=gpu, output=str(OUT / 'data'), cap_gib=a.cap_gib)
        try:
            while child.poll() is None:
                identity = OUT / 'scope_identity.json'
                if identity.exists():
                    info = json.loads(identity.read_text())
                    assert info['memory_max_bytes'] == a.cap_gib * 1024**3 and info['swap_max_bytes'] == 0
                    verified = True
                    group = Path(info['cgroup'])
                    if (group / 'memory.peak').exists():
                        peak = max(peak, int((group / 'memory.peak').read_text()) / 1024**3)
                ram = memory()
                action = guard.update(time.monotonic(), ram)
                telemetry.write(json.dumps({'time_unix': time.time(), 'available_ram_gib': ram,
                    'decision': action, 'peak_scope_gib': peak, 'guard': guard.report()}) + '\n')
                if action is not None and child.poll() is None:
                    assert os.getpgid(child.pid) == child.pid
                    if action['action'] == 'pause':
                        os.killpg(child.pid, signal.SIGSTOP)
                        emit('resource_paused', pid=child.pid)
                    elif action['action'] == 'resume':
                        os.killpg(child.pid, signal.SIGCONT)
                        emit('running', pid=child.pid, resource_action='resumed')
                    elif action['action'] == 'abort':
                        resource_stop = True
                        (OUT / 'STOP').write_text('Resource pressure stop\n')
                        os.killpg(child.pid, signal.SIGCONT)
                    elif action['action'] == 'kill':
                        resource_stop = True
                        os.killpg(child.pid, signal.SIGKILL)
                time.sleep(1)
        except BaseException:
            if child.poll() is None:
                assert os.getpgid(child.pid) == child.pid
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        code = child.wait()
    def optional_json(path):
        return json.loads(path.read_text()) if path.exists() else None
    summary = optional_json(OUT / 'data/summary.json')
    emit('finished' if code == 0 and summary and not resource_stop else 'failed', exit_code=code,
         resource_stop=resource_stop, scope_verified=verified, peak_scope_gib=peak, summary=summary,
         failure=optional_json(OUT / 'data/failure.json'))
