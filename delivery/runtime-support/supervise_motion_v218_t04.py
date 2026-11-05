"""Run the original T04 passage regression with resource telemetry and bounded retries."""
import csv
import shutil
import hashlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE=Path('/data20t/embodied/zkj/inter-nav-results/go2-online-smooth-20261027')
ROOT=Path('/data20t/embodied/zkj/inter-nav-motion-v218-20261102')
STATUS=BASE/'recognition-skill-v218-t04-ground-truth.status.json'
MIN_RAM_GIB=56.
EXCLUDED_GPU_INDICES=set()
ALLOWED_GPU_INDICES=set(range(8))  # Any same-model GPU with the unchanged free-memory threshold
MIN_GPU_MIB=14336
STABLE_SECONDS=60.


def emit(state,**details):
    row={'state':state,'time_unix':time.time(),**details}
    temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(row,indent=2));temp.replace(STATUS)
    print(json.dumps(row),flush=True)


def memory():
    data={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    return data['MemAvailable']/1024**2


def oom_count():
    p=Path('/sys/fs/cgroup/user.slice')/f'user-{os.getuid()}.slice/memory.events'
    try:return int(dict(line.split() for line in p.read_text().splitlines())['oom_kill'])
    except (OSError,KeyError,ValueError):return None


def process_memory(pid):
    try:
        lines=Path(f'/proc/{pid}/status').read_text().splitlines()
        return {line.split(':')[0]:int(line.split()[1])/1024**2
                for line in lines if line.startswith(('VmRSS:','VmHWM:','VmSwap:'))}
    except OSError:return {}


def process_state(pid):
    try:return next(line.split()[1] for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('State:'))
    except OSError:return None


def ready_resources():
    stable_since=None
    while True:
        available=memory()
        command=['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits']
        response=subprocess.run(command,capture_output=True,text=True)
        choices=[]
        if response.returncode==0:
            choices=[(int(free),int(index),int(util)) for index,free,util in csv.reader(response.stdout.splitlines())
                     if int(free)>=MIN_GPU_MIB and int(index) not in EXCLUDED_GPU_INDICES and int(index) in ALLOWED_GPU_INDICES]
        good=available>=MIN_RAM_GIB and bool(choices)
        if not good:stable_since=None
        elif stable_since is None:stable_since=time.monotonic()
        stable=0. if stable_since is None else time.monotonic()-stable_since
        emit('waiting_resources',available_ram_gib=round(available,2),required_ram_gib=MIN_RAM_GIB,
             eligible_gpus=choices,stable_seconds=round(stable,1),required_stable_seconds=STABLE_SECONDS)
        if good and stable>=STABLE_SECONDS:
            free,gpu,util=max(choices)
            return {'gpu':gpu,'free_mib':free,'available_ram_gib':available,'stable_seconds':stable,
                    'minimum_ram_gib':MIN_RAM_GIB,'time_unix':time.time(),
                    'serial_task_runs':True,'admission_policy':'v218-capped-admission-policy.json','scope_memory_max_gib':24,'scope_swap_max_gib':0}
        time.sleep(15)


def scope_memory(out):
    identity_path=out/'scope_identity.json'
    if not identity_path.exists():return {}
    identity=json.loads(identity_path.read_text());group=Path(identity['cgroup'])
    result={'pid':identity['pid'],'unit':identity['unit'],'verified_max_bytes':identity['memory_max_bytes'],'verified_swap_bytes':identity['swap_max_bytes']}
    try:
        for name in ['memory.current','memory.peak','memory.max']:
            result[name+'_gib']=int((group/name).read_text())/1024**3
        result['events']={k:int(v) for k,v in (line.split() for line in (group/'memory.events').read_text().splitlines())}
    except OSError:pass  # The scope may disappear as the verified main process exits.
    return result


def run_attempt(attempt,launch,head,admission):
    out=BASE/f'recognition-skill-v218-t04-ground-truth-try{attempt}'
    if out.exists():raise RuntimeError(f'Refusing to overwrite {out}')
    out.mkdir()
    policy_dir=out/'policy';policy_dir.mkdir()
    for filename in ['actor_checkpoint.pt','deployment_contract.json']:
        shutil.copy2(BASE/'candidate-policy-v218'/filename,policy_dir/filename)
    policy_hashes={filename:hashlib.sha256((policy_dir/filename).read_bytes()).hexdigest()
                   for filename in ['actor_checkpoint.pt','deployment_contract.json']}
    (out/'policy_source_snapshot.json').write_text(json.dumps({'files':policy_hashes,
        'training_source_commit':'8ab8daaaf925d4180a455343b46cd402b4b0e861',
        'training_run':'official-go2-v155-train1','official_pretrained_policy':False},indent=2))
    guard_source=out/'resource_pressure_guard.py';shutil.copy2(ROOT/'tools/go2_resource_pressure.py',guard_source)
    guard_spec=importlib.util.spec_from_file_location('_go2_resource_pressure',guard_source)
    guard_module=importlib.util.module_from_spec(guard_spec);guard_spec.loader.exec_module(guard_module)
    guard=guard_module.MemoryPressureGuard()
    guard_digest=hashlib.sha256(guard_source.read_bytes()).hexdigest()
    (out/'resource_launch.json').write_text(json.dumps({**launch,'source_commit':head,'resource_guard_sha256':guard_digest},indent=2))
    args=[sys.executable,'-u',str(BASE/'capture_go2_policy_state_v193.py'),str(BASE/'contact_report_launch.py'),str(BASE/'launch_online_checked.py'),
          'grutopia/demo/go2_online_interactive_navigation.py',
          '--benchmark-dir','/data20t/embodied/zkj/inter-nav-results/go2-voronoi-heading-up-20261025/collision-corrected-inputs',
          '--output',str(out),'--task-ids','T04','--gpu',str(launch['gpu']),'--max-steps','40000',
          '--global-map-hz','1','--local-map-hz','10','--floor-top','.01','--map-source','online','--interaction-source','ground_truth','--actuator-mode','explicit_dc','--path-controller','continuous',
          '--record-video','--record-every','1','--material-mode','simple','--navigation-attitude-recovery',
          '--third-person-height','1.35','--third-person-forward','.25','--third-person-fov','100',
          '--route-planner','voronoi','--navigation-heading','path','--roadmap-shape','circle',
          '--route-clearance','.30','--navigation-half-width','.22','--voronoi-clearance-weight','.30',
          '--push-start-speed','.60','--push-max-speed','.95','--turn-speed','1.0','--contact-centering-speed','.25',
          '--contact-lateral-speed','.35','--contact-yaw-tolerance','.10','--contact-front-gap','.35']
    reference=BASE/'official-policy-review-20261030/nvidia_isaaclab_4_5'
    args.extend(['--policy-path',str(out/'policy/actor_checkpoint.pt'),
        '--robot-usd',str(reference/'go2_training_flattened.usdc'),'--robot-physics-profile','isaaclab_training','--physics-backend','gpu_dynamics'])
    baseline_oom=oom_count();resource_stop=False;peak=0.
    with (out/'run.log').open('w') as log, (out/'resource_telemetry.jsonl').open('w',buffering=1) as trace:
        scope_helper=out/'scoped_exec.py';shutil.copy2(BASE/'scoped_exec_v94.py',scope_helper)
        scope_env=os.environ.copy();runtime=Path('/run/user')/str(os.getuid())
        scope_env['XDG_RUNTIME_DIR']=str(runtime);scope_env['DBUS_SESSION_BUS_ADDRESS']='unix:path='+str(runtime/'bus')
        scoped=['systemd-run','--user','--scope','--quiet','--collect','--unit=codex-go2-'+out.name,
            '--property=MemoryMax=24G','--property=MemorySwapMax=0','--',sys.executable,str(scope_helper),str(out),*args]
        child=subprocess.Popen(scoped,env=scope_env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        peak_scope=0.;scope_verified=False
        emit('running',attempt=attempt,pid=child.pid,output=str(out),launch=launch)
        try:
            while child.poll() is None:
                available=memory();stats=process_memory(child.pid);peak=max(peak,stats.get('VmRSS',0.))
                scoped_stats=scope_memory(out);scope_verified=scope_verified or bool(scoped_stats)
                peak_scope=max(peak_scope,scoped_stats.get('memory.peak_gib',0.))
                decision=guard.update(time.monotonic(),available)
                trace.write(json.dumps({'time_unix':time.time(),'available_ram_gib':available,
                    'process_memory_gib':stats,'cgroup_memory':scoped_stats,'user_oom_kill_count':oom_count(),
                    'process_state':process_state(child.pid),'pressure_guard':guard.report(),'decision':decision})+'\n')
                if decision is not None:
                    try:
                        if child.poll() is not None:break
                        assert os.getpgid(child.pid)==child.pid
                        action=decision['action']
                        if action=='pause':
                            os.killpg(child.pid,signal.SIGSTOP)
                            emit('resource_paused',attempt=attempt,pid=child.pid,output=str(out),launch=launch,
                                available_ram_gib=available,pressure_guard=guard.report(),process_state=process_state(child.pid))
                        elif action=='resume':
                            os.killpg(child.pid,signal.SIGCONT)
                            emit('running',attempt=attempt,pid=child.pid,output=str(out),launch=launch,
                                available_ram_gib=available,pressure_guard=guard.report(),resource_action='resumed')
                        elif action=='abort':
                            resource_stop=True
                            (out/'STOP').write_text('Resource guard: bounded pause expired; stop and preserve capture.\n')
                            os.killpg(child.pid,signal.SIGCONT)
                            emit('resource_stop_requested',attempt=attempt,pid=child.pid,output=str(out),
                                launch=launch,available_ram_gib=available,pressure_guard=guard.report())
                        elif action=='kill':
                            resource_stop=True;os.killpg(child.pid,signal.SIGKILL)
                            emit('resource_killed',attempt=attempt,pid=child.pid,output=str(out),
                                available_ram_gib=available,pressure_guard=guard.report(),reason=decision['reason'])
                            break
                    except ProcessLookupError:break
                time.sleep(1.)
        except BaseException:
            if child.poll() is None:
                try:
                    assert os.getpgid(child.pid)==child.pid
                    os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait()
            raise
        code=child.wait()
    after_oom=oom_count()
    outcome={'attempt':attempt,'output':str(out),'exit_code':code,'resource_stop':resource_stop,
             'peak_sampled_process_rss_gib':peak,'verified_scope_limit':scope_verified,'scope_memory_max_gib':24,'peak_observed_cgroup_gib':peak_scope,'oom_kill_before':baseline_oom,'oom_kill_after':after_oom,
             'user_oom_counter_increased':baseline_oom is not None and after_oom is not None and after_oom>baseline_oom,'resource_pressure_guard':guard.report(),'resource_guard_sha256':guard_digest}
    (out/'supervisor_outcome.json').write_text(json.dumps(outcome,indent=2))
    if code in (-9,137) or resource_stop:
        emit('resource_attempt_ended',**outcome)
        return False
    manifest_path=out/'source_manifest.json'
    source_checks={}
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        for name,digest in manifest.items():
            saved=out/'source'/name
            original=subprocess.run(['git','show',head+':'+name],cwd=ROOT,capture_output=True)
            source_checks[name]=bool(saved.exists() and original.returncode==0
                and hashlib.sha256(saved.read_bytes()).hexdigest()==digest
                and hashlib.sha256(original.stdout).hexdigest()==digest)
    (out/'source_version_audit.json').write_text(json.dumps({'commit':head,
        'valid':bool(source_checks) and all(source_checks.values()),'files':source_checks},indent=2))
    qa=[]
    for task in ['T04']:
        folder=out/task
        if not (folder/'result.json').exists() or not (folder/'video/combined.mp4').exists():
            qa.append({'task':task,'valid':False,'reason':'incomplete_task_outputs'});continue
        for script,name in [(BASE/'verify_recorded_task_dc.py','video_qa'),(ROOT/'tools/audit_visual_skill.py','visual_skill_qa'),(ROOT/'tools/audit_actual_passage.py','actual_passage_qa')]:
            with (folder/(name+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(script),str(folder)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        report=folder/'visual_skill_qa.json'
        visual=json.loads(report.read_text()) if report.exists() else {'valid':False}
        actual_report=folder/'actual_passage_qa.json'
        actual=json.loads(actual_report.read_text()) if actual_report.exists() else {'valid':False}
        qa.append({'task':task,'valid':bool(visual['valid'] and actual['valid']),
            'visual_skill':visual,'actual_passage':actual})
    emit('finished',**outcome,acceptance=qa,candidate_task_checks_passed=bool(qa) and all(row.get('valid') for row in qa),all_tasks_accepted=False)
    return True  # Functional failures require analysis, not blind reruns.


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='0d9744a8f37e8a5811f016a6aa2a6d8d75a4af02'
    assert '921 passed, 1 deselected' in (BASE/'tests-v218.log').read_text()
    assert hashlib.sha256((BASE/'candidate-policy-v218/actor_checkpoint.pt').read_bytes()).hexdigest()=='8862682da34ebb090a2e56f39254af430d5645f20cb5e3e104c72a4720671db8'
    with (BASE/'motion-v218-t04-sim.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        for attempt in range(1,2):
            admission=(BASE/'v89-initialization-admission.lock').open('w')
            emit('waiting_serial_admission',maximum_concurrent_task_runs=1)
            fcntl.flock(admission,fcntl.LOCK_EX)
            launch=ready_resources()
            subprocess.run(['git','diff','--quiet'],cwd=ROOT,check=True)
            subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT,check=True)
            current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            if current!=head:raise RuntimeError('Source revision changed while waiting; relaunch explicitly')
            try:finished=run_attempt(attempt,launch,head,admission)
            finally:admission.close()
            if finished:return
        emit('resource_retries_exhausted',attempts=1,reason='bounded_attempt_requires_resource_review')


if __name__=='__main__':
    try:main()
    except Exception as error:
        emit('supervisor_error',error=type(error).__name__,message=str(error));raise
