"""Run the remaining nine once, sequentially, preserving every result."""
from pathlib import Path
import fcntl,json,os,subprocess,time
B=Path('/data20t/embodied/zkj/inter-nav-results/go2-textured-interaction-20261103')
PY='/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python'
STATUS=B/'uniform-qv23.status.json';completed=[]
def emit(state,**details):
    report={'state':state,'time_unix':time.time(),'pid':os.getpid(),'version':'qv23','source_commit':'514cd51dc7a4400a3e7db431b1f18a7aba7e3f50','completed':completed,**details}
    tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(STATUS)
    print(json.dumps(report),flush=True)
def alive(pid):
    p=Path('/proc')/str(pid)/'stat'
    return p.exists() and p.read_text().split()[2]!='Z'
def main():
    with (B/'uniform-qv23.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for i in range(1,10):
            task=f'T{i:02d}';cohort='existing-five' if i<=5 else 'interaction-five';base=B/cohort
            name='diagnostic-qv23-'+task.lower();root=base/'runs'/(name+'-try1');folder=root/task
            assert not (base/(name+'.launch.json')).exists() and not root.exists(), 'Existing run: inspect, never restart automatically'
            emit('launching',task=task)
            subprocess.run([PY,str(B/'uniform-tools/launch_uniform_qv23_diagnostic.py'),task],check=True)
            launch=json.loads((base/(name+'.launch.json')).read_text());pid=launch['pid']
            emit('running_task',task=task,supervisor_pid=pid)
            while True:
                sp=base/(name+'.status.json');s=json.loads(sp.read_text()) if sp.exists() else {}
                if s.get('state')=='finished':break
                if s.get('state') in ['resource_retries_exhausted','supervisor_error']:
                    emit('stopped_for_environment_review',task=task,task_status=s);return
                if not alive(pid):
                    emit('stopped_for_environment_review',task=task,reason='supervisor_exited_without_finished_status',task_status=s);return
                time.sleep(10)
            while alive(pid):time.sleep(1)
            emit('auditing_task',task=task)
            if not (folder/'result.json').exists():
                emit('stopped_for_environment_review',task=task,reason='missing_result');return
            with (folder/'fluency_review.log').open('w') as log:
                flow_code=subprocess.run([PY,str(base/'tools/audit_navigation_fluency.py'),str(folder),task],stdout=log,stderr=subprocess.STDOUT).returncode
            with (folder/'functional_validation.log').open('w') as log:
                audit_code=subprocess.run([PY,str(B/'uniform-tools/audit_uniform_qv23_functional.py'),str(root),task],stdout=log,stderr=subprocess.STDOUT).returncode
            result=json.loads((folder/'result.json').read_text())
            audit_path=folder/'functional_validation.json';audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
            row={'task':task,'cohort':cohort,'run':root.name,'native_success':result.get('success'),
                 'reason':result.get('reason'),'goal_error_m':result.get('goal_error_m'),
                 'functional_checks_passed':audit.get('functional_checks_passed',False),'fluency_checks_passed':audit.get('fluency_checks_passed',False),
                 'audit_exit_code':audit_code,'flow_exit_code':flow_code,'failed_checks':[k for k,v in audit.get('checks',{}).items() if not v],
                 'failed_fluency_checks':audit.get('failed_fluency_checks',[])}
            completed.append(row);emit('task_completed',task=task)
        emit('all_nine_completed',functional_passed=sum(r['functional_checks_passed'] for r in completed),fluency_passed=sum(r['fluency_checks_passed'] for r in completed),prior_qv23_T10_accepted=True)
if __name__=='__main__':
    try:main()
    except Exception as e:emit('driver_error',error=repr(e));raise
