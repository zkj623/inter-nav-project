"""One functional trial using frozen QV23; videos waived by the user."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,subprocess
B=Path('/data20t/embodied/zkj/inter-nav-results/go2-textured-interaction-20261103')
R=Path('/data20t/embodied/zkj/inter-nav-textured-qv23-20261104')
HEAD='514cd51dc7a4400a3e7db431b1f18a7aba7e3f50'
p=argparse.ArgumentParser();p.add_argument('task',choices=[f'T{i:02d}' for i in range(1,10)]);a=p.parse_args()
base=B/('existing-five' if int(a.task[1:])<=5 else 'interaction-five')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==HEAD
test=json.loads((B/'tests-qv23.launch.json').read_text());assert test['commit']==HEAD and test['tests_passed']==1012
spec=importlib.util.spec_from_file_location('qualified_launcher',base/'tools/launch_textured_go2.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
name='diagnostic-qv23-'+a.task.lower()
assert not (base/(name+'.launch.json')).exists() and not (base/'runs'/(name+'-try1')).exists()
module.launch(base,R,a.task,'qv23',1012,prepare_only=True,diagnostic_only=True,navigation_diagnostics=True)
supervisor=base/('supervise_'+name+'.py');s=supervisor.read_text()
assert "'--record-video'" not in s and "'--material-mode','preview'" in s
assert 'for attempt in range(1,2):' in s and 'MemoryMax=24G' in s and 'MemorySwapMax=0' in s
assert "'--global-map-hz','1','--local-map-hz','10'" in s
assert 'full_recording_required' in s
s=s.replace("'full_recording_required':True", "'full_recording_required':False,'functional_validation':True")
s=s.replace('full_recording_required=True)', 'full_recording_required=False,functional_validation=True)')
compile(s,str(supervisor),'exec');supervisor.write_text(s)
wrapper=base/('run_'+name+'.sh')
with (base/(name+'.supervisor.log')).open('w') as log:
    process=subprocess.Popen(['bash',str(wrapper)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':process.pid,'task':a.task,'run_id':'qv23','source_commit':HEAD,'diagnostic_only':True,
        'functional_validation':True,'video_required':False,'maximum_attempts':1,'material_mode':'preview',
        'output':str(base/'runs'/(name+'-try1')),'status':str(base/(name+'.status.json')),
        'supervisor_sha256':hashlib.sha256(supervisor.read_bytes()).hexdigest(),
        'policy_sha256':module.POLICY,'test_log_sha256':hashlib.sha256((base/'tests-qv23.log').read_bytes()).hexdigest()}
(base/(name+'.launch.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
