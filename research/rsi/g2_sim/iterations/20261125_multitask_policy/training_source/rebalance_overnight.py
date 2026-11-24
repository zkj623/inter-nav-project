from pathlib import Path
import os, signal, subprocess, sys, time, json
R=Path('/data20t/embodied/zkj/inter-nav-results/g2-act-threecam-20261124')
W=R.parent/'g2-policy-overnight-20261124'
pid=577973
cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert str(R/'run_threecam_collection.py') in cmd
children=[int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
# Worker threads own subprocesses, so inspect every thread's children too.
for t in Path(f'/proc/{pid}/task').iterdir():
 children += [int(x) for x in (t/'children').read_text().split()]
children=sorted(set(children))
(W/'resource_transition.json').write_text(json.dumps(dict(previous_runner=pid,draining_children=children,new_collection_gpus=[5,6,6]),indent=2))
os.kill(pid,signal.SIGTERM)
print('DRAINING_EXISTING_TRIALS',children,flush=True)
def alive(p):
 try:return Path(f'/proc/{p}/stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False
while any(alive(p) for p in children):time.sleep(15)
print('TRIALS_DRAINED_RESTARTING_COLLECTION_ON5_6',flush=True)
with (R/'collection_supervisor_rebalanced.log').open('w') as log:
 subprocess.run([sys.executable,str(R/'run_threecam_collection.py'),'--gpus','5','6','6'],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
