import os,sys,subprocess,json
from pathlib import Path
root=int(sys.argv[1]);p=Path('/proc')/str(root)
assert p.stat().st_uid==os.getuid()
cmd=(p/'cmdline').read_bytes().decode().replace('\0',' ')
assert 'g2-policy-overnight-20261124/' in cmd and any(s in cmd for s in ['run_progress_guarded.py','queue_timeonly_ablation.py'])
cpus=sorted(os.sched_getaffinity(root))[:8]
lines=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,ppid='],text=True).splitlines();pairs=[list(map(int,l.split())) for l in lines]
selected={root}
while True:
 new=selected|{pid for pid,parent in pairs if parent in selected}
 if new==selected:break
 selected=new
changed=[]
for pid in selected:
 for t in (Path('/proc')/str(pid)/'task').glob('*'):
  try:os.sched_setaffinity(int(t.name),cpus);changed.append(int(t.name))
  except ProcessLookupError:pass
report=dict(root=root,pids=sorted(selected),cpus=cpus,threads_changed=len(changed))
print(json.dumps(report));(Path(__file__).parent/f'cpu_affinity_{root}.json').write_text(json.dumps(report,indent=2))
