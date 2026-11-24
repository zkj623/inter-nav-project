from pathlib import Path
import os,subprocess,json,datetime
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';changes=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  if p.stat().st_uid!=os.getuid():continue
  args=[x.decode() for x in p.read_bytes().split(b'\0') if x]
  if len(args)<2:continue
  script=Path(args[1]);name=script.name
  if not script.is_absolute():script=(p.parent/'cwd').resolve()/script
  if script.parent not in [W,R]:
   if name=='train_g2_pi05.py' and args[1]=='train_g2_pi05.py':pass
   else:continue
  priority=None
  if name.startswith(('train_g2_','act_policy_server','pi05_policy_server')):priority=0
  elif name in ['prepare_overnight_data.py','prepare_expanded_snapshot.py']:priority=7
  elif name.startswith('policy_') and name.endswith('_trial.py'):priority=2
  elif name.startswith(('realpose_','run_threecam_collection','run_wide_collection','verify_')):priority=5
  if priority is None:continue
  result=subprocess.run(['ionice','-c','2','-n',str(priority),'-p',p.parent.name],capture_output=True,text=True);changes.append(dict(pid=int(p.parent.name),script=name,priority=priority,ok=result.returncode==0,error=result.stderr.strip()))
 except (OSError,UnicodeError):pass
(W/'io_priority_revision.json').write_text(json.dumps(dict(time=datetime.datetime.now().isoformat(),changes=changes),indent=2));print(json.dumps(changes))

