from pathlib import Path
import json,os,signal,subprocess
R=Path(__file__).resolve().parent
# Stop only this experiment's driver and active children; preserve all outputs.
for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines():
 parts=line.strip().split(None,1)
 if len(parts)<2:continue
 pid=int(parts[0]);cmd=parts[1]
 if pid==os.getpid():continue
 if '/isaaclab/bin/python run_commonstart.py' in cmd or '/isaaclab/bin/python run_commonstart_extra.py' in cmd or (str(R)+'/commonstart_' in cmd and '_trial.py' in cmd) or '/isaaclab/bin/python commonstart_stool_rectangular_trial.py' in cmd:
  os.kill(pid,signal.SIGTERM);print('STOPPED_OWN_EXPERIMENT',pid,flush=True)
for p in R.glob('commonstart_*_trial.py'):
 s=p.read_text().replace("ROOT/'commonstart_trials'","ROOT/'commonstart_slow_trials'").replace("segment(last,waypoint,1.,'common_approach')","segment(last,waypoint,6.,'common_approach')")
 (R/p.name.replace('commonstart_','commonstartslow_',1)).write_text(s)
s=(R/'run_commonstart.py').read_text();s=s.replace("doc['accepted']","doc['accepted']+doc['failed']").replace("str(R/f'commonstart_{a}_trial.py')","str(R/f'commonstartslow_{a}_trial.py')")
s=s.replace("'--plan-dir','commonstart_plans'","'--plan-dir',('commonstart_extra_plans' if a in doc['failed'] else 'commonstart_plans')")
s=s.replace("f'commonstart_trials/commonstart_plans/{a}/candidate0_repeat1/result.json'","f\"commonstart_slow_trials/{'commonstart_extra_plans' if a in doc['failed'] else 'commonstart_plans'}/{a}/candidate0_repeat1/result.json\"")
s=s.replace('commonstart_events.json','commonstart_slow_events.json').replace("f'commonstart_trial_{a}.log'","f'commonstart_slow_trial_{a}.log'")
(R/'run_commonstart_slow.py').write_text(s)
