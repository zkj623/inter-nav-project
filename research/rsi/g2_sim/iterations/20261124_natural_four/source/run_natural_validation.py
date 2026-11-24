"""Require both physical and sampled-posture verification before accepting a trial."""
from pathlib import Path
import argparse,subprocess,sys,json
R=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--asset',required=True);p.add_argument('--plan-dir',default='natural_final_plans');p.add_argument('--repeat',type=int,default=1);p.add_argument('--gpu',type=int,default=2);a=p.parse_args();T=R/'realpose_trials'/a.plan_dir/a.asset/f'candidate0_repeat{a.repeat}'
subprocess.run([sys.executable,str(R/f'realpose_{a.asset}_trial.py'),'--asset',a.asset,'--plan-dir',a.plan_dir,'--repeat',str(a.repeat),'--gpu',str(a.gpu)],check=True)
subprocess.run([sys.executable,str(R/'verify_realpose.py'),'--trial-dir',str(T)],check=True)
subprocess.run([sys.executable,str(R/'verify_natural_posture.py'),str(T)],check=True)
physical=json.loads((T/'verification.json').read_text());posture=json.loads((T/'posture_verification.json').read_text());accepted=bool(physical['verified_success'] and posture['natural_verified']);(T/'acceptance.json').write_text(json.dumps({'accepted_for_natural_posture_baseline':accepted,'physical_report':'verification.json','posture_report':'posture_verification.json','note':'Acceptance is for this fixed placement; visual review and perturbation validation remain separate.'},indent=2));print('ACCEPTED',accepted)
sys.exit(0 if accepted else 2)
