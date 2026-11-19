from pathlib import Path
import ast
R=Path(__file__).resolve().parent
for mode in ['roll','soft']:
 s=(R/'liftfix_hold_trial.py').read_text().replace("ROOT/'liftfix_hold_trials'",f"ROOT/'liftpair_{mode}_trials'")
 s=s.replace("    balance = {'l': 0., 'r': 0.}","    balance = {'l': 0., 'r': 0.}\n    last_feedback_time={'l':None,'r':None}")
 s=s.replace("                latest=samples[-1]['finger_forces']","                now=step/120.\n                feedback_dt=.25 if last_feedback_time[side] is None else min(.5,max(.01,now-last_feedback_time[side]))\n                last_feedback_time[side]=now\n                latest=samples[-1]['finger_forces']")
 old="""                if min(loads)<5. and max(loads)>6.:
                    balance[side]=float(np.clip(balance[side]+np.radians(.08)*np.sign(loads[1]-loads[0]),-np.radians(8),np.radians(8)))"""
 new="""                total_load=sum(loads)
                imbalance=(loads[1]-loads[0])/max(4.,total_load)
                if total_load>4. and abs(loads[1]-loads[0])>.5:
                    balance[side]=float(np.clip(balance[side]+np.radians(.6)*feedback_dt*imbalance,-np.radians(8),np.radians(8)))"""
 assert old in s;s=s.replace(old,new)
 if mode=='soft':
  old="""                if min(loads)<5. and max(loads)<30.:squeeze[side]=min(.012,squeeze[side]+.0003)
                elif max(loads)>35.:squeeze[side]=max(0.,squeeze[side]-.0003)"""
  new="""                total_load=sum(loads)
                force_error=20.-total_load
                rate=0. if abs(force_error)<2. else float(np.clip(force_error*.00008,-.0008,.0008))
                if max(loads)>35.:rate=-.0008
                squeeze[side]=float(np.clip(squeeze[side]+rate*feedback_dt,0.,.012))"""
  assert old in s;s=s.replace(old,new)
 s=s.replace("result['feedback_revision']='",f"result['feedback_revision']='{mode}: continuous normalized load balance, max 0.6deg/s; ")
 ast.parse(s);(R/f'liftpair_{mode}_trial.py').write_text(s)
