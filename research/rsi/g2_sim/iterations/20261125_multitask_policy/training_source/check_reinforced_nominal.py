from pathlib import Path
import json,subprocess,sys,fcntl,copy
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';asset='box_reinforced'
e=json.loads((R/'standby_preparation_config.json').read_text())['execution'][asset];plan=copy.deepcopy(json.loads((R/e['plan_dir']/(asset+'.json')).read_text())['plans'][e['candidate']]);plan['q_default']=plan['q_start'].copy();plan['q_prepare']=[]
plan['collection_placement']=dict(asset=asset,index=0,seed=0,dx_m=0.,dy_m=0.,dyaw_deg=0.,group_id='box_reinforced_nominal',success=True);folder='nominal_control';O=R/folder;O.mkdir(exist_ok=True);(O/(asset+'.json')).write_text(json.dumps(dict(asset=asset,plans=[plan]),indent=2))
with (W/'gpu3_simulation.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (W/'reinforced_nominal_sim.log').open('w') as log:subprocess.run([sys.executable,str(R/f'realpose_{asset}_trial.py'),'--asset',asset,'--plan-dir',folder,'--gpu','3'],cwd=R,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
T=R/'realpose_trials'/folder/asset/'candidate0_repeat1'
for tool,args in [('verify_realpose.py',['--trial-dir']),('verify_threecam.py',[])]:
 with (W/('reinforced_nominal_'+tool+'.log')).open('w') as log:subprocess.run([sys.executable,str(R/tool),*args,str(T)],cwd=R,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
print((T/'acceptance.json').read_text(),flush=True)
