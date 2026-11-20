from pathlib import Path
import json,copy,numpy as np
R=Path(__file__).resolve().parent
exec(compile((R/'plan_commonstart.py').read_text().split('homes=[]')[0],'path_library','exec'))
d=json.loads((R/'narrowrest_candidate_config.json').read_text());old=d['standby_joint_positions_rad'];home=d['work_joint_positions_rad'];q=old.copy()
for side,sign in [('l',1),('r',-1)]:
 ns=[next(n for n in q if f'arm_{side}_joint{i}' in n) for i in range(1,4)]
 q[ns[0]]-=sign*np.pi;q[ns[1]]=-q[ns[1]];q[ns[2]]+=sign*np.pi
print('self',intersections(q),'proxy',space['collision'](q),flush=True)
out=R/'directprep_plans';out.mkdir(exist_ok=True);rows=[]
for f in sorted((R/'narrowrest_plans').glob('*.json')):
 doc=json.loads(f.read_text());p=doc['plans'][0];a=f.stem;plans[a]=p
 probes=[{n:(1-u)*q.get(n,0)+u*home.get(n,0) for n in q} for u in np.linspace(0,1,85)]
 bad=[i for i,t in enumerate(probes) if not clear(a,t)]
 row=dict(asset=a,clear=not bad,bad_samples=bad);rows.append(row);print(row,flush=True)
 p['q_default']=q.copy();p['q_prepare']=[home];p['standby_label']='CUSTOM equivalent shoulder IK branch; reduced shoulder excursion';(out/f.name).write_text(json.dumps(doc,indent=2))
def metrics(start):
 qs=[{n:(1-u)*start.get(n,0)+u*home.get(n,0) for n in start} for u in np.linspace(0,1,101)];fk=[k.fk(t,base) for t in qs];ret={}
 for s in ('l','r'):
  pts=np.array([f[f'arm_{s}_end_link'][:3,3] for f in fk]);el=np.array([f[f'arm_{s}_link4'][:3,3] for f in fk]);ret[s]=dict(wrist_path_m=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()),wrist_z_peak_m=float(pts[:,2].max()),wrist_z_start_m=float(pts[0,2]),elbow_abs_y_peak_m=float(abs(el[:,1]).max()))
 ret['total_arm_travel_deg']=float(sum(abs(np.degrees(home[n]-start[n])) for n in arms));return ret
report=dict(old_q=old,new_q=q,work_q=home,old_metrics=metrics(old),new_metrics=metrics(q),paths=rows)
(R/'directprep_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({n:report[n] for n in ['old_metrics','new_metrics']},indent=2),flush=True)
for f in R.glob('narrowrest_*_trial.py'):
 a=f.name.removeprefix('narrowrest_').removesuffix('_trial.py');s=f.read_text().replace("ROOT/'narrowrest_trials'","ROOT/'directprep_trials'");(R/f'directprep_{a}_trial.py').write_text(s)
