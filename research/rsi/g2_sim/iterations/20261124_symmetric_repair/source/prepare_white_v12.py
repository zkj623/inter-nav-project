from pathlib import Path
import json,ast,hashlib
R=Path(__file__).resolve().parent;O=R/'repair_v12';O.mkdir(exist_ok=True);T=R/'realpose_trials/repair_v7/box_white/candidate0_repeat1'
ss=json.loads((T/'samples.json').read_text());valid=[s for s in ss if s['phase'] in ['hold','contact_settle'] and s['lift_m']>.15 and s['tilt_deg']<10]
def margin(s):return min(s['finger_forces'].get(f'gripper_{side}_{finger}_link4',0) for side in ['l','r'] for finger in ['inner','outer'])
best=max(valid,key=margin);d=json.loads((R/'repair_v7/box_white.json').read_text());p=d['plans'][0];p['q_stable_hold']=best['joint_targets'];p['stable_hold_target_source']={'trial':str(T),'sample_t':best['t'],'minimum_finger_force_n':margin(best),'note':'candidate from a balanced-contact instant in a failed trial; must pass fresh full physics validation'};p['repair_revision']='v12 settle to empirically balanced joint target; avoid noisy contact rebalancing in static hold'
(O/'box_white.json').write_text(json.dumps(d,indent=2))
r=json.loads((T/'result.json').read_text());s=(R/'source_versions'/(r['source_sha256']+'.py')).read_text();needle='''            if failure is None:
                settled_updates=0''';replacement='''            if failure is None:
                balanced=plan['q_stable_hold']
                segment(last,balanced,2.,'contact_settle');last=balanced
                settled_updates=0''';assert s.count(needle)==1;s=s.replace(needle,replacement)
old="""                for _ in range(60):
                    corrected=retarget(q,last)""";new="""                for _ in range(60):
                    corrected=last.copy()""";assert s.count(old)==1;s=s.replace(old,new)
old="""                hold_start=step
                while step-hold_start<360:
                    corrected=retarget(q,last)
                    segment(last,corrected,min(.1,(360-(step-hold_start))/120.),'hold');last=corrected""";assert s.count(old)==1;s=s.replace(old,"                segment(last,last,3.,'hold')")
ast.parse(s);f=R/'realpose_box_white_trial.py';f.write_text(s);b=f.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b);print('WHITE_V12_READY',best['t'],margin(best))
