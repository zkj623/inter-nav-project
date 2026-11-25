from pathlib import Path
import json, hashlib, py_compile
W=Path('/data20t/embodied/zkj/inter-nav-results/g2-policy-overnight-20261124')
R=W.parent/'g2-act-threecam-20261124'
OUT=W/'pi05_focus_20261125';OUT.mkdir(exist_ok=True)
def replace(s,a,b):
 assert a in s,a
 return s.replace(a,b)
# Preserve every prior experiment. Use a separate trainer, checkpoint namespace and evaluator.
s=(W/'train_g2_pi05.py').read_text(encoding='utf-8-sig')
s=replace(s,'a=p.parse_args()',"p.add_argument('--init-checkpoint');p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--checkpoint-root',default='/tmp/zkj_g2_20261125_checkpoints');a=p.parse_args()")
s=replace(s,"str(W/'weights/pi05_base/params')","a.init_checkpoint or str(W/'weights/pi05_base/params')")
s=replace(s,'peak_lr=1e-4,decay_steps=a.steps,decay_lr=1e-5','peak_lr=a.lr,decay_steps=36000,decay_lr=a.lr*.1')
s=replace(s,"checkpoint_base_dir=str(W/'checkpoints')","checkpoint_base_dir=a.checkpoint_root")
s=s.replace('save_interval=1000,keep_period=1000','save_interval=1000000,keep_period=None').replace('save_interval=250,keep_period=1000','save_interval=1000000,keep_period=None')
assert 'save_interval=1000000' in s
s=replace(s,"training=dict(steps=a.steps,batch=a.batch,lr=1e-4)","training=dict(steps=a.steps,batch=a.batch,lr=a.lr,init_checkpoint=a.init_checkpoint,sampling='uniform episode and start time; original openpi visual augmentation')")
s=replace(s,'train.main(cfg)',"""# Avoid overlapping full-model writes to shared disks.
original_save=train._checkpoints.save_state
def synchronous_save(manager,*args,**kwargs):
 original_save(manager,*args,**kwargs)
 manager.wait_until_finished()
train._checkpoints.save_state=synchronous_save
(W/'pi05_focus_20261125'/('trainer_'+a.run+'.py')).write_text(Path(__file__).read_text())
train.main(cfg)
print('FOCUS_TRAINING_COMPLETE',flush=True)""")
(W/'train_pi05_focus.py').write_text(s)
# Shared executable definition of the user-requested relaxed acceptance.
rules='''from pathlib import Path
def allowed_base_contact(event):
 return (Path(event['actor']).name=='base_link' and event['impulse']<=.02
         and event.get('minimum_separation',0)>=-.003)
'''
(R/'stable_lift_rules.py').write_text(rules)
cfg=json.loads((R/'ground_lift_training_config.json').read_text())
for asset in cfg['validated_simulation_baselines']:
 source=R/f'policy_{asset}_trial.py'
 if not source.exists():continue
 s=source.read_text(encoding='utf-8-sig')
 s=replace(s,'import numpy as np','import numpy as np\nfrom stable_lift_rules import allowed_base_contact')
 s=replace(s,"(OUT/'protocol.json').write_text", "result['acceptance_revision']='stable_lift_v1: four-finger and bilateral force diagnostics only; base impulse<=0.02Ns and penetration<=3mm allowed; >=15cm lift, >=3s hold, <=20deg tilt, <=2cm drift'\n(OUT/'protocol.json').write_text")
 s=s.replace("and len(nonhand_contacts)<500:","and len(nonhand_contacts)<5000:")
 s=replace(s,"max(x['impulse'] for x in nonhand_contacts)>.001", "max((x['impulse'] for x in nonhand_contacts if not allowed_base_contact(x)),default=0.)>.001")
 s=replace(s,"len(nonhand_contacts)>=500:","len(nonhand_contacts)>=5000:")
 s=s.replace("passed=bool((plan.get('mode') not in ('bimanual_spread','bimanual_under') or four_fraction>=.9) and failure is None", "passed=bool(four_fraction>=.9 and failure is None")
 s=replace(s,"passed=bool(four_fraction>=.9 and failure is None", "passed=bool(failure is None")
 s=replace(s," and bilateral>=.9 and witness_count>0)"," and witness_count>0)")
 s=s.replace("--/plugins/carb.tasking.plugin/threadCount=16","--/plugins/carb.tasking.plugin/threadCount=8").replace("--/plugins/omni.tbb.globalcontrol/maxThreadCount=16","--/plugins/omni.tbb.globalcontrol/maxThreadCount=8").replace("--/persistent/physics/numThreads=16","--/persistent/physics/numThreads=8")
 dest=R/f'stable_policy_{asset}_trial.py';dest.write_text(s)
 sha=hashlib.sha256(dest.read_bytes()).hexdigest();(R/'source_versions'/(sha+'.py')).write_bytes(dest.read_bytes())
v=(R/'verify_realpose.py').read_text(encoding='utf-8-sig')
v=replace(v,'from pathlib import Path','from pathlib import Path\nfrom stable_lift_rules import allowed_base_contact')
v=replace(v,"audit_version=2", "audit_version='stable_lift_v1'")
v=replace(v,"if hits:reasons.append('NONHAND_GEOMETRY')","if any(h['link']!='base_link' or h['gap_m']<-.003 for h in hits):reasons.append('NONHAND_GEOMETRY')")
v=replace(v,"if max(maximp.values())>.001:reasons.append('CONTACT_IMPULSE')","nonhand_events=json.loads((D/'nonhand_contacts.json').read_text())\n if maximp['self']>.001 or any(e['impulse']>.001 and not allowed_base_contact(e) for e in nonhand_events):reasons.append('CONTACT_IMPULSE')")
v=replace(v," and d['hold_four_finger_contact_fraction']>=.9",'')
v=v.replace('verification.tmp.json','stable_verification.tmp.json').replace('verification.json','stable_verification.json')
(R/'verify_stable_lift.py').write_text(v)
e=(W/'evaluate_g2_policy.py').read_text(encoding='utf-8-sig')
e=replace(e,"f'policy_{asset}_trial.py'","f'stable_policy_{asset}_trial.py'")
e=replace(e,"R/'verify_realpose.py'","R/'verify_stable_lift.py'")
e=e.replace("T/'verification.json'","T/'stable_verification.json'")
e=replace(e,"lift_m=v.get('min_hold_lift_m',result.get('max_lift_m'))", "lift_m=v.get('min_hold_lift_m',result.get('max_lift_m')),max_lift_m=result.get('max_lift_m'),min_hold_lift_m=result.get('min_hold_lift_m'),max_hold_tilt_deg=result.get('max_hold_tilt_deg')")
e=replace(e,"print('POLICY_EVALUATION',json.dumps(results[-1]),flush=True)","print('POLICY_EVALUATION',json.dumps({k:v for k,v in results[-1].items() if k!='policy'}),flush=True)")
(W/'evaluate_pi05_stable.py').write_text(e)
for p in [W/'train_pi05_focus.py',W/'evaluate_pi05_stable.py',R/'verify_stable_lift.py',R/'stable_lift_rules.py',*R.glob('stable_policy_*_trial.py')]:py_compile.compile(str(p),doraise=True)
# Meaningful boundary checks: allow only small BASE contacts, not wrists or deep penetration.
ns={};exec(rules,ns);f=ns['allowed_base_contact']
assert f(dict(actor='/robot/base_link',impulse=.01,minimum_separation=-.001))
assert not f(dict(actor='/robot/arm_l_link7',impulse=.01,minimum_separation=-.001))
assert not f(dict(actor='/robot/base_link',impulse=.03,minimum_separation=-.001))
assert not f(dict(actor='/robot/base_link',impulse=.01,minimum_separation=-.004))
(OUT/'setup.json').write_text(json.dumps({'acceptance':'stable_lift_v1','boundary_tests':'passed','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [W/'train_pi05_focus.py',W/'evaluate_pi05_stable.py',R/'verify_stable_lift.py',R/'stable_lift_rules.py']}},indent=2))
print('FOCUS_SETUP_COMPLETE')
