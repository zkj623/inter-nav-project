from pathlib import Path
import json
R=Path(__file__).resolve().parent;old=R.parent/'g2-commonstart-20261119'
for folder in ['commonstart_slow_trials/commonstart_plans/basket_rectangular/candidate0_repeat1','commonstart_slow_trials/commonstart_rim_plans/basket_rectangular/candidate0_repeat1','commonstart_force_trials/commonstart_plans/case_handheld/candidate0_repeat1']:
 d=json.loads((old/folder/'result.json').read_text());ss=json.loads((old/folder/'samples.json').read_text());print(folder,d['failure'],d.get('surface_error_events',[])[-1:],flush=True)
 for s in ss[-2:]:print(s['phase'],s['t'],{n:round(v,4) for n,v in s['joint_positions'].items() if n in ['idx31_gripper_l_inner_joint1','idx71_gripper_r_inner_joint1']},s['finger_forces'],flush=True)
