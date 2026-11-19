from pathlib import Path
import re,ast
R=Path(__file__).resolve().parent;a='case_handheld'
s=(R/'liftpair_range_trial.py').read_text();s=re.sub(r"OUT=ROOT/[^\n]+","OUT=ROOT/'commonstart_range_trials'/args.plan_dir/args.asset/f'candidate{args.candidate}_repeat{args.repeat}'",s,count=1)
s=s.replace("        segment(qpre,q_grasp_pre,12.,'descent')","        last=qpre\n        for waypoint in plan['q_approach']:\n            segment(last,waypoint,6.,'common_approach');last=waypoint")
s=s.replace("segment(q_grasp_pre,q_grasp_pre,3.,'descent_hold')","segment(q_grasp_pre,q_grasp_pre,2.,'pregrasp_hold')")
s=s.replace("achieved=all(forces.get(f'gripper_{side}_{finger}_link4',0.)>4.","achieved=all(forces.get(f'gripper_{side}_{finger}_link4',0.)>1.")
ast.parse(s);(R/f'commonstartrange_{a}_trial.py').write_text(s)
