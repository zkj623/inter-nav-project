from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'symhome_v4_basket_rectangular_trial.py').read_text().replace("ROOT/'symhome_v4_trials'","ROOT/'symhome_v4force_trials'")
s=s.replace("    balance = {'l': 0., 'r': 0.}","    balance = {'l': 0., 'r': 0.}\n    pinch_aperture={'l':0.,'r':0.};pinch_force_time={'l':None,'r':None}")
needle='            solution,pe,re=k.solve(side,target,actual,current_base,tcp)';assert needle in s
block="""            elif plan.get('mode')=='bimanual_top':
                loads=[float(np.mean([sum(v for n,v in sample['finger_forces'].items() if n.startswith(f'gripper_{side}_{finger}_')) for sample in samples[-3:]])) for finger in ('inner','outer')]
                now=step/120.;elapsed=.25 if pinch_force_time[side] is None else min(.5,max(.01,now-pinch_force_time[side]));pinch_force_time[side]=now
                total=sum(loads)
                rate=-.04 if max(loads)>30. else float(np.clip((20.-total)*.0015,-.03,.02)) if abs(total-20.)>3. else 0.
                pinch_aperture[side]=float(np.clip(pinch_aperture[side]+rate*elapsed,-.12,0.))
                master='idx31_gripper_l_inner_joint1' if side=='l' else 'idx71_gripper_r_inner_joint1'
                corrected[master]=pinch_aperture[side]
"""
s=s.replace(needle,block+needle)
s=s.replace("    result['contact_definition']=","    result['pinch_force_feedback']='basket top pinch: total20N/hand target with3N deadband; maxbranch>30N opens; joint offset[-0.12,0]rad, max0.04rad/s opening and0.02rad/s closure; original effort/material/geometry/contact acceptance retained'\n    result['contact_definition']=")
ast.parse(s);(R/'symhome_v4force_basket_rectangular_trial.py').write_text(s)
