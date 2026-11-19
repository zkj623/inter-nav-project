from pathlib import Path
import ast
R=Path(__file__).resolve().parent
s=(R/'liftfix_trial.py').read_text().replace("ROOT/'liftfix_trials'","ROOT/'liftfix_balance_trials'")
s=s.replace("    squeeze = {'l': 0., 'r': 0.}","    squeeze = {'l': 0., 'r': 0.}\n    pinch_shift = {'l': 0., 'r': 0.}\n    pinch_axes={}\n    pre_fk=k.fk(plan['q_grasp'],np.array(plan['base_pose']))\n    for side in ('l','r'):\n        centers=[(pre_fk[f'gripper_{side}_{finger}_link4']@np.r_[fmeshes[f'gripper_{side}_{finger}_link4'].vertices.mean(0),1.])[:3] for finger in ('inner','outer')]\n        axis=pre_fk[f'arm_{side}_end_link'][:3,:3].T@(centers[0]-centers[1])\n        pinch_axes[side]=axis/max(1e-9,np.linalg.norm(axis))")
s=s.replace('if max(loads)<6.:squeeze[side]', 'if min(loads)<2. and max(loads)<30.:squeeze[side]')
needle='            solution,pe,re=k.solve(side,target,actual,current_base,tcp)'
replacement="""            elif plan.get('mode')=='bimanual_top':
                loads=[float(np.mean([sum(v for n,v in sample['finger_forces'].items() if n.startswith(f'gripper_{side}_{finger}_')) for sample in samples[-3:]])) for finger in ('inner','outer')]
                if min(loads)<2. and max(loads)>3.:
                    pinch_shift[side]=float(np.clip(pinch_shift[side]+.00025*np.sign(loads[0]-loads[1]),-.008,.008))
                target[:3,3]+=pinch_shift[side]*(desired[f'arm_{side}_end_link'][:3,:3]@pinch_axes[side])
"""+needle
assert needle in s;s=s.replace(needle,replacement)
s=s.replace("achieved=all(sum(v for n,v in forces.items() if f'gripper_{side}_' in n)>.75 for side in ('l','r'))","achieved=four_contact(samples[-1],1.)")
s=s.replace("        result['adaptive_grip_extra_m']=grip_extra","        if plan.get('mode')=='bimanual_top' and stable<3:raise TrialRejected('FOUR_BRANCH_PRELOAD_NOT_ESTABLISHED')\n        result['pinch_center_shift_m']=pinch_shift.copy()\n        result['adaptive_grip_extra_m']=grip_extra")
s=s.replace("result['contact_definition']=", "result['feedback_revision']='spread weak-finger squeeze gated by strong finger <30N; pinch centering bounded +/-8mm, 0.25mm/update; four-branch preload before lift'\n    result['contact_definition']=")
s=s.replace("                segment(last,last,3.,'hold')","                hold_start=step\n                while step-hold_start<360:\n                    corrected=retarget(q,last)\n                    segment(last,corrected,min(.1,(360-(step-hold_start))/120.),'hold');last=corrected")
ast.parse(s);(R/'liftfix_balance_trial.py').write_text(s)
