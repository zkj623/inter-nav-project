from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'torquelow_single_trial.py').read_text().replace("ROOT/'torquelow_single_trials'","ROOT/'torquelow_centered_trials'")
needle="            solution,pe,re=k.solve(side,target,actual,current_base,tcp)"
extra="""            elif plan.get('mode')=='single_top':
                loads=[float(np.mean([sum(v for n,v in sample['finger_forces'].items() if n.startswith(f'gripper_{side}_{finger}_')) for sample in samples[-3:]])) for finger in ('inner','outer')]
                if min(loads)<2. and max(loads)>3.:
                    pinch_shift[side]=float(np.clip(pinch_shift[side]+.0002*np.sign(loads[0]-loads[1]),-.012,.012))
                target[:3,3]+=pinch_shift[side]*(desired[f'arm_{side}_end_link'][:3,:3]@pinch_axes[side])
"""
assert needle in s;s=s.replace(needle,extra+needle);s=s.replace("    result['required_positive_contacts']=2","    result['required_positive_contacts']=2\n    result['pinch_centering']='weak-finger pose centering along native closing axis;0.2mm/update,max12mm; motor/material/geometry/acceptance unchanged'");s=s.replace("        result['preload_stable_updates']=stable","        result['preload_stable_updates']=stable\n        result['pinch_center_shift_m']=pinch_shift.copy()");ast.parse(s);(R/'torquelow_single_centered_trial.py').write_text(s);print('SINGLE_CENTERED_SOURCE_READY')
