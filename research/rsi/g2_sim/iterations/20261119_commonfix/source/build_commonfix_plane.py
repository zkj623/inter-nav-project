from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'commonfix_solver_case_handheld_trial.py').read_text().replace("ROOT/'commonfix_solver_trials'","ROOT/'commonfix_plane_trials'")
old="""                if total_load>4. and abs(loads[1]-loads[0])>.5:
                    balance[side]=float(np.clip(balance[side]+np.radians(1.5)*feedback_dt*imbalance,-np.radians(8),np.radians(8)))"""
new="""                positions,quaternions=fingers.get_world_poses()
                patches=[]
                for finger in ('inner','outer'):
                    name=f'gripper_{side}_{finger}_link4';index=fname.index(name)
                    actual_tip=mat(positions[index],quaternions[index]);vertices=fmeshes[name].vertices@actual_tip[:3,:3].T+actual_tip[:3,3]
                    depth=vertices@normal;patches.append(vertices[depth>depth.max()-.0002].mean(0))
                difference=patches[0]-patches[1]
                normal_error=float(np.dot(difference,normal));span=float(np.linalg.norm(difference-normal_error*normal))
                correction=float(np.clip(-.5*np.arctan2(normal_error,max(.01,span)),-np.radians(.2),np.radians(.2)))
                balance[side]=float(np.clip(balance[side]+correction,-np.radians(8),np.radians(8)))"""
# normal is needed before this feedback, and is unchanged from the baseline.
new="                normal=delta@np.array([0.,-1. if side=='l' else 1.,0.])\n"+new
assert old in s;s=s.replace(old,new)
s=s.replace("result['constraint_accuracy']=","result['measured_pad_plane_feedback']='actual simulator distal link poses + visual pad extreme 0.2mm bands; bounded wrist correction 0.2deg/update, total +/-8deg; no object pose driving'\n    result['constraint_accuracy']=")
ast.parse(s);(R/'commonfix_plane_case_handheld_trial.py').write_text(s)
