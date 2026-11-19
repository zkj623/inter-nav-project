from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'commonfix_solver_case_handheld_trial.py').read_text().replace("ROOT/'commonfix_solver_trials'","ROOT/'commonfix_center_trials'")
s=s.replace("    balance = {'l': 0., 'r': 0.}","    balance = {'l': 0., 'r': 0.}\n    planar_center=np.zeros(2)\n    preload_centering=True")
s=s.replace('    def retarget(nominal,command,extra=0.):','    def retarget(nominal,command,extra=0.):\n        nonlocal_dummy=None\n        global planar_center\n        if preload_centering:\n            displacement=np.array(obj.get_world_pose()[0])[:2]-np.array(plan[\'object_position\'])[:2]\n            planar_center=.8*planar_center+.2*np.clip(displacement,-.03,.03)')
s=s.replace("            target=desired[f'arm_{side}_end_link']@tcp","            target=desired[f'arm_{side}_end_link']@tcp\n            target[:2,3]+=planar_center")
s=s.replace('# Read object orientation only: object position is never imposed or followed.','# Bounded planar centering is measured before capture; object state is never written.')
s=s.replace("        result['adaptive_grip_extra_m']=grip_extra","        preload_centering=False\n        result['captured_planar_offset_m']=planar_center.tolist()\n        result['cartesian_feedback']='bounded measured object planar centering +/-3cm during preload; frozen thereafter; arm force balancing; no object state writes or attachment'\n        result['adaptive_grip_extra_m']=grip_extra")
ast.parse(s);(R/'commonfix_center_case_handheld_trial.py').write_text(s)
