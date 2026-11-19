from pathlib import Path
import re,ast
R=Path(__file__).resolve().parent
for a in ['basket_rectangular','case_handheld']:
 s=(R/f'commonfix_{a}_trial.py').read_text().replace("ROOT/'commonfix_trials'","ROOT/'commonfix_solver_trials'")
 needle="    # Training isolation: lock all non-arm/non-gripper joints, and anchor root."
 patch="""    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            solver=PhysxSchema.PhysxArticulationAPI.Apply(prim)
            solver.CreateSolverPositionIterationCountAttr(64);solver.CreateSolverVelocityIterationCountAttr(16)
    result['constraint_accuracy']='articulation solver 64 position / 16 velocity iterations; original geometry/material/actuator limits retained'
"""
 assert needle in s;s=s.replace(needle,patch+needle)
 # Measure native mimic constraint residuals rather than assuming exact coupling.
 needle="            rec['gravity_torques_nm']=dict(zip(names,gravity.tolist()))"
 s=s.replace(needle,needle+"\n            rec['max_mimic_error_rad']=float(max([abs(qd.get(n,0.)-(qd.get(j['mimic']['joint'],0.)*float(j['mimic'].get('multiplier',1))+float(j['mimic'].get('offset',0)))) for n,j in k.by_name.items() if j['mimic']]+[0.]))")
 ast.parse(s);(R/f'commonfix_solver_{a}_trial.py').write_text(s)
