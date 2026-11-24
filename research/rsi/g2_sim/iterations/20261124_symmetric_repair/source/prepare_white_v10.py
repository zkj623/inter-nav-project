from pathlib import Path
import json,ast,hashlib
R=Path(__file__).resolve().parent;O=R/'repair_v10';O.mkdir(exist_ok=True)
d=json.loads((R/'repair_v7/box_white.json').read_text());d['plans'][0]['repair_revision']='v10 v7 stable support geometry; higher articulation/object solver iterations128/32 for multi-finger constraint convergence';(O/'box_white.json').write_text(json.dumps(d,indent=2))
r=json.loads((R/'realpose_trials/repair_v7/box_white/candidate0_repeat1/result.json').read_text());s=(R/'source_versions'/(r['source_sha256']+'.py')).read_text();needle="    stage=Usd.Stage.Open(str(assets/'g2_contact_inertia_fixed.usdc'))"
extra='''
    result['solver_iteration_override']={'articulation_position':128,'articulation_velocity':32,'object_position':128,'object_velocity':32,'physics_hz':120}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            api=PhysxSchema.PhysxArticulationAPI.Apply(prim);api.CreateSolverPositionIterationCountAttr(128);api.CreateSolverVelocityIterationCountAttr(32)
'''
assert s.count(needle)==1;s=s.replace(needle,needle+extra).replace('bodyapi.CreateSolverPositionIterationCountAttr(32);bodyapi.CreateSolverVelocityIterationCountAttr(8)','bodyapi.CreateSolverPositionIterationCountAttr(128);bodyapi.CreateSolverVelocityIterationCountAttr(32)')
ast.parse(s);f=R/'realpose_box_white_trial.py';f.write_text(s);b=f.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b);print('WHITE_V10_READY')
