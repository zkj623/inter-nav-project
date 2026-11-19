from pathlib import Path
import json,sys,numpy as np
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':2,'physics_gpu':2,'multi_gpu':False})
from pxr import PhysxSchema
R=Path(__file__).resolve().parent
print('DECOMP',PhysxSchema.PhysxConvexDecompositionCollisionAPI.GetSchemaAttributeNames(),flush=True)
for side in ['l','r']:
 for finger in ['inner','outer']:
  n=f'gripper_{side}_{finger}_link4';d=json.loads((R/'robot_assets'/(n+'_cooked.json')).read_text());m=np.load(R/'robot_assets'/(n+'.npz'));print(n,len(d['hulls']),len(m['vertices']),len(m['faces']),flush=True)
app.close()
