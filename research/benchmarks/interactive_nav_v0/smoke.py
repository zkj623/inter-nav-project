"""Load a generated scene and test object settling; not a navigation rollout."""
import argparse,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--scene-dir',type=Path,required=True);ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--steps',type=int,default=240);a=ap.parse_args()
overlay=a.scene_dir/'physics_only.usdc'
if not overlay.exists():raise FileNotFoundError('Run prepare_physics.py first in a separate process.')
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':a.gpu,'physics_gpu':a.gpu,'multi_gpu':False})
try:
 import numpy as np,omni.usd,math
 from omni.isaac.core import World
 from omni.isaac.core.prims import RigidPrim
 from pxr import UsdGeom
 omni.usd.get_context().open_stage(str(overlay))
 for _ in range(10):app.update()
 world=World(physics_dt=1/120,rendering_dt=1/30,stage_units_in_meters=1.)
 data=json.loads((a.scene_dir/'episodes.json').read_text());bodies=[]
 for item in data['objects']:bodies.append(world.scene.add(RigidPrim('/World/Objects/'+item['instance_id'],name=item['instance_id'])))
 world.reset();before=[b.get_world_pose()[0].copy() for b in bodies]
 for _ in range(a.steps):world.step(render=False)
 result=[]
 for item,b,p0 in zip(data['objects'],bodies,before):
  p,q=b.get_world_pose();v=b.get_linear_velocity();tilt=math.degrees(math.acos(float(np.clip(1-2*(q[1]**2+q[2]**2),-1,1))));result.append({'asset_id':item['asset_id'],'initial_position':p0.tolist(),'final_position':p.tolist(),'final_orientation_wxyz':q.tolist(),'tilt_degrees':tilt,'linear_speed_m_s':float(np.linalg.norm(v)),'displacement_m':float(np.linalg.norm(p-p0)),'passed':bool(np.all(np.isfinite(p)) and abs(p[2]-p0[2])<.2 and np.linalg.norm(p[:2]-p0[:2])<.25 and np.linalg.norm(v)<.1 and tilt<15)})
 report={'scene':data['id'],'steps':a.steps,'physics_dt':1/120,'objects':result,'passed':all(x['passed'] for x in result),'scope':'passive settling only; navigation and manipulation not executed'}
 (a.scene_dir/'physics_smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
finally:app.close()
