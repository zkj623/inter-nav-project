from pathlib import Path
import json
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':6,'physics_gpu':6,'multi_gpu':False})
from omni.isaac.core import World
w=World(physics_dt=1/120,rendering_dt=1/30);w.scene.add_default_ground_plane();w.reset()
records=[]
for mode in ['legacy','explicit_physics_render']:
 for i in range(8):
  t=w.current_time;step=w.current_time_step_index
  if mode=='legacy':w.step(render=i%4==0)
  else:
   w.step(render=False)
   if i%4==0:w.render()
  records.append(dict(mode=mode,i=i,dt=w.current_time-t,physics_steps=w.current_time_step_index-step))
Path(__file__).with_name('timebase_check.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records),flush=True);app.close()
