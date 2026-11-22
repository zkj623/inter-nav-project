from pathlib import Path
import json,tempfile,numpy as np,h5py
from collection_recorder import Recorder
R=Path(__file__).resolve().parent
class Robot:
 dof_names=[f'q{i}' for i in range(16)]
 def __init__(self):self.q=np.zeros(16)
 def get_joint_positions(self):return self.q.copy()
 def get_joint_velocities(self):return np.ones(16)*120
class World:
 current_time=0.
 def render(self):pass
class Camera:
 def get_current_frame(self):return {'rendering_time':world.current_time}
 def get_rgba(self):return np.tile(np.arange(640,dtype=np.uint8)[None,:,None],(480,1,4))
world=World();robot=Robot()
with tempfile.TemporaryDirectory(dir=R,prefix='recorder_test_') as folder:
 rec=Recorder(folder,robot.dof_names,{'asset':'test'});rec.bind(Camera());rec.command(dict.fromkeys(robot.dof_names,-1),'prepare_home',robot,world);assert rec.count==0
 phases=['common_approach','test_lift','support_preload','support_settle']
 for tick in range(24):
  command=rec.command(dict.fromkeys(robot.dof_names,float(100+tick)),phases[tick//6],robot,world)
  assert command['q0']==100+(tick//6)*6
  robot.q+=1.;world.current_time+=1/120
 rec.finish({'success':True})
 with h5py.File(Path(folder)/'episode.partial.hdf5','r') as f:
  assert f['action'].shape==(4,16);assert np.array_equal(f['action'][:,0],[100,106,112,118]);assert np.array_equal(f['observations/qpos'][:,0],[0,6,12,18]);assert np.allclose(np.diff(f['timestamps'][:]),.05);assert list(f['phase'].asstr()[:])==phases
print('RECORDER_CAUSAL_TIMING_AND_ALL_TASK_PHASES_OK')
