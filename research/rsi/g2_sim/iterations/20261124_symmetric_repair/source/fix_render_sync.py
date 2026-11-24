from pathlib import Path
import ast,hashlib
R=Path(__file__).resolve().parent;f=R/'collection_recorder.py';oldbytes=f.read_bytes();(R/'source_versions'/(hashlib.sha256(oldbytes).hexdigest()+'.py')).write_bytes(oldbytes)
s=oldbytes.decode('utf-8-sig');old='   self.held=target.copy();world.render();frame=self.camera.get_current_frame();rgba=self.camera.get_rgba()'
new='''   self.held=target.copy();before_time=world.current_time;before_index=world.current_time_step_index
   for render_attempt in range(3):
    world.render();frame=self.camera.get_current_frame();rgba=self.camera.get_rgba()
    if world.current_time_step_index!=before_index or abs(world.current_time-before_time)>1e-9:raise RuntimeError('RENDER_ADVANCED_PHYSICS')
    if abs(float(world.current_time)-float(frame.get('rendering_time',world.current_time)))<=1/120+.0001:break'''
assert s.count(old)==1;s=s.replace(old,new);ast.parse(s);f.write_text(s);b=f.read_bytes();(R/'source_versions'/(hashlib.sha256(b).hexdigest()+'.py')).write_bytes(b);print('RENDER_RETRY_WITHOUT_PHYSICS_ADVANCE; TIMESTAMP_GATE_UNCHANGED')
