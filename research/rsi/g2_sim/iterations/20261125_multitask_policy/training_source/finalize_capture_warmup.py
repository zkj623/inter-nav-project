from pathlib import Path
import hashlib,json,ast
R=Path('/data20t/embodied/zkj/inter-nav-results/g2-act-threecam-20261124');W=Path(__file__).resolve().parent
s=(R/'threecam_recorder_v2.py').read_text();start=s.index('    for warmup in range(32):');end=s.index('    self.initial_camera_warmup_renders=warmup',start)
s=s[:start]+'''    consecutive_valid=0
    for warmup in range(32):
     valid=True
     for camera_name,camera in self.views.items():
      frame=camera.get_rgba()
      if frame is None or not frame.size:valid=False;break
      preview=frame[:,:,:3]
      if camera_name in self.maps:preview=self.cv2.remap(preview,*self.maps[camera_name],self.cv2.INTER_LINEAR)
      if preview.std()<=1.:valid=False;break
     consecutive_valid=consecutive_valid+1 if valid else 0
     if consecutive_valid>=2:break
     world.render()
    else:raise RuntimeError('CAMERA_INITIAL_WARMUP_FAILED')
'''+s[end:]
ast.parse(s);(R/'threecam_recorder_v3.py').write_text(s)
V=R/'source_versions';changes={}
for f in [*R.glob('realpose_*_trial.py'),*R.glob('policy_*_trial.py')]:
 raw=f.read_bytes();before=hashlib.sha256(raw).hexdigest();(V/(before+'.py')).write_bytes(raw);f.write_text(f.read_text().replace('from threecam_recorder_v2 import Recorder','from threecam_recorder_v3 import Recorder'));raw=f.read_bytes();after=hashlib.sha256(raw).hexdigest();(V/(after+'.py')).write_bytes(raw);changes[f.name]=[before,after]
f=R/'policy_threecam_recorder.py';s=f.read_text().replace('from threecam_recorder_v2 import Recorder','from threecam_recorder_v3 import Recorder');s=s.replace('  self.url=url;','  self.adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()\n  self.url=url;');s=s.replace('adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()','adapter_sha256=self.adapter_sha256');s=s.replace('self.adapter_sha256=self.adapter_sha256','self.adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()');f.write_text(s)
f=R/'verify_threecam.py';s=f.read_text().replace("['threecam_recorder.py','threecam_recorder_v2.py']","['threecam_recorder.py','threecam_recorder_v2.py','threecam_recorder_v3.py']");f.write_text(s)
for name in ['threecam_recorder_v2.py','threecam_recorder_v3.py','policy_threecam_recorder.py']:
 raw=(R/name).read_bytes();(V/(hashlib.sha256(raw).hexdigest()+'.py')).write_bytes(raw)
(W/'capture_revision_v3.json').write_text(json.dumps(dict(changes=changes,recorder_sha256=hashlib.sha256((R/'threecam_recorder_v3.py').read_bytes()).hexdigest(),reason='Warmup checks final distorted wrist crop, not padded raw image; two consecutive valid frames before physics advances. Previous rejected frame0 raw std3.48 but actual crop std0.011.'),indent=2));print('CROP_VALIDATED_WARMUP_V3_READY')
