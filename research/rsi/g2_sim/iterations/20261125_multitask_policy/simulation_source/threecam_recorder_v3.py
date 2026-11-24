"""Synchronized head and official wrist RGB on the existing held-action clock."""
from pathlib import Path
import json,hashlib,numpy as np
from collection_recorder import Recorder as BaseRecorder
class Recorder(BaseRecorder):
 def bind(self,camera):
  super().bind(camera)
  import cv2,omni.usd
  from omni.isaac.sensor import Camera
  from pxr import UsdGeom
  from g2_kinematics import Kinematics
  self.cv2=cv2;self.views={};self.maps={};R=Path(__file__).resolve().parent;self.kinematics=Kinematics(R/'generated/g2_crs_omnipicker.urdf');self.calibration=json.loads((R/'official_wrist_config.json').read_text())
  specs={'head_front':dict(parent='head_link3',xyz=[.10237,.02375,.10256],wxyz=[-.5945104121873723,-.3868316461705513,.37872914728340207,.5945442627086478],fx=305.6425170898,fy=305.6007080078,cx=318.7979431152,cy=202.4943847656,width=640,height=400)}
  for side,name in [('l','wrist_left'),('r','wrist_right')]:
   c=self.calibration['cameras'][side];specs[name]=dict(parent=c['parent'],xyz=c['xyz'],wxyz=c['wxyz'],**{n:c[n][0]/2 for n in ['fx','fy','cx','cy']},width=1280,height=1056)
   K=np.array([[c['fx'][0]/2,0,c['cx'][0]/2],[0,c['fy'][0]/2,c['cy'][0]/2],[0,0,1.]])
   yy,xx=np.mgrid[:528,:640];points=np.stack([xx,yy],-1).astype(np.float32);dist=np.array([c[n][0] for n in ['k1','k2','p1','p2','k3']]);uv=cv2.undistortPoints(points.reshape(-1,1,2),K,dist,P=K).reshape(528,640,2)
   mx=(uv[:,:,0]+320).astype(np.float32);my=(uv[:,:,1]+264).astype(np.float32);assert mx.min()>=0 and mx.max()<1279 and my.min()>=0 and my.max()<1055;self.maps[name]=(mx,my)
  self.specs=specs
  for name,c in specs.items():
   path='/World/ACT_'+name if name.startswith('wrist') else '/robot/'+c['parent']+'/ACT_'+name;v=Camera(path,resolution=(c['width'],c['height']));v.initialize();v._frequency=-1;v.set_local_pose(translation=np.array(c['xyz']),orientation=np.array(c['wxyz']),camera_axes='usd');v.set_focal_length(20.);v.set_horizontal_aperture(20*c['width']/c['fx']);v.set_vertical_aperture(20*c['height']/c['fy']);v.set_clipping_range(.01,100.)
   usd=UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath(path));padx,pady=(320,264) if name.startswith('wrist') else (0,0);usd.GetHorizontalApertureOffsetAttr().Set((c['width']/2-c['cx']-padx)*20/c['fx']);usd.GetVerticalApertureOffsetAttr().Set((c['cy']+pady-c['height']/2)*20/c['fy']);self.views[name]=v
 def command(self,target,phase,robot,world):
  if self.ticks%6==0 and (self.ticks>0 or phase in ['common_approach','pregrasp_hold','approach']):
   from scipy.spatial.transform import Rotation
   pos,quat=robot.get_world_pose();base=np.eye(4);base[:3,3]=pos;base[:3,:3]=Rotation.from_quat(np.array(quat)[[1,2,3,0]]).as_matrix();fk=self.kinematics.fk(dict(zip(robot.dof_names,robot.get_joint_positions())),base)
   for name in ['wrist_left','wrist_right']:
    c=self.specs[name];mount=np.eye(4);mount[:3,3]=c['xyz'];mount[:3,:3]=Rotation.from_quat(np.array(c['wxyz'])[[1,2,3,0]]).as_matrix();pose=fk[c['parent']]@mount;self.views[name].set_world_pose(position=pose[:3,3],orientation=Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]],camera_axes='usd')
   clock=float(world.current_time);index=world.current_time_step_index
   for _ in range(3):world.render()
   if self.count==0:
    # Sensor startup can yield one black left-wrist frame despite valid timestamps.
    # Warm up on the same physics state, never replace image contents or relax audit.
    consecutive_valid=0
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
    self.initial_camera_warmup_renders=warmup
   assert world.current_time_step_index==index and abs(float(world.current_time)-clock)<1e-9
  previous=self.count;held=super().command(target,phase,robot,world)
  if self.count==previous:return held
  f=self.file;i=self.count-1;now=float(world.current_time)
  assert np.max(abs(f['observations/qpos'][i]-np.array(robot.get_joint_positions())[[robot.dof_names.index(n) for n in self.names]]))<1e-6
  for name,camera in self.views.items():
   rgba=camera.get_rgba();assert rgba is not None and rgba.shape[:2]==(self.specs[name]['height'],self.specs[name]['width']),name+' missing frame'
   ct=float(camera.get_current_frame().get('rendering_time',-1));assert abs(now-ct)<=1/120+.0001,(name,now,ct)
   rgb=rgba[:,:,:3].astype(np.uint8).copy()
   if i==0:
    self.cv2.imwrite(str(self.out/(name+'_raw_probe.jpg')),self.cv2.cvtColor(rgb,self.cv2.COLOR_RGB2BGR));pos,quat=camera.get_world_pose(camera_axes='usd');(self.out/(name+'_camera_probe.json')).write_text(json.dumps(dict(position=np.asarray(pos).tolist(),quaternion=np.asarray(quat).tolist(),raw_std=float(rgb.std()),spec=self.specs[name]),indent=2))
    if name=='wrist_left' and 'camera_probe' in str(self.out):
     from diagnose_wrist_occlusion import inspect
     inspect(camera,self.out/'wrist_occlusion.json')
   if name in self.maps:rgb=self.cv2.remap(rgb,*self.maps[name],self.cv2.INTER_LINEAR)
   key='observations/images/'+name
   if key not in f:
    f.create_dataset(key,shape=(0,*rgb.shape),maxshape=(None,*rgb.shape),dtype='u1',chunks=(1,*rgb.shape),compression='lzf');f.create_dataset('image_timestamps/'+name,shape=(0,),maxshape=(None,),dtype='f8')
   for key,value in [(key,rgb),('image_timestamps/'+name,ct)]:ds=f[key];ds.resize(self.count,axis=0);ds[i]=value
   if i==0 or phase!=getattr(self,'last_phase',None):self.cv2.imwrite(str(self.out/(name+'_'+phase+'.jpg')),self.cv2.cvtColor(rgb,self.cv2.COLOR_RGB2BGR))
  self.last_phase=phase;f.attrs['schema_version']='g2-act-threecam-v1';f.attrs['camera_names']=json.dumps(['head_front','wrist_left','wrist_right']);f.attrs['audit_camera_names']=json.dumps(['workspace']);f.attrs['source_commit']='f649f70a0713700bbc58397c948be157aaf15e16'
  return held
 def finish(self,result):
  super().finish(result)
  m=json.loads((self.out/'metadata.json').read_text());m.update(training_camera_names=['head_front','wrist_left','wrist_right'],threecam_recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),camera_specs=getattr(self,'specs',{}),official_wrist_calibration=getattr(self,'calibration',{}),wrist_processing='official distorted half-resolution640x528 using padded pinhole render and cv2 remap; no extra image rotation',source_commit='f649f70a0713700bbc58397c948be157aaf15e16')
  asset=result.get('object_collision_asset')
  if asset:m['object_collision_sha256']=hashlib.sha256(Path(asset).read_bytes()).hexdigest()
  m['initial_camera_warmup_renders']=getattr(self,'initial_camera_warmup_renders',0)
  m['wrist_pose_follow']='Official mount transformed by actual measured joint-state FK and actual robot base pose before each render; no object-state input'
  m['render_visibility_scope']=result.get('render_visibility_scope');m['hidden_collision_debug_mesh_count']=len(result.get('collision_only_meshes_hidden_for_render',[]))
  (self.out/'metadata.json').write_text(json.dumps(m,indent=2))
