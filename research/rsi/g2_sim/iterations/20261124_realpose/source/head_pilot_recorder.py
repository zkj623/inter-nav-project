from pathlib import Path
import json,numpy as np,h5py
class HeadPilotRecorder:
    def __init__(self,out,robot,primary):
        import cv2
        from omni.isaac.sensor import Camera
        self.out=Path(out);self.primary=primary;self.robot=robot;self.count=0;self.file=None;self.previous=None;self.last_rgb=None;self.cv2=cv2
        self.camera=Camera('/robot/head_link3/Head_Front_Camera',resolution=(640,400));self.camera.initialize();self.camera._frequency=-1
        self.camera.set_local_pose(translation=np.array([.10237,.02375,.10256]),orientation=np.array([-.5945104121873723,-.3868316461705513,.37872914728340207,.5945442627086478]),camera_axes='usd')
        import omni.usd
        from pxr import UsdGeom
        usd=UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath('/robot/head_link3/Head_Front_Camera'))
        usd.GetHorizontalApertureOffsetAttr().Set((320-318.7979431152)*20/305.6425170898)
        usd.GetVerticalApertureOffsetAttr().Set((202.4943847656-200)*20/305.6007080078)
        self.camera.set_focal_length(20.);self.camera.set_horizontal_aperture(20.*640/305.6425170898);self.camera.set_vertical_aperture(20.*400/305.6007080078);self.camera.set_clipping_range(.01,100.)
    def image(self,phase,where,rgb):self.cv2.imwrite(str(self.out/(f'head_{phase}_{where}.jpg')),self.cv2.cvtColor(rgb,self.cv2.COLOR_RGB2BGR))
    def capture(self,phase,world):
        if self.primary.count==self.count:return
        rgba=self.camera.get_rgba();assert rgba is not None and rgba.shape==(400,640,4),'HEAD_FRAME_MISSING'
        rgb=rgba[:,:,:3].astype(np.uint8).copy();now=float(world.current_time);frame=self.camera.get_current_frame();ct=float(frame.get('rendering_time',now));assert abs(now-ct)<=1/120+.0001,'HEAD_TIMESTAMP_MISMATCH'
        if self.file is None:
            self.file=h5py.File(self.out/'head_pilot.hdf5','w');f=self.file;f.attrs.update(scope='new realpose validation pilot, not added to the old frozen240',camera_names=json.dumps(['head_front']),joint_names=json.dumps(self.primary.names),control_hz=20,physics_hz=120)
            f.create_dataset('observations/images/head_front',shape=(0,400,640,3),maxshape=(None,400,640,3),dtype='u1',chunks=(1,400,640,3),compression='lzf')
            for key in ['observations/qpos','observations/qvel','action']:f.create_dataset(key,shape=(0,16),maxshape=(None,16),dtype='f4',chunks=(128,16))
            for key in ['timestamps','camera_timestamps']:f.create_dataset(key,shape=(0,),maxshape=(None,),dtype='f8')
            f.create_dataset('phase',shape=(0,),maxshape=(None,),dtype=h5py.string_dtype('utf-8'))
        ids=[self.robot.dof_names.index(n) for n in self.primary.names]
        values={'observations/images/head_front':rgb,'observations/qpos':np.asarray(self.robot.get_joint_positions())[ids],'observations/qvel':np.asarray(self.robot.get_joint_velocities())[ids],'action':[self.primary.held[n] for n in self.primary.names],'timestamps':now,'camera_timestamps':ct,'phase':phase}
        for key,value in values.items():ds=self.file[key];ds.resize(self.count+1,axis=0);ds[self.count]=value
        if phase!=self.previous:
            if self.previous is not None:self.image(self.previous,'last',self.last_rgb)
            self.image(phase,'first',rgb)
        self.previous=phase;self.last_rgb=rgb;self.count+=1
    def finish(self,result):
        if self.file is None:return
        self.image(self.previous,'last',self.last_rgb);self.file.attrs['harness_success']=bool(result.get('success'));self.file.flush();self.file.close()
        (self.out/'head_pilot_metadata.json').write_text(json.dumps(dict(scope='new realpose synchronized head-camera pilot; not old240data',frames=self.count,camera_intrinsics={'fx':305.6425170898,'fy':305.6007080078,'cx':318.7979431152,'cy':202.4943847656},resolution=[640,400],camera_prim='/robot/head_link3/Head_Front_Camera',head_recorder_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),head_joint_deg=float(np.degrees(result['plan']['q_start']['idx13_head_joint3'])),control_hz=20,physics_hz=120,action_semantics='pre-action observation, absolute16rad position targets held6ticks',harness_success=result.get('success')),indent=2))
