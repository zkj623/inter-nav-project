import sys,json,math,time,traceback,hashlib
from pathlib import Path
import numpy as np
OUT=Path('/data20t/embodied/zkj/g2-nav-multiview-20261119');ASSETS=Path('/data20t/embodied/zkj/inter-nav-results/g2-migration-20261111');ACCEPT=Path('/data20t/embodied/zkj/rsi-nav-private-clone-check-20261113');sys.path[:0]=[str(ACCEPT),str(ASSETS)]
task=json.loads((OUT/'task.json').read_text());result={'task':task['id'],'success':False,'reference_path_supplied':False,'offline_grid_supplied':False,'base_pose_teleport_after_reset':False,'actuation':'physical steering and wheel joint drives','scope':'single-task G2 adaptation of acceptance online mapping and goal selection'}
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':6,'physics_gpu':6,'multi_gpu':False})
samples=[];events=[];writer=None;video_frames=0;streams={};pts=np.empty((0,3))
try:
 import omni.usd,cv2
 from omni.isaac.sensor import Camera
 from omni.isaac.core.utils.viewports import set_camera_view
 from pxr import UsdLux,PhysxSchema,PhysicsSchemaTools
 from omni.isaac.core import World
 from omni.isaac.core.articulations import Articulation
 from omni.isaac.core.utils.types import ArticulationAction
 from grutopia_extension.sensors.physx_lidar import _SynchronizedRotatingLidarPhysX,_points_array,_transform_points
 from grutopia_extension.sensors.lidar_returns import exclude_robot_returns
 from grutopia_extension.interactive_navigation.mapping import MappingConfig
 from grutopia_extension.interactive_navigation.box_push_navigation import BoxPushPlanner
 from grutopia_extension.interactive_navigation.online_navigation_map import OnlineNavigationMap
 from grutopia_extension.interactive_navigation.online_navigation_decision import OnlineGoalSelector
 from g2_kinematics import Kinematics
 from pxr import UsdGeom
 omni.usd.get_context().open_stage(str(OUT/'scene.usda'))
 stage=omni.usd.get_context().get_stage()
 for prim in stage.Traverse():
  if prim.IsA(UsdGeom.Imageable) and 'NativeCollision' not in str(prim.GetPath()) and 'CollisionMesh' not in str(prim.GetPath()):UsdGeom.Imageable(prim).CreateVisibilityAttr('inherited')
 print('UPAXIS',UsdGeom.GetStageUpAxis(stage),flush=True)
 contacts=[]
 for prim in stage.Traverse():
  if str(prim.GetPath()).startswith('/robot/') and prim.GetName().startswith(('arm_','gripper_')):
   from pxr import UsdPhysics
   if prim.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
 def on_contact(headers,data):
  for h in headers:
   a=str(PhysicsSchemaTools.intToSdfPath(h.actor0));b=str(PhysicsSchemaTools.intToSdfPath(h.actor1))
   if a.startswith('/robot') != b.startswith('/robot'):
    contacts.append({'step':int(world.current_time_step_index),'actor0':a,'actor1':b})
 import omni.physx
 contact_sub=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
 light=UsdLux.DomeLight.Define(stage,'/World/ReviewLight');light.CreateIntensityAttr(1000.)
 for _ in range(10):app.update()
 world=World(physics_dt=1/120,rendering_dt=1/10,stage_units_in_meters=1.)
 robot=world.scene.add(Articulation('/robot',name='g2'))
 yaw=task['start_yaw_rad'];start=np.array(task['start_position_m']);start[2]=.04;quat=np.array([math.cos(yaw/2),0,0,math.sin(yaw/2)])
 robot.set_default_state(position=start,orientation=quat)
 # Same synchronized PhysX LiDAR implementation as the acceptance release.
 lidar=world.scene.add(_SynchronizedRotatingLidarPhysX(prim_path='/robot/base_link/NavLidar',name='nav_lidar',translation=np.array([0.,0.,.55]),rotation_frequency=0.,fov=(360.,60.),resolution=(.5,3.),valid_range=(.25,5.)))
 lidar.add_point_cloud_data_to_frame();lidar.enable_semantics();world.reset()
 from scipy.spatial.transform import Rotation
 def aim(cam,eye,target):
  forward=np.asarray(target)-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right);up=np.cross(right,forward);qc=Rotation.from_matrix(np.column_stack([right,up,-forward])).as_quat();cam.set_world_pose(eye,np.r_[qc[3],qc[:3]],camera_axes='usd')
 cameras=[]
 for name in ('FirstPerson','ThirdPerson'):
  cam=Camera('/World/'+name,resolution=(640,360));cam.initialize();cam.set_clipping_range(.05,100.);cam.set_focal_length(1.5);cameras.append(cam)
 def update_cameras(pos,yaw):
  f=np.array([np.cos(yaw),np.sin(yaw),0.]);aim(cameras[0],pos+np.array([0,0,1.55])+.20*f,pos+np.array([0,0,1.35])+3*f);aim(cameras[1],pos+np.array([0,0,2.05])-.85*f,pos+np.array([0,0,.85])+.55*f)
 update_cameras(start,yaw)
 lamp=UsdLux.SphereLight.Define(stage,'/World/CameraLight');lamp.CreateIntensityAttr(1500.);lamp.CreateRadiusAttr(.1);lampop=UsdGeom.Xformable(lamp).AddTranslateOp();lampop.Set(tuple(start+np.array([0,0,1.7])))
 for prim in stage.Traverse():
  if 'HDR_Sphere' in str(prim.GetPath()) and prim.IsA(UsdGeom.Imageable):UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')

 writer=cv2.VideoWriter(str(OUT/'simulation-recording.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10.,(1280,720))
 assert writer.isOpened()
 for label in ('first_person','third_person','online_map','lidar_points'):
  streams[label]=cv2.VideoWriter(str(OUT/(label+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),10.,(640,360))
 (OUT/'sensor_frames').mkdir(exist_ok=True)
 names=robot.dof_names;k=Kinematics(ASSETS/'generated/g2_crs_omnipicker.urdf');pose=json.loads((OUT/'pose.json').read_text())
 q=np.array([pose.get(k.by_name[n]['mimic']['joint'],0)*float(k.by_name[n]['mimic'].get('multiplier',1))+float(k.by_name[n]['mimic'].get('offset',0)) if k.by_name[n]['mimic'] else pose.get(n,0) for n in names]);robot.set_joint_positions(q);robot.set_joint_velocities(np.zeros(len(names)))
 wheels=[i for i,n in enumerate(names) if 'wheel' in n and n.endswith('joint2')];steers=[i for i,n in enumerate(names) if 'wheel' in n and n.endswith('joint1')];active=[i for i,n in enumerate(names) if not k.by_name[n]['mimic'] and i not in wheels]
 assert len(wheels)==4 and len(steers)==4,(names,wheels,steers)
 kp=np.array([0 if i in wheels or k.by_name[n]['mimic'] else 100000 if 'body_joint' in n else 1000 for i,n in enumerate(names)],float);kd=np.array([0 if k.by_name[n]['mimic'] else 5000 if 'body_joint' in n else 100 for n in names],float)
 robot.get_articulation_controller().set_gains(kps=kp,kds=kd);robot._articulation_view.set_max_efforts(np.array([[0 if k.by_name[n]['mimic'] else k.by_name[n]['effort'] for n in names]]))
 # Bounds are a generic empty workspace, not a ground-truth occupancy prior.
 limits=task['map_limits'];res=.05;planner=BoxPushPlanner(np.ones((600,600),bool),limits,resolution=res,robot_radius=.36);config=MappingConfig(x_limits=tuple(limits[0]),y_limits=tuple(limits[1]),grid_resolution=res,robot_radius=.36,obstacle_height=(.12,1.8))
 layers=OnlineNavigationMap(planner,config,local_hz=10,global_hz=1);selector=OnlineGoalSelector();goal=np.array(task['goal_position_m'][:2]);path=[];endpoint=None;kind='wait';settled=0;last_progress=0;best=float('inf');reason='watchdog';t0=time.time()
 result['module_hashes']={n:hashlib.sha256((ACCEPT/'grutopia_extension/interactive_navigation'/n).read_bytes()).hexdigest() for n in ('online_navigation_map.py','online_navigation_decision.py','box_push_navigation.py','mapping.py')}
 result['sensor']='acceptance _SynchronizedRotatingLidarPhysX; 360x60 degrees, 0.5x3 degrees, 5 m range';result['g2_adaptations']=['wheel steering/drive backend','collision-screened navigation tucked arms initialization','0.36 m planning radius'];result['initial_map_observed_cells']=int(layers.observed.sum())
 def actuate(v,omega):
  targets=q.copy();vel=[];measured=robot.get_joint_positions();alignment=1.
  for idx in wheels:
   n=names[idx];steer=next(i for i in steers if names[i].split('_',1)[1]==n.split('_',1)[1].replace('joint2','joint1'));x=.23 if 'front' in n else -.23;y=.218 if 'lwheel' in n else -.218;vx=v-omega*y;vy=omega*x;a=math.atan2(vy,vx) if abs(vx)+abs(vy)>1e-8 else measured[steer];speed=math.hypot(vx,vy)/.07
   if a>math.pi/2:a-=math.pi;speed=-speed
   if a<-math.pi/2:a+=math.pi;speed=-speed
   targets[steer]=a;alignment=min(alignment,max(0,math.cos(a-measured[steer])));vel.append(speed)
  robot.apply_action(ArticulationAction(joint_positions=targets[active],joint_indices=np.array(active)))
  robot.apply_action(ArticulationAction(joint_velocities=np.array(vel)*alignment,joint_indices=np.array(wheels)))
 for step in range(120*240):
  if step%12==0:
   pos,orn=robot.get_world_pose();pos=np.array(pos);orn=np.array(orn);yaw=math.atan2(2*(orn[0]*orn[3]+orn[1]*orn[2]),1-2*(orn[2]**2+orn[3]**2));distance=float(np.linalg.norm(pos[:2]-goal));tilt=math.degrees(math.acos(float(np.clip(1-2*(orn[1]**2+orn[2]**2),-1,1))));speed=float(np.linalg.norm(robot.get_linear_velocity()[:2]));angular=float(np.linalg.norm(robot.get_angular_velocity()));now=step/120
   if not np.isfinite(pos).all() or tilt>20 or pos[2]<-.1:
    reason='robot_unstable';result['unstable_pose']={'position':pos.tolist(),'quaternion':orn.tolist(),'tilt':tilt,'step':step};print('UNSTABLE',result['unstable_pose'],flush=True);break
   frame=lidar.get_synchronized_frame()
   if frame:
    pts=_points_array(frame.get('point_cloud'));pts,removed=exclude_robot_returns(pts,frame['hit_paths'],'/robot');pts=_transform_points(pts,frame['position'],frame['orientation']);obs={'position':pos,'orientation':orn,'sensors':{'lidar':{'pointcloud':pts,'position':frame['position'],'physics_step':frame['physics_step'],'min_range':.25,'max_range':5.}}};layers.update(now,obs,[])
   if distance<best-.05:best=distance;last_progress=now
   if step%120==0 and now>=2 and distance>.3:
    decision,endpoint=selector.decide(layers,pos[:2],goal);kind=decision.kind;path=list(decision.path);events.append({'t':now,'kind':kind,'reason':decision.reason,'path':path,'observed':int(layers.observed.sum())});print('NAV',round(now,1),kind,'distance',round(distance,3),'observed',int(layers.observed.sum()),flush=True)
    if kind=='explore' and endpoint is not None and np.linalg.norm(pos[:2]-endpoint)<.35:selector.reached_frontier(endpoint,int(layers.observed.sum()))
   v=omega=0.
   if now>=2 and distance>.3 and path:
    pp=np.array(path)[:,:2];nearest=int(np.linalg.norm(pp-pos[:2],axis=1).argmin());target=pp[-1]
    for point in pp[nearest:]:
     if np.linalg.norm(point-pos[:2])>=.35:target=point;break
    delta=target-pos[:2];err=(math.atan2(delta[1],delta[0])-yaw+math.pi)%(2*math.pi)-math.pi;omega=float(np.clip(1.5*err,-.4,.4));v=.18*max(0,math.cos(err)) if abs(err)<.6 else 0.
   actuate(v,omega)
   samples.append({'t':now,'position':pos.tolist(),'yaw':yaw,'distance':distance,'linear_speed':speed,'angular_speed':angular,'tilt_deg':tilt,'arm_joint_max_error_rad':float(max(abs(robot.get_joint_positions()[i]-q[i]) for i,n in enumerate(names) if 'arm_' in n)),'command':[v,omega],'kind':kind,'observed':int(layers.observed.sum())})
   settled=settled+1 if distance<=.5 and speed<.03 and angular<.05 and now>2 else 0
   if settled>=10:result['success']=True;reason='goal_reached_and_stopped';break
   if now-last_progress>45:reason='no_goal_progress_45s';break
  if step%12==0:
   update_cameras(pos,yaw);lampop.Set(tuple(pos+np.array([0,0,1.7])))
  world.step(render=step%12==0)
  if step%12==0:
   panels=[]
   for cam in cameras:
    rgba=cam.get_rgba();im=np.zeros((360,640,3),np.uint8) if rgba is None or not rgba.size else cv2.cvtColor(rgba[:,:,:3].astype(np.uint8),cv2.COLOR_RGB2BGR);im=cv2.LUT(im,np.array([min(255,int((i/255.)**.67*255+7)) for i in range(256)],np.uint8));panels.append(im)
   grid=np.full((*layers.observed.shape,3),105,np.uint8);grid[layers.observed]=(235,235,235);grid[layers.global_occupied]=(50,50,50);grid[layers.local_occupied]=(40,65,220)
   def pix(xy):return tuple(np.clip(((np.array(xy)-np.array([limits[0][0],limits[1][0]]))/res).astype(int),0,599))
   for aa,bb in zip(path,path[1:]):cv2.line(grid,pix(aa[:2]),pix(bb[:2]),(255,130,20),2)
   cv2.circle(grid,pix(goal),5,(30,220,30),-1);cv2.circle(grid,pix(pos[:2]),4,(255,40,40),-1)
   # Crop the display around the robot; occupancy is generated only from online LiDAR.
   cx,cy=pix(pos[:2]);x0=max(0,min(440,cx-80));y0=max(0,min(440,cy-80));m=cv2.flip(grid[y0:y0+160,x0:x0+160],0);m=cv2.resize(m,(360,360),interpolation=cv2.INTER_NEAREST);mp=np.full((360,640,3),25,np.uint8);mp[:,:360]=m
   cv2.putText(mp,'Global: gray',(370,100),0,.5,(220,220,220),1);cv2.putText(mp,'Local: red',(370,130),0,.5,(60,80,240),1);cv2.putText(mp,'Goal: green',(370,160),0,.5,(50,220,50),1);cv2.putText(mp,'Route: blue',(370,190),0,.5,(255,150,30),1);panels.append(mp)
   pc=np.full((360,640,3),22,np.uint8)
   if len(pts):
    rel=pts[:,:3]-pos;cc,ss=np.cos(yaw),np.sin(yaw);local=np.column_stack((cc*rel[:,0]+ss*rel[:,1],-ss*rel[:,0]+cc*rel[:,1],rel[:,2]));uv=np.column_stack((320-local[:,1]*55,290-local[:,0]*40-local[:,2]*45)).astype(int);valid=(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=35)&(uv[:,1]<360);uv=uv[valid];colors=np.clip((local[valid,2]+.2)/2,0,1);pc[uv[:,1],uv[:,0]]=np.column_stack((255*(1-colors),210*np.ones(len(colors)),255*colors)).astype(np.uint8)
   cv2.circle(pc,(320,290),5,(255,255,255),-1);panels.append(pc)
   labels=('FIRST PERSON','FOLLOWING THIRD PERSON','ONLINE GLOBAL / LOCAL MAP','LIVE LIDAR POINT CLOUD')
   for im,label,key in zip(panels,labels,streams):
    cv2.rectangle(im,(0,0),(640,42),(12,12,12),-1);cv2.putText(im,label,(10,17),0,.48,(255,255,255),1);cv2.putText(im,f't={step/120:.1f}s  goal={distance:.2f}m  {kind}',(10,35),0,.45,(255,255,255),1);streams[key].write(im)
   writer.write(np.vstack((np.hstack(panels[:2]),np.hstack(panels[2:]))));video_frames+=1
   if step%120==0:np.savez_compressed(OUT/'sensor_frames'/f'{step:06d}.npz',points=pts,observed=layers.observed,global_occupied=layers.global_occupied,local_occupied=layers.local_occupied,position=pos,yaw=yaw)
 result.update(reason=reason,final_distance_m=distance,simulation_seconds=step/120,wall_seconds=time.time()-t0,final_speed_m_s=speed,final_angular_speed_rad_s=angular,maximum_tilt_deg=max([tilt]+[x['tilt_deg'] for x in samples]),scan_count=layers.scan_count,observed_cells=int(layers.observed.sum()),actual_path_length_m=sum(np.linalg.norm(np.array(b['position'][:2])-a['position'][:2]) for a,b in zip(samples,samples[1:])))
 result['arm_environment_contact_events']=len(contacts)
 (OUT/'contacts.json').write_text(json.dumps(contacts,indent=2))
 layers.save(OUT/'maps')
except Exception:
 result['error']=traceback.format_exc();print(result['error'],flush=True)
finally:
 if writer is not None:writer.release()
 for stream in streams.values():stream.release()
 result['video_frames']=video_frames;result['video_role']='new recorded rerun, not the original unrecorded trial'
 (OUT/'trajectory.json').write_text(json.dumps(samples,indent=2));(OUT/'events.json').write_text(json.dumps(events,indent=2));(OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True);app.close()
