"""Original-scale Go2 navigation with sideways parking of blocking cartons."""
import argparse,json,math,time,traceback,hashlib,shutil
from pathlib import Path
from dataclasses import replace
import numpy as np


def yaw(q):
    w,x,y,z=np.array(q)/np.linalg.norm(q)
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def angle(a):return (a+math.pi)%(2*math.pi)-math.pi


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-dir',type=Path,default=Path('/data20t/embodied/zkj/inter-nav-results/go2-floor01-20261024'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task-ids',default='T01,T02,T03,T04,T05')
    parser.add_argument('--gpu',type=int,default=7)
    parser.add_argument('--max-steps',type=int,default=30000)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--record-video',action='store_true')
    parser.add_argument('--third-person-height',type=float,default=1.35)
    parser.add_argument('--third-person-forward',type=float,default=.25)
    parser.add_argument('--third-person-fov',type=float,default=100.)
    parser.add_argument('--local-map-hz',type=float,default=10.)
    parser.add_argument('--global-map-hz',type=float,default=1.)
    parser.add_argument('--local-map-radius',type=float,default=3.)
    parser.add_argument('--record-every',type=int,default=1)
    parser.add_argument('--floor-top',type=float,default=.01,
                        help='Auxiliary collision-floor top in world metres (default: 1 cm)')
    parser.add_argument('--route-planner',choices=['voronoi','grid'],default='voronoi')
    parser.add_argument('--navigation-heading',choices=['path','fixed'],default='path')
    parser.add_argument('--path-controller',choices=['pure','guarded'],default='pure')
    parser.add_argument('--route-clearance',type=float,default=.30)
    parser.add_argument('--roadmap-shape',choices=['circle','footprint'],default='circle')
    parser.add_argument('--navigation-half-width',type=float,default=.22)
    parser.add_argument('--voronoi-clearance-weight',type=float,default=.30)
    parser.add_argument('--push-max-speed',type=float,default=.85,
                        help='Maximum forward policy request during box contact (m/s)')
    parser.add_argument('--turn-speed',type=float,default=.50,
                        help='Pure alignment yaw command (rad/s)')
    parser.add_argument('--control-config',type=Path,
                        help='Optional live JSON overrides for push_max_speed and turn_speed')
    parser.add_argument('--contact-yaw-tolerance',type=float,default=.04)
    parser.add_argument('--contact-centering-speed',type=float,default=.30)
    parser.add_argument('--contact-front-gap',type=float,default=.26)
    parser.add_argument('--calibrate-turn-before-task',action='store_true',
                        help='Run independent physical yaw trials, then reset and start the actual task')
    parser.add_argument('--material-mode',choices=['simple','preview'],default='preview',
                        help='Simple materials avoid compiling thousands of scene MDL instances')
    parser.add_argument('--start-override',type=float,nargs=2,help='Isolated contact calibration; use with one task ID')
    args=parser.parse_args()
    if not np.isfinite(args.floor_top) or not 0 < args.floor_top <= .30:
        raise ValueError('floor-top must be within (0, 0.30] m')
    floor_top=args.floor_top
    if not 1.0<=args.third_person_height<=2.5:raise ValueError('third-person-height must be in [1.0,2.5] m')
    if not -.3<=args.third_person_forward<=.6 or not 60<=args.third_person_fov<=110:
        raise ValueError('Invalid third-person framing parameters')
    if not 1<=args.local_map_hz<=20 or not .1<=args.global_map_hz<=args.local_map_hz:
        raise ValueError('Map rates must satisfy 0.1 <= global <= local <= 20 Hz')
    if not np.isfinite(args.route_clearance) or not .20<=args.route_clearance<=.45:
        raise ValueError('route-clearance must be within 0.20-0.45 m')
    if not .20<=args.navigation_half_width<=.32:raise ValueError('navigation-half-width must be in [0.20,0.32] m')
    if not 0<=args.voronoi_clearance_weight<=2:raise ValueError('voronoi-clearance-weight must be in [0,2]')
    spawn_height=floor_top+.40
    if args.record_every != 1:
        raise ValueError('Complete task videos capture every rendered frame; use --record-every 1')
    if args.calibrate_turn_before_task and not args.record_video:
        raise ValueError('Yaw calibration records every render; enable --record-video')
    if not .30 <= args.push_max_speed <= 1.0:
        raise ValueError('push-max-speed must be within 0.30-1.00 m/s')
    if not .25 <= args.turn_speed <= 1.0:
        raise ValueError('turn-speed must be within 0.25-1.00 rad/s')
    if not .02<=args.contact_yaw_tolerance<=.10 or not .20<=args.contact_centering_speed<=.50:
        raise ValueError('Contact alignment parameters exceed calibrated bounds')
    if not .10<=args.contact_front_gap<=.45:raise ValueError('contact-front-gap must be within 0.10-0.45 m')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():raise ValueError('Use a new output directory')
    from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
    from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
    from grutopia_extension.interactive_navigation.go2_exploration_tracking import ExplorationPathTracker as Go2VelocityPathTracker
    from grutopia_extension.interactive_navigation.go2_exploration_tracking import decision_buffer_recovery
    from grutopia_extension.interactive_navigation.go2_contact_control import (
        nose_aligned_body_target,approach_alignment_ready,contact_centering_velocity,contact_braking_required,contact_pose_ready,safe_contact_retreat,push_centerline_velocity,ContactAlignmentGate)
    from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
        ExplorationVoronoiRoutes,ForwardExplorationPathTracker,update_task_cameras,body_rotation_clear,body_motion_clear,navigation_body_blocked,body_translation_recovery_clear)
    from grutopia_extension.interactive_navigation.layered_navigation_map import LayeredNavigationMap
    from grutopia_extension.interactive_navigation.go2_rollout_health import Go2RolloutHealth
    from grutopia_extension.interactive_navigation.local_path_guidance import local_guidance_path
    from scipy.ndimage import distance_transform_edt
    from grutopia.demo.go2_semantic_exploration import build_grscene_floor
    from grutopia_extension.interactive_navigation.mapping_runtime import MapNavigationRuntime
    from grutopia_extension.interactive_navigation.point_navigation_profiles import load_point_navigation_profile
    from grutopia_extension.interactive_navigation.go2_navigation_runner import build_go2_navigation_config,Go2NavigationRunConfig
    profile=load_point_navigation_profile(str(args.benchmark_dir/'scene_profile.json'))
    layout=json.loads((args.benchmark_dir/'layout.json').read_text())['objects']
    original_metadata=json.loads((args.benchmark_dir/'original_metadata.json').read_text())
    grid=np.load(args.benchmark_dir/'navigation_grid.npz')
    planner=SidePushPlanner(grid['free'],[profile.mapping.x_limits,profile.mapping.y_limits],
        passages=json.loads((args.benchmark_dir/'passages.json').read_text()))
    if 'box_free' in grid:planner.box_free=grid['box_free']
    planner.navigation_route_radius=args.route_clearance
    planner.navigation_half_width=args.navigation_half_width
    planner.voronoi_clearance_weight=args.voronoi_clearance_weight
    planner.full_navigation_footprint=True
    planner.layered_local_control=True
    base_static_free=planner.free.copy()
    routes=ExplorationVoronoiRoutes(planner);layers=None
    def ordinary_path(start,goal,boxes,local=False):
        def plan(view_boxes):
            return routes.plan(start,goal,view_boxes) if args.route_planner=='voronoi' else planner.path(start,goal,view_boxes)
        if layers is not None and local and layers.global_path:
            path,detail=local_guidance_path(planner,start,layers.global_path,boxes,
                planner.robot_heading if planner.robot_heading is not None else 0.,args.local_map_radius)
            routes.last_plan={**detail,'map_scope':'local_dynamic'}
            layers.local_path=detail.get('local_prefix',())
            layers.local_replans+=bool(path)
            return path
        if layers is not None and not local:
            with layers.global_planning() as global_boxes:path=plan(global_boxes)
        else:path=plan(boxes)
        if layers is not None:
            if local:layers.local_path=path
            else:layers.global_path=path;layers.local_path=();layers.guidance_revision=layers.global_revision
        routes.last_plan['map_scope']='local_dynamic' if local else 'global_guidance'
        return path
    def ordinary_tracker(start,path,fixed_heading):
        return (ForwardExplorationPathTracker(start,path,guard_motion=args.path_controller=='guarded') if args.navigation_heading=='path'
            else Go2VelocityPathTracker(start,path,fixed_heading=fixed_heading))
    tasks=[t for t in json.loads((args.benchmark_dir/'tasks_preflight.json').read_text())['tasks']
           if t['id'] in args.task_ids.split(',')]
    if not tasks or any(t['decision']=='failed' for t in tasks):raise ValueError('Rejected task configuration')
    if args.start_override:
        if len(tasks)!=1:raise ValueError('start-override requires one task')
        tasks[0]={**tasks[0],'start':args.start_override,'calibration_only':True}
    (args.output/'tasks.json').write_text(json.dumps({'robot':'go2','scale':1.0,'tasks':tasks},indent=2))
    if args.preflight_only:return 0
    repo=Path(__file__).resolve().parents[2]
    manifest={}
    for rel in ['grutopia/demo/go2_side_clearance_navigation.py','grutopia_extension/interactive_navigation/side_push_navigation.py',
        'grutopia_extension/interactive_navigation/box_push_navigation.py','grutopia_extension/controllers/isaac_go2_ctrl.py',
        'grutopia_extension/interactive_navigation/go2_path_tracking.py','grutopia_extension/interactive_navigation/task_video_recording.py',
        'grutopia_extension/interactive_navigation/carton_materials.py',
        'grutopia_extension/interactive_navigation/go2_exploration_tracking.py',
        'grutopia_extension/interactive_navigation/go2_contact_control.py',
        'grutopia_extension/interactive_navigation/go2_voronoi_navigation.py',
        'grutopia_extension/interactive_navigation/layered_navigation_map.py',
        'grutopia_extension/interactive_navigation/go2_rollout_health.py',
        'grutopia_extension/interactive_navigation/local_path_guidance.py',
        'grutopia_extension/interactive_navigation/local_turn_maneuver.py',
        'grutopia_extension/interactive_navigation/semantic_voronoi.py',
        'grutopia_extension/controllers/go2_policy.py',
        'grutopia_extension/interactive_navigation/go2_turn_calibration.py',
        'grutopia/demo/go2_semantic_exploration.py',
        'grutopia_extension/interactive_navigation/mapping_runtime.py',
        'grutopia_extension/interactive_navigation/go2_navigation_runner.py',
        'grutopia_extension/interactive_navigation/visualization.py',
        'grutopia_extension/configs/robots/go2.py','grutopia_extension/robots/go2.py']:
        target=args.output/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repo/rel,target);manifest[rel]=hashlib.sha256((repo/rel).read_bytes()).hexdigest()
    (args.output/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    profile=replace(profile,start=(*tasks[0]['start'],spawn_height),goals=tuple((*t['goal'],floor_top+.32) for t in tasks),
                    success_distance=.20)
    import torch
    # A single robot uses tiny batches; a large CPU thread pool adds latency.
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(2)
    from grutopia.core.runtime import SimulatorRuntime
    from grutopia.core.gym_env import Env
    from grutopia_extension import import_extensions
    from grutopia_extension.interactive_navigation.point_navigation import PointNavigationComponent,PointNavigationConfig
    from grutopia_extension.interactive_navigation.scene_paths import configure_scene_mdl_paths
    config=Go2NavigationRunConfig(gpu=args.gpu,ground_height=floor_top,
        max_steps=args.max_steps,rendering_interval=4,use_fabric=False,lidar_max_range=5.0,
        material_mode=args.material_mode,generate_fallback_asset=False)
    configure_scene_mdl_paths(profile.scene_asset_path)
    simulation_config=build_go2_navigation_config(profile,config,objects=build_grscene_floor(profile,floor_top=floor_top))
    runtime=SimulatorRuntime(config_class=simulation_config,headless=True,
        active_gpu=args.gpu,physics_gpu=args.gpu)
    env=None;results=[];recorder=None;camera_trace=None
    try:
        import_extensions(('controllers','objects','robots','sensors','tasks'))
        env=Env(runtime);obs,_=env.reset()
        from omni.isaac.core.utils.stage import get_current_stage
        from omni.isaac.core.prims import RigidPrim
        from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf,Sdf
        stage=get_current_stage()
        from grutopia_extension.interactive_navigation.go2_navigation_runner import _apply_high_friction_material
        _apply_high_friction_material('/World/env_0/objects/grscene_go2_floor')
        active=next(iter(env.runner.current_tasks.values()));robot=next(iter(active.robots.values()))
        from grutopia_extension.interactive_navigation.carton_materials import restore_carton_materials
        material_audit=restore_carton_materials(stage,'/World/env_0/scene',[o['id'] for o in layout])
        (args.output/'carton_material_audit.json').write_text(json.dumps(material_audit,indent=2))
        print(json.dumps({'event':'carton_materials_restored','boxes':len(material_audit)}),flush=True)
        bodies={}
        for prim in stage.Traverse():
            if '/InteractiveBoxes/B' in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                assert prim.GetAttribute('physics:kinematicEnabled').Get() is False
                body=RigidPrim(str(prim.GetPath()),name='tracked_'+prim.GetName());body.initialize()
                bodies[prim.GetName()]=body
        assert set(bodies)=={o['id'] for o in layout}
        first_camera=third_camera=None;recording_enabled=False;recording_tick=0;record_phase='start';record_component=None;task_health=None
        if args.record_video:
            from grutopia_extension.interactive_navigation.visualization import FollowingRobotCamera
            from grutopia_extension.interactive_navigation.go2_navigation_runner import GO2_FIRST_PERSON_OFFSET,GO2_FIRST_PERSON_PITCH_DEGREES
            from grutopia_extension.interactive_navigation.task_video_recording import TaskVideoRecorder
            from pxr import UsdLux
            for p in ['/World/env_0/scene/BaseScene/Meshes/Base/ceiling']:
                prim=stage.GetPrimAtPath(p)
                if prim:UsdGeom.Imageable(prim).MakeInvisible()
            light=UsdLux.DomeLight.Define(stage,'/World/Go2NavigationVideoDome');light.CreateIntensityAttr(900)
            print(json.dumps({'event':'video_camera_initialization','material_mode':args.material_mode}),flush=True)
            first_camera=FollowingRobotCamera(resolution=(1280,736),offset=GO2_FIRST_PERSON_OFFSET,pitch_degrees=GO2_FIRST_PERSON_PITCH_DEGREES,
                enable_bounding_boxes=False,prim_path='/World/Go2TaskFirstCamera')
            third_camera=FollowingRobotCamera(resolution=(1280,736),offset=(args.third_person_forward,0.,args.third_person_height),pitch_degrees=90,
                horizontal_fov_degrees=args.third_person_fov,enable_bounding_boxes=False,prim_path='/World/Go2TaskThirdCamera')
            print(json.dumps({'event':'video_cameras_ready'}),flush=True)
            (args.output/'recording_config.json').write_text(json.dumps({
                'full_task_recording':True,'capture':'every rendered observation','fps':25,
                'physics_dt':.005,'rendering_dt':.020,'rendering_interval':4,
                'effective_env_ticks_per_render':5,'scene_scale':list(profile.scene_scale),
                'material_mode':args.material_mode,'ground_height':floor_top,
                'spawn_base_height':spawn_height,
                'lidar_max_range':config.lidar_max_range,
                'push_max_speed':args.push_max_speed,
                'turn_speed':args.turn_speed,
                'contact_yaw_tolerance':args.contact_yaw_tolerance,
                'contact_centering_speed':args.contact_centering_speed,
                'contact_front_gap_m':args.contact_front_gap,'contact_braking_horizon_s':1.0,
                'contact_centering_axes':'lateral correction before longitudinal approach',
                'live_control_config':str(args.control_config) if args.control_config else None,
                'third_person_offset':[args.third_person_forward,0.,args.third_person_height],'third_person_pitch_degrees':90,
                'third_person_fov_degrees':args.third_person_fov,
                'global_map_hz':args.global_map_hz,'local_map_hz':args.local_map_hz,
                'local_map_radius_m':args.local_map_radius,
                'third_person_orientation':'dog forward up, actual yaw following',
                'route_planner':args.route_planner,'navigation_heading':args.navigation_heading,
                'path_controller':args.path_controller,
                'route_clearance_m':args.route_clearance,
                'roadmap_shape':args.roadmap_shape,
                'navigation_half_width_m':args.navigation_half_width,
                'voronoi_clearance_weight_m':args.voronoi_clearance_weight,
                'local_collision_model':'actual_yaw_full_footprint_and_motion_sweep',
                'ordinary_forward_speed_law':'exploration_cosine_squared',
                'first_person_offset':list(GO2_FIRST_PERSON_OFFSET),
                'first_person_pitch_degrees':GO2_FIRST_PERSON_PITCH_DEGREES,
                'episode_start':'after initialization standby, before the first decision',
                'episode_end':'after success/failure and final stationary frames'},indent=2))
        def box_states():
            states=[]
            for o in layout:
                p,q=bodies[o['id']].get_world_pose();w,x,y,z=np.array(q)/np.linalg.norm(q)
                matrix=np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                    [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                    [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
                dimensions=tuple(map(float,np.abs(matrix)@o['dimensions']))
                states.append(BoxState(o['id'],tuple(map(float,p[:2])),dimensions,0.,o['mass_kg']))
            return states
        def box_height_states():
            heights=[]
            for item in layout:
                p,q=bodies[item['id']].get_world_pose();w,x,y,z=np.asarray(q)/np.linalg.norm(q)
                vertical=np.array([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)])
                half=float(np.dot(np.abs(vertical),item['dimensions'])/2)
                heights.append({'id':item['id'],'center_z_m':float(p[2]),'bottom_z_m':float(p[2]-half),
                                'top_z_m':float(p[2]+half),'bottom_above_aux_floor_m':float(p[2]-half-floor_top)})
            return heights
        def tick(action):
            nonlocal recording_tick
            if task_health is not None and not task_health.healthy:action=stop
            if first_camera:
                position,orientation=robot.get_world_pose()
                update_task_cameras(first_camera,third_camera,position,orientation)
            observation=env.step(action)[0]
            if task_health is not None:task_health.observe(observation['position'],observation['orientation'])
            if layers is not None:
                now=robot._go2_locomotion_control._simulation_time()
                now=float(now) if now is not None else layers.now+.008
                if now-layers.last_local>=layers.local_period-1e-8:
                    layers.update(now,observation,box_states())
                    if record_component is not None:layers.synchronize_display(record_component.mapping,step)
            if recording_enabled and recorder:
                if observation.get('render',False):
                    observation.setdefault('sensors',{})['recording_camera']=first_camera.get_data()
                    observation['sensors']['overview_camera']=third_camera.get_data()
                    from types import SimpleNamespace
                    recorder.capture(recording_tick,f"{recorder.task['id']} {record_phase}",observation,
                        SimpleNamespace(robot_position=tuple(map(float,observation['position']))),
                        record_component.mapping,target_context=record_component)
                    camera_trace.write(json.dumps({'frame_index':recorder.frame_count-1,'step':recording_tick,
                        'robot_yaw_rad':yaw(observation['orientation']),
                        'camera_yaw_rad':float(third_camera._smoothed_yaw)})+'\n')
                recording_tick+=1
            return observation
        stop={'move_by_speed':[0.,0.,0.]}
        def markers(task):
            for name,point,color in [('Start',task['start'],(.05,.75,.25)),('Goal',task['goal'],(.95,.1,.1))]:
                path='/World/TaskMarkers/'+name
                marker=UsdGeom.Cylinder.Define(stage,path);marker.CreateRadiusAttr(.10);marker.CreateHeightAttr(.012)
                xf=UsdGeom.Xformable(marker)
                if not xf.GetOrderedXformOps():xf.AddTranslateOp()
                xf.GetOrderedXformOps()[0].Set(Gf.Vec3d(point[0],point[1],floor_top+.015))
                marker.CreateDisplayColorAttr([Gf.Vec3f(*color)])
                marker.GetPrim().SetCustomDataByKey('task_id',task['id'])
        def crossed_passage(position,orientation,box_id):
            passage=planner.passages[box_id];axis=np.array(passage['axis'])
            door=original_metadata[passage['door_id']]
            boundary=max(np.dot(door['min_points'][:2],axis),np.dot(door['max_points'][:2],axis))
            heading=yaw(orientation);forward=np.array([math.cos(heading),math.sin(heading)])
            side=np.array([-forward[1],forward[0]])
            rear=max(.32*np.dot(forward,axis),-.50*np.dot(forward,axis))+.25*abs(np.dot(side,axis))
            return float(np.dot(position[:2],axis)-rear-boundary)>.02
        if args.calibrate_turn_before_task:
            from grutopia_extension.interactive_navigation.go2_turn_calibration import calibrate_turn_requests
            report=calibrate_turn_requests(robot,tick,third_camera,planner,box_states(),args.output/'turn_calibration',ground_height=floor_top)
            args.turn_speed=float(report['selected_turn_speed'])
            if args.control_config:
                controls=json.loads(args.control_config.read_text());controls['turn_speed']=args.turn_speed
                temporary=args.control_config.with_suffix('.tmp');temporary.write_text(json.dumps(controls,indent=2));temporary.replace(args.control_config)
            recording=json.loads((args.output/'recording_config.json').read_text());recording['turn_speed']=args.turn_speed
            recording['turn_calibration']='turn_calibration/summary.json'
            (args.output/'recording_config.json').write_text(json.dumps(recording,indent=2))
            print(json.dumps({'event':'turn_calibration_completed','selected_turn_speed':args.turn_speed}),flush=True)
        tuning_stamp=None
        for task in tasks:
            recording_enabled=False
            layers=None;record_component=None;task_health=None
            planner.free=base_static_free.copy()
            planner.static_clear=distance_transform_edt(planner.free)*planner.resolution>planner.radius
            routes=ExplorationVoronoiRoutes(planner)
            if first_camera:
                for camera in [first_camera,third_camera]:camera._smoothed_position=None;camera._smoothed_yaw=None
            folder=args.output/task['id'];folder.mkdir(exist_ok=True)
            events=[];samples=[];phase='decide';phase_start=0;plan=None;path=();push_count=0
            success=False;reason='global_step_limit';maximum_tilt=0.;start_time=time.time();initial_positions={}
            eventfile=(folder/'events.jsonl').open('w')
            def event(name,**values):
                row={'event':name,'task':task['id'],**values};events.append(row)
                eventfile.write(json.dumps(row)+'\n');eventfile.flush();print(json.dumps(row),flush=True)
            initial_heading={'T01':0.,'T02':math.pi,'T03':math.pi/2,'T04':0.,'T05':math.pi}[task['id']]
            planner.navigation_pose_heading=initial_heading if args.roadmap_shape=='footprint' else None
            nav_heading=initial_heading;turn_point=None;completed_box=None;passage_certificate=None
            art=robot.isaac_robot;root_p,_=art.get_world_pose();base_p,_=robot.get_world_pose()
            desired=np.array((*task['start'],spawn_height))
            art.set_world_pose(position=root_p+desired-base_p,
                orientation=np.array([math.cos(initial_heading/2),0.,0.,math.sin(initial_heading/2)]))
            art.set_joint_positions(art.get_joints_default_state().positions)
            art.set_joint_velocities(np.zeros(art.num_dof));robot.get_robot_base().set_linear_velocity(np.zeros(3))
            robot.get_robot_base().set_angular_velocity(np.zeros(3));robot._go2_locomotion_control.reset()
            for o in layout:
                a=math.radians(o['yaw_degrees'])/2;b=bodies[o['id']]
                box_position=np.array(o['position']);box_position[2]+=floor_top
                b.set_world_pose(position=box_position,orientation=np.array([math.cos(a),0,0,math.sin(a)]))
                b.set_linear_velocity(np.zeros(3));b.set_angular_velocity(np.zeros(3))
            markers(task)
            for _ in range(400):obs=tick(stop)
            p=np.array(obs['position']);q=np.array(obs['orientation']);q/=np.linalg.norm(q)
            tilt=math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
            startup_ok=(tilt<15 and floor_top+.24<p[2]<floor_top+.50 and np.linalg.norm(p[:2]-task['start'])<.25)
            event('startup',position=p.tolist(),tilt_degrees=tilt,passed=bool(startup_ok),boxes=[vars(b) for b in box_states()],
                  ground_height=floor_top,box_heights=box_height_states())
            if not startup_ok:
                results.append({**task,'success':False,'reason':'startup_unstable','position':p.tolist()});eventfile.close();continue
            task_health=Go2RolloutHealth(floor_top)
            component=PointNavigationComponent(PointNavigationConfig(goal=(*task['goal'],floor_top+.32),mapping=profile.mapping,
                safe_base_height=floor_top+.32,fall_height=floor_top+.20,success_distance=.20,max_steps=args.max_steps,
                use_scene_graph=False,prefer_voronoi_paths=True))
            record_component=component
            layers=LayeredNavigationMap(planner,profile.mapping,local_hz=args.local_map_hz,
                global_hz=args.global_map_hz,local_radius=args.local_map_radius)
            layers.update(float(robot._go2_locomotion_control._simulation_time() or 0.),obs,box_states(),force=True)
            layers.synchronize_display(component.mapping,0)
            if args.record_video:
                recorder=TaskVideoRecorder(str(folder/'video'),task=task,capture_interval=1,
                    fps=25,show_scene_graph=False,show_semantic_visuals=False,record_robot_topdown=False)
                record_component=component;recording_tick=0;recording_enabled=True
                camera_trace=(folder/'video/camera_heading.jsonl').open('w')
            release_start=None;planned_positions={};last_action=stop;last_tracking=None;tracker=None;realign_count=0;decision_recovery_start=None
            last_route_replan=-100
            decision_recovery_tracker=None
            contact_pause_until=-1;contact_stable_ticks=0;push_lateral_velocity=0.
            contact_alignment=ContactAlignmentGate()
            for step in range(args.max_steps):
                iteration_tracking=None
                if (args.output/'STOP').exists():reason='stop_requested';break
                if phase not in ['push_center','contact_brake']:contact_pause_until=-1;contact_stable_ticks=0
                if phase!='push_center':contact_alignment=ContactAlignmentGate()
                if args.control_config and step%200==0:
                    stamp=args.control_config.stat().st_mtime_ns
                    if stamp!=tuning_stamp:
                        tuning=json.loads(args.control_config.read_text())
                        values={key:float(tuning.get(key,getattr(args,key))) for key in
                                ['push_max_speed','turn_speed','contact_yaw_tolerance','contact_centering_speed']}
                        if not .30<=values['push_max_speed']<=1.0 or not .25<=values['turn_speed']<=1.0:
                            raise ValueError('Live controls exceed policy request bounds')
                        if not .02<=values['contact_yaw_tolerance']<=.10 or not .20<=values['contact_centering_speed']<=.50:
                            raise ValueError('Live contact controls exceed calibrated bounds')
                        for key,value in values.items():setattr(args,key,value)
                        tuning_stamp=stamp
                        event('control_parameters_updated',step=step,**values)
                        with (args.output/'control_parameter_history.jsonl').open('a') as history:
                            history.write(json.dumps({'task':task['id'],'step':step,**values})+'\n')
                record_phase=phase
                p=np.array(obs['position']);q=np.array(obs['orientation']);heading=yaw(q)
                q/=np.linalg.norm(q);tilt=math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
                maximum_tilt=max(maximum_tilt,tilt);current=box_states();by_id={b.id:b for b in current}
                distance=float(np.linalg.norm(p[:2]-task['goal']))
                if step%800==0:
                    layers.save(folder/'live')
                    (folder/'live'/'state.json').write_text(json.dumps({'step':step,'phase':phase,
                        'position':p.tolist(),'orientation':q.tolist(),'decision':layers.decision,
                        'tracking':last_tracking,'path':list(getattr(tracker,'_planned_path',()) or ()),
                        'waypoint_index':int(getattr(tracker,'_waypoint_index',0)),
                        'global_path':layers.global_path,'goal':task['goal']}))
                if step%8==0:
                    component.update(step,layers.filter_robot_returns(obs))
                    layers.synchronize_display(component.mapping,step)
                if phase not in ['navigate','approach']:component.mapping._planned_path=()
                if step%20==0:samples.append({'step':step,'phase':phase,'robot':p.tolist(),'orientation':q.tolist(),
                    'boxes':{b.id:list(b.xy) for b in current},'previous_action':last_action,'previous_tracking':last_tracking})
                if p[2]<floor_top+.20 or tilt>30 or not task_health.healthy:reason='robot_unstable';break
                planner.robot_heading=None
                crossed=completed_box is None or crossed_passage(p,q,completed_box)
                if distance<.20 and crossed and phase in ['navigate','decide','cross_passage'] and not planner.blocked(current)[planner.cell(task['goal'])]:
                    # Validate the pose after ALL stationary recording frames.
                    # Checking before an extra tail can report success even when
                    # the gait drifts outside the final goal tolerance afterwards.
                    record_phase='goal_settling'
                    for _ in range(260 if args.record_video else 200):obs=tick(stop)
                    if not task_health.healthy:reason='robot_unstable';break
                    crossed=completed_box is None or crossed_passage(np.array(obs['position']),np.array(obs['orientation']),completed_box)
                    settled_error=float(np.linalg.norm(np.array(obs['position'][:2])-task['goal']))
                    event('goal_settled',step=step,distance=settled_error,full_body_crossed=bool(crossed))
                    if settled_error<.25 and crossed:
                        success=True;reason='goal_reached';break
                    # The observation changed during settling. Start the next
                    # control iteration from that fresh pose instead of stale p.
                    last_action=stop
                    continue
                if phase=='decide':
                    recovery=None
                    if navigation_body_blocked(planner,current,heading)[planner.cell(p[:2])]:
                        if decision_recovery_tracker is None:
                            decision_recovery_tracker=ForwardExplorationPathTracker(p[:2],[(*task['goal'],0.)],guard_motion=False)
                        recovery=decision_recovery_tracker.action(p[:2],heading,planner,current)
                    else:decision_recovery_tracker=None
                    if recovery is None and args.roadmap_shape=='circle':
                        with layers.global_planning() as global_boxes:
                            attachment=decision_buffer_recovery(p[:2],heading,planner,global_boxes,task['goal'],heading)
                        if attachment is not None:
                            command=attachment[0]['move_by_speed']
                            if body_motion_clear(planner,current,p[:2],heading,command[0],command[2],horizon=.5,lateral=command[1]):
                                recovery=attachment;recovery[1]['controller']='rejoin_global_clearance'
                    if recovery is not None:
                        if decision_recovery_start is None:decision_recovery_start=step
                        action,tracking=recovery;record_phase='recover_navigation_buffer'
                        if step%100==0:event('decision_buffer_recovery',step=step,position=p.tolist(),**tracking,command=action['move_by_speed'])
                        if step-decision_recovery_start>2500:reason='decision_buffer_recovery_timeout';break
                        obs=tick(action);last_action=action;last_tracking=tracking;continue
                    decision_recovery_start=None
                    planner.robot_heading=heading if args.navigation_heading=='path' else nav_heading
                    layers.refresh_global(current,reason='decision')
                    with layers.global_planning() as global_boxes:
                        decision=planner.decide(p[:2],task['goal'],global_boxes)
                    layers.decision=decision.explanation
                    event('decision',step=step,kind=decision.kind,reason=decision.reason,
                          plan=vars(decision.push) if decision.push else None,explanation=decision.explanation,
                          global_map_revision=layers.global_revision,local_map_revision=layers.local_revision)
                    if decision.kind=='failed':reason=decision.reason;break
                    if decision.kind=='navigate':path=decision.path;phase='navigate'
                    else:plan=decision.push;path=plan.approach_path;phase='approach'
                    planner.robot_heading=heading if args.navigation_heading=='path' else nav_heading
                    endpoint=task['goal'] if phase=='navigate' else plan.approach_xy
                    path=ordinary_path(p[:2],endpoint,current)
                    if not path:reason='body_footprint_blocks_navigation';break
                    tracker=ordinary_tracker(p[:2],path,nav_heading)
                    event('ordinary_route',step=step,path=path,**(routes.last_plan if args.route_planner=='voronoi' else {'engine':'grid_astar'}))
                    phase_start=step;planned_positions={b.id:np.array(b.xy) for b in current};action=stop
                elif phase in ['navigate','approach']:
                    planner.robot_heading=heading if args.navigation_heading=='path' else nav_heading
                    if (step%200==0 and any(np.linalg.norm(np.array(b.xy)-planned_positions[b.id])>.08 for b in current)
                            and not planner.blocked(current)[planner.cell(p[:2])]):
                        phase='decide';obs=tick(stop);last_action=stop;continue
                    approach_error=abs(float(np.cross(np.array(plan.direction),p[:2]-plan.approach_xy))) if phase=='approach' else 0.
                    # The rough gait stalls under the generic 0.08 m/s lateral
                    # pose request. Hand the safe final approach to the contact
                    # stages, which center the nose and validate the settled pose.
                    # Contact has a prescribed push heading. Checking the goal
                    # footprint at the current walking yaw can reject a safe
                    # east/west-facing approach just because the final short
                    # navigation segment points north toward the door wall.
                    if (phase=='approach' and approach_alignment_ready(p[:2],plan.approach_xy,planner,
                            navigation_body_blocked(planner,current,math.atan2(plan.direction[1],plan.direction[0])))
                            and body_rotation_clear(planner,current,p[:2],heading,
                                heading+angle(math.atan2(plan.direction[1],plan.direction[0])-heading))):
                        event('approach_alignment_handoff',step=step,position=p.tolist(),
                              approach=list(plan.approach_xy),distance=float(np.linalg.norm(p[:2]-plan.approach_xy)))
                        # Use the same guarded preparation as every later
                        # realignment. A separate yaw/line-only align stage
                        # could walk into the box while turning and then keep
                        # commanding yaw with its front paws trapped against it.
                        phase='push_center';phase_start=step;action=stop;push_count+=1
                        contact_alignment=ContactAlignmentGate()
                        event('initial_contact_prepare',step=step,position=p.tolist())
                        event('side_push_begin',step=step,box_id=plan.box_id,passage=planner.active_passages[plan.box_id])
                    else:
                        action,tracking=tracker.action(p[:2],heading,planner,current)
                        if (tracking.get('needs_replan')
                                and (np.linalg.norm(p[:2]-endpoint)>.30 or tracking.get('buffer_cleared_replan'))
                                and step-last_route_replan>=100):
                            last_route_replan=step
                            replanned=ordinary_path(p[:2],endpoint,current,local=True)
                            if replanned:
                                tracker=ordinary_tracker(p[:2],replanned,nav_heading)
                                action,tracking=tracker.action(p[:2],heading,planner,current)
                                event('path_replanned_from_actual_pose',step=step,map_scope='local_dynamic',
                                      global_map_revision=layers.global_revision,local_map_revision=layers.local_revision)
                            elif layers.global_revision!=getattr(layers,'guidance_revision',layers.global_revision):
                                phase='decide';action=stop
                                event('global_replan_after_local_block',step=step,global_map_revision=layers.global_revision)
                        iteration_tracking=tracking
                        if tracking.get('recovering_buffer'):record_phase='navigation_buffer_recovery'
                        if tracking.get('local_turn_maneuver'):record_phase=tracking['controller']
                        component.mapping._planned_path=tracker._planned_path[tracker._waypoint_index:]
                        if step%400==0:event('path_tracking',step=step,**tracking)
                    if step-phase_start>16000:reason='navigation_timeout';break
                elif phase=='push':
                    box=by_id[plan.box_id];direction=np.array(plan.direction)
                    c,s=math.cos(heading),math.sin(heading)
                    body_v=robot._go2_locomotion_control.body_lin_vel[:2]
                    measured_world_v=np.array([c*body_v[0]-s*body_v[1],s*body_v[0]+c*body_v[1]])
                    tangent=np.array([-direction[1],direction[0]])
                    push_lateral_velocity=.92*push_lateral_velocity+.08*float(np.dot(measured_world_v,tangent))
                    moved=np.array(box.xy)-plan.initial_xy;along=float(np.dot(moved,direction))
                    side_error=float(np.cross(direction,moved));clearance=planner.clearance(box)
                    if abs(side_error)>.15:reason='box_left_safe_push_line';break
                    release=p[:2]-.45*direction
                    certificate=None
                    if step%20==0 and along>.25:
                        with layers.global_planning():
                            full_route=planner.path(task['start'],task['goal'],current)
                            release_route=planner.path(release,task['goal'],current) if full_route else ()
                        if release_route:
                            certificate=planner.certify_cleared_passage(box,full_route,task['start'],current)
                    if certificate:
                        layers.refresh_global(current,reason='side_push_cleared_passage')
                        phase='release';phase_start=step;release_start=p[:2].copy();action=stop
                        passage_certificate=certificate
                        event('side_push_end',step=step,box_id=box.id,position=list(box.xy),clearance=certificate,
                              nominal_center_clearance=clearance,actual_push_distance_m=along)
                    else:
                        desired_heading=math.atan2(direction[1],direction[0]);error=angle(desired_heading-heading)
                        tangent=np.array([-direction[1],direction[0]])
                        nose=p[:2]+.33*np.array([math.cos(heading),math.sin(heading)])
                        cross_error=float(np.dot(np.array(box.xy)-nose,tangent))
                        if abs(error)>.18 or abs(cross_error)>.06:
                            extent=float(np.dot(np.abs(direction),np.array(box.dimensions[:2]))/2)
                            front_gap=float(np.dot(np.array(box.xy)-nose,direction)-extent)
                            # Already separated from the box: center directly.
                            # An unnecessary retreat can enter furniture behind us.
                            # Front paws extend ~0.17 m beyond the nose. Any
                            # lateral/yaw correction needs their clearance too.
                            phase='contact_brake';contact_stable_ticks=0
                            phase_start=step;realign_origin=p[:2].copy()
                            realign_count+=1;action=stop
                            event('contact_realign_begin',step=step,heading_error=error,contact_offset=cross_error,
                                  front_gap=front_gap,retraction_required=front_gap<=.26)
                        else:
                            force_speed=min(args.push_max_speed,.30+.00045*(step-phase_start))
                            v=push_centerline_velocity(direction,force_speed,cross_error,push_lateral_velocity)
                            c,s=math.cos(heading),math.sin(heading)
                            action={'move_by_speed':[float(c*v[0]+s*v[1]),float(-s*v[0]+c*v[1]),float(np.clip(1.8*error,-.45,.45))]}
                        if along>plan.push_distance+.25:reason='safe_push_distance_exceeded';break
                        if step-phase_start>2000 and along<.025:reason='no_box_motion';break
                        if step-phase_start>8000:reason='push_timeout';break
                        if step%200==0:event('push_progress',step=step,position=p.tolist(),box=list(box.xy),clearance=clearance,
                            command=action['move_by_speed'],actual_body_velocity=robot._go2_locomotion_control.body_lin_vel.tolist())
                elif phase=='contact_brake':
                    # The rough gait continues moving after a zero request.
                    # Settle before changing from a forward push to lateral/yaw
                    # correction; otherwise that switch can drag a paw sideways
                    # into the carton or the wall behind the approach lane.
                    action=stop
                    speed=float(np.linalg.norm(robot._go2_locomotion_control.body_lin_vel[:2]))
                    contact_stable_ticks=contact_stable_ticks+1 if speed<.08 else 0
                    if contact_stable_ticks>=40:
                        box=by_id[plan.box_id];direction=np.asarray(plan.direction)
                        nose=p[:2]+.33*np.array([math.cos(heading),math.sin(heading)])
                        extent=float(np.dot(np.abs(direction),np.asarray(box.dimensions[:2]))/2)
                        front_gap=float(np.dot(np.asarray(box.xy)-nose,direction)-extent)
                        phase='push_retract' if front_gap<.30 else 'push_center'
                        phase_start=step;realign_origin=p[:2].copy()
                        event('contact_direction_switch_settled',step=step,front_gap=front_gap,next_phase=phase)
                    if step-phase_start>600:reason='contact_braking_timeout';break
                elif phase=='push_retract':
                    direction=np.array(plan.direction)
                    backed=float(np.dot(realign_origin-p[:2],direction))
                    box=by_id[plan.box_id]
                    extent=float(np.dot(np.abs(direction),np.array(box.dimensions[:2]))/2)
                    nose=p[:2]+.33*np.array([math.cos(heading),math.sin(heading)])
                    front_gap=float(np.dot(np.array(box.xy)-nose,direction)-extent)
                    if front_gap>=max(.30,args.contact_front_gap):
                        phase='push_center';phase_start=step;action=stop
                    else:
                        others=[b for b in current if b.id!=plan.box_id]
                        retreat=safe_contact_retreat(planner,others,p[:2],heading,direction,max(.30,args.contact_front_gap)-front_gap)
                        if retreat is None:reason='contact_retreat_blocked_by_other_geometry';break
                        v,retreat_target=retreat;c,s=math.cos(heading),math.sin(heading)
                        command=[float(c*v[0]+s*v[1]),float(-s*v[0]+c*v[1]),0.]
                        if step%100==0:event('contact_retreat_target',step=step,front_gap=front_gap,target=retreat_target.tolist())
                        action={'move_by_speed':command}
                    if backed>.70:reason='contact_retreat_distance_exceeded';break
                    if step-phase_start>1800:reason='contact_retraction_timeout';break
                elif phase=='push_center':
                    box=by_id[plan.box_id];direction=np.array(plan.direction)
                    extent=float(np.dot(np.abs(direction),np.array(box.dimensions[:2]))/2)
                    center_target=nose_aligned_body_target(box.xy,box.dimensions,direction,heading,front_gap=args.contact_front_gap)
                    desired_heading=math.atan2(direction[1],direction[0]);error=angle(desired_heading-heading)
                    delta=center_target-p[:2];centering_error=float(np.linalg.norm(delta))
                    c,s=math.cos(heading),math.sin(heading)
                    body_velocity=robot._go2_locomotion_control.body_lin_vel[:2]
                    world_velocity=np.array([c*body_velocity[0]-s*body_velocity[1],s*body_velocity[0]+c*body_velocity[1]])
                    brake=contact_braking_required(delta,world_velocity)
                    tangent=np.array([-direction[1],direction[0]])
                    nose=p[:2]+.33*np.array([math.cos(heading),math.sin(heading)])
                    contact_error=abs(float(np.dot(np.array(box.xy)-nose,tangent)))
                    front_gap=float(np.dot(np.array(box.xy)-nose,direction)-extent)
                    alignment_axis=contact_alignment.select(contact_error,error,args.contact_yaw_tolerance)
                    ready=alignment_axis is None and contact_pose_ready(p,heading,box.xy,box.dimensions,direction,args.contact_front_gap,args.contact_yaw_tolerance)
                    motion_mode='hold' if ready else (alignment_axis or 'longitudinal')
                    mode_settling=contact_alignment.settling_required(motion_mode,step)
                    # A turning gait has oscillating COM linear velocity.
                    # Braking on every such sample repeatedly cancelled yaw.
                    # Settle on transitions, then finish the same yaw action;
                    # retain predictive braking for translation toward a pose.
                    slowing=brake and alignment_axis!='yaw'
                    if front_gap<.26 and (contact_error>.035 or abs(error)>args.contact_yaw_tolerance):
                        phase='contact_brake';phase_start=step;contact_stable_ticks=0;action=stop
                        event('paw_clearance_before_alignment',step=step,front_gap=front_gap,contact_offset=contact_error)
                    elif ready or slowing or mode_settling or step<contact_pause_until:
                        if contact_pause_until<step and slowing:contact_pause_until=step+100
                        action=stop
                        contact_stable_ticks=contact_stable_ticks+1 if ready and np.linalg.norm(body_velocity)<.08 else 0
                        if contact_stable_ticks>=40 and not mode_settling:
                            phase='push';phase_start=step
                            event('contact_realign_end',step=step,position=p.tolist(),contact_offset=contact_error)
                    elif alignment_axis=='yaw':
                        contact_stable_ticks=0
                        action={'move_by_speed':[0.,0.,math.copysign(args.turn_speed,error)]}
                    else:
                        contact_stable_ticks=0
                        v=contact_centering_velocity(delta,direction,args.contact_centering_speed)
                        action={'move_by_speed':[float(c*v[0]+s*v[1]),float(-s*v[0]+c*v[1]),0.]}
                    if step%200==0:event('contact_center_tracking',step=step,position=p.tolist(),target=center_target.tolist(),
                        position_error=centering_error,contact_offset=contact_error,heading_error=error,command=action['move_by_speed'],
                        actual_body_velocity=body_velocity.tolist(),safe_contact_pose=bool(ready),pause_until=contact_pause_until,
                        alignment_axis=alignment_axis,motion_mode=motion_mode,mode_settling=bool(mode_settling))
                    if step-phase_start>2500:reason='contact_centering_timeout';break
                elif phase=='release':
                    direction=np.array(plan.direction);backed=float(np.dot(release_start-p[:2],direction))
                    if backed>=.45:
                        for _ in range(120):obs=tick(stop)
                        current=box_states();box=next(b for b in current if b.id==plan.box_id)
                        with layers.global_planning():
                            route=planner.path(np.array(obs['position'][:2]),task['goal'],current)
                            full=planner.path(task['start'],task['goal'],current)
                        clear=planner.certify_cleared_passage(box,full,task['start'],current)
                        event('passage_check',step=step,clearance=clear,route_available=bool(route),full_route_available=bool(full))
                        if not clear or not route or not full:reason='side_passage_not_clear';break
                        passage_certificate=clear
                        completed_box=box.id
                        nav_heading=math.atan2(plan.direction[1],plan.direction[0])
                        turn_point,path,turn_blocked=planner.turning_staging(np.array(obs['position'][:2]),box.id,current,nav_heading,
                            passage_center=clear['center'])
                        if not path:reason='no_safe_turning_staging_path';break
                        # Withdraw along the passage axis before moving toward
                        # its centerline, keeping the body away from the carton.
                        axis=np.array(planner.passages[box.id]['axis'])
                        origin=np.array(obs['position'][:2])
                        side_point=origin+np.dot(turn_point-origin,axis)*axis
                        first_leg=planner.path(origin,side_point,current)
                        second_leg=planner.path(side_point,turn_point,current)
                        if first_leg and second_leg:path=[*first_leg,*second_leg]
                        tracker=Go2VelocityPathTracker(obs['position'][:2],path,fixed_heading=nav_heading)
                        phase='turn_staging';phase_start=step;action=stop
                        event('turn_staging_selected',step=step,position=turn_point.tolist(),path=path)
                    else:
                        v=-.18*direction;c,s=math.cos(heading),math.sin(heading)
                        desired_heading=math.atan2(direction[1],direction[0])
                        action={'move_by_speed':[float(c*v[0]+s*v[1]),float(-s*v[0]+c*v[1]),float(np.clip(angle(desired_heading-heading)*.6,-.25,.25))]}
                    if step-phase_start>3000:reason='release_timeout';break
                elif phase=='turn_staging':
                    planner.robot_heading=nav_heading
                    if np.linalg.norm(p[:2]-turn_point)<.20 and not turn_blocked[planner.cell(p[:2])]:
                        turn_point=p[:2].copy()
                        for _ in range(120):obs=tick(stop)
                        phase='turn_to_passage';phase_start=step;action=stop
                    else:
                        action,tracking=tracker.action(p[:2],heading,planner,current)
                        command=action['move_by_speed']
                        error=angle(nav_heading-heading)
                        if tracking['recovering_buffer']:
                            # Recover translation first; an in-place correction
                            # must never cancel the command to leave an obstacle.
                            command[2]=0.
                        elif abs(error)>.10:
                            action={'move_by_speed':[0.,0.,math.copysign(args.turn_speed,error)]}
                        else:
                            # The policy stalls below ~0.16 m/s in lateral motion;
                            # mixed high lateral/yaw commands after contact are unstable.
                            limit=.20+.10*min((step-phase_start)/300,1.)
                            command[1]=float(np.clip(command[1],-limit,limit))
                            if abs(command[1])>.06:command[1]=math.copysign(limit,command[1])
                            if abs(command[0])<.03:command[0]=0.
                        if step%200==0:event('staging_tracking',step=step,position=p.tolist(),**tracking,command=action['move_by_speed'])
                    if step-phase_start>4000:reason='turn_staging_timeout';break
                elif phase=='turn_to_passage':
                    axis=np.array(planner.passages[completed_box]['axis'])
                    desired_heading=math.atan2(axis[1],axis[0]);error=angle(desired_heading-heading)
                    v=np.zeros(2);c,s=math.cos(heading),math.sin(heading)
                    rotation=math.copysign(args.turn_speed,error) if abs(error)>=.08 else 0.
                    action={'move_by_speed':[float(c*v[0]+s*v[1]),float(-s*v[0]+c*v[1]),rotation]}
                    if abs(error)<.08:
                        for _ in range(120):obs=tick(stop)
                        # Use a fresh collision-checked route through the opened
                        # doorway; blind lateral centering ignored nearby furniture.
                        nav_heading=desired_heading;phase='decide';phase_start=step;action=stop
                        if args.roadmap_shape=='footprint':planner.navigation_pose_heading=desired_heading
                        layers.refresh_global(box_states(),reason='replan_through_cleared_passage')
                        event('aligned_to_cleared_passage',step=step,position=p.tolist())
                    if step-phase_start>4000:reason='passage_alignment_timeout';break
                elif phase=='cross_passage':
                    passage=planner.passages[completed_box]
                    axis=np.array(passage['axis']);normal=np.array([-axis[1],axis[0]])
                    center=np.array(passage['center'])
                    lateral_error=float(np.dot(center-p[:2],normal))
                    heading_error=angle(nav_heading-heading)
                    if crossed:
                        phase='decide';action=stop
                        event('full_body_passage_crossed',step=step,position=p.tolist())
                    else:
                        # Center the body before advancing through the doorway.
                        # A small proportional sidestep is below the gait policy's
                        # effective speed and cannot correct lateral drift.
                        if abs(lateral_error)>.045:
                            v=.30*np.sign(lateral_error)*normal;rotation=0.
                        elif abs(heading_error)>.08:
                            v=np.zeros(2);rotation=math.copysign(args.turn_speed,heading_error)
                        else:
                            v=.24*axis;rotation=float(np.clip(heading_error*.7,-.12,.12))
                        planner.robot_heading=heading
                        blocked=planner.blocked(current)
                        target=p[:2]+v*.4
                        from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner
                        if not blocked[planner.cell(p[:2])] and not AStarMapPlanner.segment_clear(planner.cell(p[:2]),planner.cell(target),blocked):
                            reason='cleared_passage_tracking_blocked';break
                        c,s=math.cos(heading),math.sin(heading)
                        action={'move_by_speed':[float(c*v[0]+s*v[1]),float(-s*v[0]+c*v[1]),rotation]}
                        if step%200==0:event('passage_tracking',step=step,position=p.tolist(),lateral_error=lateral_error,command=action['move_by_speed'])
                    if step-phase_start>5000:reason='passage_traversal_timeout';break
                obs=tick(action);last_action=action;last_tracking=iteration_tracking
                if step%800==0:event('progress',step=step,phase=phase,position=list(map(float,obs['position'])),distance=distance)
            if not task_health.healthy:success=False;reason='robot_unstable'
            record_phase='goal_reached' if success else reason
            if not success:
                for _ in range(160 if args.record_video else 100):obs=tick(stop)
            final_q=np.array(obs['orientation']);final_q/=np.linalg.norm(final_q)
            final_tilt=math.degrees(math.acos(np.clip(1-2*(final_q[1]**2+final_q[2]**2),-1,1)))
            if success and (obs['position'][2]<floor_top+.20 or final_tilt>30):
                success=False;reason='goal_pose_unstable'
            if success and np.linalg.norm(np.array(obs['position'][:2])-task['goal'])>=.25:
                success=False;reason='goal_drifted_after_settling'
            result={**task,'success':success,'reason':reason,'steps':step+1,'push_count':push_count,
                'final_position':list(map(float,obs['position'])),'goal_error_m':float(np.linalg.norm(np.array(obs['position'][:2])-task['goal'])),
                'final_orientation':final_q.tolist(),'final_tilt_degrees':final_tilt,
                'maximum_tilt_degrees':max(maximum_tilt,task_health.max_tilt),'wall_seconds':time.time()-start_time,
                'rollout_health':task_health.report(),
                'contact_realignments':realign_count,
                'actual_path_length_m':float(sum(np.linalg.norm(np.array(b['robot'][:2])-a['robot'][:2]) for a,b in zip(samples,samples[1:]))),
                'final_boxes':[vars(b) for b in box_states()],'final_box_heights':box_height_states(),
                'ground_height':floor_top,'teleports_during_rollout':False,
                'ordinary_route_planning':routes.statistics(),
                'navigation_heading':args.navigation_heading,
                'path_controller':args.path_controller,
                'full_body_passage_crossed':(crossed_passage(np.array(obs['position']),np.array(obs['orientation']),completed_box)
                    if completed_box is not None else (False if task['decision']=='push' else None)),
                'go2_policy_inferences':int(robot._go2_locomotion_control.policy_inferences)}
            layers.save(folder)
            result['layered_mapping']=layers.statistics()
            result['passage_certificate']=passage_certificate
            if recorder:
                recording_enabled=False;camera_trace.close();camera_trace=None;recorder.close();recorder=None
                result['video_directory']=str(folder/'video')
            event('task_end',**result);eventfile.close()
            (folder/'result.json').write_text(json.dumps(result,indent=2));(folder/'trajectory.json').write_text(json.dumps(samples))
            results.append(result);(args.output/'summary.json').write_text(json.dumps({'robot':'go2','scale':1.0,'tasks':results,'complete':False},indent=2))
        (args.output/'summary.json').write_text(json.dumps({'robot':'go2','scale':1.0,'tasks':results,'complete':True,
            'success_count':sum(r['success'] for r in results),'total':len(results)},indent=2))
    except Exception as error:
        traceback.print_exc();(args.output/'error.json').write_text(json.dumps({'error':str(error),'traceback':traceback.format_exc()},indent=2));raise
    finally:
        if camera_trace:camera_trace.close()
        if recorder:recorder.close()
        if env is not None:env.close()
        runtime.simulation_app.close()
    return 0 if all(r['success'] for r in results) else 1


if __name__=='__main__':raise SystemExit(main())
