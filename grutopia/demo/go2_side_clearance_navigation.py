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
    parser.add_argument('--path-controller',choices=['pure','guarded','continuous'],default='continuous')
    parser.add_argument('--map-source',choices=['prior','online'],default='online')
    parser.add_argument('--actuator-mode',choices=['implicit_pd','explicit_dc'],default=None,
                        help='Online mode defaults to per-physics-step DC torque control; legacy mode remains available')
    parser.add_argument('--forward-stop-feedback',action='store_true',
                        help='Use checked persistent velocity feedback after forward-navigation stops')
    parser.add_argument('--navigation-attitude-recovery',action='store_true',
                        help='Stop on rising navigation tilt, settle with the original policy, then replan')
    parser.add_argument('--route-clearance',type=float,default=.30)
    parser.add_argument('--roadmap-shape',choices=['circle','footprint'],default='circle')
    parser.add_argument('--navigation-half-width',type=float,default=.22)
    parser.add_argument('--voronoi-clearance-weight',type=float,default=.30)
    parser.add_argument('--push-start-speed',type=float,default=.60,
                        help='Forward policy request that initiates a contact stride (m/s)')
    parser.add_argument('--push-max-speed',type=float,default=.85,
                        help='Maximum forward policy request during box contact (m/s)')
    parser.add_argument('--turn-speed',type=float,default=.50,
                        help='Pure alignment yaw command (rad/s)')
    parser.add_argument('--control-config',type=Path,
                        help='Optional live JSON overrides for push_max_speed and turn_speed')
    parser.add_argument('--contact-yaw-tolerance',type=float,default=.04)
    parser.add_argument('--contact-centering-speed',type=float,default=.30)
    parser.add_argument('--contact-lateral-speed',type=float,default=.35,
                        help='Calibrated sideways stepping request during contact preparation (m/s)')
    parser.add_argument('--contact-front-gap',type=float,default=.26)
    parser.add_argument('--calibrate-turn-before-task',action='store_true',
                        help='Run independent physical yaw trials, then reset and start the actual task')
    parser.add_argument('--material-mode',choices=['simple','preview'],default='preview',
                        help='Simple materials avoid compiling thousands of scene MDL instances')
    parser.add_argument('--start-override',type=float,nargs=2,help='Isolated contact calibration; use with one task ID')
    parser.add_argument('--interaction-catalog',type=Path,default=Path('grutopia/demo/profiles/interactive_object_catalog.json'))
    parser.add_argument('--interaction-source',choices=['ground_truth','vision'],default='ground_truth',
                        help='Use known box eligibility from the layout, or require live visual classification')
    args=parser.parse_args()
    if args.actuator_mode is None:args.actuator_mode='explicit_dc' if args.map_source=='online' else 'implicit_pd'
    if args.forward_stop_feedback and (args.actuator_mode!='explicit_dc' or args.path_controller!='continuous'):
        raise ValueError('forward-stop-feedback requires continuous navigation and explicit_dc')
    from grutopia_extension.interactive_navigation.go2_stop_feedback import ForwardStopFeedback
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
    if not .30<=args.push_start_speed<=args.push_max_speed:
        raise ValueError('push-start-speed must be between 0.30 and push-max-speed')
    if not .25 <= args.turn_speed <= 1.0:
        raise ValueError('turn-speed must be within 0.25-1.00 rad/s')
    if not .02<=args.contact_yaw_tolerance<=.10 or not .20<=args.contact_centering_speed<=.50:
        raise ValueError('Contact alignment parameters exceed calibrated bounds')
    if not .10<=args.contact_front_gap<=.45:raise ValueError('contact-front-gap must be within 0.10-0.45 m')
    if not .35<=args.contact_lateral_speed<=.50:raise ValueError('contact-lateral-speed must be within 0.35-0.50 m/s')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():raise ValueError('Use a new output directory')
    from grutopia_extension.interactive_navigation.interaction_catalog import InteractionCatalog
    from grutopia_extension.interactive_navigation.rgb_object_recognition import RGBObjectRecognizer,ObjectGeometry
    from grutopia_extension.interactive_navigation.interaction_recognition_runtime import InteractionRecognitionRuntime
    from grutopia_extension.interactive_navigation.ground_truth_interaction import GroundTruthInteractionRuntime
    from grutopia_extension.interactive_navigation.interaction_skill import SkillRegistry,SkillContext
    from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill,PushAsideConfig
    from grutopia_extension.interactive_navigation.object_inspection import ObjectInspectionPlanner
    catalog=InteractionCatalog.load(args.interaction_catalog)
    recognition=None
    if args.interaction_source=='vision':
        recognizer=RGBObjectRecognizer(labels=catalog.classification_labels,proposal_query=catalog.proposal_query)
        recognizer.check_ready()
        recognition=InteractionRecognitionRuntime(recognizer,catalog,args.output)
    registry=SkillRegistry();registry.register(PushAsideSkill)
    from grutopia_extension.interactive_navigation.go2_contact_turn import ContactTurnControl,checked_contact_turn_command
    from grutopia_extension.interactive_navigation.go2_passage_guidance import PassageEntryGuide
    from grutopia_extension.interactive_navigation.go2_contact_control import damped_contact_centering_velocity,contact_heading_hold,ContactYawServo
    from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
    from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
    from grutopia_extension.interactive_navigation.go2_exploration_tracking import ExplorationPathTracker as Go2VelocityPathTracker
    from grutopia_extension.interactive_navigation.go2_exploration_tracking import decision_buffer_recovery
    from grutopia_extension.interactive_navigation.go2_contact_control import (
        nose_aligned_body_target,approach_alignment_ready,contact_centering_velocity,contact_axis_delta,contact_braking_required,contact_pose_ready,safe_contact_retreat,observed_contact_retreat,push_centerline_velocity,push_completion_state,ContactAlignmentGate,CONTACT_BRAKING_HORIZON_S)
    from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
        ExplorationVoronoiRoutes,ForwardExplorationPathTracker,update_task_cameras,body_rotation_clear,body_motion_clear,navigation_body_blocked,body_translation_recovery_clear)
    from grutopia_extension.interactive_navigation.layered_navigation_map import LayeredNavigationMap
    from grutopia_extension.interactive_navigation.go2_rollout_health import Go2RolloutHealth
    from grutopia_extension.interactive_navigation.go2_actuation import physics_steps_per_command
    from grutopia_extension.interactive_navigation.go2_passage_confirmation import observed_passage_routes
    from grutopia_extension.interactive_navigation.online_navigation_map import OnlineNavigationMap
    from grutopia_extension.interactive_navigation.online_navigation_decision import OnlineGoalSelector
    from grutopia_extension.interactive_navigation.smooth_go2_navigation import SmoothExplorationTracker,VelocitySlewLimiter,smooth_recovery_action
    from grutopia_extension.interactive_navigation.go2_interaction_motion import navigation_recovery_enabled,MeasuredStopGate,ObservedSubgoalArrival,checked_skill_brake
    from grutopia_extension.interactive_navigation.go2_attitude_recovery import NavigationAttitudeRecovery
    from grutopia_extension.interactive_navigation.local_path_guidance import local_guidance_path
    from scipy.ndimage import distance_transform_edt
    from grutopia.demo.go2_semantic_exploration import build_grscene_floor
    from grutopia_extension.interactive_navigation.mapping_runtime import MapNavigationRuntime
    from grutopia_extension.interactive_navigation.point_navigation_profiles import load_point_navigation_profile
    from grutopia_extension.interactive_navigation.go2_navigation_runner import build_go2_navigation_config,Go2NavigationRunConfig
    profile=load_point_navigation_profile(str(args.benchmark_dir/'scene_profile.json'))
    layout=json.loads((args.benchmark_dir/'layout.json').read_text())['objects']
    if args.interaction_source=='ground_truth':
        recognition=GroundTruthInteractionRuntime(catalog,args.benchmark_dir/'layout.json',args.output)
    if args.map_source=='online':
        # Bounds and requested S/G are known; architectural geometry is not.
        # The simulator asset/layout below is only the physical environment.
        profile=replace(profile,mapping=replace(profile.mapping,grid_resolution=.05,
            obstacle_height=(floor_top+.05,floor_top+.50),sticky_free_scale=.25,ray_traversal='exact'))
        from grutopia_extension.interactive_navigation.mapping import OccupancyGridMap
        blank=OccupancyGridMap(profile.mapping)
        planner=SidePushPlanner(np.zeros_like(blank.observed),[profile.mapping.x_limits,profile.mapping.y_limits],
            resolution=profile.mapping.grid_resolution,passages={})
        original_metadata={}
    else:
        original_metadata=json.loads((args.benchmark_dir/'original_metadata.json').read_text())
        grid=np.load(args.benchmark_dir/'navigation_grid.npz')
        planner=SidePushPlanner(grid['free'],[profile.mapping.x_limits,profile.mapping.y_limits],
            passages=json.loads((args.benchmark_dir/'passages.json').read_text()))
        if 'box_free' in grid:planner.box_free=grid['box_free']
    planner.interactable_ids=recognition.eligible_ids(0.)
    planner.navigation_route_radius=args.route_clearance
    planner.navigation_half_width=args.navigation_half_width
    planner.voronoi_clearance_weight=args.voronoi_clearance_weight
    planner.full_navigation_footprint=True
    planner.layered_local_control=True
    base_static_free=planner.free.copy()
    routes=ExplorationVoronoiRoutes(planner);layers=None;smooth_limiter=None;control_log=None
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
    def ordinary_tracker(start,path,fixed_heading,short_goal=False):
        if args.path_controller=='continuous':
            return SmoothExplorationTracker(start,path,smooth_limiter,final_goal_slow_radius=.30 if short_goal else .75)
        return (ForwardExplorationPathTracker(start,path,guard_motion=args.path_controller=='guarded') if args.navigation_heading=='path'
            else Go2VelocityPathTracker(start,path,fixed_heading=fixed_heading))
    tasks=[t for t in json.loads((args.benchmark_dir/'tasks_preflight.json').read_text())['tasks']
           if t['id'] in args.task_ids.split(',')]
    if args.map_source=='online':tasks=[{k:t[k] for k in ['id','start','goal']} for t in tasks]
    if not tasks or any(t.get('decision')=='failed' for t in tasks):raise ValueError('Rejected task configuration')
    if args.start_override:
        if len(tasks)!=1:raise ValueError('start-override requires one task')
        tasks[0]={**tasks[0],'start':args.start_override,'calibration_only':True}
    (args.output/'tasks.json').write_text(json.dumps({'robot':'go2','scale':1.0,'tasks':tasks},indent=2))
    if args.preflight_only:return 0
    repo=Path(__file__).resolve().parents[2]
    manifest={}
    shutil.copy2(args.interaction_catalog,args.output/'interaction_catalog.json')
    for rel in ['grutopia_extension/interactive_navigation/interaction_catalog.py',
        'grutopia_extension/interactive_navigation/rgb_object_recognition.py',
        'grutopia_extension/interactive_navigation/interaction_recognition_runtime.py',
        'grutopia_extension/interactive_navigation/ground_truth_interaction.py',
        'grutopia_extension/interactive_navigation/interaction_skill.py',
        'grutopia_extension/interactive_navigation/push_aside_skill.py',
        'grutopia_extension/interactive_navigation/object_inspection.py',
        'grutopia/demo/go2_side_clearance_navigation.py','grutopia_extension/interactive_navigation/side_push_navigation.py',
        'grutopia/demo/go2_online_interactive_navigation.py',
        'grutopia_extension/interactive_navigation/box_push_navigation.py','grutopia_extension/controllers/isaac_go2_ctrl.py',
        'grutopia_extension/interactive_navigation/go2_path_tracking.py','grutopia_extension/interactive_navigation/task_video_recording.py',
        'grutopia_extension/interactive_navigation/carton_materials.py',
        'grutopia_extension/interactive_navigation/material_fallback.py',
        'grutopia_extension/interactive_navigation/go2_exploration_tracking.py',
        'grutopia_extension/interactive_navigation/go2_contact_control.py',
        'grutopia_extension/interactive_navigation/go2_contact_turn.py',
        'grutopia_extension/interactive_navigation/go2_heading_trim.py',
        'grutopia_extension/interactive_navigation/go2_voronoi_navigation.py',
        'grutopia_extension/interactive_navigation/layered_navigation_map.py',
        'grutopia_extension/interactive_navigation/go2_rollout_health.py',
        'grutopia_extension/interactive_navigation/go2_attitude_recovery.py',
        'grutopia_extension/interactive_navigation/go2_interaction_motion.py',
        'grutopia_extension/interactive_navigation/go2_stop_feedback.py',
        'grutopia_extension/interactive_navigation/online_navigation_map.py',
        'grutopia_extension/interactive_navigation/online_lidar_evidence.py',
        'grutopia_extension/interactive_navigation/online_navigation_decision.py',
        'grutopia_extension/interactive_navigation/smooth_go2_navigation.py',
        'grutopia_extension/interactive_navigation/go2_recovery_settling.py',
        'grutopia_extension/interactive_navigation/local_staging_control.py',
        'grutopia_extension/interactive_navigation/go2_turn_response.py',
        'grutopia_extension/interactive_navigation/go2_passage_confirmation.py',
        'grutopia_extension/interactive_navigation/go2_passage_guidance.py',
        'grutopia_extension/interactive_navigation/go2_actuation.py',
        'grutopia_extension/controllers/go2_move_by_speed_controller.py',
        'grutopia_extension/configs/controllers/go2_move_by_speed_controller.py',
        'grutopia_extension/interactive_navigation/local_path_guidance.py',
        'grutopia_extension/interactive_navigation/local_turn_maneuver.py',
        'grutopia_extension/interactive_navigation/semantic_voronoi.py',
        'grutopia_extension/controllers/go2_policy.py',
        'grutopia_extension/interactive_navigation/go2_turn_calibration.py',
        'grutopia/demo/go2_semantic_exploration.py',
        'grutopia_extension/interactive_navigation/mapping_runtime.py',
        'grutopia_extension/interactive_navigation/go2_navigation_runner.py',
        'grutopia_extension/interactive_navigation/visualization.py',
        'grutopia_extension/configs/objects/__init__.py','grutopia_extension/objects/fixed_cube.py',
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
        actuator_mode=args.actuator_mode,
        lidar_resolution=(.5,2.5) if args.map_source=='online' else (2.,5.),
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
        native=robot.isaac_robot._articulation_view._physics_view
        native_gains={'stiffness':np.asarray(native.get_dof_stiffnesses()).tolist(),
                      'damping':np.asarray(native.get_dof_dampings()).tolist()}
        if args.actuator_mode=='explicit_dc' and any(np.any(np.asarray(v)!=0.) for v in native_gains.values()):
            raise RuntimeError('Explicit DC control cannot run with active implicit joint drives')
        (args.output/'actuator_contract.json').write_text(json.dumps({'mode':args.actuator_mode,
            'policy_frequency_hz':50,'physics_frequency_hz':200,
            'rendering_dt':simulation_config.simulator.rendering_dt,
            'rendering_interval':simulation_config.simulator.rendering_interval,**native_gains},indent=2))
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
        if args.record_video or args.interaction_source=='vision':
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
                enable_depth=args.interaction_source=='vision',enable_point_cloud_image=args.interaction_source=='vision',enable_semantics=False,
                enable_bounding_boxes=False,prim_path='/World/Go2TaskFirstCamera')
            third_camera=FollowingRobotCamera(resolution=(1280,736),offset=(args.third_person_forward,0.,args.third_person_height),pitch_degrees=90,
                horizontal_fov_degrees=args.third_person_fov,enable_bounding_boxes=False,prim_path='/World/Go2TaskThirdCamera') if args.record_video else None
            print(json.dumps({'event':'video_cameras_ready'}),flush=True)
            (args.output/'recording_config.json').write_text(json.dumps({
                'recognition_required':args.interaction_source=='vision','interaction_source':args.interaction_source,
                'interaction_skill':'push_aside/1.0',
                'full_task_recording':bool(args.record_video),'capture':'every rendered observation','fps':25,
                'physics_dt':.005,'rendering_dt':float(simulation_config.simulator.rendering_dt),
                'rendering_interval':simulation_config.simulator.rendering_interval,
                'effective_env_ticks_per_render':5,'scene_scale':list(profile.scene_scale),
                'actuator_mode':args.actuator_mode,
                'physics_steps_per_command_cycle':([1,2,1,2,2] if args.actuator_mode=='explicit_dc' else None),
                'physics_steps_per_render':8,
                'material_mode':args.material_mode,'ground_height':floor_top,
                'spawn_base_height':spawn_height,
                'lidar_max_range':config.lidar_max_range,
                'lidar_resolution_degrees':list(config.lidar_resolution),
                'push_max_speed':args.push_max_speed,'push_start_speed':args.push_start_speed,
                'turn_speed':args.turn_speed,
                'contact_yaw_tolerance':args.contact_yaw_tolerance,
                'contact_centering_speed':args.contact_centering_speed,
                'contact_lateral_speed':args.contact_lateral_speed,
                'contact_lateral_brake_horizon_s':.65 if args.path_controller=='continuous' else CONTACT_BRAKING_HORIZON_S,
                'contact_front_gap_m':args.contact_front_gap,'contact_braking_horizon_s':CONTACT_BRAKING_HORIZON_S,
                'contact_centering_axes':'lateral correction before longitudinal approach',
                'live_control_config':str(args.control_config) if args.control_config else None,
                'third_person_offset':[args.third_person_forward,0.,args.third_person_height],'third_person_pitch_degrees':90,
                'third_person_fov_degrees':args.third_person_fov,
                'global_map_hz':args.global_map_hz,'local_map_hz':args.local_map_hz,
                'local_map_radius_m':args.local_map_radius,
                'third_person_orientation':'dog forward up, actual yaw following',
                'route_planner':args.route_planner,'navigation_heading':args.navigation_heading,
                'path_controller':args.path_controller,
                'navigation_attitude_recovery':args.navigation_attitude_recovery,
                'forward_stop_feedback':args.forward_stop_feedback,
                'map_source':args.map_source,'initial_map':'unknown' if args.map_source=='online' else 'scene_prior',
                'uses_navigation_grid':args.map_source=='prior','uses_static_metadata':args.map_source=='prior',
                'normal_navigation_max_forward':.50 if args.path_controller=='continuous' else .40,
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
        def planning_boxes():
            return list(layers.current_boxes) if args.map_source=='online' and layers is not None else box_states()
        def save_lidar_sample(observation,path):
            lidar=observation.get('sensors',{}).get('lidar',{})
            np.savez_compressed(path,points=np.asarray(lidar.get('pointcloud',[]),dtype=np.float32),
                origin=np.asarray(lidar.get('position',observation['position'])),
                base_position=np.asarray(observation['position']),base_orientation=np.asarray(observation['orientation']),
                physics_step=int(lidar.get('physics_step',-1)),max_range=float(lidar.get('max_range',5.)),
                self_hits_removed=int(lidar.get('self_hits_removed',0)))
        command_tick_index=0;forward_stop_servo=None
        def tick(action):
            nonlocal recording_tick,command_tick_index
            if task_health is not None and not task_health.healthy:action=stop
            command_time=float(robot._go2_locomotion_control._simulation_time() or 0.)
            if smooth_limiter is not None:smooth_limiter.accept(action.get('move_by_speed',[0.,0.,0.]),command_time)
            microsteps=physics_steps_per_command(command_tick_index,args.actuator_mode)
            physical_commands=[];feedback_details=[]
            for _ in range(microsteps):
                healthy=task_health is None or task_health.healthy
                requested=stop if not healthy else action
                micro_action={k:v for k,v in requested.items() if k!='_forward_stop_request'}
                if forward_stop_servo is not None:
                    ctrl=robot._go2_locomotion_control;when=float(ctrl._simulation_time() or 0.)
                    def checked_feedback(target,limiter,check_time):
                        if layers is None:return np.zeros(3),False
                        position,orientation=robot.get_world_pose()
                        return checked_contact_turn_command(target,position,yaw(orientation),planner,planning_boxes(),limiter,check_time)
                    contact_phase=(phase=='skill' and skill is not None and skill.state in
                        {'engage','push','settle_push_progress','brake_after_push','brake_for_recovery'})
                    command=forward_stop_servo.apply(micro_action.get('move_by_speed',[0.,0.,0.]),
                        [*ctrl.body_lin_vel[:2],ctrl.body_ang_vel[2]],when,
                        stop_requested=bool(requested.get('_forward_stop_request',False)),
                        feedback_allowed=not contact_phase,healthy=healthy,checker=checked_feedback)
                    micro_action['move_by_speed']=command.tolist()
                    feedback_details.append(dict(forward_stop_servo.last_detail))
                physical_commands.append(list(micro_action.get('move_by_speed',[0.,0.,0.])))
                if first_camera:
                    position,orientation=robot.get_world_pose()
                    update_task_cameras(first_camera,third_camera,position,orientation)
                observation=env.step(micro_action)[0]
                if task_health is not None:task_health.observe(observation['position'],observation['orientation'])
            command_tick_index+=1
            now=float(robot._go2_locomotion_control._simulation_time() or 0.)
            if smooth_limiter is not None:
                controller=robot._go2_locomotion_control
                smooth_limiter.observe([*controller.body_lin_vel[:2],controller.body_ang_vel[2]],now)
            if layers is not None:
                now=robot._go2_locomotion_control._simulation_time()
                now=float(now) if now is not None else layers.now+.008
                if now-layers.last_local>=layers.local_period-1e-8:
                    layers.update(now,observation,box_states())
                    if record_component is not None:layers.synchronize_display(record_component.mapping,step)
            if control_log is not None:
                diagnostic=robot._go2_locomotion_control.get_obs()
                root_position,root_orientation=robot.isaac_robot.get_world_pose()
                control_log.write(json.dumps({'time_s':now,'command_time_s':command_time,'phase':record_phase,'command':action.get('move_by_speed'),
                    'policy_command':diagnostic['policy_command'].tolist(),
                    'body_velocity':diagnostic['body_lin_vel'].tolist(),'body_angular_velocity':diagnostic['body_ang_vel'].tolist(),
                    'position':list(map(float,observation['position'])),'orientation':list(map(float,observation['orientation'])),
                    'controller_root_position':list(map(float,root_position)),
                    'controller_root_orientation':list(map(float,root_orientation)),
                    'policy_inferences':diagnostic['policy_inferences'],
                    'actuator_mode':args.actuator_mode,'physics_substeps':microsteps,'physical_commands':physical_commands,
                    'forward_stop_requested':bool(action.get('_forward_stop_request',False)),
                    'forward_stop_feedback':feedback_details,
                    'observed_cells':int(layers.observed.sum()) if hasattr(layers,'observed') else None})+'\n')
            if layers is not None:
                recognized=recognition.poll()
                planner.interactable_ids=recognition.eligible_ids(now)
                layers.recognized_objects=recognition.snapshot(now)
                layers.interaction_source=recognition.source
                layers.interactable_ids=planner.interactable_ids
                if recognized is not None:
                    event('visual_recognition',time_s=now,details=recognized,eligible_ids=sorted(planner.interactable_ids))
            if observation.get('render',False) and layers is not None and first_camera is not None:
                geometry=[]
                if args.interaction_source=='vision':
                    for tracked in layers.current_boxes:
                        center,_=bodies[tracked.id].get_world_pose()
                        geometry.append(ObjectGeometry(tracked.id,(*tracked.xy,float(center[2])),tracked.dimensions))
                wants=recognition.wants_frame(now,geometry)
                packet=first_camera.get_data(include_geometry=wants)
                observation.setdefault('sensors',{})['recording_camera']=packet
                if wants and packet.get('rendering_frame') is not None:
                    recognition.submit(packet,geometry,packet['rendering_frame'],now)
            if recording_enabled and recorder:
                if observation.get('render',False):
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
            if args.map_source=='online':boundary=passage['boundary']
            else:
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
        if args.actuator_mode=='explicit_dc':env.runner.render_trigger=0
        command_tick_index=0
        for task in tasks:
            record_phase='start'
            step=0  # Sensor-map warmup runs before the first control iteration.
            recording_enabled=False
            layers=None;record_component=None;task_health=None
            planner.free=base_static_free.copy()
            planner.static_clear=distance_transform_edt(planner.free)*planner.resolution>planner.radius
            routes=ExplorationVoronoiRoutes(planner)
            if first_camera:
                for camera in [first_camera,third_camera]:
                    if camera is not None:camera._smoothed_position=None;camera._smoothed_yaw=None
            folder=args.output/task['id'];folder.mkdir(exist_ok=True)
            recognition.reset(task['id']);planner.interactable_ids=recognition.eligible_ids(0.)
            inspection_planner=ObjectInspectionPlanner();inspection=None;inspection_turn=None
            skill=None;skill_reports=[];subtracker=None;subtarget=None;sub_started=None;sub_reached=False
            control_log=(folder/'control_trace.jsonl').open('w',buffering=1)
            smooth_limiter=VelocitySlewLimiter() if args.path_controller=='continuous' else None
            forward_stop_servo=ForwardStopFeedback() if args.forward_stop_feedback else None
            online_selector=OnlineGoalSelector();goal_kind='task';online_wait_start=None;online_next_decision_time=0.
            if args.map_source=='online':planner.passages={};planner.active_passages={}
            events=[];samples=[];phase='decide';phase_start=0;plan=None;path=();push_count=0
            success=False;reason='global_step_limit';maximum_tilt=0.;start_time=time.time();initial_positions={}
            eventfile=(folder/'events.jsonl').open('w')
            def event(event_type,**values):
                row={'event':event_type,'task':task['id'],**values};events.append(row)
                eventfile.write(json.dumps(row)+'\n');eventfile.flush();print(json.dumps(row),flush=True)
            event('interaction_eligibility_source',source=recognition.source,known_interactable_ids=sorted(planner.interactable_ids))
            initial_heading={'T01':0.,'T02':math.pi,'T03':math.pi/2,'T04':0.,'T05':math.pi}[task['id']]
            planner.navigation_pose_heading=initial_heading if args.roadmap_shape=='footprint' else None
            nav_heading=initial_heading;turn_point=None;completed_box=None;passage_certificate=None;passage_guide=None
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
            map_class=OnlineNavigationMap if args.map_source=='online' else LayeredNavigationMap
            layers=map_class(planner,profile.mapping,local_hz=args.local_map_hz,
                global_hz=args.global_map_hz,local_radius=args.local_map_radius)
            if args.map_source=='online':
                layers.save(folder/'initial_unknown')
                event('online_map_initial',observed_cells=int(layers.observed.sum()),occupied_cells=int(layers.global_occupied.sum()),
                      scene_prior_loaded=False,box_count=len(layers.current_boxes))
            layers.update(float(robot._go2_locomotion_control._simulation_time() or 0.),obs,box_states(),force=True)
            if args.map_source=='online':save_lidar_sample(obs,folder/'lidar_initial.npz')
            layers.synchronize_display(component.mapping,0)
            if args.record_video:
                recorder=TaskVideoRecorder(str(folder/'video'),task=task,capture_interval=1,
                    fps=25,show_scene_graph=False,show_semantic_visuals=False,record_robot_topdown=False)
                record_component=component;recording_tick=0;recording_enabled=True
                camera_trace=(folder/'video/camera_heading.jsonl').open('w')
            if args.map_source=='online':
                record_phase='map_warmup'
                for _ in range(120):obs=tick(stop)
                save_lidar_sample(obs,folder/'lidar_warmed.npz')
                layers.refresh_global(reason='map_warmup_completed')
                event('online_map_warmed',**layers.statistics())
            release_start=None;planned_positions={};last_action=stop;last_tracking=None;tracker=None;realign_count=0;decision_recovery_start=None
            last_route_replan=-100
            decision_recovery_tracker=None
            contact_pause_until=-1;contact_stable_ticks=0;push_lateral_velocity=0.
            retreat_wait_since=None;contact_turn=None;contact_yaw_servo=ContactYawServo(args.contact_yaw_tolerance)
            contact_alignment=ContactAlignmentGate()
            attitude_recovery=NavigationAttitudeRecovery()
            sub_arrival=ObservedSubgoalArrival();inspection_settle=MeasuredStopGate()
            def observation_brake(position,heading,boxes,now):
                if smooth_limiter is None:return stop
                command,_=checked_contact_turn_command([0.,0.,0.],position,heading,planner,boxes,smooth_limiter,now)
                action={'move_by_speed':command.tolist()}
                if args.forward_stop_feedback:action['_forward_stop_request']=True
                return action
            def inspection_stopped(position,tilt,now):
                ctrl=robot._go2_locomotion_control
                return inspection_settle.update(position,ctrl.body_lin_vel,ctrl.body_ang_vel,tilt,now)
            def subnavigation(target,position,heading,boxes,now):
                nonlocal subtracker,subtarget,sub_started
                target=np.asarray(target,float)
                if subtarget is None or np.linalg.norm(target-subtarget)>.02:
                    subtracker=None;subtarget=target.copy();sub_started=now;sub_arrival.reset()
                if now-sub_started>80:return stop,False,{'failed':'subnavigation_timeout'}
                ctrl=robot._go2_locomotion_control
                arrival=sub_arrival.update(target,position,ctrl.body_lin_vel,ctrl.body_ang_vel,tilt,now,
                    heading=heading,command=None if smooth_limiter is None else smooth_limiter.command)
                if arrival['replan']:subtracker=None
                if arrival['subgoal_braking']:
                    if arrival['subgoal_reached'] and smooth_limiter is not None:
                        # Arrival is measured and settled: an old route's turn
                        # maneuver must not survive into the next skill/route.
                        smooth_limiter.clear_guidance(position,now)
                    return observation_brake(position,heading,boxes,now),arrival['subgoal_reached'],arrival
                planner.robot_heading=heading
                if subtracker is None:
                    route=()
                    separation=float(np.linalg.norm(position[:2]-target))
                    if separation<=2.:
                        with layers.global_planning() as view_boxes:
                            connection=planner.path(position[:2],target,view_boxes)
                        if connection and planner.length(position[:2],connection)<=1.25*separation+.10:
                            route=connection;layers.global_path=route;layers.local_path=()
                            layers.guidance_revision=layers.global_revision
                    if not route:route=ordinary_path(position[:2],target,boxes)
                    if not route:return stop,False,{'needs_replan':True}
                    subtracker=ordinary_tracker(position[:2],route,heading,short_goal=True)
                action,detail=subtracker.action(position[:2],heading,planner,boxes)
                if detail.get('needs_replan') and step%100==0:
                    route=ordinary_path(position[:2],target,boxes,local=True)
                    if route:subtracker=ordinary_tracker(position[:2],route,heading,short_goal=True)
                component.mapping._planned_path=subtracker._planned_path[subtracker._waypoint_index:]
                return action,False,detail

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
                maximum_tilt=max(maximum_tilt,tilt);current=planning_boxes();by_id={b.id:b for b in current}
                planner.control_time=float(robot._go2_locomotion_control._simulation_time() or 0.)
                if args.map_source=='online' and goal_kind=='frontier' and phase=='navigate':record_phase='explore_navigate'
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
                if args.navigation_attitude_recovery:
                    recovery=attitude_recovery.update(p,q,
                        enabled=navigation_recovery_enabled(phase,getattr(skill,'state',None)))
                    if recovery['started']:
                        event('navigation_attitude_recovery_begin',step=step,phase=phase,
                              position=p.tolist(),**recovery)
                    if recovery['state']=='failed':
                        event('navigation_attitude_recovery_failed',step=step,position=p.tolist(),**recovery)
                        reason='navigation_attitude_recovery_timeout';break
                    if recovery['state']=='hold':
                        record_phase='navigation_attitude_recovery'
                        obs=tick(stop);last_action=stop;last_tracking=recovery;continue
                    if recovery['state']=='released':
                        event('navigation_attitude_recovery_end',step=step,position=p.tolist(),**recovery)
                        # Gait settling can drift. Replan on the latest maps
                        # instead of continuing a stale local turning target.
                        if phase not in ['skill','inspect_navigate','inspect_turn','inspect_wait']:phase='decide'
                        phase_start=step;tracker=None;subtracker=None;sub_reached=False
                        sub_arrival.reset();inspection_settle.reset();inspection_turn=None
                        if passage_guide is not None:passage_guide.reset_arrival()
                        if phase=='inspect_wait':inspection_wait_frames=set();inspection_view_started=planner.control_time
                        decision_recovery_tracker=None;decision_recovery_start=None
                        if smooth_limiter is not None:smooth_limiter.clear_guidance(p,planner.control_time)
                        obs=tick(stop);last_action=stop;last_tracking=recovery;continue
                planner.robot_heading=None
                crossed=completed_box is None or crossed_passage(p,q,completed_box)
                if passage_guide is not None and crossed:passage_guide=None
                if (passage_guide is not None and not passage_guide.entry_reached
                        and phase in ['navigate','decide']):
                    ctrl=robot._go2_locomotion_control
                    braking_before=passage_guide.arrival.braking
                    arrival=passage_guide.observe_entry(p,ctrl.body_lin_vel,ctrl.body_ang_vel,tilt,
                        planner.control_time,heading=heading,
                        command=None if smooth_limiter is None else smooth_limiter.command)
                    if arrival['subgoal_braking'] and not braking_before:
                        event('passage_entry_braking',step=step,position=p.tolist(),**arrival)
                    if arrival['entry_advanced']:
                        event('cleared_passage_entry_reached',step=step,position=p.tolist(),
                            entry=passage_guide.entry.tolist(),measured_settled=True,**arrival)
                        if smooth_limiter is not None:smooth_limiter.clear_guidance(p,planner.control_time)
                        phase='decide';tracker=None
                    elif arrival['replan']:phase='decide';tracker=None
                    if arrival['subgoal_braking']:
                        record_phase='passage_entry_settling'
                        action=observation_brake(p,heading,current,planner.control_time)
                        obs=tick(action);last_action=action;last_tracking=arrival;continue
                navigation_goal=passage_guide.target if passage_guide is not None else task['goal']
                if distance<.20 and crossed and phase in ['navigate','decide','cross_passage'] and not planner.blocked(current)[planner.cell(task['goal'])]:
                    # Validate the pose after ALL stationary recording frames.
                    # Checking before an extra tail can report success even when
                    # the gait drifts outside the final goal tolerance afterwards.
                    record_phase='goal_settling'
                    goal_stop=MeasuredStopGate();measured_stopped=False
                    minimum_hold=260 if args.record_video else 200
                    for held in range(1000):
                        braking=observation_brake(np.asarray(obs['position']),yaw(obs['orientation']),planning_boxes(),
                            float(robot._go2_locomotion_control._simulation_time() or planner.control_time))
                        obs=tick(braking)
                        ctrl=robot._go2_locomotion_control
                        settle_q=np.asarray(obs['orientation']);settle_q=settle_q/np.linalg.norm(settle_q)
                        settle_tilt=math.degrees(math.acos(np.clip(1-2*(settle_q[1]**2+settle_q[2]**2),-1,1)))
                        measured_stopped=goal_stop.update(obs['position'],ctrl.body_lin_vel,ctrl.body_ang_vel,
                            settle_tilt,float(ctrl._simulation_time() or planner.control_time))
                        if held+1>=minimum_hold and (measured_stopped or not task_health.healthy):break
                    if not task_health.healthy:reason='robot_unstable';break
                    if not measured_stopped:
                        reason='goal_settling_timeout'
                        event('goal_not_settled',step=step,position=list(map(float,obs['position'])))
                        break
                    crossed=completed_box is None or crossed_passage(np.array(obs['position']),np.array(obs['orientation']),completed_box)
                    settled_error=float(np.linalg.norm(np.array(obs['position'][:2])-task['goal']))
                    event('goal_settled',step=step,distance=settled_error,full_body_crossed=bool(crossed),measured_stopped=True)
                    if settled_error<.25 and crossed:
                        success=True;reason='goal_reached';break
                    # The observation changed during settling. Start the next
                    # control iteration from that fresh pose instead of stale p.
                    last_action=stop
                    continue
                if phase=='decide':
                    if args.map_source=='online' and online_wait_start is not None and planner.control_time<online_next_decision_time:
                        if step-online_wait_start>1200:reason='no_observed_route_or_frontier';break
                        obs=tick(stop);last_action=stop;continue
                    recovery=None
                    if navigation_body_blocked(planner,current,heading)[planner.cell(p[:2])]:
                        if decision_recovery_tracker is None:
                            decision_recovery_tracker=ForwardExplorationPathTracker(p[:2],[(*navigation_goal,0.)],guard_motion=False)
                        recovery=decision_recovery_tracker.action(p[:2],heading,planner,current)
                    else:decision_recovery_tracker=None
                    if recovery is not None and recovery[1].get('needs_replan'):recovery=None
                    if recovery is None and args.roadmap_shape=='circle':
                        local_motion_free=planner.free.copy()
                        with layers.global_planning() as global_boxes:
                            attachment=decision_buffer_recovery(p[:2],heading,planner,global_boxes,navigation_goal,heading,
                                local_free=local_motion_free,local_boxes=current,
                                require_goal_connection=args.map_source!='online')
                        if attachment is not None:
                            # The helper already checked the complete local
                            # body connection and monotone overlap reduction.
                            recovery=attachment;recovery[1]['controller']='rejoin_global_clearance'
                    if recovery is not None:
                        online_wait_start=None
                        if decision_recovery_start is None:decision_recovery_start=step
                        action,tracking=recovery;record_phase='recover_navigation_buffer'
                        if smooth_limiter is not None:
                            action=smooth_recovery_action(action,p[:2],heading,planner,current,smooth_limiter,planner.control_time)
                        if step%100==0:event('decision_buffer_recovery',step=step,position=p.tolist(),**tracking,command=action['move_by_speed'])
                        if step-decision_recovery_start>2500:reason='decision_buffer_recovery_timeout';break
                        obs=tick(action);last_action=action;last_tracking=tracking;continue
                    decision_recovery_start=None
                    if smooth_limiter is not None and last_tracking and last_tracking.get('recovering_buffer'):
                        if not smooth_limiter.recovery_gate.leave_recovery(smooth_limiter.measured,planner.control_time):
                            record_phase='navigation_recovery_settle'
                            obs=tick(stop);last_action=stop;continue
                        smooth_limiter.clear_guidance(p,planner.control_time)
                    planner.robot_heading=heading if args.navigation_heading=='path' else nav_heading
                    if passage_guide is not None:
                        with layers.global_planning() as global_boxes:
                            saved_heading=planner.robot_heading
                            try:
                                planner.robot_heading=None
                                path=passage_guide.path(planner,p[:2],global_boxes)
                            finally:planner.robot_heading=saved_heading
                        if not path:reason='verified_passage_connection_lost';break
                        endpoint=passage_guide.target;goal_kind='task' if passage_guide.entry_reached else 'passage_entry'
                        phase='navigate';phase_start=step;planned_positions={b.id:np.asarray(b.xy) for b in current}
                        layers.global_path=path;layers.guidance_revision=layers.global_revision
                        layers.decision={**(layers.decision or {}),'kind':'navigate',
                            'execution_state':'follow_cleared_corridor' if passage_guide.entry_reached else 'approach_cleared_corridor_entry',
                            'navigation_subgoal':endpoint.tolist(),'requested_goal':task['goal']}
                        tracker=ordinary_tracker(p[:2],path,nav_heading,short_goal=not passage_guide.entry_reached)
                        action,tracking=tracker.action(p[:2],heading,planner,current)
                        component.mapping._planned_path=tracker._planned_path[tracker._waypoint_index:]
                        event('verified_passage_route',step=step,path=path,entry=passage_guide.entry.tolist(),
                              exit=passage_guide.exit.tolist(),entry_reached=passage_guide.entry_reached)
                        obs=tick(action);last_action=action;last_tracking=tracking;continue
                    layers.refresh_global(current,reason='decision')
                    inspection=(inspection_planner.propose(layers,p[:2],heading,task['goal'],recognition.store,planner.control_time)
                                if args.interaction_source=='vision' else None)
                    if inspection is not None:
                        phase='inspect_navigate';phase_start=step;subtracker=None;subtarget=None
                        sub_arrival.reset();inspection_settle.reset()
                        layers.decision={'kind':'inspect','reason':inspection.reason,'object_id':inspection.object_id,
                            'recognized_category':'unknown','interaction_allowed':False,'execution_state':'move_to_rgb_view'}
                        event('inspection_begin',step=step,time_s=planner.control_time,**vars(inspection))
                        obs=tick(stop);last_action=stop;continue
                    if args.map_source=='online':
                        decision,online_endpoint=online_selector.decide(layers,p[:2],task['goal'])
                        if decision.kind=='wait':
                            layers.decision=decision.explanation
                            if online_wait_start is None:online_wait_start=step
                            online_next_decision_time=planner.control_time+layers.global_period
                            if step%200==0:event('online_observation_wait',step=step,reason=decision.reason,**layers.statistics())
                            if step-online_wait_start>1200:reason='no_observed_route_or_frontier';break
                            obs=tick(stop);last_action=stop;continue
                        online_wait_start=None
                    else:
                        with layers.global_planning() as global_boxes:
                            decision=planner.decide(p[:2],task['goal'],global_boxes)
                    layers.decision=decision.explanation
                    event('decision',step=step,kind=decision.kind,reason=decision.reason,
                          position=p.tolist(),orientation=q.tolist(),
                          plan=vars(decision.push) if decision.push else None,explanation=decision.explanation,
                          global_map_revision=layers.global_revision,local_map_revision=layers.local_revision)
                    if decision.kind=='failed':reason=decision.reason;break
                    if decision.kind in ['navigate','explore']:
                        path=decision.path;phase='navigate';goal_kind='frontier' if decision.kind=='explore' else 'task'
                    else:
                        plan=decision.push
                        permission=recognition.permission(plan.box_id,'push_aside',planner.control_time)
                        if permission is None:
                            event('interaction_permission_denied',step=step,object_id=plan.box_id)
                            obs=tick(stop);last_action=stop;continue
                        skill=registry.create(permission,plan=plan,task_start=task['start'],task_goal=task['goal'],
                            config=PushAsideConfig(front_gap=max(.30,args.contact_front_gap),push_max_speed=args.push_max_speed,push_start_speed=args.push_start_speed,
                                lateral_speed=args.contact_lateral_speed,forward_speed=args.contact_centering_speed,
                                 yaw_tolerance=args.contact_yaw_tolerance))
                        phase='skill';push_count+=1;subtracker=None;subtarget=None;sub_reached=False
                        event('skill_start',step=step,time_s=planner.control_time,permission=vars(permission),plan=vars(plan))
                        obs=tick(stop);last_action=stop;continue
                    planner.robot_heading=heading if args.navigation_heading=='path' else nav_heading
                    endpoint=(online_endpoint if args.map_source=='online' else task['goal']) if phase=='navigate' else plan.approach_xy
                    path=ordinary_path(p[:2],endpoint,current)
                    if not path:reason='body_footprint_blocks_navigation';break
                    tracker=ordinary_tracker(p[:2],path,nav_heading)
                    event('ordinary_route',step=step,path=path,**(routes.last_plan if args.route_planner=='voronoi' else {'engine':'grid_astar'}))
                    phase_start=step;planned_positions={b.id:np.array(b.xy) for b in current};action=stop
                    if args.path_controller=='continuous':
                        action,iteration_tracking=tracker.action(p[:2],heading,planner,current)
                        component.mapping._planned_path=tracker._planned_path[tracker._waypoint_index:]
                elif phase in ['navigate','approach']:
                    planner.robot_heading=heading if args.navigation_heading=='path' else nav_heading
                    if args.map_source=='online' and phase=='navigate' and goal_kind=='frontier' and not getattr(tracker,'staging_active',False):
                        reconsider=np.linalg.norm(p[:2]-endpoint)<.35
                        if step%125==0:
                            reconsider |= online_selector.task_decision(layers,p[:2],task['goal']) is not None
                        if reconsider:
                            if np.linalg.norm(p[:2]-endpoint)<.35:online_selector.reached_frontier(endpoint,layers.observed.sum())
                            phase='decide';obs=tick(last_action);continue
                    if (step%200==0 and not getattr(tracker,'staging_active',False) and any(b.id not in planned_positions or np.linalg.norm(np.array(b.xy)-planned_positions[b.id])>.08 for b in current)
                            and not planner.blocked(current)[planner.cell(p[:2])]):
                        # A newly observed distant box requires reconsidering
                        # the route, not an unconditional zero-speed pulse.
                        # The old route's next twist is checked against all
                        # current obstacles before the next decision tick.
                        action,tracking=(tracker.action(p[:2],heading,planner,current)
                                         if args.path_controller=='continuous' else (stop,{}))
                        phase='decide';obs=tick(action);last_action=action;last_tracking=tracking;continue
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
                                and (step-last_route_replan>=100 or tracking.get('staging_completed'))):
                            last_route_replan=step
                            replanned=ordinary_path(p[:2],endpoint,current,local=True)
                            if replanned:
                                tracker=ordinary_tracker(p[:2],replanned,nav_heading,short_goal=goal_kind=='passage_entry')
                                action,tracking=tracker.action(p[:2],heading,planner,current)
                                event('path_replanned_from_actual_pose',step=step,map_scope='local_dynamic',
                                      global_map_revision=layers.global_revision,local_map_revision=layers.local_revision,
                                      position=p.tolist(),orientation=q.tolist(),path=[list(map(float,point)) for point in replanned])
                            elif layers.global_revision!=getattr(layers,'guidance_revision',layers.global_revision):
                                phase='decide';action=stop
                                event('global_replan_after_local_block',step=step,global_map_revision=layers.global_revision)
                        iteration_tracking=tracking
                        if args.map_source=='online' and last_tracking and last_tracking.get('observation_motion') and not tracking.get('observation_motion'):
                            if not (folder/'lidar_after_observation.npz').exists():save_lidar_sample(obs,folder/'lidar_after_observation.npz')
                        if args.map_source=='online' and tracking.get('needs_replan') and tracking.get('blocked_ticks',0)>30:
                            if goal_kind=='frontier':online_selector.reached_frontier(endpoint,layers.observed.sum())
                            phase='decide'
                        if tracking.get('recovering_buffer'):record_phase='navigation_buffer_recovery'
                        if tracking.get('local_turn_maneuver'):record_phase=tracking['controller']
                        if tracking.get('observation_motion'):record_phase='observe_local_map'
                        component.mapping._planned_path=tracker._planned_path[tracker._waypoint_index:]
                        if tracking.get('local_staging'):
                            record_phase=tracking['staging_phase']
                            component.mapping._planned_path=(tuple(tracking['target']),)
                        if step%400==0:event('path_tracking',step=step,**tracking)
                    if step-phase_start>16000:reason='navigation_timeout';break
                elif phase=='inspect_navigate':
                    action,arrived,iteration_tracking=subnavigation(inspection.position,p,heading,current,planner.control_time)
                    if step%200==0:event('inspection_navigation_tracking',step=step,position=p.tolist(),tracking=iteration_tracking)
                    if iteration_tracking.get('failed'):
                        event('inspection_view_unreachable',step=step,reason=iteration_tracking['failed'])
                        phase='decide';action=stop
                    elif arrived:
                        phase='inspect_turn';phase_start=step;inspection_turn=None;inspection_settle.reset();action=stop
                elif phase=='inspect_turn':
                    if inspection_turn is None and not inspection_stopped(p,tilt,planner.control_time):
                        action=observation_brake(p,heading,current,planner.control_time)
                    else:
                        if inspection_turn is None:
                            inspection_turn=ContactTurnControl(p,heading,inspection.heading,planner,current,planner.control_time)
                        answer=inspection_turn.update(p,heading,planner,current,smooth_limiter,planner.control_time)
                        action=stop if answer is None else {'move_by_speed':answer[0].tolist()}
                        if abs(angle(inspection.heading-heading))<.10:
                            phase='inspect_wait';phase_start=step;inspection_wait_frames=set();inspection_view_started=planner.control_time
                            inspection_settle.reset();action=observation_brake(p,heading,current,planner.control_time)
                        elif inspection_turn.failed:
                            phase='decide';action=stop
                            event('inspection_turn_retry',step=step,object_id=inspection.object_id)
                    if step-phase_start>6000:phase='decide';action=stop
                elif phase=='inspect_wait':
                    action=observation_brake(p,heading,current,planner.control_time)
                    layers.decision['execution_state']='RGB_DINO_SAM_Qwen_confirm_two_frames'
                    recognized=recognition.store.confirmed(inspection.object_id,planner.control_time)
                    debug=recognition.last_debug or {}
                    if debug.get('observed_at',-1.)>=inspection_view_started and any(
                            d.get('object_id')==inspection.object_id and d.get('status')=='classified'
                            for d in debug.get('detections',[])):
                        inspection_wait_frames.add(debug['frame_id'])
                    fresh=[] if recognized is None else [e for e in recognized.evidence if e.observed_at>=inspection_view_started]
                    settled=inspection_stopped(p,tilt,planner.control_time)
                    if recognized is not None and len(fresh)>=2 and settled:
                        inspection_planner.review(inspection.object_id,recognized.category,planner.control_time)
                        event('inspection_classified',step=step,time_s=planner.control_time,object_id=inspection.object_id,
                            category=recognized.category,allowed=inspection.object_id in planner.interactable_ids,
                            fresh_view_frames=[e.frame_id for e in fresh])
                        phase='decide'
                    elif (len(inspection_wait_frames)>=3 and settled) or step-phase_start>3500:
                        event('inspection_unknown_new_view',step=step,object_id=inspection.object_id)
                        phase='decide'
                elif phase=='skill':
                    ctx=SkillContext(planner.control_time,p,heading,robot._go2_locomotion_control.body_lin_vel,
                        float(robot._go2_locomotion_control.body_ang_vel[2]),tilt,task_health.healthy,
                        by_id.get(plan.box_id),current,planner,layers,smooth_limiter,sub_reached,
                        body_angular_velocity=robot._go2_locomotion_control.body_ang_vel)
                    command=skill.step(ctx)
                    for transition in command.events:event('skill_transition',step=step,**transition)
                    layers.decision={**(layers.decision or {}),'skill':'push_aside','skill_state':command.state,
                        'skill_retry':skill.retries,'recognized_category':skill.permission.category,
                        'interaction_allowed':True,'interaction_source':skill.permission.source,'execution_state':command.state}
                    record_phase='skill_'+command.state
                    action={'move_by_speed':checked_skill_brake(command,ctx).tolist()};sub_reached=False
                    if command.navigation_goal is not None:
                        action,sub_reached,iteration_tracking=subnavigation(command.navigation_goal,p,heading,current,planner.control_time)
                        if iteration_tracking.get('failed'):
                            skill._fail(ctx,iteration_tracking['failed'])
                    if step%200==0:event('skill_feedback',step=step,state=command.state,position=p.tolist(),
                        box=list(by_id[plan.box_id].xy) if plan.box_id in by_id else None,command=action['move_by_speed'],
                        requested_command=list(command.velocity),retry=skill.retries,**skill.feedback(ctx))
                    if command.status!='running':
                        report=skill.report();skill_reports.append(report);event('skill_end',step=step,time_s=planner.control_time,**report)
                        if command.status!='succeeded':reason='push_aside:'+command.reason;break
                        completed_box=plan.box_id;passage_certificate=command.certificate
                        passage_guide=PassageEntryGuide(passage_certificate,task['goal'])
                        layers.refresh_global(reason='skill_success_replan_through_passage')
                        phase='decide';phase_start=step;tracker=None;action=stop
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
                'navigation_attitude_recovery':{'enabled':args.navigation_attitude_recovery,**attitude_recovery.report()},
                'contact_realignments':realign_count,
                'actual_path_length_m':float(sum(np.linalg.norm(np.array(b['robot'][:2])-a['robot'][:2]) for a,b in zip(samples,samples[1:]))),
                'final_boxes':[vars(b) for b in box_states()],'final_box_heights':box_height_states(),
                'ground_height':floor_top,'teleports_during_rollout':False,
                'ordinary_route_planning':{**routes.statistics(),'map_source':args.map_source},
                'navigation_heading':args.navigation_heading,
                'path_controller':args.path_controller,
                'actuator_mode':args.actuator_mode,
                'full_body_passage_crossed':(crossed_passage(np.array(obs['position']),np.array(obs['orientation']),completed_box)
                    if completed_box is not None else (False if task.get('decision')=='push' else None)),
                'go2_policy_inferences':int(robot._go2_locomotion_control.policy_inferences)}
            layers.save(folder)
            result['layered_mapping']=layers.statistics()
            result['passage_certificate']=passage_certificate
            result['interaction_source']=args.interaction_source
            result['recognition']=recognition.save(float(robot._go2_locomotion_control._simulation_time() or planner.control_time))
            result['interaction_skills']=skill_reports
            if skill is not None and not skill_reports:result['interaction_skills']=[skill.report()]
            result['map_source']=args.map_source
            result['goal_source']='requested_coordinates'
            if control_log is not None:control_log.close();control_log=None
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
        recognition.close()
        if camera_trace:camera_trace.close()
        if recorder:recorder.close()
        if env is not None:env.close()
        runtime.simulation_app.close()
    return 0 if all(r['success'] for r in results) else 1


if __name__=='__main__':raise SystemExit(main())
