"""Isolated adapter: keep frozen robot/body gates, use cooked object finger geometry."""
from dataclasses import replace
from types import MethodType
import numpy as np
if __package__:
    from .cooked_geometry import CookedGeometry,grasp_target_errors
else:
    from cooked_geometry import CookedGeometry,grasp_target_errors


def install_geometry_audit():
    from grutopia_extension.manipulation.r1lite_bimanual_lift_sim import R1LiteBimanualLift
    from grutopia_extension.interaction_clearance import primitives
    old_sample=R1LiteBimanualLift.collision_sample
    old_check=primitives.check_collision_transforms
    def ensure(env):
        if hasattr(env,'object_geometry'):return
        geometry=CookedGeometry.from_stage(env.stage,'/World/LiftObject',env.box.dimensions,
                                          env.output/'cooked_object_geometry.json')
        env.object_geometry=geometry
        model=env.collision_model;original=model.box_clearances
        def native_gaps(model,transforms,box_transform,dimensions,include_fingers=False):
            root=np.asarray(box_transform).copy()
            root[:3,3]-=root[:3,:3]@np.array((0.,0.,dimensions[2]/2))
            native_box=root.copy()
            native_box[:3,3]+=root[:3,:3]@geometry.bounds.mean(axis=0)
            native_dimensions=geometry.bounds[1]-geometry.bounds[0]
            distances=original(transforms,native_box,native_dimensions,include_fingers)
            if include_fingers:
                model.last_finger_bbox_depth=max(0.,-min(v for n,v in distances.items() if 'gripper_finger' in n))
                inv=np.linalg.inv(root)
                for name in distances:
                    if 'gripper_finger' in name:
                        distances[name]=geometry.gap(model.envelopes[name],model.vertices[name],inv@transforms[name])
            return distances
        model.box_clearances=MethodType(native_gaps,model)
        env.finger_geometry='same_PhysX_cooked_convex_union_v2'
        def bottom(env):
            p,q=env.object.get_world_pose()
            return float(p[2]+(geometry.points@env.rotation(q).T)[:,2].min())
        env.object_bottom=MethodType(bottom,env)
    def sample(env):
        ensure(env);before=len(env.physics_audit)
        try:return old_sample(env)
        finally:
            if len(env.physics_audit)>before:
                env.physics_audit[-1]['maximum_finger_bbox_penetration_m']=env.collision_model.last_finger_bbox_depth
                env.physics_audit[-1]['cooked_geometry_sha256']=env.object_geometry.sha256
    def check(env,transforms,box,minimum=.015):
        # This probe only performs grasp/lift without intentional object rotation.
        # Legacy certification flattened approach orientation to yaw; preserve the
        # measured full orientation of the actual cooked object at this plan step.
        box=np.asarray(box).copy()
        box[:3,:3]=env.rotation(env.object.get_world_pose()[1])
        details=old_check(env,transforms,box,minimum)
        if not hasattr(env,'object_geometry'):raise RuntimeError('Native finger geometry not initialized')
        gaps=env.collision_model.box_clearances(transforms,box,env.box.dimensions,include_fingers=True)
        finger={n:g for n,g in gaps.items() if 'gripper_finger' in n}
        details.update(finger_clearance_geometry='same_PhysX_cooked_convex_union_v2',
            minimum_finger_gap_m=min(finger.values()),finger_gaps_m=finger)
        if min(finger.values()) < -.002:
            raise primitives.InternalArmPathError(details,'PLANNED_NATIVE_FINGER_PENETRATION')
        return details
    R1LiteBimanualLift.collision_sample=sample
    primitives.check_collision_transforms=check


def configure_surface_grasp(env):
    from grutopia_extension.interaction_clearance.task import yaw_rotation
    from grutopia_extension.interaction_clearance.motion_geometry import grasp_frames
    from grutopia_extension.interaction_clearance.primitives import ArmPlanPolicy
    from scipy.spatial.transform import Rotation
    import json
    profile=env.object_geometry.grasp_profile(env.box.grasp_height)
    # Exact-surface targeting uses 1 mm preload inside the unchanged 2 mm gate.
    env.box=replace(env.box,grasp_height=profile['height_m'],squeeze=.001)
    env.surface_grasp=profile
    (env.output/'surface_grasp.json').write_text(json.dumps(dict(profile=profile,preload_m=.001,
        physical_penetration_limit_m=.002,patch_source='PhysX cooked convex hull sections'),indent=2))
    def targets(center,half_width,height,rotation):
        half=profile['half_width_m']+half_width-env.task.dimensions[1]/2
        origin=np.asarray(center)+np.array((0.,0.,height-env.box.grasp_height))
        local=np.array([[profile['x_m'],profile['center_y_m']+half,env.box.grasp_height],
                        [profile['x_m'],profile['center_y_m']-half,env.box.grasp_height]])
        return origin+local@rotation.T,origin
    def solve(env,center,half_width,height,seed,object_rotation=None,preserve_contact_normal=False):
        bp,bq=env.base.get_world_pose();br=env.rotation(bq)
        if object_rotation is None:
            q=env.object.get_world_pose()[1]
            object_rotation=env.rotation(q)
        rotation=np.asarray(object_rotation)
        positions,origin=targets(center,half_width,height,rotation)
        frames=grasp_frames(np.zeros(3),0.,1.,1.)[2:]
        obstacle=np.eye(4);obstacle[:3,:3]=br.T@rotation
        obstacle[:3,3]=br.T@(origin+rotation@np.array((0.,0.,env.box.dimensions[2]/2))-bp)
        solution=env.bimanual.solve(br.T@(positions[0]-bp),br.T@(positions[1]-bp),
            br.T@rotation@frames[0],br.T@rotation@frames[1],seed,env.collision_model,
            obstacle,env.box.dimensions,clearance=.05,
            upper_limits=np.r_[min(env.bimanual.upper[0],0.),env.bimanual.upper[1:]],reseed_torso=True,
            pose_weights=(24.,.6),
            **({'contact_normal':br.T@rotation[:,1]} if preserve_contact_normal else {}))
        if env.recorder:
            env.recorder.event('native_surface_ik',dict(height_m=height,target_positions=positions.tolist(),
                attempts=env.bimanual.last_solve_diagnostics))
        return solution
    env.solve_both=MethodType(solve,env)
    def check_squeeze(planner,target,center):
        environment=planner.environment
        p,q=environment.object.get_world_pose();rotation=environment.rotation(q)
        desired,_=targets(center,environment.task.dimensions[1]/2-environment.box.squeeze,
                          environment.box.grasp_height,rotation)
        bp,bq=environment.base.get_world_pose();br=environment.rotation(bq)
        actual=np.array([bp+br@t[:3,3] for t in environment.bimanual.forward(target[:15])])
        tolerance=min(.001,environment.box.squeeze/3)
        # Retain the legacy normal-direction preload reserve; bound tangential
        # error separately inside the 8 mm supported surface patch.
        details=grasp_target_errors(actual,desired,rotation[:,1],tolerance)
        if not details['passed']:
            raise RuntimeError('NATIVE_SURFACE_GRASP_TARGET_INACCURATE: '+str(details))
        return details
    ArmPlanPolicy.check_squeeze_target=check_squeeze
    return profile
