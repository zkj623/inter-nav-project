import math
from types import SimpleNamespace as NS
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.interaction_skill import SkillRegistry,SkillContext
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill
from grutopia_extension.interactive_navigation.go2_attitude_recovery import NavigationAttitudeRecovery


def make_skill():
    permission=InteractionPermission('B1','box','push_aside',('rgb1','rgb2'),('Qwen',),0.)
    plan=NS(box_id='B1',direction=(1.,0.),push_distance=1.)
    return PushAsideSkill(permission,plan,(0.,0.),(2.,0.))


def context(box=None,now=0.):
    return SkillContext(now,np.array([0.,0.,.4]),0.,np.zeros(3),0.,0.,True,box,[],None,None,None)


def test_skill_requires_recognition_permission_and_matching_target():
    registry=SkillRegistry();registry.register(PushAsideSkill)
    with pytest.raises(ValueError):registry.create(None)
    with pytest.raises(ValueError):
        PushAsideSkill(InteractionPermission('B1','box','push_aside',('one',),('Qwen',),0.),
                       NS(box_id='B1'),(0,0),(1,0))
    with pytest.raises(ValueError):
        PushAsideSkill(InteractionPermission('B2','box','push_aside',('one','two'),('Qwen',),0.),
                       NS(box_id='B1'),(0,0),(1,0))


def test_lost_track_returns_safe_failure_before_dereferencing_box():
    skill=make_skill();command=skill.step(context())
    assert command.status=='failed' and command.reason=='target_track_lost'
    assert command.velocity==(0.,0.,0.)


def test_retry_resets_old_release_origin():
    skill=make_skill();skill.release_origin=np.array([123.,0.])
    command=skill._retry(context(),'alignment_drift')
    assert skill.release_origin is None and skill.retries==1
    assert command.velocity==(0.,0.,0.)


def test_thirty_degree_hard_health_limit_is_unchanged():
    skill=make_skill();ctx=context(NS(id='B1',xy=(1.,0.)))
    ctx.tilt_degrees=30.
    assert skill.step(ctx).reason=='robot_unstable'


def quaternion(degrees):
    a=math.radians(degrees)/2
    return [math.cos(a),0.,math.sin(a),0.]


def test_stable_moderate_tilt_can_resume_but_sliding_cannot():
    for sliding in [False,True]:
        recovery=NavigationAttitudeRecovery(minimum_hold_ticks=6,stable_ticks=3,maximum_hold_ticks=20)
        recovery.update([0,0,.4],quaternion(20.))
        for i in range(6):answer=recovery.update([i*.03 if sliding else 0,0,.4],quaternion(12.))
        assert answer['state']==('hold' if sliding else 'released')


def test_reengagement_does_not_mistake_previous_push_for_current_contact():
    skill=make_skill();skill.state='engage';skill.started=0.;skill.state_since=0.
    box=NS(id='B1',xy=(1.18,0.),dimensions=(1.,1.,1.))
    skill.initial_box=np.array([.98,0.]);skill.engage_box_position=np.array(box.xy)
    ctx=context(box,1.)
    ctx.planner=NS(clearance=lambda box:{'side_parked':False})
    ctx.limiter=NS(preview=lambda command,now:command)
    command=skill.step(ctx)
    assert command.state=='engage' and command.velocity[0]<=.30


def test_progress_watchdog_detects_stalling_after_partial_movement():
    skill=make_skill();skill.state='push';skill.started=0.;skill.state_since=0.
    skill.initial_box=np.array([.8,0.]);skill.push_progress=.20;skill.push_progress_time=0.
    ctx=context(NS(id='B1',xy=(1.,0.),dimensions=(1.,1.,1.)),7.)
    ctx.planner=NS(clearance=lambda box:{'side_parked':False})
    ctx.limiter=NS(preview=lambda command,now:command)
    command=skill.step(ctx)
    assert command.state=='brake_for_recovery' and skill.release_reason=='box_motion_stalled'
    assert command.velocity==(0.,0.,0.)


def test_nominal_clearance_does_not_stop_before_the_planned_parking_position():
    skill=make_skill();skill.state='push';skill.started=0.;skill.state_since=0.
    skill.initial_box=np.array([.8,0.]);skill.push_progress=.63;skill.push_progress_time=.9
    ctx=context(NS(id='B1',xy=(1.43,0.),dimensions=(1.,1.,1.)),1.)
    ctx.position=np.array([.6,0.,.4])
    ctx.planner=NS(clearance=lambda box:{'side_parked':True,'passage_clearance_margin_m':.04})
    ctx.limiter=NS(preview=lambda command,now:command)
    command=skill.step(ctx)
    assert command.state=='push' and command.velocity[0]>0.


@pytest.mark.parametrize('state',['turn_at_stage','refine_heading','align_lateral'])
def test_precontact_turn_recovers_before_hard_health_limit(state):
    skill=make_skill();skill.state=state
    ctx=context(NS(id='B1',xy=(1.,0.)))
    ctx.tilt_degrees=20.
    command=skill.step(ctx)
    assert command.state=='brake_for_recovery' and skill.release_reason=='preparation_attitude'
    assert command.velocity==(0.,0.,0.) and skill.retries==1


def test_current_aligned_pose_can_be_reused_without_navigation():
    from contextlib import contextmanager
    from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
    from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
    box=BoxState('B1',(1.,0.),(1.,1.,1.))
    planner=SidePushPlanner(np.ones((80,80),bool),[[-2,2],[-2,2]],resolution=.05,
                           passages={'B1':{'axis':[0.,1.],'center':[1.,0.]}})
    @contextmanager
    def global_planning():yield [box]
    ctx=context(box);ctx.position=np.array([-.3,-.5,.4]);ctx.planner=planner;ctx.boxes=[box]
    ctx.layers=NS(global_planning=global_planning)
    skill=make_skill()
    assert skill._reuse_alignment_pose(ctx)
    np.testing.assert_allclose(skill.stage,ctx.position[:2])


def test_horizontal_route_length_accepts_mixed_xy_xyz_waypoints():
    from grutopia_extension.interactive_navigation.box_push_navigation import BoxPushPlanner
    assert BoxPushPlanner.length((0.,0.,9.),[(3.,4.),(6.,8.,-1.)])==10.
    assert BoxPushPlanner.length((2.,3.),[(2.,3.)])==0.
    assert math.isinf(BoxPushPlanner.length((2.,3.),[]))


def test_actual_safe_contact_pose_ends_alignment_without_millimetre_chasing():
    from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
    from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
    box=BoxState('B1',(1.,0.),(1.,1.,1.))
    planner=SidePushPlanner(np.ones((160,160),bool),[[-4,4],[-4,4]],resolution=.05)
    skill=make_skill();skill.state='align_lateral';skill.state_since=0.;skill.started=0.
    skill.initial_box=np.array(box.xy);skill.anchor=np.array([-.18,0.])
    ctx=context(box,1.);ctx.position=np.array([-.18,-.040,.4]);ctx.heading=.03
    ctx.planner=planner;ctx.boxes=[box]
    command=skill.step(ctx)
    assert command.state=='settle_contact' and command.velocity==(0.,0.,0.)
    assert command.events[-1]['reason']=='actual_contact_pose_ready'


def test_contact_reach_accounts_for_front_feet_not_just_the_nose():
    skill=make_skill();skill.state='engage';skill.started=0.;skill.state_since=0.
    box=NS(id='B1',xy=(1.01,0.),dimensions=(1.,1.,1.))
    skill.initial_box=np.array(box.xy);skill.engage_box_position=np.array(box.xy)
    ctx=context(box,1.);ctx.planner=NS(clearance=lambda box:{'side_parked':False})
    ctx.limiter=NS(preview=lambda command,now:command)
    command=skill.step(ctx)
    assert command.state=='push' and command.velocity[0]<=.30


def alignment_context(state,gap=.46,lateral=.12,heading=0.):
    from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
    from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
    from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
    box=BoxState('B1',(1.,0.),(1.,1.,1.))
    ctx=context(box,1.);ctx.heading=heading
    ctx.position=np.array([1.-.5-gap-.33*math.cos(heading),-lateral-.33*math.sin(heading),.4])
    ctx.planner=SidePushPlanner(np.ones((160,160),bool),[[-4,4],[-4,4]],resolution=.05)
    ctx.boxes=[box];ctx.limiter=VelocitySlewLimiter();ctx.limiter.accept([0.,0.,0.],.99)
    skill=make_skill();skill.state=state;skill.state_since=0.;skill.started=0.
    skill.initial_box=np.array(box.xy);skill.anchor=np.array([-.18,0.])
    return skill,ctx


def test_safe_standoff_stops_longitudinal_chasing_before_lateral_alignment():
    skill,ctx=alignment_context('align_longitudinal')
    command=skill.step(ctx)
    assert command.velocity==(0.,0.,0.) and command.state=='align_longitudinal'
    ctx.now+=.31
    command=skill.step(ctx)
    assert command.state=='align_lateral' and command.velocity==(0.,0.,0.)
    assert command.events[-1]['reason']=='safe_standoff_reached'
    assert abs(skill._box_frame(ctx)[1])>.055  # still needs lateral work


def test_standoff_outside_safe_interval_still_requires_longitudinal_motion():
    skill,ctx=alignment_context('align_longitudinal',gap=.55)
    command=skill.step(ctx)
    assert command.state=='align_longitudinal' and command.velocity[0]>0.


def test_fine_heading_brakes_and_waits_for_measured_settling_before_translation():
    skill,ctx=alignment_context('refine_heading',heading=-.097)
    command=skill.step(ctx)
    assert command.velocity[0]==command.velocity[1]==0. and command.velocity[2]>0.
    ctx.heading=-.03;ctx.yaw_rate=.12;ctx.now+=.5
    assert skill.step(ctx).state=='refine_heading'
    ctx.yaw_rate=0.;ctx.now+=.5
    assert skill.step(ctx).state=='refine_heading'
    ctx.now+=.31
    assert skill.step(ctx).state=='align_longitudinal'


def test_settling_restarts_on_body_motion_even_when_measured_speed_is_low():
    skill,ctx=alignment_context('settle_contact',lateral=0.)
    assert not skill._quiet(ctx)
    ctx.now+=.31;ctx.position[0]+=.03
    assert not skill._quiet(ctx)
    ctx.now+=.31
    assert skill._quiet(ctx)
    ctx.limiter.command=np.array([.10,0.,0.])
    assert not skill._quiet(ctx)


def test_predictive_braking_finishes_before_requesting_another_alignment_stride():
    skill,ctx=alignment_context('align_lateral')
    ctx.body_velocity=np.array([0.,.30,0.]);ctx.limiter.command=np.array([0.,.35,0.])
    target=ctx.position[:2]+np.array([0.,.25])
    command,done=skill._translation(ctx,target,skill.tangent,.35,.035)
    assert not done and np.all(command==0.) and skill.translation_braking
    ctx.body_velocity=np.array([0.,.04,0.]);ctx.limiter.command=np.zeros(3);ctx.now+=.3
    command,done=skill._translation(ctx,target,skill.tangent,.35,.035)
    assert not done and np.all(command==0.) and skill.translation_braking
    ctx.now+=.31
    command,done=skill._translation(ctx,target,skill.tangent,.35,.035)
    assert not done and np.all(command==0.) and not skill.translation_braking
    ctx.now+=.01
    command,done=skill._translation(ctx,target,skill.tangent,.35,.035)
    assert not done and command[1]>0.


def test_recovery_reuses_observed_aligned_pose_and_settles_before_correcting():
    skill,ctx=alignment_context('prepare',gap=.46,lateral=.12,heading=.03)
    command=skill.step(ctx)
    assert command.state=='settle_alignment' and command.navigation_goal is None
    assert command.events[-1]['reason']=='reuse_observed_aligned_pose'
    np.testing.assert_allclose(skill.stage,ctx.position[:2])


def test_reuse_rejects_unobserved_alignment_corridor():
    skill,ctx=alignment_context('prepare',gap=.70,lateral=.12)
    row,col=ctx.planner.cell([-.18,0.]);ctx.planner.free[row-2:row+3,col-2:col+3]=False
    assert not skill._reuse_alignment_pose(ctx)


def test_longitudinal_drift_brakes_before_switching_from_sidestep():
    skill,ctx=alignment_context('align_lateral',gap=.65,lateral=.12)
    ctx.body_velocity=np.array([0.,.3,0.]);ctx.limiter.command=np.array([0.,.35,0.])
    command=skill.step(ctx)
    assert command.state=='settle_alignment' and command.velocity==(0.,0.,0.)
    ctx.now+=1.
    assert skill.step(ctx).state=='settle_alignment'


def test_settling_rejects_roll_pitch_motion_and_vertical_bounce():
    skill,ctx=alignment_context('settle_contact',lateral=0.)
    ctx.body_angular_velocity=np.array([.2,0.,0.])
    assert not skill._quiet(ctx)
    ctx.now+=.5;ctx.body_angular_velocity=np.zeros(3);ctx.body_velocity[2]=.1
    assert not skill._quiet(ctx)
    ctx.now+=.5;ctx.body_velocity=np.zeros(3)
    assert not skill._quiet(ctx)
    ctx.now+=.31
    assert skill._quiet(ctx)
