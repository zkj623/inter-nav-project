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
    assert command.state=='engage' and command.velocity[0]<=skill.config.push_start_speed


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
    assert command.state=='push' and command.velocity[0]<=skill.config.push_start_speed


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
    skill,ctx=alignment_context('refine_heading',heading=-.15)
    command=skill.step(ctx)
    assert command.velocity==(0.,0.,0.)
    ctx.now+=1.21;skill.step(ctx)
    ctx.now+=.01;command=skill.step(ctx)
    assert command.velocity[0]==command.velocity[1]==0. and command.velocity[2]>0.
    ctx.heading=-.03;ctx.yaw_rate=.12;ctx.now+=.5
    assert skill.step(ctx).state=='refine_heading'
    ctx.yaw_rate=0.;ctx.now+=.5
    assert skill.step(ctx).state=='refine_heading'
    ctx.now+=1.21
    assert skill.step(ctx).state=='align_lateral'  # already stopped and within the safe standoff


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


def test_nearby_safe_stage_handoff_waits_for_settling_instead_of_chasing_nominal_point():
    skill,ctx=alignment_context('navigate_to_stage',gap=.46,lateral=.12)
    skill.stage=ctx.position[:2]+np.array([0.,.19])
    ctx.body_velocity=np.array([0.,.12,0.]);ctx.limiter.command=np.array([0.,.2,0.])
    command=skill.step(ctx)
    assert command.state=='settle_at_stage' and command.navigation_goal is None
    assert command.velocity==(0.,0.,0.) and command.events[-1]['reason']=='actual_safe_stage_pose'
    ctx.now+=.5
    assert skill.step(ctx).state=='settle_at_stage'


def test_near_stage_is_not_accepted_when_full_turn_intersects_obstacle():
    skill,ctx=alignment_context('navigate_to_stage',gap=.46,lateral=0.,heading=math.pi/2)
    skill.stage=ctx.position[:2]+np.array([.19,0.])
    ctx.planner.free[:,ctx.planner.cell([.40,0.])[1]:]=False
    command=skill.step(ctx)
    assert command.state=='navigate_to_stage' and command.navigation_goal is not None


def test_missing_stage_attempts_bounded_observed_recovery_before_failing():
    skill,ctx=alignment_context('prepare',heading=math.pi)
    skill._select_stage=lambda context:False
    command=skill.step(ctx)
    assert command.state=='recover_preparation' and command.status=='running'
    ctx.now+=13.
    assert skill.step(ctx).reason=='no_observed_preparation_after_recovery'


def test_observed_T04_preparation_overlap_gets_a_checked_escape_command():
    from pathlib import Path
    from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
    from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
    with np.load(Path(__file__).parent/'fixtures/t04_preparation_buffer_v49.npz') as fixture:
        p=SidePushPlanner(fixture['free'].copy(),fixture['limits'],resolution=.05)
        p.planning_observed=fixture['observed'].copy()
        ctx=context(NS(id='B1',xy=fixture['box_xy'].copy(),dimensions=fixture['box_dimensions'].copy()),1.)
        ctx.position=fixture['position'].copy();ctx.heading=float(fixture['heading'])
    p.navigation_half_width=.22;p.layered_local_control=True;p.full_navigation_footprint=True;p.control_time=1.
    ctx.planner=p;ctx.boxes=[];ctx.limiter=VelocitySlewLimiter();ctx.limiter.accept([0.,0.,0.],.99)
    skill=make_skill();skill.direction=np.array([-1.,0.]);skill.tangent=np.array([0.,-1.]);skill.push_heading=math.pi
    skill._select_stage=lambda context:False
    command=skill.step(ctx)
    assert command.state=='recover_preparation'
    ctx.now+=.01;p.control_time=ctx.now
    before=p.free.copy();command=skill.step(ctx)
    assert command.state=='recover_preparation' and command.status=='running'
    assert command.velocity[0]<0. and command.velocity[2]==0.
    assert skill.recovery_target is not None
    np.testing.assert_array_equal(p.free,before)


def test_small_heading_recovery_still_requires_full_observed_turn_and_contact_alignment():
    skill,ctx=alignment_context('prepare',gap=.60,lateral=.12,heading=.30)
    assert skill._reuse_alignment_pose(ctx)
    assert abs(skill._box_frame(ctx)[2])>skill.config.yaw_tolerance
    command=skill.step(ctx)
    assert command.state=='settle_at_stage'  # larger error needs coarse turn first


def test_timed_out_heading_pose_is_not_immediately_reused():
    skill,ctx=alignment_context('refine_heading',gap=.46,lateral=.12,heading=.03)
    assert skill._reuse_alignment_pose(ctx)
    command=skill._retry(ctx,'heading_refinement_timeout')
    assert command.state=='brake_for_recovery'
    assert not skill._reuse_alignment_pose(ctx)


def test_nearby_stage_handoff_cannot_reuse_a_rejected_actual_pose():
    skill,ctx=alignment_context('navigate_to_stage',gap=.46,lateral=.12)
    skill.stage=ctx.position[:2]+np.array([0.,.19]);skill.used_stages=[ctx.position[:2].copy()]
    command=skill.step(ctx)
    assert command.state=='navigate_to_stage' and command.navigation_goal is not None


def test_coarse_turn_hands_off_to_centering_without_unnecessary_stationary_trim():
    skill,ctx=alignment_context('turn_at_stage',gap=.46,lateral=.12,heading=.11)
    skill.turn=NS(update=lambda *args:None,failed=False)
    command=skill.step(ctx)
    assert command.state=='align_lateral' and command.status=='running'
    assert command.events[-1]['reused_turn_stop_for_standoff'] is True
    assert abs(skill._box_frame(ctx)[2])>skill.config.yaw_tolerance  # contact is not yet allowed
    ctx.now+=.01
    command=skill.step(ctx)
    assert command.state=='align_lateral' and command.velocity[1]>0.
    assert skill.retries==0  # checked centering starts without another idle dwell


@pytest.mark.parametrize('gap',[.15,.55])
def test_completed_turn_cannot_skip_an_unsafe_longitudinal_standoff(gap):
    skill,ctx=alignment_context('turn_at_stage',gap=gap,lateral=.12,heading=.11)
    skill.turn=NS(update=lambda *args:None,failed=False)
    command=skill.step(ctx)
    assert command.state=='align_longitudinal'
    assert command.events[-1]['reused_turn_stop_for_standoff'] is False


def test_completed_turn_rechecks_the_current_observed_body_footprint():
    skill,ctx=alignment_context('turn_at_stage',gap=.46,lateral=.12,heading=.11)
    skill.turn=NS(update=lambda *args:None,failed=False)
    row,col=ctx.planner.cell(ctx.position);ctx.planner.free[row,col]=False
    assert skill.step(ctx).state=='align_longitudinal'


def test_unfinished_turn_does_not_handoff_even_at_a_safe_standoff():
    skill,ctx=alignment_context('turn_at_stage',gap=.46,lateral=.12,heading=.11)
    skill.turn=NS(update=lambda *args:(np.zeros(3),{}),failed=False)
    assert skill.step(ctx).state=='turn_at_stage'


def test_final_contact_still_requires_heading_tolerance_and_uses_checked_trim():
    skill,ctx=alignment_context('settle_contact',gap=.46,lateral=0.,heading=.12)
    command=skill.step(ctx)
    assert command.state=='refine_heading' and command.velocity==(0.,0.,0.)
    assert command.events[-1]['reason']=='contact_heading_outside_tolerance'


def test_predicted_push_stop_must_settle_and_complete_measured_shortfall():
    skill=make_skill();skill.state='push';skill.started=0.;skill.state_since=0.
    skill.initial_box=np.array([0.,0.]);skill.push_progress=.8;skill.push_progress_time=1.
    skill.engage_box_position=skill.initial_box.copy()
    box=NS(id='B1',xy=(.83,0.),dimensions=(1.,1.,1.))
    ctx=context(box,1.)
    ctx.planner=NS(clearance=lambda box:{'side_parked':False})
    ctx.limiter=NS(preview=lambda command,now:command,measured=np.array([.4,0.,0.]),command=np.array([.8,0.,0.]))
    ctx.body_velocity=np.array([.4,0.,0.])
    result=skill.step(ctx)
    assert result.state=='settle_push_progress' and result.velocity==(0.,0.,0.)
    assert skill.push_brakes==1 and skill.status=='running'
    ctx.now=1.2
    assert skill.step(ctx).state=='settle_push_progress'
    ctx.body_velocity=np.zeros(3);ctx.limiter.command=np.zeros(3)
    ctx.now=1.3;skill.step(ctx)
    ctx.now=1.8
    assert skill.step(ctx).state=='engage'
    assert skill.release_reason!='side_parked'


def test_measured_park_after_predictive_braking_releases_without_another_push():
    skill=make_skill();skill.state='settle_push_progress';skill.started=0.;skill.state_since=0.
    skill.initial_box=np.array([0.,0.])
    ctx=context(NS(id='B1',xy=(.99,0.),dimensions=(1.,1.,1.)),3.1)
    ctx.planner=NS(clearance=lambda box:{'side_parked':True,'passage_clearance_margin_m':.05})
    ctx.limiter=NS(command=np.zeros(3))
    assert skill.step(ctx).state=='settle_push_progress'
    ctx.now=3.6
    assert skill.step(ctx).state=='retreat'
    assert skill.release_reason=='side_parked'


def test_previous_push_distance_does_not_brake_a_resume_before_box_motion():
    skill=make_skill();skill.state='engage';skill.started=0.;skill.state_since=0.
    skill.initial_box=np.array([.535,0.]);skill.engage_box_position=np.array([1.155,0.])
    skill.plan.push_distance=.85
    ctx=context(NS(id='B1',xy=(1.155,0.),dimensions=(.85,.55,.95)),1.)
    ctx.position=np.array([.22,0.,.4]);ctx.body_velocity=np.array([.45,0.,0.])
    ctx.planner=NS(clearance=lambda box:{'side_parked':False})
    ctx.limiter=NS(preview=lambda command,now:command,measured=np.array([.45,0.,0.]),command=np.array([.6,0.,0.]))
    command=skill.step(ctx)
    assert command.state=='push' and command.velocity[0]>0
    assert skill.push_brakes==0
    ctx.now=1.1;ctx.box.xy=(1.185,0.)
    assert skill.step(ctx).state=='settle_push_progress'
    assert skill.push_brakes==1


def test_retry_remembers_whether_it_came_from_actual_box_contact():
    for state,contact in [('settle_at_stage',False),('align_lateral',False),('push',True),('engage',True)]:
        skill=make_skill();skill.state=state
        result=skill._retry(context(),'diagnostic_recovery')
        assert skill.recovery_from_contact is contact
        assert result.events[-1]['recovery_source_state']==state
        assert result.events[-1]['recovery_from_contact'] is contact


@pytest.mark.parametrize('state',['settle_at_stage','settle_alignment','settle_recovered_preparation','settle_contact'])
def test_stationary_preparation_holds_through_transient_tilt_without_a_retry(state):
    skill,ctx=alignment_context(state,gap=.35,lateral=0.)
    ctx.layers=NS(refresh_global=lambda **kwargs:None)
    ctx.tilt_degrees=20.
    command=skill.step(ctx)
    assert command.state==state and command.velocity==(0.,0.,0.) and skill.retries==0
    ctx.now+=.2;ctx.tilt_degrees=14.
    assert skill.step(ctx).state==state  # original measured quiet interval still required
    ctx.now+=.41
    command=skill.step(ctx)
    assert command.state!=state and skill.retries==0


def test_stationary_preparation_preserves_hard_health_and_total_skill_deadlines():
    for violation in ['health','deadline']:
        skill,ctx=alignment_context('settle_recovered_preparation')
        ctx.tilt_degrees=30. if violation=='health' else 20.
        if violation=='deadline':ctx.now=skill.config.max_skill_seconds+.01
        command=skill.step(ctx)
        assert command.status=='failed' and command.velocity==(0.,0.,0.)
        assert command.reason==('robot_unstable' if violation=='health' else 'skill_timeout')


def test_stationary_preparation_cannot_wait_forever_below_hard_health_limit():
    skill,ctx=alignment_context('settle_recovered_preparation');ctx.tilt_degrees=20.
    ctx.now=skill.config.max_alignment_seconds+.01
    command=skill.step(ctx)
    assert command.state=='brake_for_recovery' and skill.retries==1
    assert skill.release_reason=='preparation_settling_timeout'


def test_completed_alignment_stop_hands_off_without_repeating_safe_standoff_wait():
    skill,ctx=alignment_context('settle_alignment',gap=.46,lateral=.3)
    assert skill.step(ctx).state=='settle_alignment'
    ctx.now+=.31
    assert skill.step(ctx).state=='align_lateral'
    ctx.now+=.01
    command=skill.step(ctx)
    assert command.state=='align_lateral' and command.velocity[1]>0.



def test_lateral_alignment_tracks_validated_body_centerline_while_heading_is_corrected():
    skill,ctx=alignment_context('align_lateral',gap=.35,lateral=.30,heading=.12)
    captured=[]
    def translate(context,target,axis,speed,tolerance):
        captured.append(np.array(target));return np.zeros(3),False
    skill._translation=translate
    assert skill.step(ctx).state=='align_lateral'
    assert len(captured)==1
    assert captured[0][1]==pytest.approx(ctx.box.xy[1])
    assert captured[0][0]==pytest.approx(ctx.position[0])


def test_body_already_on_centerline_is_not_moved_to_compensate_an_unfinished_yaw():
    heading=.12
    skill,ctx=alignment_context('align_lateral',gap=.35,lateral=-.33*math.sin(heading),heading=heading)
    assert abs(ctx.position[1]-ctx.box.xy[1])<1e-9
    assert abs(skill._box_frame(ctx)[1])>.035
    command=skill.step(ctx)
    assert command.velocity==(0.,0.,0.) and command.state=='align_lateral'
    ctx.now+=.31
    assert skill.step(ctx).state=='settle_contact'
    ctx.now+=.01
    assert skill.step(ctx).state=='refine_heading'  # actual heading condition still enforced
