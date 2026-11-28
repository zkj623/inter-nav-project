"""SDK and simulation share one skill while keeping explicit plant contracts."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from deploy.interaction.profile import ContactProfile
from deploy.tests.contact_fixtures import sdk_profile
from deploy.interaction.motion import SdkHeadingTrim as HeadingTrimControl, SdkNavigationMotion
from grutopia_extension.interactive_navigation.body_geometry import BodyFootprint
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import navigation_footprint_samples
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
from grutopia_extension.interactive_navigation.interaction_skill import SkillContext
from deploy.interaction.box_motion import ObservedBoxMotion
from deploy.interaction.skill_backend import create_push_skill, SdkPushBackend
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideConfig, PushAsideSkill
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter


def profile(**changes):
    # Synthetic test fixture, never a deploy calibration record.
    value=replace(sdk_profile(),backend='sdk',calibration_reference='synthetic-unit-test',
        max_forward_mps=.25,max_reverse_mps=.15,max_lateral_mps=.12,max_yaw_rps=.30,
        forward_mps=.15,reverse_mps=.12,lateral_mps=.10,push_start_mps=.15,push_max_mps=.20,
        min_translation_mps=.05,min_yaw_rps=.08,stall_seconds=2.)
    return replace(value,**changes)


def permission(**changes):
    p=InteractionPermission('B1','box','push_aside',('lidar:1:1','lidar:1:2'),(),0.,
        source='registered_box_lidar',source_reference='test-registry.json#B1',source_digest='a'*64)
    return replace(p,**changes)


def fixture(state='engage'):
    pr=profile()
    planner=SidePushPlanner(np.ones((160,160),bool),[[-4,4],[-4,4]],resolution=.05)
    SdkNavigationMotion(pr).bind(planner)
    planner.active_passages={'B1':{'axis':[0,1],'center':[.7,0]}}
    box=BoxState('B1',(.7,0.),(.4,.6,.6),local_dimensions=(.4,.6,.6),orientation=(1,0,0,0),center_z=.3)
    limiter=VelocitySlewLimiter(pr.linear_acceleration_mps2,pr.yaw_acceleration_rps2)
    limiter.accept([.1,0,0],0.)
    skill=create_push_skill(permission(),NS(box_id='B1',direction=(1.,0.),push_distance=.6),
        (0.,-1.),(0.,1.),pr)
    skill.state=state;skill.started=0.;skill.state_since=0.;skill.initial_box=np.array(box.xy)
    skill.engage_box_position=np.array(box.xy);skill.contact_started=0.
    ctx=SkillContext(.2,np.array([0.,0.,.4]),0.,np.zeros(3),0.,0.,True,box,[box],planner,None,limiter,
        body_angular_velocity=np.zeros(3),orientation=(1,0,0,0),contact_motion_check=lambda cmd,ctx:True,
        box_observed_at=.2,box_observation_id="scan:2",map_observed_at=.2,map_observation_id=2)
    return skill,ctx


def prime_stationary_observations(skill,ctx):
    ctx.now=1.2
    for index,stamp in enumerate(np.linspace(.3,1.1,9)):
        skill.backend.box_motion.observe(ctx.box,stamp,f'stationary:{index}',ctx.now)
    ctx.box_observed_at=1.1;ctx.box_observation_id='stationary:8'
    ctx.map_observed_at=1.2;ctx.map_observation_id=12
    skill.last_box_motion_time=skill.backend.box_motion.last_motion_at


def test_sdk_profile_requires_explicit_fields_and_calibration_provenance():
    with pytest.raises(ValueError,match='explicit fields'):
        ContactProfile.from_dict({'backend':'sdk','calibration_reference':'x'})
    with pytest.raises(ValueError,match='calibration_reference'):
        replace(sdk_profile(), calibration_reference='')
    assert ContactProfile.from_dict(profile().to_dict())==profile()
    with pytest.raises(ValueError,match='Unknown'):
        ContactProfile.from_dict({'verified':True})


@pytest.mark.parametrize('changes',[{'push_max_mps':1.},{'head_x_min_m':.9},
    {'stall_seconds':float('nan')},{'max_contact_seconds':1.},{'min_yaw_rps':.8}])
def test_invalid_sdk_response_or_geometry_is_rejected(changes):
    with pytest.raises(ValueError):profile(**changes)


def test_translation_limits_preserve_direction_and_do_not_reduce_yaw():
    pr=profile()
    value=pr.limit([1.,.5,.2])
    assert value[1]/value[0]==pytest.approx(.5)
    assert value[2]==.2 and value[0]<=pr.max_forward_mps and value[1]<=pr.max_lateral_mps
    assert pr.limit([-1.,0.,2.])[0]==-pr.max_reverse_mps
    assert pr.limit([-1.,0.,2.])[2]==pr.max_yaw_rps


def test_profile_geometry_controls_complete_navigation_footprint():
    pr=profile(foot_x_min_m=-.40,foot_x_max_m=.60,foot_half_width_m=.30)
    xx,yy=navigation_footprint_samples(NS(body_geometry=BodyFootprint(pr.foot_x_min_m,pr.foot_x_max_m,pr.foot_half_width_m),resolution=.05,navigation_half_width=.01))
    assert (xx.min(),xx.max(),yy.min(),yy.max())==(-.4,.6,-.3,.3)
    skill,ctx=fixture();skill=create_push_skill(permission(),skill.plan,(0,0),(1,0),pr)
    assert skill.config.front_contact_reach==pytest.approx(.60-pr.nose_offset_m)
    assert skill.config.reverse_speed==pr.reverse_mps


@pytest.mark.parametrize('changes',[{'evidence_frames':('rgb1','rgb2')},
    {'evidence_frames':('lidar:1:1',)}, {'classifier_models':('fake Qwen',)},
    {'source_digest':'invalid'},{'source_reference':' '}])
def test_registered_permission_cannot_forge_visual_or_missing_provenance(changes):
    with pytest.raises(ValueError,match='LiDAR provenance'):
        PushAsideSkill(permission(**changes),NS(box_id='B1'),(0,0),(1,0),backend=SdkPushBackend(profile()))


def test_sdk_yaw_stall_never_uses_simulation_pulse_or_priming_arc():
    pr=profile();planner=SidePushPlanner(np.ones((160,160),bool),[[-4,4],[-4,4]],resolution=.05)
    SdkNavigationMotion(pr).bind(planner)
    limiter=VelocitySlewLimiter();control=HeadingTrimControl(.25,profile=pr)
    control.update([0,0,.4],0.,planner,[],limiter,0.,np.zeros(3),np.zeros(3),0.)
    control.update([0,0,.4],0.,planner,[],limiter,.5,np.zeros(3),np.zeros(3),0.)
    command,done,_=control.update([0,0,.4],0.,planner,[],limiter,2.6,np.zeros(3),np.zeros(3),0.,policy_command=[0,0,1])
    assert control.failed=='sdk_heading_stalled' and not done
    assert control.last_detail['contact_backend'] == 'sdk'
    assert np.all(command==0.)


@pytest.mark.parametrize('refining', [False, True])
def test_upper_short_end_aims_for_six_degrees_and_accepts_settled_rebound(refining):
    from deploy.app.config import ROOT, load
    field = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')['interaction']
    base, ctx = fixture()
    skill = create_push_skill(permission(), base.plan, (0, 0), (1, 0), ContactProfile.from_dict(field['contact_profile']),
                              upper_short_end=True)
    turn = skill.backend.heading_trim(skill) if refining else skill.backend.new_turn(skill, ctx)
    limiter = VelocitySlewLimiter()

    def tick(degrees, now, measured_yaw=0.):
        velocity = np.array([0., 0., measured_yaw])
        limiter.observe(velocity, now)
        command, done, _ = turn.update(ctx.position, np.deg2rad(degrees), ctx.planner, [], limiter, now,
                                       np.zeros(3), velocity, 0.)
        limiter.accept(command, now)
        return command[2], done

    for t in np.arange(0., 1., .1):
        _, done = tick(7.4, t)
        assert not done
    yaw_request, done = tick(6.7, 1.)
    assert yaw_request < 0. and not done and not turn.braking
    tick(5.9, 1.1)
    assert turn.braking
    for t in np.arange(1.2, 2.2, .1):
        _, done = tick(6.7, t, measured_yaw=.12)
        assert not done
    for t in np.arange(2.2, 3.2, .1):
        _, done = tick(6.7, t)
    assert done


def test_sdk_contact_requires_scan_collision_check_and_stays_bounded():
    skill,ctx=fixture();ctx.contact_motion_check=None
    result=skill.step(ctx)
    assert result.status=='failed' and result.reason=='contact_motion_check_missing'
    skill,ctx=fixture();result=skill.step(ctx)
    assert result.status=='running' and 0<result.velocity[0]<=skill.backend.profile.push_max_mps


@pytest.mark.parametrize('kind',['callback','wall','unknown','box_sweep'])
def test_sdk_contact_blocks_non_target_geometry_and_unknown_without_map_edits(kind):
    skill,ctx=fixture()
    if kind=='callback':ctx.contact_motion_check=lambda cmd,ctx:False
    elif kind=='wall':ctx.boxes.append(BoxState('wall',(.35,.15),(.3,.2,1.)))
    elif kind=='unknown':ctx.planner.free[ctx.planner.cell((.1,.1))]=False
    else:ctx.planner.box_free[ctx.planner.cell((.93,0.))]=False
    original=ctx.planner.free.copy();result=skill.step(ctx)
    assert result.status=='failed' and result.velocity==(0.,0.,0.)
    np.testing.assert_array_equal(original,ctx.planner.free)


def test_sdk_contact_budget_covers_recovery_and_cannot_restart_on_retry():
    skill,ctx=fixture();skill._retry(ctx,'alignment')
    assert skill.contact_started==0.
    ctx.now=skill.backend.profile.max_contact_seconds
    result=skill.step(ctx)
    assert result.reason=='contact_budget_exhausted' and result.velocity==(0.,0.,0.)


def test_sdk_box_stall_fails_without_a_higher_speed_retry():
    skill,ctx=fixture('push');skill.push_progress=0.;skill.push_progress_time=0.;ctx.now=2.1
    result=skill.step(ctx)
    assert result.reason=='box_motion_stalled' and skill.retries==0 and result.velocity==(0.,0.,0.)


def test_sdk_brake_requires_measured_settle_instead_of_elapsed_shortcut():
    skill,ctx=fixture('brake_after_push');ctx.now=2.;ctx.body_velocity=np.array([.2,0,0])
    result=skill.step(ctx)
    assert result.state=='brake_after_push' and result.velocity==(0.,0.,0.)


def test_sdk_reverse_profile_applies_without_policy_feedback(monkeypatch):
    skill,ctx=fixture('retreat');skill.release_origin=np.zeros(2);skill.release_reason='alignment'
    seen=[]
    def retreat(*args,**kwargs):
        seen.append(kwargs['speed']);return np.array([-kwargs['speed'],0,0]),np.array([-.2,0])
    monkeypatch.setattr('grutopia_extension.interactive_navigation.push_aside_skill.observed_contact_retreat',retreat)
    monkeypatch.setattr(skill,'_observed_passage_effect',lambda *args,**kwargs:False)
    result=skill.step(ctx)
    assert seen==[skill.backend.profile.reverse_mps]
    assert result.velocity[0]==-skill.backend.profile.reverse_mps


def test_sdk_recovery_uses_measured_profile_without_native_stride_floor():
    from grutopia_extension.interactive_navigation.go2_voronoi_navigation import calibrated_buffer_translation
    _,ctx=fixture()
    command=calibrated_buffer_translation(ctx.planner,[],[0,0],0.,[-.2,0])
    assert command[0]==-ctx.planner.motion_backend.profile.reverse_mps and command[1]==0.


def test_sdk_contact_requires_the_measured_vertical_face():
    skill,ctx=fixture()
    assert skill._contact_face_clear(ctx)
    ctx.box=replace(ctx.box,local_dimensions=(.4,.6,.1),center_z=.05)
    assert not skill._contact_face_clear(ctx)


def test_sdk_passage_success_requires_scan_after_last_box_motion(monkeypatch):
    skill,ctx=fixture('verify_passage');ctx.position=np.array([-.3,0.,.4]);ctx.limiter.accept([0,0,0],0.)
    skill.initial_box=np.array([.1,0.]);skill.initial_box_geometry=replace(ctx.box,xy=(.1,0.))
    prime_stationary_observations(skill,ctx)
    skill.last_box=np.array(ctx.box.xy);skill.last_box_time=.1
    skill.last_verify=-1e9;skill.release_reason='side_parked'
    ctx.map_observed_at=.3
    ctx.layers=NS(last_local=.1,visible_box_ids=['B1'],current_boxes=[ctx.box],planner=ctx.planner,
        refresh_global=lambda **kwargs:None)
    ctx.planner.passages={'B1':{'axis':[0,1],'center':[.1,0]}}
    proof={'axis':[0,1],'center':[.1,0],'body_margin_m':.04}
    monkeypatch.setattr('grutopia_extension.interactive_navigation.push_aside_skill.observed_passage_routes',lambda *args:(proof,[(1,1)],[(1,1)]))
    monkeypatch.setattr('grutopia_extension.interactive_navigation.push_aside_skill.body_rotation_clear',lambda *args,**kwargs:False)
    first=skill.step(ctx)
    assert first.status=='running' and first.certificate is None
    ctx.now=1.5;ctx.layers.last_local=1.45;ctx.map_observed_at=1.45;ctx.map_observation_id=15
    ctx.box_observed_at=1.4;ctx.box_observation_id='new:14'
    second=skill.step(ctx)
    assert second.status=='succeeded'
    assert second.certificate['confirming_observation_time_s']>second.certificate['last_box_motion_time_s']


def test_sdk_profile_rejects_simulated_ground_truth_permission():
    with pytest.raises(ValueError,match='simulated ground-truth'):
        create_push_skill(permission(source='ground_truth_layout',evidence_frames=()),
            NS(box_id='B1'),(0,0),(1,0),profile())


def test_body_motion_clear_accepts_margin_and_protects_the_expanded_envelope():
    from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_motion_clear
    _,ctx=fixture()
    # Wall at x=.60 lies outside the .50 m front envelope, inside a .15 m reserve.
    ctx.planner.free[:,ctx.planner.cell((.60,0.))[1]:]=False
    assert body_motion_clear(ctx.planner,[],[0.,0.],0.,0.,0.)
    assert body_motion_clear(ctx.planner,[],[0.,0.],0.,0.,0.,margin=0.)
    assert not body_motion_clear(ctx.planner,[],[0.,0.],0.,0.,0.,margin=.15)


def test_staging_planning_hook_yields_before_mutating_the_preparation_target():
    skill,ctx=fixture('prepare')
    class PlanningRequired(Exception):pass
    reasons=[]
    def boundary(reason):
        reasons.append(reason);raise PlanningRequired
    ctx.planner.before_contact_planning=boundary
    skill.stage=np.array([9.,8.]);skill.anchor=np.array([7.,6.]);skill.axis=np.array([0.,1.])
    with pytest.raises(PlanningRequired):skill._select_stage(ctx)
    assert reasons==['skill_prepare_stage'] and skill.state=='prepare'
    np.testing.assert_array_equal(skill.stage,[9.,8.]);np.testing.assert_array_equal(skill.anchor,[7.,6.])
    np.testing.assert_array_equal(skill.axis,[0.,1.])
    assert skill.events==[]


def test_passage_planning_hook_preserves_probe_and_passage_state_for_retry():
    skill,ctx=fixture('retreat');ctx.position=np.array([-.3,0.,.4]);ctx.limiter.accept([0,0,0],0.)
    skill.initial_box=np.array([.1,0.]);skill.initial_box_geometry=replace(ctx.box,xy=(.1,0.))
    prime_stationary_observations(skill,ctx)
    skill.last_box=np.array(ctx.box.xy);skill.last_box_time=.1
    skill.release_origin=np.zeros(2);skill.release_reason='side_parked';skill.last_release_probe=-10.
    current={'axis':[0,1],'center':[.1,0]};ctx.planner.passages={'B1':current.copy()}
    ctx.layers=NS(last_local=.15,visible_box_ids=['B1'],current_boxes=[ctx.box],planner=ctx.planner)
    class PlanningRequired(Exception):pass
    def boundary(reason):
        assert reason=='skill_passage_confirmation';raise PlanningRequired
    ctx.planner.before_contact_planning=boundary
    with pytest.raises(PlanningRequired):skill.step(ctx)
    assert skill.state=='retreat' and skill.last_release_probe==-10. and skill.certificate is None
    assert ctx.planner.passages=={'B1':current}
    assert skill.events==[]


def test_sdk_stationary_lidar_jitter_settles_using_distinct_source_scans():
    pr=profile(box_position_tolerance_m=.004,box_settle_seconds=.8,box_max_settled_speed_mps=.002)
    observer=ObservedBoxMotion(pr)
    _,ctx=fixture()
    for index in range(21):
        t=index*.1
        box=replace(ctx.box,xy=(.7+(-1)**index*.0005,0.))
        observer.observe(box,t,f'scan:{index}',t+.02)
    assert observer.settled
    assert observer.last_motion_at<observer.latest_at
    assert observer.diagnostic['independent_scans']>=3
    assert observer.diagnostic['max_surface_speed_mps']<=pr.box_max_settled_speed_mps
    assert observer.diagnostic['calibration_reference']=='synthetic-unit-test'


@pytest.mark.parametrize('mode',['translation','rotation'])
def test_sdk_sustained_small_frame_changes_are_not_mistaken_for_static_jitter(mode):
    pr=profile(box_position_tolerance_m=.005,box_settle_seconds=.8,box_max_settled_speed_mps=.002)
    observer=ObservedBoxMotion(pr)
    _,ctx=fixture()
    for index in range(31):
        t=index*.1
        if mode=='translation':
            box=replace(ctx.box,xy=(.7+.004*t,0.))  # 0.4 mm/frame, well below the 5 mm bound.
        else:
            yaw=.02*t
            box=replace(ctx.box,orientation=(np.cos(yaw/2),0.,0.,np.sin(yaw/2)))
        observer.observe(box,t,f'scan:{index}',t+.02)
        if t>=pr.box_settle_seconds:assert not observer.settled
    assert observer.last_motion_at==observer.latest_at
    assert observer.diagnostic['max_surface_speed_mps']>pr.box_max_settled_speed_mps


def test_sdk_repeated_or_relabelled_source_frames_do_not_accumulate_a_stability_window():
    observer=ObservedBoxMotion(profile());_,ctx=fixture()
    observer.observe(ctx.box,1.,'scan:1',1.)
    for now in [1.5,2.,10.]:observer.observe(ctx.box,1.,'scan:1',now)
    assert not observer.settled and observer.diagnostic['window_seconds']==0.
    with pytest.raises(ValueError,match='monotonically'):
        observer.observe(ctx.box,1.,'scan:other',10.)
    with pytest.raises(ValueError,match='repeated box source'):
        observer.observe(replace(ctx.box,xy=(.71,0)),1.,'scan:1',10.)


def test_sdk_confirmation_rejects_processing_newer_but_source_older_map():
    skill,ctx=fixture('verify_passage');prime_stationary_observations(skill,ctx)
    skill.initial_box_geometry=replace(ctx.box,xy=(.1,0.));skill.engage_box_position=np.array([.1,0.])
    ctx.layers=NS(last_local=1000.,visible_box_ids=['B1'])
    ctx.map_observed_at=skill.backend.box_motion.last_motion_at
    assert not skill._observed_passage_effect(ctx,.6)
    ctx.map_observed_at=skill.backend.box_motion.latest_at-.01
    assert not skill._observed_passage_effect(ctx,.6)


def test_sdk_stability_window_restarts_after_unobserved_gap():
    skill,ctx=fixture();prime_stationary_observations(skill,ctx)
    assert skill.backend.box_motion.settled
    skill.backend.box_motion.observe(ctx.box,3.,'scan:after-gap',3.)
    assert not skill.backend.box_motion.settled and skill.backend.box_motion.diagnostic['independent_scans']==1


@pytest.mark.parametrize('motion_source', ['requested', 'measured'])
def test_yaw_collision_check_covers_measured_rotational_stop_response(motion_source):
    pr = profile(translation_stop_seconds=.1, yaw_stop_seconds=1.5)
    planner = SidePushPlanner(np.ones((160, 160), bool), [[-4, 4], [-4, 4]], resolution=.05)
    backend = SdkNavigationMotion(pr)
    backend.bind(planner)
    # Clear at the current pose, hit by the front corner during yaw coast.
    planner.free[planner.cell((.4, .4))] = False
    command = np.array([0., 0., .3]) if motion_source == 'requested' else np.zeros(3)
    measured = np.array([0., 0., .3]) if motion_source == 'measured' else np.zeros(3)
    limiter = NS(preview=lambda value, now: np.asarray(value), measured=measured,
                 linear_acceleration=pr.linear_acceleration_mps2)
    output, safe = backend.checked_contact_command(command, [0., 0., .4], 0., planner, [], limiter, 1.)
    assert not safe
    np.testing.assert_array_equal(output, np.zeros(3))


def test_local_sdk_yaw_command_uses_the_same_stop_envelope():
    pr = profile(translation_stop_seconds=.1, yaw_stop_seconds=1.5)
    planner = SidePushPlanner(np.ones((160, 160), bool), [[-4, 4], [-4, 4]], resolution=.05)
    backend = SdkNavigationMotion(pr)
    backend.bind(planner)
    planner.free[planner.cell((.4, .4))] = False
    command, detail = backend.local_action({'move_by_speed': [0., 0., .3]}, {}, [0., 0.], 0., planner, [])
    assert detail['needs_replan']
    np.testing.assert_array_equal(command['move_by_speed'], np.zeros(3))


@pytest.mark.parametrize('stop', ['displacement', 'travel', 'time'])
def test_first_push_low_box_actual_contact_command_and_stop_threshold(stop):
    from deploy.app.config import ROOT, load
    from deploy.interaction.first_push import FirstPushSkill
    field = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')['interaction']
    p = ContactProfile.from_dict(field['contact_profile'])
    _, ctx = fixture()
    SdkNavigationMotion(p).bind(ctx.planner)
    ctx.box = BoxState('B1', (.515, 0.), (.36, .86, .36), local_dimensions=(.30,.80,.30),
                       orientation=(1.,0.,0.,0.), center_z=.15)
    ctx.boxes = [ctx.box]
    ctx.position = np.array([0.,0.,.311])
    ctx.limiter.accept(np.zeros(3), ctx.now-.10)
    skill = create_push_skill(permission(), NS(box_id='B1', direction=(1.,0.), push_distance=.6),
                              (-1.,0.),(1.,0.),p)
    assert isinstance(skill, FirstPushSkill)
    skill.state, skill.started, skill.state_since, skill.contact_started = 'engage', .1, .1, .1
    skill.initial_box = np.array(ctx.box.xy)
    skill.engage_box_position = np.array(ctx.box.xy)
    assert skill._box_frame(ctx)[0] == pytest.approx(.025)
    assert skill._contact_face_clear(ctx, moving=True)
    high = replace(ctx, position=np.array([0.,0.,.40]))
    assert not skill._contact_face_clear(high, moving=True)
    command = skill.step(ctx)
    assert command.status == 'running' and 0. < np.linalg.norm(command.velocity[:2]) <= p.push_max_mps
    if stop == 'displacement':
        ctx.now = ctx.box_observed_at = 1.0
        ctx.box_observation_id = 'scan:3'
        ctx.box = replace(ctx.box, xy=(.56,0.))
        assert skill.step(ctx).status == 'running'
        ctx.box = replace(ctx.box, xy=(.62,0.))
    elif stop == 'travel':
        ctx.position[0] += .07
    else:
        ctx.now += skill.stall_seconds
    result = skill.step(ctx)
    assert result.status == 'checking' and not any(result.velocity)
    assert result.reason == {'displacement':'box_reached_trial_distance', 'travel':'first_push_body_travel_limit',
                             'time':'first_push_time_limit'}[stop]
    for i in range(1,7):
        t = ctx.now+i*.12
        observed = skill.check_stopped_box(t, ctx.position, ctx.box, t, str(i), t)
        if observed is not None:
            break
    assert observed is (stop == 'displacement')
    assert skill.report()['passage_certificate'] is None


def test_trial_selects_box_faces_at_a_non_grid_heading():
    from deploy.interaction.first_push import RegisteredFacePlanner
    from math import cos, sin, radians
    angle=radians(23)
    planner=RegisteredFacePlanner(np.ones((60,60),bool),[[-3,3],[-3,3]],resolution=.1)
    aabb=(.3*cos(angle)+.8*sin(angle)+.06, .3*sin(angle)+.8*cos(angle)+.06, .36)
    box=BoxState('B1',(0.,0.),aabb, local_dimensions=(.30,.80,.30),
                 orientation=(cos(angle/2),0.,0.,sin(angle/2)), center_z=.15)
    oriented=planner.oriented_box(box)
    assert oriented.yaw == pytest.approx(23) and oriented.dimensions[:2] == pytest.approx((.36,.86))
    for direction in planner.push_directions(oriented):
        axes=np.array([[cos(angle),sin(angle)],[-sin(angle),cos(angle)]])
        assert max(abs(axes@direction)) == pytest.approx(1.)


@pytest.mark.parametrize('yaw', [12., 192., 90.])
def test_field_plan_only_approaches_the_upper_short_end(yaw):
    from deploy.interaction.first_push import RegisteredFacePlanner
    planner = RegisteredFacePlanner(np.ones((60, 60), bool), [[-3, 3], [-3, 3]], resolution=.1)
    angle = np.deg2rad(yaw)
    dims = (.3, .8, .3)
    c, s = abs(np.cos(angle)), abs(np.sin(angle))
    box = BoxState('B1', (0., 0.), (c*.3+s*.8, s*.3+c*.8, .3), local_dimensions=dims,
                   orientation=(np.cos(angle/2), 0., 0., np.sin(angle/2)), center_z=.15)
    # All push-space observations are pending. The walking route is known.
    unverified = np.zeros_like(planner.free)
    plan = planner.first_contact_plan((-1.5, 0.), [box], {'B1'}, unverified, profile(), upper_end_goal=(1.5, 0.))
    if yaw == 90.:  # Both short ends face sideways.
        assert plan is None
        return
    assert plan.approach_xy[1] > box.xy[1] and plan.direction[1] < -.95
    assert abs(np.dot(plan.direction, [np.cos(angle), np.sin(angle)])) < 1e-6
    planner.static_clear[planner.cell(plan.approach_xy)] = False
    assert planner.first_contact_plan((-1.5, 0.), [box], {'B1'}, unverified, profile(), upper_end_goal=(1.5, 0.)) is None


def test_first_push_moves_the_turn_away_from_a_blocked_front_corner():
    from deploy.interaction.first_push import RegisteredFacePlanner
    # Field frame 945: the base was within the 16 cm arrival band. The front
    # left foot crossed the (1.15, -1.15) cell about halfway through the turn.
    planner = RegisteredFacePlanner(np.ones((39, 66), bool), [[-1.4, 5.2], [-2.8, 1.1]], resolution=.1)
    SdkNavigationMotion(profile()).bind(planner)
    position = np.array([.625, -.907])
    heading, target = np.deg2rad([-67., -2.])
    direction = np.array([np.cos(target), np.sin(target)])
    plan = NS(direction=direction, approach_xy=np.array([.626, -1.036]))
    box = BoxState('box-1', (1.487, -1.036), (.314, .805, .30))
    planner.free[planner.cell((1.15, -1.15))] = False
    backoff = planner.turn_standoff(position, heading, plan, [box], 0., .16)
    assert backoff == .25
    stage = plan.approach_xy - backoff*direction
    assert planner.turn_standoff(stage, heading, plan, [box], backoff, .16) == backoff
    planner.free[:] = False
    assert planner.turn_standoff(position, heading, plan, [box], 0., .16) is None


@pytest.mark.parametrize('wall', [False, True])
def test_field_sidestep_keeps_a_safe_speed_when_its_next_acceleration_is_blocked(wall):
    from deploy.app.config import ROOT, load
    cfg = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    p = ContactProfile.from_dict(cfg['interaction']['contact_profile'])
    _, ctx = fixture()
    SdkNavigationMotion(p).bind(ctx.planner)
    ctx.boxes = []
    if wall:
        # The target at Y=.18 fits, but the body cannot sweep far beyond it.
        row, _ = ctx.planner.cell((0., .49))
        ctx.planner.free[row:, :] = False
    skill = create_push_skill(permission(), NS(box_id='B1', direction=(1., 0.), push_distance=.1),
                              (0., 0.), (1., 0.), p, upper_short_end=True)
    skill.state, skill.push_heading = 'align_lateral', 0.
    ctx.limiter.linear_acceleration = cfg['navigation']['linear_acceleration']
    ctx.limiter.accept(np.zeros(3), 0.)
    for i in range(1, 16):
        ctx.now = i * .1
        v, done = skill.backend.translate(skill, ctx, np.array([0., .18]), np.array([0., 1.]), p.lateral_mps, .03)
        assert not done and skill.backend.motion_clear(skill, ctx, v, contact=False)
        ctx.limiter.accept(v, ctx.now)
    if wall:
        assert .10 < v[1] < p.lateral_mps
    else:
        assert v[1] == pytest.approx(.40)


def test_upper_face_uses_a_side_turn_when_the_center_is_only_clear_after_turning():
    from deploy.interaction.first_push import RegisteredFacePlanner
    from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_rotation_clear, body_translation_recovery_clear
    planner = RegisteredFacePlanner(np.ones((60, 60), bool), [[-3, 3], [-3, 3]], resolution=.1)
    p = profile()
    SdkNavigationMotion(p).bind(planner)
    dims = (.30, .80, .30)
    box = BoxState('B1', (0., 0.), dims, local_dimensions=dims, orientation=(1., 0., 0., 0.), center_z=.15)
    anchor = np.array([0., .40 + p.nose_offset_m + PushAsideConfig.front_gap])
    start = anchor - [1., 0.]
    planner.free[planner.cell(anchor + [.43, 0.])] = False
    plan = planner.first_contact_plan(start, [box], {'B1'}, planner.free, p, upper_end_goal=(1.5, 0.))
    assert plan.direction == (0., -1.) and plan.approach_xy[0] == pytest.approx(-.10)
    assert not body_rotation_clear(planner, [box], anchor, 0., -np.pi/2)
    assert body_rotation_clear(planner, [box], plan.approach_xy, 0., -np.pi/2)
    assert body_translation_recovery_clear(planner, [box], plan.approach_xy, -np.pi/2, anchor)


def test_first_push_checks_where_the_head_will_land_on_a_narrow_face():
    from deploy.app.config import ROOT, load
    p = ContactProfile.from_dict(load(interaction_path=ROOT/'configs/box_30x80x30.yaml')['interaction']['contact_profile'])
    original, ctx = fixture()
    skill = create_push_skill(permission(), original.plan, (0., 0.), (1., 0.), p)
    ctx.box = replace(ctx.box, xy=(1.1, 0.), local_dimensions=(.8, .3, .3), center_z=.15)
    ctx.position = np.array([0., 0., .311])
    assert skill._contact_face_clear(ctx)
    angle = np.deg2rad(12.)
    ctx.heading, ctx.orientation = angle, (np.cos(angle/2), 0., 0., np.sin(angle/2))
    assert abs(angle) < skill.config.yaw_tolerance
    assert not skill._contact_face_clear(ctx)


@pytest.mark.parametrize('blocked_layer,tight_reserve',
                         [(None, False), ('free', False), ('box_free', False), (None, True)])
def test_first_push_needs_recent_face_after_measured_settling(blocked_layer, tight_reserve):
    from deploy.app.config import ROOT, load
    p = ContactProfile.from_dict(load(interaction_path=ROOT/'configs/box_30x80x30.yaml')['interaction']['contact_profile'])
    _, ctx = fixture()
    SdkNavigationMotion(p).bind(ctx.planner)
    ctx.box = BoxState('B1',(.82,0.),(.36,.86,.36),local_dimensions=(.3,.8,.3),
                       orientation=(1.,0.,0.,0.),center_z=.15)
    ctx.position=np.array([0.,0.,.31]); ctx.boxes=[ctx.box]
    ctx.limiter.accept(np.zeros(3),0.)
    skill=create_push_skill(permission(),NS(box_id='B1',direction=(1.,0.),push_distance=.6),
                            (-1.,0.),(1.,0.),p)
    skill.state, skill.started, skill.state_since='settle_contact',.1,.1
    skill.initial_box=np.array(ctx.box.xy)
    # This cell is beyond the fitted box, inside its forward short-push sweep.
    cell = ctx.planner.cell((1.19, 0.))
    if blocked_layer:
        getattr(ctx.planner, blocked_layer)[cell] = False
    if tight_reserve:
        # This cell lies beyond the slower push's braking reserve. Leave it
        # occupied and verify that the skill selects the executable speed.
        ctx.planner.box_free[ctx.planner.cell((1.39, 0.))] = False
    saw_block = False
    for i in range(1,20):
        if blocked_layer and i == 13:
            getattr(ctx.planner, blocked_layer)[cell] = True
        ctx.now=.1+i*.10
        ctx.map_observed_at=ctx.now
        ctx.box_observation_id=str(i) if i%2==0 else None
        ctx.box_observed_at=ctx.now if ctx.box_observation_id else None
        command=skill.step(ctx)
        assert not any(command.velocity)
        if blocked_layer and i < 13:
            assert skill.contact_started is None
            saw_block |= skill.precontact_wait == 'push_path_blocked_or_unknown'
        if command.state=='engage':
            assert ctx.box_observation_id and skill.backend.box_motion.settled
            break
        assert skill.contact_started is None
    assert command.state=='engage'
    assert skill.config.push_max_speed == (p.push_start_mps if tight_reserve else p.push_max_mps)
    if blocked_layer:
        assert saw_block
