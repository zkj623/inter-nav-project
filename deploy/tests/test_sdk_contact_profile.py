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
