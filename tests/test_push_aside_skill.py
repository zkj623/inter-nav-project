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


def test_precontact_turn_recovers_before_hard_health_limit():
    skill=make_skill();skill.state='turn_at_stage'
    ctx=context(NS(id='B1',xy=(1.,0.)))
    ctx.tilt_degrees=20.
    command=skill.step(ctx)
    assert command.state=='brake_for_recovery' and skill.release_reason=='preparation_attitude'
    assert command.velocity==(0.,0.,0.) and skill.retries==1
