from types import SimpleNamespace as NS
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.go2_interaction_motion import checked_skill_action
from grutopia_extension.interactive_navigation.go2_stop_feedback import ForwardStopFeedback
from grutopia_extension.interactive_navigation.go2_contact_turn import checked_contact_turn_command
from grutopia_extension.interactive_navigation.interaction_skill import SkillCommand
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter


def context():
    planner=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05)
    planner.full_navigation_footprint=True;planner.navigation_half_width=.22
    limiter=VelocitySlewLimiter();limiter.accept([.35,0,0],0.)
    limiter.observe([.26,0,0],0.)
    return NS(position=np.array([0.,0.,.4]),heading=0.,planner=planner,boxes=[],box=None,
              limiter=limiter,now=.01,healthy=True,tilt_degrees=14.)


def checker(ctx):
    return lambda target,limiter,now:checked_contact_turn_command(
        target,ctx.position,ctx.heading,ctx.planner,ctx.boxes,limiter,now)


def test_subnavigation_stop_survives_nonzero_skill_braking_residual_and_later_hold():
    ctx=context();servo=ForwardStopFeedback();check=checker(ctx)
    servo.apply([.35,0,0],[.26,0,0],0.)
    servo.apply([.3435,0,0],[.26,0,0],.01,stop_requested=True,checker=check)
    assert servo.active
    previous=servo.limiter.command.copy()
    for tick in range(2,102):
        ctx.now=tick*.01
        skill=SkillCommand('settle_at_stage')
        action=checked_skill_action(skill,ctx,forward_stop_feedback=True)
        if tick==2:assert action['move_by_speed'][0]>0.  # This used to release the hold.
        sent=servo.apply(action['move_by_speed'],[.15,0,0],ctx.now,
            stop_requested=action.get('_forward_stop_request',False),checker=check)
        assert servo.active and servo.last_detail['mode']=='forward_hold'
        assert np.linalg.norm(sent[:2]-previous[:2])<=.0065+1e-8
        ctx.limiter.accept(action['move_by_speed'],ctx.now);previous=sent
    assert sent[0]<0.
    ctx.now=1.02
    action=checked_skill_action(SkillCommand('turn_at_stage',(0.,0.,1.)),ctx,forward_stop_feedback=True)
    servo.apply(action['move_by_speed'],[0,0,0],ctx.now,
        stop_requested=action.get('_forward_stop_request',False),checker=check)
    assert not servo.active


@pytest.mark.parametrize('prior',[[0,.35,0],[0,0,1.],[-.35,0,0]])
def test_skill_stop_does_not_expand_forward_eligibility(prior):
    ctx=context();ctx.limiter.accept([0,0,0],0.);ctx.limiter.observe([0,0,0],.01)
    servo=ForwardStopFeedback();servo.apply(prior,prior,0.)
    action=checked_skill_action(SkillCommand('settle_contact'),ctx,forward_stop_feedback=True)
    servo.apply(action['move_by_speed'],[0,0,0],ctx.now,
        stop_requested=action.get('_forward_stop_request',False))
    assert not servo.active


@pytest.mark.parametrize('state',['engage','push','settle_push_progress','brake_after_push'])
def test_contact_phase_does_not_arm_skill_stop_feedback(state):
    action=checked_skill_action(SkillCommand(state),context(),forward_stop_feedback=True)
    assert not action.get('_forward_stop_request',False)


def test_recovery_source_controls_stop_intent_and_navigation_keeps_its_controller():
    command=SkillCommand('brake_for_recovery');ctx=context()
    assert checked_skill_action(command,ctx,forward_stop_feedback=True,recovery_from_contact=False)['_forward_stop_request']
    assert not checked_skill_action(command,ctx,forward_stop_feedback=True,recovery_from_contact=True).get('_forward_stop_request',False)
    action=checked_skill_action(SkillCommand('navigate_to_stage',navigation_goal=(1.,2.)),ctx,forward_stop_feedback=True)
    assert not action.get('_forward_stop_request',False)
    assert not checked_skill_action(SkillCommand('settle_at_stage'),ctx,forward_stop_feedback=False).get('_forward_stop_request',False)


def test_new_obstacle_and_health_failure_still_prevent_feedback_motion():
    ctx=context();servo=ForwardStopFeedback();servo.apply([.35,0,0],[.26,0,0],0.)
    ctx.planner.free[:]=False
    action=checked_skill_action(SkillCommand('settle_at_stage'),ctx,forward_stop_feedback=True)
    sent=servo.apply(action['move_by_speed'],[.26,0,0],ctx.now,
        stop_requested=action.get('_forward_stop_request',False),checker=checker(ctx))
    np.testing.assert_array_equal(sent,[0,0,0]);assert not servo.last_detail['motion_safe']
    ctx.healthy=False
    action=checked_skill_action(SkillCommand('settle_at_stage'),ctx,forward_stop_feedback=True)
    assert not action.get('_forward_stop_request',False)
    sent=servo.apply(action['move_by_speed'],[.26,0,0],.02,healthy=False,checker=checker(ctx))
    np.testing.assert_array_equal(sent,[0,0,0]);assert not servo.active
