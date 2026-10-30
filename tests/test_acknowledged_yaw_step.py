import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

import grutopia_extension.interactive_navigation.go2_heading_trim as module
from grutopia_extension.interactive_navigation.go2_heading_trim import HeadingTrimControl,YawStepPulse
from grutopia_extension.interactive_navigation.interaction_skill import SkillContext
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter


@pytest.mark.parametrize('direction',[1.,-1.])
def test_transient_body_alignment_is_not_an_acknowledged_yaw_step(direction):
    pulse=YawStepPulse(direction*.14,-direction*.14,0.)
    request,detail=pulse.update(0.,[0.,0.,0.],.1)
    assert request[2]==direction*.55 and not detail['input_complete']
    assert not pulse.update(0.,[0.,0.,direction*.55],.2)[1]['input_complete']
    assert not pulse.update(0.,[0.,0.,direction*.55],.239)[1]['input_complete']
    request,detail=pulse.update(0.,[0.,0.,direction*.55],.241)
    assert detail['input_complete'] and np.all(request==0.)


def test_wrong_sign_and_interrupted_delivery_do_not_accumulate_hold_time():
    pulse=YawStepPulse(.14,-.14,0.)
    assert not pulse.update(-.1,[0.,0.,-.55],.1)[1]['input_complete']
    pulse.update(-.1,[0.,0.,.55],.2)
    pulse.update(-.1,[0.,0.,.50],.23)
    pulse.update(-.1,[0.,0.,.55],.25)
    assert not pulse.update(-.1,[0.,0.,.55],.289)[1]['input_complete']
    assert pulse.update(-.1,[0.,0.,.55],.291)[1]['input_complete']


@pytest.mark.parametrize('feedback,heading,now,reason',[
    ([0.,0.,0.],-.14,3.01,'heading_step_input_timeout'),
    ([0.,0.,.55],.32,.1,'heading_step_excursion_limit'),
    (None,-.14,.1,'heading_step_feedback_unavailable'),
    ([0.,0.,float('nan')],-.14,.1,'heading_step_feedback_unavailable'),
])
def test_missing_delivery_or_excessive_excursion_stops_the_request(feedback,heading,now,reason):
    pulse=YawStepPulse(.14,-.14,0.)
    request,detail=pulse.update(heading,feedback,now)
    assert detail['failure']==reason and not detail['input_complete']
    assert np.all(request==0.)


def setup(monkeypatch,heading=-.14):
    planner=SidePushPlanner(np.ones((100,100),bool),[[-2.5,2.5],[-2.5,2.5]],resolution=.05)
    planner.navigation_half_width=.22;planner.full_navigation_footprint=True
    planner.layered_local_control=True;planner.planning_scope='local';planner.planning_observed=np.ones_like(planner.free)
    original=module.body_motion_clear
    monkeypatch.setattr(module,'body_motion_clear',lambda p,b,x,h,v,w,**kw:False if abs(v)>1e-8 else original(p,b,x,h,v,w,**kw))
    limiter=VelocitySlewLimiter();limiter.accept([0.,0.,0.],0.);limiter.observe([0.,0.,0.],0.)
    control=HeadingTrimControl(0.,.05,acceptance_tolerance=.10,stalled_seconds=0.)
    position=np.array([0.,0.,.41])
    def step(now,angle=heading,feedback=(0.,0.,0.),position_override=None):
        pos=position if position_override is None else np.asarray(position_override,float)
        planner.control_time=now
        command,done,detail=control.update(pos,angle,planner,[],limiter,now,np.zeros(3),np.zeros(3),0.,policy_command=feedback)
        limiter.accept(command,now);limiter.observe([0.,0.,0.],now)
        return command,done,detail
    return control,planner,limiter,step


def test_measured_initial_stop_precedes_any_pulse(monkeypatch):
    control,_,_,step=setup(monkeypatch)
    assert np.all(step(0.)[0]==0.) and control.yaw_step is None
    assert np.all(step(1.)[0]==0.) and control.yaw_step is None
    step(1.21)
    assert not control.braking and control.yaw_step is None
    command,done,detail=step(1.22)
    assert not done and control.yaw_step is not None and command[2]>0.
    assert detail['mode']=='calibrated_yaw_step'


def test_pulse_delivery_still_requires_a_new_actual_heading_and_stop_check(monkeypatch):
    control,_,limiter,step=setup(monkeypatch)
    step(0.);step(1.21);step(1.22)
    limiter.accept([0.,0.,.55],1.29)
    assert not step(1.3,angle=-.02,feedback=(0.,0.,.55))[1]
    assert control.yaw_step is not None  # transient body angle does not stop it
    command,done,_=step(1.341,angle=-.02,feedback=(0.,0.,.55))
    assert not done and control.braking and control.yaw_step is None
    limiter.accept([0.,0.,0.],1.39)
    assert not step(1.4,angle=-.063)[1]
    assert not step(2.5,angle=-.063)[1]
    assert step(2.61,angle=-.063)[1]


def test_already_acceptable_stopped_heading_needs_no_pulse(monkeypatch):
    control,_,_,step=setup(monkeypatch,heading=-.063)
    step(0.)
    assert step(1.21)[1] and control.yaw_step_count==0


def test_missing_policy_feedback_keeps_the_legacy_controller_path(monkeypatch):
    control,_,_,step=setup(monkeypatch)
    step(0.,feedback=None);step(1.21,feedback=None)
    command,done,_=step(1.22,feedback=None)
    assert command[2]>0 and not done
    assert control.yaw_step is None and control.yaw_step_count==0


def test_new_obstacle_blocks_a_pulse_without_faking_delivery(monkeypatch):
    control,planner,_,step=setup(monkeypatch)
    step(0.);step(1.21);step(1.22)
    planner.free[:]=False;planner.box_free[:]=False
    command,done,_=step(1.3)
    assert np.all(command==0.) and not done
    command,done,_=step(4.23)
    assert np.all(command==0.) and not done
    assert control.failed=='heading_step_input_timeout'


def test_translation_budget_interrupts_the_pulse(monkeypatch):
    control,_,_,step=setup(monkeypatch)
    step(0.);step(1.21);step(1.22)
    step(1.3,position_override=[.26,0.,.41])
    assert control.travel_budget_exhausted and control.braking and control.yaw_step is None


def test_runner_context_copies_actual_policy_input_not_requested_velocity():
    source=Path(__file__).parents[1]/'grutopia/demo/go2_side_clearance_navigation.py'
    calls=[n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id=='SkillContext']
    assert len(calls)==1
    actual=np.array([0.,0.,.1]);requested=np.array([0.,0.,.55])
    ctrl=NS(body_lin_vel=np.zeros(3),body_ang_vel=np.zeros(3),policy_command=actual,
        observation_dim=45,policy=NS(go2_deployment_contract={'observation_profile':'unitree_go2_45'}))
    names={'SkillContext':SkillContext,'policy_feedback_for_heading_step':module.policy_feedback_for_heading_step,'planner':NS(control_time=1.),'p':np.array([0.,0.,.41]),'heading':0.,
        'robot':NS(_go2_locomotion_control=ctrl),'tilt':0.,'task_health':NS(healthy=True),
        'by_id':{'B1':None},'plan':NS(box_id='B1'),'current':[],'layers':None,
        'smooth_limiter':NS(command=requested),'sub_reached':False,'q':np.array([1.,0.,0.,0.])}
    context=eval(compile(ast.Expression(calls[0]),str(source),'eval'),names)
    np.testing.assert_array_equal(context.policy_command,actual)
    assert not np.array_equal(context.policy_command,requested)
    actual[2]=.55
    assert context.policy_command[2]==.1


def test_push_skill_forwards_the_actual_policy_feedback():
    from test_push_aside_skill import alignment_context
    skill,context=alignment_context('refine_heading',heading=-.14)
    context.policy_command=np.array([0.,0.,.23]);captured={}
    def update(*args,**kwargs):
        captured.update(kwargs)
        return np.zeros(3),False,{}
    skill.heading_trim=NS(update=update,failed=None)
    skill.step(context)
    np.testing.assert_array_equal(captured['policy_command'],context.policy_command)


@pytest.mark.parametrize('tilt,expected_state',[(18.1,'brake_for_recovery'),(30.,'failed')])
def test_existing_attitude_guards_precede_any_heading_step(tilt,expected_state):
    from test_push_aside_skill import alignment_context
    skill,context=alignment_context('refine_heading',heading=-.14)
    context.policy_command=np.array([0.,0.,.55]);context.tilt_degrees=tilt
    def unexpected(*args,**kwargs):raise AssertionError('heading controller must not run past the attitude gate')
    skill.heading_trim=NS(update=unexpected,failed=None)
    assert skill.step(context).state==expected_state


@pytest.mark.parametrize('dimension,contract,expected',[
    (235,{},False),(45,{},False),(235,{'observation_profile':'unitree_go2_45'},False),
    (45,{'observation_profile':'unitree_go2_45'},True),
])
def test_full_task_feedback_only_enables_the_validated_unitree_profile(dimension,contract,expected):
    controller=NS(observation_dim=dimension,policy=NS(go2_deployment_contract=contract),
                  policy_command=np.array([0.,0.,.55]))
    actual=module.policy_feedback_for_heading_step(controller)
    assert (actual is not None)==expected
    if expected:
        np.testing.assert_array_equal(actual,controller.policy_command)
        actual[2]=0.
        assert controller.policy_command[2]==.55
