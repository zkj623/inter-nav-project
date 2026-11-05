import numpy as np
from test_push_aside_skill import alignment_context
from grutopia_extension.interactive_navigation.go2_contact_translation_step import ContactTranslationStep


def stalled_context(policy_feedback=True):
    skill,ctx=alignment_context('align_longitudinal',gap=.205,lateral=.795)
    ctx.now=2.;ctx.policy_command=np.zeros(3) if policy_feedback else None
    skill.progress_mark=ctx.position[:2].copy();skill.progress_since=0.;skill.translation_braking=True
    skill.settled_since=1.5;skill.settled_position=ctx.position.copy();skill.settled_tilt=ctx.tilt_degrees
    return skill,ctx


def test_settled_reverse_plateau_arms_bounded_input_without_resetting_alignment_deadline():
    skill,ctx=stalled_context();deadline_start=skill.state_since
    command=skill.step(ctx)
    assert command.state=='align_longitudinal' and command.velocity==(0.,0.,0.)
    assert skill.longitudinal_step is not None and skill.longitudinal_step.minimum_progress_m==.08
    assert skill.state_since==deadline_start and command.events[-1]['reason']=='settled_longitudinal_plateau'
    other,unknown=stalled_context(False);other.step(unknown)
    assert other.longitudinal_step is None


def test_transient_safe_standoff_does_not_cancel_acknowledged_input_requirement():
    skill,ctx=alignment_context('align_longitudinal',gap=.245,lateral=.795)
    ctx.now=2.;ctx.policy_command=np.array([-.35,0.,0.])
    origin=ctx.position[:2]+[.045,0.]
    step=ContactTranslationStep(origin,[-1.,0.],1.9,minimum_progress_m=.08,axis_name='longitudinal')
    skill.longitudinal_step=step
    command=skill.step(ctx)
    assert skill.longitudinal_step is step and not step.complete and command.state=='align_longitudinal'
    assert command.velocity[0]<0. and command.velocity[2]==0.


def test_longitudinal_input_completion_waits_for_actual_stop_before_lateral_work():
    skill,ctx=alignment_context('align_longitudinal',gap=.28,lateral=.795)
    origin=ctx.position[:2]+[.09,0.];step=ContactTranslationStep(origin,[-1.,0.],1.8,minimum_progress_m=.08,axis_name='longitudinal')
    step.update(origin,0.,[-.35,0.,0.],1.8);step.update(ctx.position,0.,[-.35,0.,0.],1.85)
    assert step.complete
    skill.longitudinal_step=step;ctx.now=2.;ctx.policy_command=np.zeros(3);ctx.body_velocity=np.array([-.10,0.,0.])
    assert skill.step(ctx).state=='align_longitudinal' and skill.longitudinal_step is step
    ctx.body_velocity[:]=0.;ctx.now=2.1;skill.step(ctx)
    ctx.now=2.41;command=skill.step(ctx)
    assert command.state=='align_longitudinal' and skill.longitudinal_step is None
    assert command.events[-1]['reason']=='longitudinal_step_measured_stop'
    # Production adds the physical step when forwarding each skill event.
    # The diagnostic detail must not shadow that keyword and abort recording.
    def emit(name,step,**payload):return name,step,payload
    name,tick,payload=emit('skill_transition',step=42,**command.events[-1])
    assert tick==42 and payload['longitudinal_step']['input_acknowledged']
    ctx.now=2.42;assert skill.step(ctx).state=='align_lateral'


def test_longitudinal_step_preserves_existing_input_and_travel_bounds():
    timeout=ContactTranslationStep([0.,0.],[-1.,0.],0.,minimum_progress_m=.08,axis_name='longitudinal')
    command,_=timeout.update([0.,0.],0.,[0.,0.,0.],3.01)
    assert timeout.failed=='longitudinal_step_input_timeout' and np.array_equal(command,[0.,0.])
    travel=ContactTranslationStep([0.,0.],[-1.,0.],0.,axis_name='longitudinal')
    command,_=travel.update([-.26,0.],0.,[-.35,0.,0.],.1)
    assert travel.failed=='longitudinal_step_translation_limit' and np.array_equal(command,[0.,0.])
