import math
import numpy as np
import pytest

from grutopia_extension.interactive_navigation.go2_contact_translation_step import ContactTranslationStep
from test_push_aside_skill import alignment_context


def test_acknowledgement_uses_actual_world_direction_after_body_rotation():
    step=ContactTranslationStep([0.,0.],[0.,1.],0.)
    command,detail=step.update([0.,.02],math.pi/2,[.15,0.,0.],.2)
    np.testing.assert_allclose(command,[0.,.35]);assert not detail['input_complete']
    assert not step.update([0.,.03],math.pi/2,[.35,0.,0.],.4)[1]['input_complete']
    assert not step.update([0.,.03],math.pi/2,[.35,0.,0.],.439)[1]['input_complete']
    command,detail=step.update([0.,.03],math.pi/2,[.35,0.,0.],.441)
    assert detail['input_complete'];np.testing.assert_array_equal(command,[0.,0.])


def test_wrong_sign_or_interrupted_input_does_not_accumulate_hold():
    step=ContactTranslationStep([0.,0.],[0.,1.],0.)
    assert not step.update([0.,0.],0.,[0.,-.35,0.],.1)[1]['input_complete']
    step.update([0.,0.],0.,[0.,.35,0.],.2)
    step.update([0.,0.],0.,[0.,.20,0.],.23)
    assert not step.update([0.,0.],0.,[0.,.35,0.],.25)[1]['input_complete']
    assert not step.update([0.,0.],0.,[0.,.35,0.],.289)[1]['input_complete']
    assert step.update([0.,0.],0.,[0.,.35,0.],.291)[1]['input_complete']


@pytest.mark.parametrize('position,feedback,now,reason',[
    ([0.,0.],None,.1,'lateral_step_feedback_unavailable'),
    ([0.,0.],[0.,float('nan'),0.],.1,'lateral_step_feedback_unavailable'),
    ([0.,0.],[0.,0.,0.],3.01,'lateral_step_input_timeout'),
    ([.251,0.],[0.,.35,0.],.1,'lateral_step_translation_limit'),
])
def test_failed_step_emits_no_motion(position,feedback,now,reason):
    step=ContactTranslationStep([0.,0.],[0.,1.],0.)
    command,detail=step.update(position,0.,feedback,now)
    assert detail['failure']==reason and not detail['input_complete']
    np.testing.assert_array_equal(command,[0.,0.])


def corrective_context(profile=True):
    skill,ctx=alignment_context('settle_contact',gap=.35,lateral=.12,heading=0.)
    if profile:ctx.policy_command=np.zeros(3)
    assert skill.step(ctx).state=='settle_contact'
    ctx.now+=.41
    assert skill.step(ctx).state=='align_lateral'
    return skill,ctx


def test_transient_outer_face_band_does_not_cancel_corrective_input():
    skill,ctx=corrective_context();assert skill.lateral_step is not None
    ctx.position[1]+=.03;ctx.now+=.01;ctx.limiter.last_time=ctx.now-.01
    assert skill._contact_ready(ctx)
    command=skill.step(ctx)
    assert command.state=='align_lateral' and command.velocity[1]>0
    assert skill.lateral_step is not None and command.certificate is None


def test_inner_position_can_brake_but_cannot_skip_measured_settling():
    skill,ctx=corrective_context();ctx.position[1]+=.10;ctx.now+=.01
    command=skill.step(ctx)
    assert command.state=='settle_contact' and command.velocity==(0.,0.,0.)
    assert skill.lateral_step is None
    ctx.now+=.1;ctx.body_velocity[1]=.20
    assert skill.step(ctx).state=='settle_contact'
    ctx.now+=.1;ctx.body_velocity[:]=0;ctx.limiter.command[:]=0
    assert skill.step(ctx).state=='settle_contact'
    ctx.now+=.41
    assert skill.step(ctx).state=='engage'


def test_delivered_input_alone_does_not_certify_contact_pose():
    skill,ctx=corrective_context();ctx.policy_command=np.array([0.,.35,0.])
    ctx.now+=.1;assert skill.step(ctx).state=='align_lateral'
    ctx.now+=.041;command=skill.step(ctx)
    assert command.state=='settle_contact' and command.certificate is None
    assert not skill._contact_ready(ctx)


def test_legacy_profile_retains_previous_contact_handoff():
    skill,ctx=corrective_context(profile=False);assert skill.lateral_step is None
    ctx.position[1]+=.03;ctx.now+=.01
    assert skill.step(ctx).state=='settle_contact'


def test_blocked_geometry_cannot_fake_delivered_input():
    skill,ctx=corrective_context();ctx.planner.free[:]=False;ctx.planner.box_free[:]=False
    ctx.now+=.1;command=skill.step(ctx)
    assert command.velocity==(0.,0.,0.) and skill.lateral_step is not None
    ctx.now+=3.;command=skill.step(ctx)
    assert command.status!='succeeded' and skill.lateral_step is None
    assert skill.retries==1
