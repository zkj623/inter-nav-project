import numpy as np
import pytest
from grutopia_extension.interactive_navigation.go2_contact_translation_step import ContactTranslationStep
from test_contact_translation_step import corrective_context


def test_acknowledgment_is_not_directional_motion_and_progress_is_signed():
    step=ContactTranslationStep([0,0],[0,1],0.,minimum_progress_m=.06)
    step.update([0,.01],0.,[0,.35,0],.1)
    _,d=step.update([0,.01],0.,[0,.35,0],.15)
    assert d['input_acknowledged'] and not d['step_complete']
    assert not step.update([0,-.06],0.,[0,.35,0],.2)[1]['step_complete']
    command,d=step.update([0,.06],0.,[0,0,0],.25)
    assert d['step_complete'] and d['directional_progress_m']==pytest.approx(.06)
    np.testing.assert_array_equal(command,[0,0])


@pytest.mark.parametrize('required',[-.01,float('nan'),float('inf'),.251])
def test_required_progress_cannot_exceed_the_existing_bound(required):
    with pytest.raises(ValueError,match='progress'):
        ContactTranslationStep([0,0],[0,1],0.,minimum_progress_m=required)


def test_received_input_without_motion_still_uses_original_three_second_limit():
    step=ContactTranslationStep([0,0],[0,1],0.,minimum_progress_m=.08)
    step.update([0,0],0.,[0,.35,0],.1);step.update([0,0],0.,[0,.35,0],.15)
    command,d=step.update([0,.01],0.,[0,.35,0],3.01)
    assert d['failure']=='lateral_step_progress_timeout' and not d['step_complete']
    np.testing.assert_array_equal(command,[0,0])


def test_short_corrective_step_requests_translation_and_retains_heading_gate():
    skill,ctx=corrective_context();ctx.heading=.04;ctx.now+=.1
    ctx.limiter.command=np.array([0,.35,0]);ctx.limiter.last_time=ctx.now-.01
    command=skill.step(ctx)
    assert command.state=='align_lateral' and command.velocity[2]==0.
    assert skill.lateral_step.minimum_progress_m==pytest.approx(.08)
    ctx.heading=.181;ctx.now+=.1
    assert skill.step(ctx).state=='brake_for_recovery'
