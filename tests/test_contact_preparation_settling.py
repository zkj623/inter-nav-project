import numpy as np

from test_push_aside_skill import alignment_context


def test_contact_settle_does_not_start_heading_work_while_still_sliding():
    skill,ctx=alignment_context('settle_contact',gap=.55,lateral=.17,heading=.15)
    ctx.body_velocity=np.array([0.,.23,0.])
    ctx.limiter.command=np.array([0.,.30,0.])
    command=skill.step(ctx)
    assert command.state=='settle_contact'
    assert command.velocity==(0.,0.,0.)
    assert skill.heading_trim is None


def test_stopped_lateral_overshoot_is_corrected_before_heading_trim():
    skill,ctx=alignment_context('settle_contact',gap=.55,lateral=.17,heading=.15)
    assert skill.step(ctx).state=='settle_contact'
    ctx.now+=.41
    command=skill.step(ctx)
    assert command.state=='align_lateral'
    assert command.velocity==(0.,0.,0.)
    assert skill.heading_trim is None


def test_already_valid_engagement_standoff_does_not_chase_an_anchor():
    skill,ctx=alignment_context('align_longitudinal',gap=.55,lateral=.12)
    assert .55<skill.config.max_engage_standoff
    command=skill.step(ctx)
    assert command.state=='align_longitudinal' and command.velocity==(0.,0.,0.)
    ctx.now+=.31
    command=skill.step(ctx)
    assert command.state=='align_lateral' and command.velocity==(0.,0.,0.)


def test_lateral_work_uses_the_same_admissible_standoff():
    skill,ctx=alignment_context('align_lateral',gap=.60,lateral=.20)
    command=skill.step(ctx)
    assert command.state=='align_lateral'
    assert command.velocity[1]>0.


def test_too_close_pose_withdraws_before_attempting_lateral_correction():
    skill,ctx=alignment_context('settle_contact',gap=.15,lateral=.17,heading=.15)
    assert skill.step(ctx).state=='settle_contact'
    ctx.now+=.41
    command=skill.step(ctx)
    assert command.state=='align_longitudinal'
    assert command.velocity==(0.,0.,0.)


def test_valid_contact_pose_still_requires_a_complete_stop_before_engaging():
    skill,ctx=alignment_context('settle_contact',gap=.46,lateral=0.,heading=0.)
    assert skill.step(ctx).state=='settle_contact'
    ctx.now+=.41
    assert skill.step(ctx).state=='engage'


def test_lateral_hysteresis_does_not_expand_actual_engagement_limit():
    skill,ctx=alignment_context('settle_contact',gap=.64,lateral=0.,heading=0.)
    assert not skill._contact_ready(ctx)
    assert skill.step(ctx).state=='settle_contact'
    ctx.now+=.41
    assert skill.step(ctx).state=='align_longitudinal'
    assert skill.config.max_engage_standoff==.60
