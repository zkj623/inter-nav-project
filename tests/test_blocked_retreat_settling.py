"""A late safety rejection must give the measured-stop recovery a chance."""
import numpy as np
from test_blocked_contact_recovery import scenario
import grutopia_extension.interactive_navigation.push_aside_skill as module
from grutopia_extension.interactive_navigation.go2_stop_feedback import skill_has_active_contact


def test_late_moving_rejection_brakes_then_uses_existing_safe_recovery(monkeypatch):
    skill,ctx=scenario();skill.state_since=ctx.now-5.15
    ctx.body_velocity[:]=[-.069,.020,0.]
    ctx.limiter.accept([-.32,-.027,0.],ctx.now)
    monkeypatch.setattr(module,'observed_contact_retreat',lambda *a,**k:None)
    command=skill.step(ctx)
    assert command.state=='settle_blocked_retreat' and command.status=='running'
    assert command.velocity==(0.,0.,0.) and command.certificate is None
    assert skill.release_pose_recoveries==0
    ctx.now+=.2;ctx.layers.last_local=ctx.now
    assert skill.step(ctx).state=='settle_blocked_retreat'
    ctx.body_velocity[:]=0.;ctx.limiter.accept([0.,0.,0.],ctx.now)
    assert skill.step(ctx).state=='settle_blocked_retreat'
    ctx.now+=.39;ctx.layers.last_local=ctx.now
    assert skill.step(ctx).state=='settle_blocked_retreat'
    ctx.now+=.02;ctx.layers.last_local=ctx.now
    command=skill.step(ctx)
    assert command.state=='recover_release_pose' and command.status=='running'
    assert command.events[-1]['reason']=='blocked_contact_withdrawal'
    assert command.certificate is None and skill.release_pose_recoveries==1


def test_unsettled_withdrawal_still_has_existing_braking_deadline(monkeypatch):
    skill,ctx=scenario();skill.state_since=ctx.now-20.
    ctx.body_velocity[0]=-.12
    monkeypatch.setattr(module,'observed_contact_retreat',lambda *a,**k:None)
    assert skill.step(ctx).state=='settle_blocked_retreat'
    ctx.now+=skill.config.max_push_braking_seconds+.01;ctx.layers.last_local=ctx.now
    result=skill.step(ctx)
    assert result.status=='failed' and result.reason=='blocked_retreat_not_settled'


def test_quiet_but_unavailable_recovery_remains_bounded(monkeypatch):
    skill,ctx=scenario()
    monkeypatch.setattr(module,'observed_contact_retreat',lambda *a,**k:None)
    monkeypatch.setattr(skill,'_begin_release_pose_recovery',lambda *a,**k:False)
    assert skill.step(ctx).state=='settle_blocked_retreat'
    ctx.now+=5.01;ctx.layers.last_local=ctx.now
    result=skill.step(ctx)
    assert result.status=='failed' and result.reason=='no_safe_observed_retreat'


def test_reopened_retreat_is_rechecked_after_stop_without_geometry_changes(monkeypatch):
    skill,ctx=scenario();original_free=ctx.planner.free.copy();boxes=ctx.boxes
    available=False
    def checked(planner,seen,position,heading,direction,limiter,now,**kwargs):
        assert planner is ctx.planner and seen is boxes
        return (np.array([-.1,0.,0.]),position[:2]+[.1,0.]) if available else None
    monkeypatch.setattr(module,'observed_contact_retreat',checked)
    monkeypatch.setattr(skill,'_begin_release_pose_recovery',lambda *a,**k:False)
    assert skill.step(ctx).state=='settle_blocked_retreat'
    available=True;ctx.now+=.41;ctx.layers.last_local=ctx.now
    result=skill.step(ctx)
    assert result.state=='retreat' and result.velocity==(0.,0.,0.)
    assert result.events[-1]['reason']=='retry_withdrawal_after_measured_stop'
    np.testing.assert_array_equal(ctx.planner.free,original_free)


def test_settling_transition_does_not_reset_total_skill_deadline(monkeypatch):
    skill,ctx=scenario();started=skill.started
    monkeypatch.setattr(module,'observed_contact_retreat',lambda *a,**k:None)
    assert skill.step(ctx).state=='settle_blocked_retreat'
    assert skill.started==started
    ctx.now=started+skill.config.max_skill_seconds+.01
    result=skill.step(ctx)
    assert result.status=='failed' and result.reason=='skill_timeout'


def test_blocked_withdrawal_stop_keeps_contact_feedback_mode():
    assert skill_has_active_contact('settle_blocked_retreat',False)
    assert skill_has_active_contact('settle_blocked_retreat',True)
