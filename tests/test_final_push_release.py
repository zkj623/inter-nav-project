"""A spent push budget still requires release and fresh effect verification."""
import numpy as np
import pytest

from test_recovered_passage_effect import scenario


def exhausted_push():
    skill, ctx = scenario('push')
    skill.retries = skill.config.max_retries
    return skill, ctx


def test_last_partial_push_brakes_for_release_without_issuing_another_push():
    skill, ctx = exhausted_push()
    result = skill._retry(ctx, 'box_motion_stalled')
    assert result.status == 'running' and result.state == 'brake_for_recovery'
    assert result.velocity == (0., 0., 0.)
    assert skill.config.max_retries == 4
    assert skill.terminal_release_reason == 'retry_limit:box_motion_stalled'
    assert result.certificate is None


@pytest.mark.parametrize('missing', ['movement', 'contact', 'original', 'motion_time'])
def test_final_release_does_not_grant_a_new_attempt_without_a_real_push(missing):
    skill, ctx = exhausted_push()
    if missing == 'movement': skill.initial_box = np.asarray(ctx.box.xy)
    if missing == 'contact': skill.engage_box_position = None
    if missing == 'original': skill.initial_box_geometry = None
    if missing == 'motion_time': skill.last_box_motion_time = None
    result = skill._retry(ctx, 'box_motion_stalled')
    assert result.status == 'failed' and result.certificate is None


def test_final_release_accepts_only_the_existing_stopped_observed_effect():
    skill, ctx = exhausted_push()
    assert skill._retry(ctx, 'box_motion_stalled').state == 'brake_for_recovery'
    # Model a completed physical withdrawal; the certificate still uses the
    # full real map/box geometry from the fixture, not a mocked success flag.
    skill.state = 'settle_after_retreat'
    skill.state_since = ctx.now
    assert skill.step(ctx).status == 'running'
    ctx.now += .41
    ctx.layers.last_local = ctx.now
    result = skill.step(ctx)
    assert result.status == 'succeeded'
    assert result.certificate['original_box_would_block_corridor']
    assert result.certificate['confirmation_stage'] == 'settle_after_retreat'
    assert result.certificate['actual_push_distance_m'] == .76


@pytest.mark.parametrize('invalid', ['global_route', 'original_obstruction', 'visibility', 'new_scan'])
def test_final_release_fails_without_proof_and_never_restarts_preparation(invalid):
    skill, ctx = exhausted_push()
    skill._retry(ctx, 'box_motion_stalled')
    if invalid == 'global_route': ctx.layers.global_free[:] = False
    if invalid == 'original_obstruction': skill.initial_box_geometry = ctx.box
    if invalid == 'visibility': ctx.layers.visible_box_ids = []
    if invalid == 'new_scan': skill.last_box_motion_time = ctx.now + 10.
    skill.state = 'settle_after_retreat'
    skill.state_since = ctx.now
    assert skill.step(ctx).status == 'running'
    ctx.now += .41
    ctx.layers.last_local = ctx.now
    result = skill.step(ctx)
    assert result.status == 'failed'
    assert result.reason == 'retry_limit:box_motion_stalled'
    assert result.velocity == (0., 0., 0.) and result.certificate is None


def test_a_second_recovery_failure_does_not_reopen_the_spent_budget():
    skill, ctx = exhausted_push()
    skill._retry(ctx, 'box_motion_stalled')
    assert skill._retry(ctx, 'released_corridor_not_settled').status == 'failed'


def test_terminal_release_keeps_the_hard_health_limit():
    skill, ctx = exhausted_push()
    skill._retry(ctx, 'box_motion_stalled')
    ctx.tilt_degrees = 30.
    result = skill.step(ctx)
    assert result.status == 'failed' and result.reason == 'robot_unstable'
