import json
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.go2_stationary_hold import StationaryJointHold, stationary_hold_permitted


def update(hold, now, *, target=None, command=(0, 0, 0), allowed=True,
           position=(0, 0, .42), linear=(0, 0, 0), angular=(0, 0, 0), tilt=3.):
    target = np.zeros(12, np.float32) if target is None else np.asarray(target, np.float32)
    return hold.update(target, command, position, linear, angular, tilt, now, allowed=allowed)


def captured():
    hold = StationaryJointHold()
    update(hold, 0.)
    target = np.arange(12, dtype=np.float32) * .01
    update(hold, .4, target=target)
    assert hold.state == 'holding'
    return hold, target


def test_capture_needs_complete_measured_stop_and_copies_supported_target():
    hold = StationaryJointHold()
    for now in [0., .2, .39]:
        update(hold, now)
        assert hold.state == 'tracking'
    supported = np.arange(12, dtype=np.float32) * .01
    result = update(hold, .4, target=supported)
    result[:] = 20.
    supported[:] = 30.
    actual = update(hold, 1., target=np.ones(12))
    np.testing.assert_allclose(actual, np.arange(12) * .01)
    assert hold.capture_count == 1
    json.dumps(hold.report(), allow_nan=False)


@pytest.mark.parametrize('body', [
    {'linear': (0, 0, .06)}, {'angular': (.11, 0, 0)},
    {'command': (.01, 0, 0)}, {'allowed': False}, {'position': (0, 0, .20)},
])
def test_moving_unpermitted_or_unsupported_body_does_not_capture(body):
    hold = StationaryJointHold()
    for now in np.arange(0., 1., .02):
        update(hold, now, **body)
    assert hold.state == 'tracking' and hold.capture_count == 0


@pytest.mark.parametrize('release_request', [{'command': (.4, 0, 0)}, {'allowed': False}])
def test_release_has_no_initial_target_jump_and_tracks_current_actor(release_request):
    hold, supported = captured()
    np.testing.assert_allclose(update(hold, .5, target=np.ones(12), **release_request), supported)
    assert hold.state == 'releasing'
    np.testing.assert_allclose(update(hold, .7, target=np.full(12, 2.), **release_request), .5 * supported + 1.)
    np.testing.assert_allclose(update(hold, .9, target=np.full(12, 3.), **release_request), 3.)
    assert hold.state == 'tracking' and hold.release_count == 1


def test_zero_request_during_release_does_not_recapture_mid_transition():
    hold, supported = captured()
    update(hold, .5, command=(.2, 0, 0))
    update(hold, .7)
    assert hold.state == 'releasing'
    update(hold, .9)
    assert hold.state == 'tracking'
    update(hold, 1.)
    update(hold, 1.39)
    assert hold.state == 'tracking'
    update(hold, 1.41)
    assert hold.state == 'holding' and hold.capture_count == 2


def test_episode_reset_and_clock_rewind_cannot_keep_old_target():
    hold, _ = captured()
    np.testing.assert_array_equal(update(hold, .1, target=np.ones(12)), np.ones(12))
    assert hold.state == 'tracking' and hold.capture_count == 0
    hold.reset()
    assert hold.held_target is None and hold.last_time is None


def test_loss_of_body_support_releases_to_policy():
    hold, _ = captured()
    update(hold, .5, tilt=18.)
    assert hold.state == 'releasing' and hold.reason == 'body_support_changed'


@pytest.mark.parametrize('state', ['engage', 'push', 'settle_push_progress', 'brake_after_push',
                                  'retreat', 'settle_after_retreat', 'recover_release_pose',
                                  'settle_released_corridor', 'unknown'])
def test_contact_and_unknown_skill_states_cannot_capture_hold(state):
    assert not stationary_hold_permitted('skill', state)


def test_free_space_and_recovery_origin_permissions_are_distinguished():
    assert stationary_hold_permitted('navigate')
    assert stationary_hold_permitted('skill', 'settle_at_stage')
    assert stationary_hold_permitted('skill', 'settle_contact')
    assert stationary_hold_permitted('skill', 'brake_for_recovery', False)
    assert not stationary_hold_permitted('skill', 'brake_for_recovery', True)
    assert not stationary_hold_permitted('navigate', healthy=False)
    assert not stationary_hold_permitted('unknown')


def test_opt_in_reaches_both_controller_trees_and_requires_explicit_actuation():
    from grutopia_extension.interactive_navigation.go2_navigation_runner import Go2NavigationRunConfig, build_go2_navigation_config
    from grutopia_extension.interactive_navigation.point_navigation_profiles import programmatic_go2_profile
    assert not Go2NavigationRunConfig().stationary_joint_hold
    with pytest.raises(ValueError, match='explicit_dc'):
        Go2NavigationRunConfig(stationary_joint_hold=True)
    cfg = build_go2_navigation_config(programmatic_go2_profile(),
        Go2NavigationRunConfig(actuator_mode='explicit_dc', stationary_joint_hold=True))
    robot = cfg.task_config.episodes[0].robots[0]
    assert robot.controllers[0].stationary_joint_hold
    assert robot.controllers[1].sub_controllers[0].sub_controllers[0].stationary_joint_hold
