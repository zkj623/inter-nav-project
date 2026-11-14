"""Synthetic deploy integration: measured-box adapter -> shared skill -> writer lease.

These are offline interface and planning fixtures, never evidence of contact
physics or robot acceptance. Only terminal skill output is stubbed where a
passage certificate/stop handoff is under test.
"""
from dataclasses import replace

import numpy as np
import pytest

from deploy.app.navigation import NavigationEngine
from deploy.app.tasks import Tasks
from deploy.perception.registered_boxes import RegisteredBoxPerception
from deploy.robot.observation import Snapshot
from grutopia_extension.interactive_navigation.box_push_navigation import BoxDecision, PushPlan
from deploy.interaction.profile import ContactProfile
from deploy.tests.contact_fixtures import sdk_profile
from grutopia_extension.interactive_navigation.interaction_skill import SkillCommand
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner


def settings(push_allowed=True):
    # Explicitly synthetic fixture; no production calibration is emitted.
    profile = replace(sdk_profile(), backend='sdk', calibration_reference='synthetic-integration-only',
        max_forward_mps=.25, max_reverse_mps=.15, max_lateral_mps=.12, max_yaw_rps=.30,
        forward_mps=.15, reverse_mps=.12, lateral_mps=.10, push_start_mps=.15,
        push_max_mps=.20, min_translation_mps=.05, min_yaw_rps=.08,
        linear_acceleration_mps2=.5, yaw_acceleration_rps2=1.5, stall_seconds=2.)
    tracking = {'geometry_verified': True, 'registry_reference': 'synthetic-integration-boxes',
                'boxes': [{'id': 'B1', 'dimensions_m': [.4, .3, .5],
                           'search_region': [-.3, .3, -.3, .3], 'push_allowed': push_allowed}]}
    interaction = {'source': 'registered_box_lidar', 'contact_profile': profile.to_dict(),
                   'max_skill_seconds': 90.}
    navigation = {'mapping': {'x_limits': [-3., 3.], 'y_limits': [-3., 3.], 'grid_resolution': .1,
                             'robot_radius': .25, 'obstacle_height': [.05, .8]},
                  'max_forward_speed': .3, 'max_lateral_speed': .15, 'max_yaw_speed': .4,
                  'linear_acceleration': .5, 'yaw_acceleration': 1.5, 'success_distance': .25,
                  'interaction_enabled': True, 'route_clearance': .25,
                  'tick_hz': 10., 'task_timeout_s': 180., 'fall_height': .16, 'safe_base_height': .22}
    return navigation, interaction, tracking


def scan(index, stamp, *, center=(0., 0.), include_box=True):
    u, z = np.meshgrid(np.linspace(-1, 1, 18), np.linspace(.03, .495, 14))
    front = np.c_[u.ravel()*.2, np.full(u.size, -.15), z.ravel()]
    side = np.c_[np.full(u.size, -.2), u.ravel()*.15, z.ravel()]
    points = np.vstack([front, side])
    points[:, :2] += center
    points += np.random.default_rng(4).normal(0., .002, points.shape)
    if not include_box:
        points = np.array([[2., -2., .4], [2., -1.9, .4], [2., -1.8, .4]])
    observation = {'position': np.array([-.6, -2., .4]), 'lidar_captured_at': stamp,
        'orientation': np.array([2**-.5, 0., 0., 2**-.5]), 'velocity': (0., 0., 0.),
        'body_linear_velocity': np.zeros(3), 'body_angular_velocity': np.zeros(3),
        'sensors': {'lidar': {'pointcloud': points, 'position': np.array([-.6, -1.718, .4]),
                              'physics_step': index, 'max_range': 5., 'min_range': .1}}}
    return Snapshot(observation, stamp, stamp, stamp, None, (0., 0., 0.), index,
                    planning_frame='experiment', coordinate_transform={'source_frame': 'synthetic_odom'})


def observed_layout(engine, *, open_room=False):
    """Deterministic observed grid: narrow exit with a lower side parking bay."""
    x, y = np.meshgrid(np.arange(-3, 3, .1)+.05, np.arange(-3, 3, .1)+.05)
    free = (np.ones(x.shape, bool) if open_room else
            (abs(x) < .55) | ((y < .35) & (y > -1.3)) | ((y <= -1.3) & (abs(x) < 1.5)))
    layers = engine.layers
    layers.evidence.log_odds[:] = np.where(free, -4., 4.)
    layers.observed[:] = True
    layers._latest_static[:] = ~free
    layers.local_hits[:] = False
    layers.refresh_global(reason='synthetic_observed_grid_fixture')
    layers._bind_local()


@pytest.fixture
def setup():
    navigation, interaction, tracking = settings()
    provider = RegisteredBoxPerception(tracking, synchronous=True)
    engine = NavigationEngine(navigation, interaction=interaction)
    engine.set_goal((0., 2., .4), 0.)
    for index in (1, 2):
        packet = scan(index, index*.1)
        boxes, recognition = provider.update(packet, index*.1)
        engine.observe(packet.observation, index*.1, boxes)
        observed_layout(engine)
    try:
        yield provider, engine, packet, boxes, recognition
    finally:
        provider.close()


def dispatch(setup):
    provider, engine, packet, boxes, recognition = setup
    velocity = engine.update(packet.observation, packet.lidar_at, boxes, recognition)
    assert isinstance(engine.skill, PushAsideSkill) and engine.phase == 'skill'
    np.testing.assert_array_equal(velocity, np.zeros(3))
    return engine.skill


def test_registered_provider_dispatches_real_side_push_planner_and_shared_skill_without_semantics(setup, monkeypatch):
    provider, engine, packet, boxes, recognition = setup
    def no_semantics(*args, **kwargs):
        pytest.fail('Registered LiDAR must not invoke RGB-D inspection/classification')
    monkeypatch.setattr(engine.inspection_planner, 'propose', no_semantics)
    monkeypatch.setattr(engine.catalog, 'permission', no_semantics)
    skill = dispatch(setup)
    assert isinstance(engine.planner, SidePushPlanner)
    assert engine.decision['kind'] == 'push'
    assert engine.decision['before_path'] == ()
    assert engine.decision['after_path'] and engine.decision['candidate_count'] > 0
    assert engine.decision['opens_connectivity'] is True
    assert skill.permission.source == 'registered_box_lidar'
    assert skill.permission.classifier_models == ()
    assert skill.permission.source_digest == provider.source_digest
    assert skill.backend.profile.backend == 'sdk'
    assert engine.planner.interactable_ids == {'B1'}
    assert engine.layers.statistics()['movable_pose_source'] == 'registered_box_lidar'
    assert engine.layers.statistics()['uses_simulator_object_poses'] is False
    lease = engine.interaction_command
    assert lease['source_expires_at'] == pytest.approx(packet.lidar_at+.25)
    assert lease['deadline'] == pytest.approx(packet.lidar_at+90.)
    assert lease['limits']['max_vx'] == .25 and lease['limits']['max_wz'] == .30


def test_open_room_uses_actual_planner_detour_instead_of_forcing_a_push(setup):
    provider, engine, packet, boxes, recognition = setup
    observed_layout(engine, open_room=True)
    velocity = engine.update(packet.observation, packet.lidar_at, boxes, recognition)
    assert engine.decision['kind'] == 'navigate'
    assert engine.skill is None and not engine.execution.active
    assert np.isfinite(velocity).all()


def test_expired_registration_geometry_stops_before_shared_skill_step(setup, monkeypatch):
    provider, engine, packet, boxes, recognition = setup
    skill = dispatch(setup)
    monkeypatch.setattr(skill, 'step', lambda *a: pytest.fail('Expired box must not enter a contact step'))
    # Repeated sensor data cannot extend the captured-at deadline.
    assert provider.update(packet, .451)[0] == ()
    assert engine.update(packet.observation, .451, (), provider) is None
    assert engine.status == 'failed'
    assert engine.result['reason'] == 'push_aside:live_geometry_or_recognition_lost'
    np.testing.assert_array_equal(engine.limiter.command, np.zeros(3))


def test_unregistered_box_is_never_given_permission_even_when_selector_requests_it(setup, monkeypatch):
    provider, engine, packet, boxes, recognition = setup
    unknown_plan = PushPlan('unknown', (0., 0.), (.7, 0.), (1., 0.), (-.8, 0.), ((-.8, 0.),), .7, 3.)
    monkeypatch.setattr(engine.selector, 'decide', lambda *a: (
        BoxDecision('push', 'synthetic_unknown_candidate', push=unknown_plan, explanation={}), (0., 2.)))
    velocity = engine.update(packet.observation, .2, boxes, recognition)
    assert provider.permission('unknown', .2) is None
    assert engine.skill is None and not engine.execution.active
    assert np.all(velocity == 0.)


def test_push_disallowed_registry_still_maps_obstacle_without_permission():
    navigation, interaction, tracking = settings(push_allowed=False)
    provider = RegisteredBoxPerception(tracking, synchronous=True)
    engine = NavigationEngine(navigation, interaction=interaction)
    engine.set_goal((0., 2., .4), 0.)
    try:
        for index in (1, 2):
            packet = scan(index, index*.1)
            boxes, recognition = provider.update(packet, index*.1)
            engine.observe(packet.observation, index*.1, boxes)
        assert len(boxes) == 1 and engine.layers.current_boxes
        assert engine._permission('B1', provider) is None
        assert not provider.eligible_ids(.2)
    finally:
        provider.close()


def test_fresh_capture_updates_execution_metadata_without_renewing_contact_budget(setup, monkeypatch):
    provider, engine, packet, boxes, recognition = setup
    skill = dispatch(setup)
    # Only the motion result is stubbed: actual tracking/permission/context and
    # deadline plumbing are exercised through NavigationEngine.update.
    skill.state = 'engage'
    monkeypatch.setattr(skill, 'step', lambda ctx: SkillCommand('engage'))
    initial_id = engine.interaction_command['id']
    for index in (3, 4):
        packet = scan(index, index*.1)
        boxes, recognition = provider.update(packet, index*.1)
        velocity = engine.update(packet.observation, index*.1, boxes, recognition)
        assert np.all(velocity == 0.)
        assert engine.interaction_command['source_expires_at'] == pytest.approx(index*.1+.25)
        assert engine.interaction_command['id'] == initial_id
        assert engine.interaction_command['deadline'] == pytest.approx(.3+skill.backend.profile.max_contact_seconds)
    assert engine._contact_context.policy_command is None
    assert engine._contact_context.box.local_dimensions == (.4, .3, .5)
    assert engine._contact_context.contact_motion_check is not None


def test_pause_resume_retires_contact_execution_and_requires_new_lidar_confirmation():
    navigation, interaction, tracking = settings()
    provider = RegisteredBoxPerception(tracking, synchronous=True)
    tasks = Tasks(navigation, perception=provider, interaction=interaction)
    try:
        tasks.start('navigate', (0., 2., .4), 0.)
        first_generation = provider.generation
        for index in (1, 2):
            packet = scan(index, index*.1)
            boxes, recognition = provider.update(packet, index*.1)
            tasks.engine.observe(packet.observation, index*.1, boxes)
            observed_layout(tasks.engine)
        tasks.engine.update(packet.observation, .2, boxes, recognition)
        assert tasks.engine.execution.active
        previous_id = tasks.engine.execution.identity
        tasks.pause()
        assert tasks.engine.skill is None and not tasks.engine.execution.active
        tasks.resume(.25)
        assert provider.generation == first_generation+1
        assert provider.permission('B1', .25) is None
        assert tasks.engine.current_proposals == {}
        packet = scan(3, .3)
        assert provider.update(packet, .3)[0] == ()
        packet = scan(4, .4)
        boxes, recognition = provider.update(packet, .4)
        tasks.engine.observe(packet.observation, .4, boxes)
        observed_layout(tasks.engine)
        tasks.engine.update(packet.observation, .4, boxes, recognition)
        assert tasks.engine.execution.identity != previous_id
        assert tasks.engine.phase == 'skill'
        assert all(f'lidar:{provider.generation}:' in f for f in tasks.engine.skill.permission.evidence_frames)
    finally:
        provider.close()


@pytest.mark.parametrize('certificate', [None, {'corridor_start': [0., -1.], 'corridor_end': [0., 1.],
    'side_parked': True, 'original_box_would_block_corridor': True,
    'confirming_observation_time_s': .3, 'last_box_motion_time_s': .2}])
def test_skill_success_requires_certificate_and_writer_stop_handoff(setup, monkeypatch, certificate):
    provider, engine, packet, boxes, recognition = setup
    skill = dispatch(setup)
    # A synthetic terminal skill result tests orchestration only. It is NOT a
    # fabricated runtime certificate and does not establish physical success.
    monkeypatch.setattr(skill, 'step', lambda ctx: SkillCommand('completed', status='succeeded', certificate=certificate))
    monkeypatch.setattr(skill, 'report', lambda: {'synthetic_terminal_result': True})
    packet = scan(3, .3)
    boxes, recognition = provider.update(packet, .3)
    velocity = engine.update(packet.observation, .3, boxes, recognition)
    if certificate is None:
        assert engine.status == 'failed' and 'missing_passage_certificate' in engine.result['reason']
        assert velocity is None
        return
    assert np.all(velocity == 0.)
    assert engine.skill is None and engine.phase == 'decide'
    assert engine.execution.active and engine.execution.awaiting_stop
    assert engine.passage.certificate == certificate
    assert engine.skill_reports[-1]['object_pose_feedback'] == 'registered_box_lidar'
    engine.interaction_stopped()
    assert not engine.execution.active and not engine.execution.awaiting_stop
    assert engine.interaction_command is None
    assert engine.velocity_limits['max_vx'] == navigation_limits()['max_forward_speed']


def navigation_limits():
    return settings()[0]


def test_centered_contact_keeps_same_skill_with_current_bounded_face_measurements(setup, monkeypatch):
    from deploy.tests.test_registered_boxes import bounded_face_cloud
    provider, engine, packet, boxes, recognition = setup
    skill = dispatch(setup)
    initial_identity = engine.execution.identity
    skill.state = 'engage'
    monkeypatch.setattr(skill, 'step', lambda context: SkillCommand('engage'))
    for index in (3, 4, 5):
        center = (.025*(index-3), 0.)
        points, origin = bounded_face_cloud(center=center, yaw=0., depth_to_face=.15)
        packet = scan(index, index*.1)
        packet.observation['sensors']['lidar'].update(pointcloud=points, position=origin)
        packet.observation.update(position=np.array([origin[0]-.282, origin[1], .4]),
                                  orientation=np.array([1., 0., 0., 0.]))
        boxes, recognition = provider.update(packet, index*.1)
        assert len(boxes) == 1 and boxes[0].xy == pytest.approx(center, abs=.008)
        command = engine.update(packet.observation, index*.1, boxes, recognition)
        assert command is not None and np.all(command == 0.)
        assert engine.skill is skill and engine.execution.identity == initial_identity
        assert engine.interaction_command['source_expires_at'] == pytest.approx(index*.1+.25)
        fit = provider.last_delivery['detections'][0]
        assert fit['geometry_source'] == 'current_lidar_bounded_single_face'
        assert fit['boundary_free_ray_counts'][0] >= 3 and fit['boundary_free_ray_counts'][1] >= 3


def _async_controller(monkeypatch):
    import time
    from types import SimpleNamespace
    from deploy.app import controller as module
    from deploy.app.config import load
    from deploy.app.controller import NavigationController
    from deploy.tests.test_controller import Motion, Events
    navigation, interaction, tracking = settings()
    provider = RegisteredBoxPerception(tracking)  # Real one-worker asynchronous path.
    tasks = Tasks(navigation, perception=provider, interaction=interaction)
    tasks.start('navigate', (0., 2., .4), 10.)
    def wait_result():
        deadline = time.monotonic()+2.
        while provider.results.empty() and time.monotonic() < deadline:
            time.sleep(.002)
        assert not provider.results.empty()
    for index in (1, 2, 3):
        now = 10.+index*.1
        packet = scan(index, now)
        tasks.observe_only(packet, now)
        wait_result()
        tasks.observe_only(packet, now+.01)
    observed_layout(tasks.engine)
    boxes = tuple(tasks.engine.current_proposals.values())
    tasks.engine.update(packet.observation, now+.01, boxes, provider)
    assert isinstance(tasks.engine.skill, PushAsideSkill)
    config = load()
    config['navigation'], config['interaction'] = navigation, interaction
    motion, events = Motion(), Events()
    motion.state = 'active'
    clock = [now+.01]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    controller = NavigationController(config, tasks, motion, events, provider)
    return controller, provider, clock, wait_result, events


def test_async_tracking_continues_during_long_planned_stop_without_advancing_skill(monkeypatch):
    controller, provider, clock, wait_result, events = _async_controller(monkeypatch)
    tasks, engine = controller.tasks, controller.tasks.engine
    skill, identity = engine.skill, engine.execution.identity
    deadline, generation = engine.execution.deadline, provider.generation
    calls = []
    monkeypatch.setattr(skill, 'step', lambda ctx: calls.append(ctx.now) or SkillCommand('prepare'))
    controller._hold('synthetic_stop_for_stage_planning')
    hold = controller.planning_hold
    try:
        # More than 3x the .25 s geometry age limit, with real worker results
        # consumed throughout. The writer remains in its stop handshake.
        for index in range(4, 12):
            now = 10.+index*.1
            packet = scan(index, now)
            clock[0] = now
            assert controller.step(packet, now) is None
            wait_result()
            clock[0] = now+.01
            assert controller.step(packet, now+.01) is None
            assert provider.permission('B1', now+.01) is not None
            assert provider.expires_at('B1') == pytest.approx(now+.25)
            assert engine.execution.deadline == deadline
        assert calls == [] and engine.skill is skill
        assert engine.execution.identity == identity and provider.generation == generation
        assert not controller.motion.commands
        assert any(e['event'] == 'perception_result' and e['captured_at'] > 11. for e in events.items)
        controller.motion.state = 'stopped'
        controller.motion_events([{'state': 'stopped', 'detail': hold['token'], 'worker_at': 11.11}])
        # Preparation and observe_only share one perception sample in a cycle.
        original_update, perceptions = provider.update, []
        def counted(packet, now):
            perceptions.append((packet.sequence, now))
            return original_update(packet, now)
        monkeypatch.setattr(provider, 'update', counted)
        for index in (12, 13):
            now = 10.+index*.1
            clock[0] = now
            packet = scan(index, now)
            assert controller.step(packet, now) is None
            wait_result()
        assert perceptions == [(12, 11.2), (13, 11.3)]
        assert controller.planning_hold is None and controller.motion.state == 'active'
        assert tasks.state == 'active' and engine.skill is skill
        assert engine.execution.identity == identity and engine.execution.deadline == deadline
        assert controller.fresh_cycle_required and not controller.motion.commands
    finally:
        provider.close()


def test_slow_held_preparation_cannot_retime_old_target_or_arm_without_fresh_fit(monkeypatch):
    controller, provider, clock, wait_result, _ = _async_controller(monkeypatch)
    engine = controller.tasks.engine
    skill = engine.skill
    def slow_preparation(context):
        clock[0] += .6  # Main thread cannot acquire another snapshot during this work.
        return SkillCommand('prepare')
    monkeypatch.setattr(skill, 'step', slow_preparation)
    controller._hold('synthetic_slow_stage_planning')
    hold = controller.planning_hold
    controller.motion.state = 'stopped'
    controller.motion_events([{'state': 'stopped', 'detail': hold['token'], 'worker_at': 10.32}])
    try:
        clock[0] = 10.4
        packet = scan(4, 10.4)
        controller.step(packet, 10.4)
        assert clock[0] == pytest.approx(11.)
        assert controller.motion.state == 'stopped' and not controller.motion.commands
        assert hold['prepared_at'] == pytest.approx(11.)
        wait_result()
        # Even though this new snapshot is fresh, an old async fitting result
        # and stale identity are not silently renewed after the .6 s gap.
        clock[0] = 11.1
        controller.step(scan(5, 11.1), 11.1)
        assert controller.motion.state == 'stopped' and not controller.motion.commands
        assert engine.status == 'failed'
        assert 'live_geometry_or_recognition_lost' in engine.result['reason']
        assert provider.permission('B1', 11.1) is None
    finally:
        provider.close()


def test_skill_context_keeps_box_capture_time_and_scan_identity_when_consumed_later(monkeypatch):
    controller, provider, clock, wait_result, _ = _async_controller(monkeypatch)
    engine, tasks = controller.tasks.engine, controller.tasks
    engine.skill.state = 'engage'
    monkeypatch.setattr(engine.skill, 'step', lambda context: SkillCommand('engage'))
    try:
        packet = scan(4, 10.4)
        tasks.observe_only(packet, 10.4)
        wait_result()
        clock[0] = 10.43
        tasks.observe_only(packet, 10.43)
        engine.update(packet.observation, 10.43, tuple(engine.current_proposals.values()), provider)
        context = engine._contact_context
        assert context.now == 10.43
        assert context.box_observed_at == 10.4
        assert context.box_observed_at != context.now
        assert context.box_observation_id == provider.permission('B1', 10.43).evidence_frames[-1]
        assert context.box_observation_id.endswith(':4')
        assert context.map_observed_at == 10.4
        assert str(context.map_observation_id).endswith('4')
    finally:
        provider.close()
