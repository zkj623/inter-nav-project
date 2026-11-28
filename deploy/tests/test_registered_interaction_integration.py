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


def test_registered_provider_dispatches_real_side_push_planner_and_shared_skill_without_semantics(setup):
    provider, engine, packet, boxes, recognition = setup
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


def test_only_registered_scan_evidence_can_authorize_contact(setup):
    from grutopia_extension.interactive_navigation.interaction_catalog import RecognitionEvidence, RecognitionStore
    provider, engine, packet, _, _ = setup
    assert engine._permission('B1', provider) is not None
    semantic = RecognitionStore()
    for frame in ('first', 'second'):
        semantic.observe(RecognitionEvidence('B1', 'box', frame, packet.lidar_at, .9,
                                             'Qwen-VL', '/observed/crop.jpg', 'rgb_dino_qwen_vl'))
    assert semantic.confirmed('B1', packet.lidar_at) is not None
    assert engine._permission('B1', semantic) is None


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


def test_known_registry_box_keeps_map_occupancy_across_occlusion_and_goals():
    from deploy.perception.box_memory import BoxMapMemory
    navigation, interaction, registry = settings()
    memory = BoxMapMemory()
    interaction['boxes'] = registry['boxes']
    provider = RegisteredBoxPerception(registry, synchronous=True, box_memory=memory)
    tasks = Tasks(navigation, provider, interaction=interaction, box_memory=memory)
    try:
        for i in (1, 2):
            tasks.observe_only(scan(i, i*.1), i*.1)
        original = tasks.engine.layers.detected_boxes['B1']
        tasks.start('navigate', (0., 2., .4), .3)
        tasks.observe_only(scan(3, 5., include_box=False), 5.)
        assert tasks.engine.layers.current_boxes[0].xy == pytest.approx(original.xy)
        assert tasks.engine.layers.tracked_occupied.any()
        assert not tasks.engine.latest_visible_ids and provider.permission('B1', 5.) is None
        tasks.cancel()
        tasks.start('navigate', (.5, 1., .4), 6.)
        assert 'B1' in tasks.engine.layers.detected_boxes
        assert not tasks.engine.execution.active and not tasks.engine.current_proposals
        assert not BoxMapMemory().locations()
    finally:
        provider.close()


def test_approach_keeps_the_plan_through_jitter_and_invalidates_a_real_shift():
    from deploy.app.config import load, ROOT
    from deploy.perception.box_memory import BoxMapMemory
    nav, interaction, registry = settings()
    field = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')['interaction']
    interaction.update(contact_profile=field['contact_profile'], boxes=registry['boxes'], first_push=True)
    memory = BoxMapMemory()
    provider = RegisteredBoxPerception(registry, synchronous=True, box_memory=memory)
    engine = NavigationEngine(nav, interaction, box_memory=memory)
    engine.set_goal((0., 2., .4), 0.)
    try:
        for i, shift in enumerate([0., 0., .015, -.01, .08], 1):
            now = i*.15
            packet = scan(i, now, center=(shift, 0.))
            boxes, recognition = provider.update(packet, now)
            engine.observe(packet.observation, now, boxes)
            observed_layout(engine)
            if i == 1:
                continue
            engine.update(packet.observation, now, boxes, recognition)
            if i == 2:
                plan = engine.box_approach
                assert plan is not None
            elif i < 5:
                assert engine.box_approach is plan
                assert engine.layers.detected_boxes['B1'] is engine.layers.approach_reference
                assert engine.current_proposals['B1'].xy[0] == pytest.approx(shift, abs=.005)
        assert engine.phase == 'decide' and engine.box_approach is None
        assert sum(e['event'] == 'box_approach_geometry_changed' for e in engine.events) == 1
    finally:
        provider.close()


@pytest.mark.parametrize('blocked', [None, 'turn', 'sidestep', 'exact_center_only'])
def test_stopped_short_of_side_stage_checks_turn_and_centering_before_replanning(blocked, monkeypatch):
    from deploy.app.config import ROOT, load
    from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
    config = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    engine = NavigationEngine(config['navigation'], config['interaction'])
    engine.set_goal((4., 0., .32), 0.)
    engine.full_motion_feedback = True
    engine.position = np.array([1.20, .10, .32])
    engine.heading = engine.tilt = 0.
    engine.angular = np.zeros(3)
    engine.linear = np.array([.20, 0., 0.])
    box = BoxState('box-1', (1.50, -1.), (.30, .80, .30), local_dimensions=(.30, .80, .30),
                   orientation=(1., 0., 0., 0.), center_z=.15)
    engine.box_approach = PushPlan(box.id, box.xy, (1.50, -1.10), (0., -1.), (1.32, .10), (), .10, 0.)
    engine.phase = 'box_approach'
    engine.layers.current_boxes = [box]
    engine.layers.detected_boxes[box.id] = engine.layers.approach_reference = box
    engine.layers.decision = {}
    engine.planner.free = np.ones_like(engine.planner.free)
    if blocked:
        obstacle = {'turn': (1.30, .10), 'sidestep': (1.60, -.40), 'exact_center_only': (1.70, -.40)}[blocked]
        engine.planner.free[engine.planner.cell(obstacle)] = False
    if blocked == 'exact_center_only':
        from copy import copy
        from deploy.interaction.motion import SdkNavigationMotion
        from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_translation_recovery_clear
        check = copy(engine.planner)
        SdkNavigationMotion(engine.contact_profile).bind(check)
        def reachable(x):
            return body_translation_recovery_clear(check, [box], engine.position, -np.pi/2, (x, .10))
        assert not reachable(1.50) and reachable(1.45)
    routes = []
    monkeypatch.setattr(engine, '_route', lambda target: routes.append(target.copy()) or [])
    engine.limiter.accept(engine.linear, 0.)
    for i in range(1, 12):
        engine.now = i * .10
        if i >= 3:
            engine.linear = np.zeros(3)
        engine.limiter.observe(engine.linear, engine.now)
        engine._approach_box_face(None)
        if i < 6:
            assert engine.phase == 'box_approach'
        if engine.phase == 'box_turn' or routes:
            break
    accepted = [r for r in engine.events if r['event'] == 'box_turn_from_nearby_stage']
    if blocked in ('turn', 'sidestep'):
        assert not accepted and routes and engine.phase == 'box_approach'
    else:
        assert not routes and engine.phase == 'box_turn'
        assert accepted[0]['stage_error_m'] == pytest.approx(.12)


def test_upper_short_end_waits_for_box_even_with_a_clear_nearby_goal():
    from deploy.app.config import ROOT, load
    from deploy.tests.test_online_navigation import observation
    config = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    engine = NavigationEngine(config['navigation'], config['interaction'])
    packet = observation(1, position=(0., 0., .32), points=[(.8, 0., .03)])
    packet.update(body_linear_velocity=np.zeros(3), body_angular_velocity=np.zeros(3))
    engine.set_goal((.1, 0., .32), 0.)
    for now in (.1, .6, 1.2):
        command = engine.update(packet, now)
        assert engine.decision['reason'] == 'waiting_for_registered_box'
        assert engine.status == 'running' and engine.tracker is None
        np.testing.assert_array_equal(command, np.zeros(3))

    # Re-entering the same goal after an interrupted contact must not skip
    # the required displacement measurement or announce goal success.
    engine.trial_used = True
    engine.set_goal((.1, 0., .32), 2.)
    engine.update(packet, 2.1)
    assert engine.status == 'failed'
    assert engine.result['reason'] == 'push_aside:first_push_displacement_unconfirmed'


def test_first_push_dispatch_review_and_same_goal_replan():
    from deploy.app.config import load, ROOT
    from deploy.interaction.first_push import FirstPushSkill
    from deploy.perception.box_memory import BoxMapMemory
    nav, interaction, registry = settings()
    field = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')['interaction']
    interaction.update(contact_profile=field['contact_profile'], boxes=registry['boxes'], first_push=True)
    memory = BoxMapMemory()
    provider = RegisteredBoxPerception(registry, synchronous=True, box_memory=memory)
    engine = NavigationEngine(nav, interaction, box_memory=memory)
    goal = (0., 2., .4)
    engine.set_goal(goal, 0.)
    def measure(i, center=(0., 0.)):
        t = i*.12
        packet = scan(i, t, center=center)
        obs = dict(packet.observation, lidar_captured_at=t)
        boxes, recognition = provider.update(packet, t)
        return obs, t, boxes, recognition
    try:
        for i in (1, 2):
            obs, t, boxes, recognition = measure(i)
            engine.observe(obs, t, boxes)
            observed_layout(engine)
        engine.update(obs, t, boxes, recognition)
        # The full approach/turn/scan sequence is exercised below. Start this
        # review fixture at the handoff, with the plan chosen by navigation.
        engine._start_push_skill(engine.box_approach, recognition.permission('B1', t), recognition)
        skill = engine.skill
        assert isinstance(skill, FirstPushSkill) and not engine.execution.active
        # Starting at a settled approach pose isolates the first-contact handoff.
        skill.state, skill.started, skill.state_since = 'settle_contact', t, t
        skill.initial_box = np.array(boxes[0].xy)
        skill.engage_box_position = skill.initial_box.copy()
        skill.state, skill.contact_started = 'engage', t
        engine.execution.start('B1', t)
        engine.trial_used = True
        for i in range(3, 7):
            offset = min(.045, (i-2)*.012)
            obs, t, boxes, recognition = measure(i, offset*skill.direction)
            engine.observe(obs, t, boxes)
        skill.stop_trial(t, engine.position, 'box_reached_trial_distance', boxes[0])
        engine._stop_first_push()
        assert engine.execution.awaiting_stop and not engine.passage
        assert engine.update(obs, t, boxes, recognition).tolist() == [0., 0., 0.]
        engine.interaction_stopped()
        for i in range(7, 21):
            obs, t, boxes, recognition = measure(i, .045*skill.direction)
            engine.update(obs, t, boxes, recognition)
            if engine.trial_report:
                break
        assert engine.trial_report['reason'] == 'first_push_observed'
        assert engine.trial_report['first_push']['displacement_m'] == pytest.approx(.045, abs=.01)
        assert engine.goal == goal and engine.phase == 'decide' and engine.passage is None
        assert engine.trial_used and not engine.execution.active
        engine.set_goal((.3, 1., .4), t)
        assert engine.trial_used
    finally:
        provider.close()


@pytest.mark.parametrize('from_memory,initial_error_deg,face_mode',
                         [(False, None, None), (True, None, None), (True, 12., None),
                          (True, None, 'upper_short_end'), (True, 4.8, 'upper_short_end'),
                          (True, 6.7, 'upper_short_end'), (True, 4.8, 'auto'), (True, 4.8, 'auto_retry')],
                         ids=['visible_face', 'remembered_face', 'imperfect_contact_pose', 'upper_short_end',
                              'lid_only', 'lid_only_settled_at_6_7_deg', 'auto_long_face', 'auto_wait_after_only_clear_side_vetoed'])
def test_first_push_in_open_room_from_navigation_to_measured_result(from_memory, initial_error_deg, face_mode, monkeypatch):
    from deploy.app.config import ROOT, load
    from deploy.interaction.config import tracking_config
    from deploy.interaction.first_push import TRIAL_DISTANCE_M, TRIAL_YAW_TOLERANCE
    from deploy.perception.box_memory import BoxMapMemory
    from deploy.tests.test_contact_guard import run_worker
    from deploy.tests.test_motion_protocol import ReplyPipe, protocol_client
    upper_end = face_mode == 'upper_short_end'
    timed = face_mode is not None
    retry = face_mode == 'auto_retry'
    preset = 'box_30x80x30_auto_face.yaml' if face_mode == 'auto' or retry else 'box_30x80x30.yaml'
    config = load(interaction_path=ROOT/'configs'/preset)
    field = config['interaction']
    if not timed:
        field = dict(field, first_push_face=None)
        config['runtime']['control'].pop('field_push_launch_speeds')
        config['runtime']['control'].pop('timed_push_speeds')
    nav = config['navigation']
    mount = config['robot']['lidar_origin_offset_base_m']
    memory = BoxMapMemory()
    engine = NavigationEngine(nav, field, box_memory=memory)
    if retry:
        # Reject the robot's full straight stroke at the field boundary.
        boundary = type(engine.world_boundary)
        contains_motion = boundary.contains_motion

        def boundary_check(self, xy, yaw, velocity, horizon, corners):
            if tuple(velocity) == (1., 0., 0.) and horizon >= .53:
                return False
            return contains_motion(self, xy, yaw, velocity, horizon, corners)

        monkeypatch.setattr(boundary, 'contains_motion', boundary_check)
    provider = RegisteredBoxPerception(tracking_config(field), synchronous=True, box_memory=memory,
                                       world_boundary=engine.world_boundary)
    # Prior scans surveyed the empty floor. Live synthetic returns below describe
    # the two faces of the registered carton, with its 80 cm side along Y.
    layers = engine.layers
    layers.observed[layers.interior] = True
    layers.evidence.log_odds[layers.interior] = -4.
    layers.refresh_global(reason='surveyed_room')
    layers._bind_local()
    evidence = engine.contact_collision.evidence
    evidence.observed[:] = True
    evidence.log_odds[:] = -4.
    center = np.array([1.1, -.55])
    depth, width = (.80, .30) if upper_end else (.30, .80)
    # The fixture describes contact in forward/left coordinates. In the
    # timed cases forward maps to world -Y, across the route to the goal.
    frame = np.array([[0., 1.], [-1., 0.]]) if timed else np.eye(2)
    offset = np.array([2., .24]) if timed else np.zeros(2)
    initial = center.copy()
    # The field failure started at the standing point, still 67° off the face.
    pose = np.array([center[0]-depth/2-.72 if from_memory else .05, -.59, .311])
    yaw = np.deg2rad(-67.) if from_memory else 0.
    near_aligned = initial_error_deg is not None
    lid_only = timed and near_aligned
    if near_aligned:
        pose[1] = center[1] + (.12 if lid_only else .05)
        yaw = np.deg2rad(initial_error_deg)
    velocity = np.zeros(3)
    pinned = None
    saw_face_on_way = turned = contact_request = False
    waiting_scans = 0
    peak_sidestep = 0.
    writer = protocol_client(ReplyPipe())
    stop_at = None
    turn_rates = []
    first_contact_error = None
    centered_from_lid = False
    engage_frame = None
    return_target = None
    precontact_fits_y = []
    height, u = np.meshgrid(np.linspace(.035, .29, 9), np.linspace(-1., 1., 22))
    front = np.c_[np.full(u.size, -depth/2), u.ravel()*width/2, height.ravel()]
    side = np.c_[u.ravel()*depth/2, np.full(u.size, -width/2), height.ravel()]
    faces = np.vstack([front, side])
    along, across = np.meshgrid(np.linspace(-depth/2, depth/2, 8), np.linspace(-width/2, width/2, 6))
    lid = np.c_[along.ravel(), across.ravel(), np.full(along.size, .30)]
    try:
        for i in range(1, 451):
            # In the memory case the initial survey predates this approach.
            now = i*.12 + (2. if from_memory and i > 2 else 0.)
            early_contact_tick = engage_frame is not None and i == engage_frame[0] + 1
            if early_contact_tick:
                # Interval between handoff and first forward request in the field run.
                now = engage_frame[1] + .099610656
            c, s = np.cos(yaw), np.sin(yaw)
            axes = np.array([[c, -s], [s, c]])
            scan_pose = pose.copy()
            if i <= 2:
                scan_pose[1] = center[1]-.85
            origin = scan_pose + np.r_[axes @ mount[:2], mount[2]]
            measured_center = center.copy()
            if near_aligned and not lid_only and i > 2:
                measured_center[1] += .011 if i % 2 else -.011
            if i <= 2:
                # Initial survey from a second viewpoint before the navigation trial.
                points = faces.copy()
                points[:, :2] += measured_center
            elif from_memory and (i == 3 or engine.phase == 'box_turn'
                                  or (engine.phase == 'box_confirm' and waiting_scans < 3)):
                points = np.array([[1.9, -.9, .01], [1.9, -.8, .01], [1.9, -.7, .01]])
                if engine.phase == 'box_confirm':
                    waiting_scans += 1
            elif initial_error_deg == 6.7 and engine.phase in ('timed_push', 'push_settle'):
                # Valid ground rays without an identifiable box through the end of the stroke.
                points = np.array([[1.9, -.9, .01], [1.9, -.8, .01], [1.9, -.7, .01]])
            else:
                points = (lid if lid_only else front).copy()
                points[:, :2] += measured_center
                # Observed background rays pass outside the two ends of the front face.
                crossings = np.array([[measured_center[0]-depth/2, measured_center[1]+edge*(width/2+.045), z] for edge in (-1, 1)
                                      for z in np.linspace(.09, .274, 30)])
                wall_scale = (center[0]+depth/2+.35-origin[0]) / (center[0]-depth/2-origin[0])
                floor_scale = origin[2] / np.maximum(.001, origin[2]-crossings[:, 2])
                scale = np.minimum(wall_scale, floor_scale)[:, None]
                points = np.vstack([points, origin + scale*(crossings-origin)])
            points += np.random.default_rng(4).normal(0., .001, points.shape)
            scan_pose[:2] = offset + frame @ scan_pose[:2]
            origin[:2] = offset + frame @ origin[:2]
            points[:, :2] = offset + points[:, :2] @ frame.T
            world_yaw = yaw - (np.pi/2 if timed else 0.)
            obs = dict(position=scan_pose, orientation=np.array([np.cos(world_yaw/2), 0., 0., np.sin(world_yaw/2)]),
                       velocity=velocity.copy(), body_linear_velocity=np.array([*velocity[:2], 0.]),
                       body_angular_velocity=np.array([0., 0., velocity[2]]), lidar_captured_at=now,
                       sensors={'lidar': dict(pointcloud=points, physics_step=i, max_range=6., min_range=.10,
                                              position=origin)})
            packet = Snapshot(obs, now, now, now, None, tuple(velocity), i, planning_frame="experiment",
                              coordinate_transform={"source_frame": "synthetic_odom"})
            boxes, recognition = provider.update(packet, now)
            if lid_only and boxes and i > 2:
                evidence = recognition.freeze()
                assert pinned.box_id in evidence.top_returns(i, now)
                assert not evidence.top_returns(i-1, now)
                assert not evidence.top_returns(i, now+1.)
            obs = packet.navigation_view(nav['mapping']['point_height'], engine.world_boundary).observation
            if i <= 2:
                engine.observe(obs, now, boxes)
                continue
            if i == 3:
                engine.set_goal((4., 0., .31) if timed else (2.4, -.75, .31), now)
            previous_phase = engine.phase
            if timed and previous_phase == 'timed_push' and not engine.trial_used:
                # Deliver the previous tick's successful SDK move. Playback of the
                # actual worker below independently checks this same command.
                lease = engine.interaction_command
                engine.push_speed_sent(pinned.box_id, interaction_id=lease['id'],
                                       worker_at=engine.skill.timed_push.planned_at + .01)
            if initial_error_deg == 6.7 and previous_phase in ('timed_push', 'push_settle'):
                boxes, recognition = [], None
            velocity = engine.update(obs, now, boxes, recognition)
            if retry and engine.rejected_push_faces:
                assert engine.status == 'running' and engine.result is None
                assert not engine.execution.active and not engine.trial_used and engine.interaction_command is None
                if previous_phase == 'skill':
                    np.testing.assert_array_equal(velocity, np.zeros(3))
                    assert engine.phase == 'decide' and engine.skill is None
                    assert engine.events[-1]['reason'] == 'robot_push_outside_world_boundary'
                if engine.decision.get('reason') == 'no_reachable_push_face':
                    # This room's other three pushes cannot open the route to (4, 0).
                    assert engine.box_approach is None and engine.goal[:2] == (4., 0.)
                    np.testing.assert_array_equal(velocity, np.zeros(3))
                    return
            if timed and return_target is None and engine.phase == 'timed_push':
                return_target = engine.position[:2].copy()
            if lid_only and engage_frame is None and engine.execution.active:
                engage_frame = (i, now, boxes, recognition.freeze())
                assert engine.interaction_command['phase'] == 'timed_push'
                np.testing.assert_allclose(velocity, [engine.skill.timed_push.approach_speed, 0., 0.])
            if early_contact_tick:
                assert engine.layers.last_scan == engine.contact_collision.last_scan == i
                assert velocity[0] > 0. and engine.phase == 'timed_push'
            if engage_frame is not None and i == engage_frame[0] + 2:
                assert velocity[0] > 0. and engine.trial_used
            if pinned is None and engine.box_approach is not None:
                pinned = engine.box_approach
            if engine.box_approach is not None:
                assert engine.box_approach is pinned and engine.interaction_command is None
                assert engine.layers.detected_boxes[pinned.box_id] is engine.layers.approach_reference
                if engine.phase == 'box_approach' and engine._permission(pinned.box_id, recognition):
                    saw_face_on_way = True
                if previous_phase == 'box_turn' and velocity is not None:
                    np.testing.assert_allclose(velocity[:2], 0.)
                    turned |= abs(velocity[2]) > .01
                    turn_rates.append(abs(velocity[2]))
                if previous_phase == 'box_confirm' and not boxes:
                    np.testing.assert_allclose(velocity, 0.)
            if engine.phase == 'skill':
                assert engine.skill.plan is pinned
                assert engine.planner.motion_backend.profile == engine.contact_profile
                if boxes and not engine.trial_used:
                    precontact_fits_y.append(boxes[0].xy[1])
                if lid_only and engine.skill.state == 'align_lateral' and velocity is not None:
                    centered_from_lid |= abs(velocity[1]) > .02
                    peak_sidestep = max(peak_sidestep, abs(velocity[1]))
                    if abs(velocity[1]) > .12:
                        assert abs(velocity[2]) < 1e-8
            if velocity is not None and not engine.execution.awaiting_stop and stop_at is None:
                writer.submit(velocity, now, now+.25, interaction=engine.interaction_command)
            if (engine.interaction_command or {}).get('phase') in ('engage', 'push', 'timed_push') and velocity is not None:
                contact_request |= velocity[0] > 0.
                if contact_request and first_contact_error is None:
                    first_contact_error = engine.skill._box_frame(engine._contact_context)
            if engine.execution.awaiting_stop:
                stop_at = now
                engine.interaction_stopped()
            if engine.trial_report or velocity is None:
                break
            if (lid_only and engine.skill is not None and engine.skill.contact_started is None
                    and abs(velocity[1]) > abs(velocity[0])):
                # A low lateral request can lean the body without producing
                # a footstep. Exercise a 0.25 m/s deadband in this model.
                velocity = velocity.copy()
                velocity[:2] *= max(0., 1. - .25 / abs(velocity[1]))
            pose[:2] += .12 * (axes @ velocity[:2])
            if from_memory and previous_phase == 'box_turn':
                # Field-like under-response: about .02 rad/s at a .32 request
                # and .39 rad/s at .60. Feed actual yaw back on the next tick.
                velocity = velocity.copy()
                velocity[2] = np.sign(velocity[2])*np.interp(abs(velocity[2]), [0., .32, .60], [0., .02, .39])
            yaw += .12*velocity[2]
            # Ideal non-slip forward contact only moves the box once the nose reaches it.
            if engine.execution.active and abs(pose[1]-center[1]) < width/2:
                center[0] = max(center[0], pose[0] + .34*np.cos(yaw) + depth/2)
        assert not retry, (engine.result, engine.phase, engine.decision, engine.rejected_push_faces)
        assert engine.trial_report, (engine.result, engine.phase, engine.decision, pose, center)
        expected = .40 * 4 / 3 if timed else TRIAL_DISTANCE_M
        assert pinned.push_distance == pytest.approx(expected)
        if timed:
            assert engine.phase == 'post_push_reobserve' and engine.trial_used
            assert engine.trial_report['reason'] == 'timed_push_released'
            assert engine.trial_report['first_push']['displacement_m'] is None
            assert engine.trial_report['actual_push_distance_m'] is None
            assert np.linalg.norm(engine.position[:2] - return_target) > expected - .03
            assert center[0]-initial[0] >= expected - .03
            assert pinned.box_id not in memory.locations() and pinned.box_id not in engine.layers.detected_boxes
            assert not engine.execution.active and engine.interaction_command is None
            # A repeated stopped scan cannot renew the old box or start ordinary motion.
            np.testing.assert_allclose(engine.update(obs, now, boxes, recognition), 0.)
            assert engine.phase == 'post_push_reobserve' and pinned.box_id not in engine.layers.detected_boxes
        else:
            assert engine.trial_report['reason'] == 'first_push_observed'
            displacement = engine.trial_report['first_push']['displacement_m']
            assert .02 <= displacement <= expected + .03
            assert displacement == pytest.approx(center[0]-initial[0], abs=.008)
            assert not engine.trial_used and engine.phase == 'decide'
        assert contact_request
        aligned = next(e for e in engine.events if e['event'] == 'box_face_aligned')
        confirmed = next(e for e in engine.events if e['event'] == 'box_face_confirmed')
        assert abs(aligned['heading_error_rad']) < TRIAL_YAW_TOLERANCE
        assert confirmed['scan_at'] > aligned['time_s']
        packets = [(p['generated_at'], p) for p in writer.pipe.sent]
        assert stop_at is not None
        sequence = [(packets[0][0], dict(op='arm')), *packets,
                    (stop_at, dict(op='stop', reason='interaction_completed'))]
        if timed:
            nav_start = now + .12
            sequence.append((nav_start, dict(op='arm')))
            lidar = obs['sensors']['lidar']
            nav_pose, nav_yaw = engine.position.copy(), engine.heading
            command = np.zeros(3)
            for step in range(100):
                t = nav_start + step*.1
                c, s = np.cos(nav_yaw), np.sin(nav_yaw)
                axes = np.array([[c, -s], [s, c]])
                nav_pose[:2] += .1 * (axes @ command[:2])
                nav_yaw += .1 * command[2]
                fresh = dict(obs, position=nav_pose.copy(), orientation=[np.cos(nav_yaw/2), 0., 0., np.sin(nav_yaw/2)],
                             velocity=command.copy(), body_linear_velocity=[*command[:2], 0.],
                             body_angular_velocity=[0., 0., command[2]], lidar_captured_at=t,
                             sensors={'lidar': dict(lidar, physics_step=i+1+step,
                                                    position=nav_pose + np.r_[axes @ mount[:2], mount[2]])})
                command = engine.update(fresh, t, boxes, recognition)
                assert engine.goal[:2] == (4., 0.) and engine.interaction_command is None
                assert command is not None, engine.result
                writer.submit(command, t, t+.25)
                sequence.append((t, writer.pipe.sent[-1]))
                if engine.phase == 'navigate' and np.linalg.norm(command[:2]) > .01:
                    break
            assert engine.decision['reason'] == 'post_push_route_observed'
            assert engine.phase == 'navigate' and np.linalg.norm(command[:2]) > .01
        reports, moves = run_worker(monkeypatch, sequence, worker_control=config['runtime']['control'])
        if timed:
            assert any(t >= nav_start and np.linalg.norm(v[:2]) > .005 for t, v in moves)
            sent_start = engine.trial_report['timed_push']['sdk_started_at']
            assert all(v[0] >= 0. for t, v in moves if sent_start <= t < stop_at)
        assert all(r['accepted'] for r in reports if r.get('kind') == 'command_ack')
        sent, = [r for r in reports if r.get('event') == 'push_speed_sent']
        if engage_frame is not None:
            assert engage_frame[1] - 1e-9 <= sent['worker_at'] < engage_frame[1] + .099610656
            planned = next(p for t, p in packets if t == engage_frame[1])
            assert planned['velocity'][0] == pytest.approx(.35)
            assert sent['velocity'][0] == pytest.approx(config['runtime']['control']['field_push_launch_speeds'][pinned.box_id])
        engine.push_speed_sent(sent['object_id'])
        assert engine.trial_used
        assert sent['object_id'] == pinned.box_id and sent['worker_at'] > confirmed['time_s']
        assert 0 < sent['velocity'][0] <= field['contact_profile']['push_max_mps']
        assert any(t == sent['worker_at'] and np.array_equal(v, sent['velocity']) for t, v in moves)
        peak_forward = max(v[0] for t, v in moves if sent['worker_at'] <= t < stop_at)
        if timed:
            assert peak_forward == pytest.approx(field['timed_push_speed_mps'])
        else:
            assert .10 <= peak_forward <= field['contact_profile']['push_max_mps']+1e-8
        if from_memory:
            assert waiting_scans == 3
            if not timed:
                assert saw_face_on_way
            if lid_only:
                assert centered_from_lid and not turned
                assert peak_sidestep >= .29
                feet = [r['sidestep_feedback'] for r in reports if 'sidestep_feedback' in r]
                assert max(abs(row['sdk_velocity'][1]) for row in feet) >= .29
                assert all(row['foot_force'] == [22, 0, 0, 21] and len(row['foot_z_body_m']) == 4 for row in feet)
                assert confirmed['scan_at'] > aligned['time_s']
                assert np.rad2deg(abs(aligned['heading_error_rad'])) == pytest.approx(initial_error_deg, abs=.25)
                assert engage_frame[3].top_returns(engage_frame[3].entries[pinned.box_id]['track']['scan_id'], engage_frame[1])
            elif near_aligned:
                assert not turned
                assert np.ptp(precontact_fits_y) > .018
                _, lateral, error = first_contact_error
                assert abs(lateral) > .08 and np.deg2rad(10.) < abs(error) < TRIAL_YAW_TOLERANCE
            else:
                assert turned and .50 <= max(turn_rates) <= field['contact_profile']['max_yaw_rps']
            assert sum(event.get('event') == 'decision' and event.get('kind') == 'push'
                       for event in engine.events) == 1
        if face_mode == 'auto':
            np.testing.assert_allclose(pinned.direction, [0., -1.], atol=.03)
            assert abs(np.dot(pinned.direction, pinned.long_axis)) < .03
            assert any(event.get('reason') == 'automatic_face_approach' for event in engine.events)
        elif upper_end:
            np.testing.assert_allclose(pinned.direction, [0., -1.], atol=.03)
            assert pinned.approach_xy[1] > pinned.initial_xy[1]
            assert pinned.destination_xy[1] < pinned.initial_xy[1]
            assert any(event.get('reason') == 'upper_short_end_approach' for event in engine.events)
        else:
            assert any(event.get('reason') == 'registered_first_contact' for event in engine.events)
    finally:
        provider.close()
