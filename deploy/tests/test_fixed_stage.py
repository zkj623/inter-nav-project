"""A pinned side stage near a wall must be reached with measured body motion."""
import math
from dataclasses import replace

import numpy as np
import pytest

from deploy.app.config import ROOT, load
from deploy.app.navigation import NavigationEngine
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState, PushPlan


@pytest.mark.parametrize('lateral_response,blocked', [(1., False), (.5, False), (0., False), (1., True)])
def test_stalled_final_leg_keeps_its_target_and_requires_settled_arrival(lateral_response, blocked, monkeypatch):
    cfg = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    engine = NavigationEngine(cfg['navigation'], cfg['interaction'])
    engine.set_goal((4., 0., .32), 0.)
    engine.full_motion_feedback = True
    stage = np.array([1.17, .335])
    direction = np.array([.055, -math.sqrt(1-.055**2)])
    box = BoxState('box-1', (1.43, -.77), (.40, .88, .36), local_dimensions=(.30, .80, .30),
                   orientation=(1., 0., 0., 0.), center_z=.15)
    plan = PushPlan(box.id, box.xy, tuple(box.xy + .30*direction), tuple(direction), tuple(stage), (), .30, 0.)
    engine.box_approach = plan
    engine.layers.current_boxes = [box]
    engine.layers.detected_boxes[box.id] = engine.layers.approach_reference = box
    engine.layers.decision = {}
    engine.phase = 'box_approach'
    engine.layers.observed[:] = True
    engine.layers.global_observed[:] = True
    engine.planner.free[:] = True
    engine.planner.free[:, engine.planner.cell((1.75, .35))[1]] = False
    monkeypatch.setattr(engine, '_route', lambda target: [tuple(target)])
    # A forward request moves the trunk up to 6 cm without advancing the feet.
    # After rotation only the portion above the simulated lateral deadband steps.
    base = np.array([.99, .31])
    pos = base.copy()
    lean = np.zeros(2)
    velocity = np.zeros(2)
    yaw, yaw_rate = .11, 0.
    peak = 0.
    for i in range(1, 530):
        engine.now = engine.planner.control_time = .1*i
        engine.position, engine.heading, engine.tilt = np.r_[pos, .32], yaw, 0.
        engine.linear, engine.angular = np.r_[velocity, 0.], np.array([0., 0., yaw_rate])
        engine.limiter.observe([*velocity, yaw_rate], engine.now, body_state=(engine.position, engine.linear, engine.angular, 0.))
        phase = engine.phase
        if blocked and phase == 'box_stage_brake':
            engine.planner.free[engine.planner.cell((1.17, -.10))] = False
        cmd = engine._approach_box_face(None)
        assert engine.box_approach is plan and engine.interaction_command is None
        if cmd is None or engine.phase == 'box_confirm':
            break
        if phase == 'box_stage_translate':
            peak = max(peak, abs(cmd[1]))
        c, s = math.cos(yaw), math.sin(yaw)
        axes = np.array([[c, -s], [s, c]])
        base += .1 * axes[:, 1] * lateral_response * np.sign(cmd[1]) * max(0., abs(cmd[1])-.25)
        lean += .27 * (axes @ (.06*np.clip(cmd[:2]/[.18, .30], -1., 1.) * [1., 1./3]) - lean)
        new_pos = base+lean
        velocity, pos = axes.T @ (new_pos-pos)/.1, new_pos
        yaw_rate = np.sign(cmd[2]) * np.interp(abs(cmd[2]), [0., .32, .60, .70], [0., .02, .39, .45])
        yaw += .1*yaw_rate
    events = [e['event'] for e in engine.events]
    assert events.count('box_stage_stalled') == 1
    if blocked or not lateral_response:
        reason = 'fixed_stage_turn_blocked' if blocked else 'fixed_stage_no_progress'
        assert engine.result['reason'] == 'push_aside:'+reason
        assert 'box_stage_reached_after_sidestep' not in events
    else:
        assert engine.phase == 'box_confirm' and peak >= .29
        assert np.linalg.norm(pos-stage) < .08 and np.linalg.norm(base-stage) < .08
        assert events.index('box_stage_reached_after_sidestep') < events.index('box_face_aligned')


@pytest.mark.parametrize('responds', [True, False])
def test_turn_rebound_requires_net_angular_progress(responds):
    from deploy.interaction.motion import SdkHeadingTrim, SdkNavigationMotion
    from deploy.tests.test_sdk_contact_profile import fixture
    from deploy.interaction.profile import ContactProfile
    cfg = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    profile = ContactProfile.from_dict(cfg['interaction']['contact_profile'])
    _, ctx = fixture()
    SdkNavigationMotion(profile).bind(ctx.planner)
    ctx.limiter.yaw_acceleration = profile.yaw_acceleration_rps2
    turn = SdkHeadingTrim(0., math.radians(6), acceptance_tolerance=math.radians(7),
                          profile=profile, compensate_rebound=True)
    grounded = heading = math.radians(8.6)
    lean = rate = 0.
    peak = 0.
    for i in range(200):
        now = .1*i
        ctx.limiter.observe([0., 0., rate], now)
        cmd, done, detail = turn.update(ctx.position, heading, ctx.planner, [], ctx.limiter, now,
                                        np.zeros(3), np.array([0., 0., rate]), 0.)
        ctx.limiter.accept(cmd, now)
        peak = max(peak, abs(cmd[2]))
        if done or turn.failed:
            break
        if responds:
            grounded += .1*np.sign(cmd[2])*max(0., abs(cmd[2])-.52)
        lean += .3*(math.radians(3.6)*np.clip(cmd[2]/.5, -1., 1.) - lean)
        new_heading = grounded + lean
        rate, heading = (new_heading-heading)/.1, new_heading
    assert detail['measured_rebound_rad'] > 0.
    if responds:
        assert done and abs(grounded) < math.radians(7) and peak > .52
    else:
        assert turn.failed == 'sdk_heading_stalled' and not done


@pytest.mark.parametrize('stepping', [True, False])
def test_lid_centering_tracks_net_motion_through_yaw_jitter(stepping):
    from deploy.interaction.motion import SdkNavigationMotion
    from deploy.interaction.profile import ContactProfile
    from deploy.interaction.skill_backend import create_push_skill
    from deploy.tests.test_sdk_contact_profile import fixture, permission
    cfg = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    profile = ContactProfile.from_dict(cfg['interaction']['contact_profile'])
    base, ctx = fixture()
    skill = create_push_skill(permission(), base.plan, (0, 0), (1, 0), profile, upper_short_end=True)
    skill.state = 'align_lateral'
    ctx.box = replace(ctx.box, xy=(1.1, 0.))
    ctx.boxes = [ctx.box]
    SdkNavigationMotion(profile).bind(ctx.planner)
    ctx.limiter.linear_acceleration = cfg['navigation']['linear_acceleration']
    ctx.limiter.accept(np.zeros(3), 0.)
    target = ctx.position[:2] + [0., .18]
    grounded = ctx.position[:2].copy()
    lean = np.zeros(2)
    for i in range(1, 201):
        ctx.now = i*.1
        ctx.heading = math.radians(6.7 if i%2 else 7.2)
        cmd, done = skill.backend.translate(skill, ctx, target, np.array([0., 1.]), profile.lateral_mps, .03)
        assert skill.state == 'align_lateral' and cmd[2] == 0.
        result = skill.backend.after_step(skill, ctx, skill._output(cmd))
        if result.state == 'recheck_lid':
            assert not stepping and result.status == 'running' and not any(result.velocity)
            assert result.events[-1]['reason'] == 'lid_centering_no_progress'
            return
        ctx.limiter.accept(cmd, ctx.now)
        if done:
            break
        c, s = math.cos(ctx.heading), math.sin(ctx.heading)
        axes = np.array([[c, -s], [s, c]])
        if stepping:
            grounded += .1 * axes @ (np.sign(cmd[:2])*np.maximum(np.abs(cmd[:2])-.25, 0.))
        lean += .3 * (axes @ (.02*np.clip(cmd[:2]/.3, -1., 1.))-lean)
        pos = grounded+lean
        ctx.body_velocity[:2] = axes.T @ (pos-ctx.position[:2])/.1
        ctx.position[:2] = pos
        ctx.limiter.observe([*ctx.body_velocity[:2], 0.], ctx.now)
    assert stepping and done and abs(grounded[1]-target[1]) < .03


@pytest.mark.parametrize('travel,rotation', [(.12, 0.), (0., 22.)])
def test_stationary_box_guidance_requires_independent_fits_before_replacing_stage(travel, rotation, monkeypatch):
    from deploy.perception.box_memory import BoxMapMemory
    from deploy.tests.test_online_navigation import observation
    cfg = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    memory = BoxMapMemory()
    engine = NavigationEngine(cfg['navigation'], cfg['interaction'], box_memory=memory)
    engine.set_goal((4., 0., .32), 0.)
    key = 'box-1'
    plan = PushPlan(key, (1.3, -.85), (1.3, -1.15), (0., -1.), (1.3, .25), (), .30, 0.)
    monkeypatch.setattr(engine, '_subnavigation', lambda target: (np.zeros(3), False, None))
    def feed(stamp, x, degrees, *, window=False):
        angle = math.radians(degrees)
        row = dict(object_id=key, valid=True, center=[1.3+x, -.85, .15], captured_at=stamp, expires_at=stamp+.38)
        if window:
            row.update(dimensions_m=(.30, .80, .30), track_id=key, long_axis_rad=angle+math.pi/2)
        else:
            row.update(local_dimensions=(.30, .80, .30), geometry_verified=True, yaw_rad=angle)
        memory.observe(dict(geometry_source='window_two_faces' if window else 'current_lidar_cuboid_surfaces',
                            detections=[row]), stamp)
        box = memory.locations()[key].box
        obs = observation(round(stamp*10), position=(1., .3, .32))
        engine.observe(obs, stamp, [box])
        assert engine.current_proposals[key] is box
        return box

    anchor = feed(.1, 0., 0.)
    engine.layers.approach_reference = anchor
    engine.box_approach, engine.phase = plan, 'box_approach'
    engine.layers.decision = {'kind':'push'}
    for stamp, x, yaw in ((.2, .021, 5.), (.3, .03, 178.), (.4, -.03, -6.),
                          (.5, .16, 19.), (.6, -.008, 1.), (.7, 0., 179.)):
        feed(stamp, x, yaw)
        engine._approach_box_face(None)
        assert engine.box_approach is plan
    # A window and a lid fit of the same physical sweep count once.
    for window in (True, False, False, False):
        feed(.8, travel, rotation, window=window)
        engine._approach_box_face(None)
        assert engine.box_approach is plan
    feed(.9, travel, rotation)
    engine._approach_box_face(None)
    assert engine.box_approach is plan
    feed(1., travel, rotation)
    engine._approach_box_face(None)
    assert engine.box_approach is None and engine.phase == 'decide'
    assert sum(e['event']=='box_approach_geometry_changed' for e in engine.events) == 1
    engine.push_speed_sent(key)
    assert key not in engine.guidance_poses


def test_usable_pose_goes_straight_to_turn_and_blocked_stage_has_alternatives():
    from deploy.interaction.motion import SdkSidestep
    cfg = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    engine = NavigationEngine(cfg['navigation'], cfg['interaction'])
    engine.set_goal((4., 0., .32), 0.)
    box = BoxState('box-1', (1.34, -.86), (.30, .80, .30), local_dimensions=(.30, .80, .30),
                   orientation=(1., 0., 0., 0.), center_z=.15)
    plan = PushPlan(box.id, box.xy, (1.34, -1.16), (0., -1.), (1.44, .23), (), .30, 0.)
    engine.layers.decision = {'kind':'push'}
    engine.box_approach, engine.phase = plan, 'box_stage_brake'
    engine.full_motion_feedback = True
    engine.position = np.array([1.15, .32, .32])
    engine.heading = engine.tilt = 0.
    engine.linear = engine.angular = np.zeros(3)
    engine.layers.current_boxes = [box]
    engine.layers.detected_boxes[box.id] = engine.layers.approach_reference = box
    engine.planner.free[:] = engine.planner.box_free[:] = engine.planner.static_clear[:] = True
    engine.layers.observed[:] = True
    engine.planner.robot_heading = engine.heading
    engine.stage_sidestep = SdkSidestep(plan.approach_xy, .08, 0., engine.contact_profile)
    for i in range(1,12):
        engine.now = engine.planner.control_time = i/10
        engine.limiter.observe(np.zeros(3), engine.now)
        engine._approach_box_face(None)
        if engine.phase == 'box_turn':
            break
    assert engine.phase == 'box_turn' and engine.stage_sidestep is None
    handoff = next(e for e in engine.events if e['event']=='box_turn_from_nearby_stage')
    assert handoff['stage_error_m'] > .29
    # A different stop with a blocked turn must keep the chosen face, skip
    # rejected positions and issue a reachable replacement before trying again.
    engine._relocate_upper_stage('turn_blocked', np.zeros(3))
    assert engine.phase == 'box_approach' and engine.box_approach.direction == plan.direction
    assert engine.box_approach.initial_xy == plan.initial_xy
    stage = np.asarray(engine.box_approach.approach_xy)
    assert all(np.linalg.norm(stage-xy) > .08 for xy in engine.rejected_stages)
    assert engine.box_approach.approach_path


def test_stopped_short_end_uses_measured_face_even_when_approach_grid_is_occupied():
    from deploy.interaction.motion import SdkNavigationMotion
    from deploy.interaction.skill_backend import create_push_skill
    from deploy.tests.test_sdk_contact_profile import fixture, permission

    cfg = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    engine = NavigationEngine(cfg['navigation'], cfg['interaction'])
    base, ctx = fixture()
    skill = create_push_skill(permission(), base.plan, (0., 0.), (4., 0.), engine.contact_profile, upper_short_end=True)
    ctx.planner = engine.planner
    SdkNavigationMotion(engine.contact_profile).bind(ctx.planner)
    ctx.planner.free = np.ones_like(ctx.planner.free)
    ctx.box = replace(ctx.box, xy=(1.12, 0.), dimensions=(1.04, .54, .30),
                      local_dimensions=(.80, .30, .30), center_z=.15)
    ctx.boxes = [ctx.box]
    ctx.position = np.array([0., .05, .32])
    cell = ctx.planner.cell((.64, .05))
    ctx.planner.free[cell] = False
    ctx.boxes.append(BoxState('B2', (.64, .05), (.12, .12, .30)))
    ctx.planner.mixed_box_cells = {ctx.box.id: ~ctx.planner.free}
    assert skill._contact_ready(ctx)
    ctx.box_observation_id = None
    assert not skill._contact_ready(ctx)


def test_lid_sidestep_without_progress_holds_its_station_until_a_new_lid():
    from deploy.interaction.motion import SdkSidestep
    from deploy.tests.test_sdk_contact_profile import permission

    cfg = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    engine = NavigationEngine(cfg['navigation'], cfg['interaction'])
    engine.set_goal((4., 0., .32), 0.)
    box = BoxState('box-1', (1.34, -.86), (.30, .80, .30), local_dimensions=(.30, .80, .30),
                   orientation=(1., 0., 0., 0.), center_z=.15)
    plan = PushPlan(box.id, box.xy, (1.34, -1.16), (0., -1.), (1.44, .23), (), .30, 0.)
    engine.task_start = (0., 0.)
    engine.layers.decision = {'kind': 'push'}
    engine.full_motion_feedback = True
    engine.position = np.array([1.17, .32, .32])
    engine.heading = engine.planner.robot_heading = -math.pi/2
    engine.orientation = (math.sqrt(.5), 0., 0., -math.sqrt(.5))
    engine.tilt = 0.
    engine.linear = engine.angular = np.zeros(3)
    engine.layers.current_boxes = [box]
    engine.layers.detected_boxes[box.id] = box
    engine.planner.free[:] = engine.planner.box_free[:] = engine.planner.static_clear[:] = True
    engine.layers.observed[:] = True
    engine._start_push_skill(plan, permission(object_id=box.id), None)
    skill = engine.skill
    skill.state = 'align_lateral'
    skill.backend.centering = SdkSidestep(plan.approach_xy, .03, 0., engine.contact_profile)
    skill.backend.centering.progress_at = 0.
    engine.now = engine.contact_profile.stall_seconds + .1
    engine._skill_step(None)
    assert engine.phase == 'skill' and engine.skill is skill and skill.state == 'recheck_lid'
    assert not engine.rejected_stages and engine.box_approach is None
    for _ in range(12):
        engine.now += .1
        engine.limiter.observe(np.zeros(3), engine.now)
        np.testing.assert_array_equal(engine._skill_step(None), np.zeros(3))
    assert skill.precontact_wait == 'waiting_for_stopped_lid' and skill.plan is plan
    assert engine.result is None and engine.interaction_command is None and not engine.trial_used
    assert all(e.get('event') != 'box_stage_reposition' for e in engine.events)


@pytest.mark.parametrize('landing,z,blocked,ready', [
    (.08, .3455, None, True), (-.08, .3455, None, True), (.081, .3455, None, False),
    (.04, .40, None, False), (.07, .3455, 'scan', True), (.07, .3455, 'map', True),
])
def test_blocked_sidestep_hands_a_confirmed_short_face_to_the_timed_controller(landing, z, blocked, ready, monkeypatch):
    from deploy.interaction.motion import SdkNavigationMotion
    from deploy.interaction.profile import ContactProfile
    from deploy.interaction.skill_backend import create_push_skill
    from deploy.tests.test_sdk_contact_profile import fixture, permission

    cfg = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    profile = ContactProfile.from_dict(cfg['interaction']['contact_profile'])
    original, ctx = fixture()
    skill = create_push_skill(permission(), original.plan, (0., 0.), (1., 0.), profile, upper_short_end=True)
    skill.state = 'align_lateral'
    skill.started = skill.state_since = 0.
    ctx.box = replace(ctx.box, xy=(1.10, 0.), dimensions=(.86, .36, .36), local_dimensions=(.80, .30, .30), center_z=.15)
    ctx.boxes = [ctx.box]
    skill.initial_box = np.array(ctx.box.xy)
    # A six-degree heading points the nose away from its current lateral
    # coordinate. Place its projected landing exactly at the requested offset.
    ctx.heading = math.radians(-6.)
    ctx.orientation = (math.cos(ctx.heading/2), 0., 0., math.sin(ctx.heading/2))
    ctx.position = np.array([0., landing - .70*math.tan(ctx.heading), z])
    SdkNavigationMotion(profile).bind(ctx.planner)
    actual_sweep = skill.backend.motion_clear
    monkeypatch.setattr(skill.backend, 'motion_clear',
                        lambda skill, ctx, v, *, contact: contact and actual_sweep(skill, ctx, v, contact=True))
    ctx.limiter.accept([0., .12, 0.], .1)
    result = skill.backend.after_step(skill, ctx, skill._output([0., .12, 0.]))
    assert result.state == 'recheck_lid' and not any(result.velocity)
    assert result.events[-1]['reason'] == 'sdk_body_or_box_sweep_blocked'
    ctx.contact_motion_check = lambda velocity, ctx: blocked != 'scan'
    if blocked == 'map':
        ctx.planner.free[ctx.planner.cell((.52, 0.))] = False
    ctx.body_velocity[1] = .14
    for i in range(3, 18):
        ctx.now = ctx.box_observed_at = ctx.map_observed_at = i/10
        ctx.box_observation_id = f'scan:{i}'
        ctx.box_top_observed = i < 7
        if i >= 7:
            ctx.body_velocity[:] = 0.
            ctx.limiter.accept(np.zeros(3), ctx.now)
        assert not any(skill.step(ctx).velocity)
        assert skill.contact_started is None
    stopped = skill.recheck_stopped_at
    assert stopped >= 1.0 and skill.precontact_wait == 'waiting_for_stopped_lid'
    ctx.box_top_observed = True
    ctx.box_observed_at = stopped
    assert not any(skill.step(ctx).velocity) and skill.contact_started is None
    ctx.box_observed_at, ctx.map_observed_at = ctx.now, stopped
    assert not any(skill.step(ctx).velocity) and skill.precontact_wait == 'waiting_for_push_map'
    ctx.now = ctx.map_observed_at = ctx.box_observed_at = 1.8
    ctx.box_observation_id = 'scan:18'
    result = skill.step(ctx)
    if not ready:
        assert not any(result.velocity) and result.state == 'recheck_lid' and skill.contact_started is None
        return
    # The engine owns the timed command and hands it to the writer on this same tick.
    assert result.state == 'engage' and result.events[-1]['scan_at'] > stopped
    assert result.events[-1]['landing_offset_m'] == pytest.approx(landing)
    assert skill.contact_started == ctx.now


def test_sweep_limited_lateral_command_from_field_run_stops_for_lid_recheck():
    from deploy.interaction.profile import ContactProfile
    from deploy.interaction.skill_backend import create_push_skill
    from deploy.tests.test_sdk_contact_profile import fixture, permission

    cfg = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')
    profile = ContactProfile.from_dict(cfg['interaction']['contact_profile'])
    base, ctx = fixture()
    skill = create_push_skill(permission(), base.plan, (0., 0.), (1., 0.), profile, upper_short_end=True)
    skill.state = 'align_lateral'
    skill.backend.centering_detail = dict(limited_by_sweep=True, braking=False, measured_closing_mps=0.)
    command = skill.backend.after_step(skill, ctx, skill._output((-.00046, .04811, 0.)))
    assert command.state == 'recheck_lid' and not any(command.velocity)
    assert command.events[-1]['reason'] == 'sdk_body_or_box_sweep_blocked'
