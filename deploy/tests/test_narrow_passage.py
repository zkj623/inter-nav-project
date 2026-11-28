"""Offline navigation through measured straight throats on a 10 cm grid."""
import math

import numpy as np
import pytest

from deploy.app.navigation import NavigationEngine
from deploy.app.config import ROOT, load
from deploy.tests.test_online_navigation import config, observation
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState


def scene(width=.60, *, start=(-1.5, 0.), goal=(1.5, 0.), angle=0., interaction=None):
    engine = NavigationEngine(config(route_clearance=.40), interaction)
    engine.set_goal((*goal, .31), 0.)
    packet = observation(position=(*start, .31), points=[(-2.8, -2.8, .2)])
    packet['orientation'] = np.array([math.cos(angle/2), 0., 0., math.sin(angle/2)])
    engine.observe(packet, 0.)
    y, x = np.mgrid[:60, :60]*.1-2.95
    free = (abs(x) > .70) | (abs(y) < width/2)
    layers = engine.layers
    layers.evidence.log_odds[:] = np.where(free, -4., 4.)
    layers.observed[:] = True
    layers._latest_static[:] = ~free
    layers.local_hits[:] = False
    layers.refresh_global(reason='fixture')
    layers._bind_local()
    return engine, packet


def drive(engine, packet, steps=1000, start_time=0.):
    pose = packet['position'].copy()
    yaw = 2*math.atan2(packet['orientation'][3], packet['orientation'][0])
    twist = np.zeros(3)
    stages = []
    for frame in range(steps):
        t = start_time + (1+frame)*.1
        packet.update(position=pose.copy(), orientation=np.array([math.cos(yaw/2), 0., 0., math.sin(yaw/2)]),
                      body_linear_velocity=np.array([*twist[:2], 0.]),
                      body_angular_velocity=np.array([0., 0., twist[2]]))
        twist = engine.update(packet, t)
        if twist is None:
            break
        if engine.phase == 'cross_passage':
            stages.append((engine.narrow_stage, pose.copy(), yaw, twist.copy()))
        c, s = math.cos(yaw), math.sin(yaw)
        pose[:2] += .1*np.array([c*twist[0]-s*twist[1], s*twist[0]+c*twist[1]])
        yaw += .1*twist[2]
    return stages


def test_straight_throat_enters_aligned_and_reaches_original_goal():
    engine, packet = scene(start=(-1.7, -.15), angle=.35)
    stages = drive(engine, packet)
    assert engine.result and engine.result['success'], (engine.phase, engine.decision, engine.last_tracking)
    assert any(stage == 'cross' and abs(pose[0]) < .6 and abs(yaw) < .06 and cmd[0] > 0
               for stage, pose, yaw, cmd in stages)
    assert any(e['event'] == 'narrow_passage_crossed' for e in engine.events)


@pytest.mark.parametrize('width', [.40, 1.20])
def test_impassable_and_ordinary_corridors_do_not_select_special_mode(width):
    engine, packet = scene(width)
    engine.update(packet, .1)
    assert engine.narrow is None
    if width > 1:
        assert engine.phase == 'navigate'


def test_map_change_while_crossing_stops_the_command():
    engine, packet = scene(start=(-.3, 0.))
    engine.update(packet, .1)
    assert engine.phase == 'cross_passage'
    engine.layers.observed[:, 32:34] = False
    cmd = engine.update(packet, .2)
    assert not np.any(cmd) and engine.phase == 'decide'
    assert engine.decision['reason'] == 'narrow_passage_changed'


@pytest.mark.parametrize('field_floor', [True, False])
def test_push_reopens_throat_from_confirmed_field_floor_or_new_rays(field_floor):
    profile = load(interaction_path=ROOT/'configs/box_30x80x30.yaml')['interaction']
    if not field_floor:
        profile = dict(profile, first_push_face=None)
    engine, _ = scene(interaction=profile)
    layers = engine.layers
    # A shallow recess in one wall accommodates the shifted box.
    y, x = np.mgrid[:60, :60]*.1-2.95
    recess = (abs(x) < .41) & (y > .29) & (y < .55)
    layers.evidence.log_odds[recess] = -4.
    layers._latest_static[recess] = False
    start = (-1.5, 0., .31)

    def see_box(step, center_y):
        box = BoxState('box-1', (0., center_y), (.8, .3, .3), local_dimensions=(.8, .3, .3),
                       orientation=(1., 0., 0., 0.), center_z=.15)
        packet = observation(step, start, points=[(-.39, center_y+d, .20) for d in (-.03, 0., .03)])
        packet['sensors']['lidar']['position'] = np.array(start)
        engine.observe(packet, .2*step, [box])
        layers.refresh_global(reason='post_push_observation')
        assert layers.detected_boxes[box.id].xy == box.xy

    see_box(1, .285)
    assert not engine._try_narrow_passage()
    engine.push_speed_sent('box-1')
    see_box(2, .35)
    assert bool(layers.observed[engine.planner.cell((.25, .15))]) == field_floor
    if field_floor:
        targets = [(-2.8, -2.8, .2)]  # No ray across the vacated strip.
    else:
        assert not engine._try_narrow_passage()
        targets = [(2., y, .25) for y in np.linspace(-.7, .64, 300)]
    packet = observation(3, start, points=targets)
    packet['sensors']['lidar']['position'] = np.array(start)
    engine.observe(packet, .6)
    layers.refresh_global(reason='first_push_reobserve_route')
    engine.trial_used = True
    engine.trial_report = {'reason': 'first_push_observed'}
    original_goal = engine.goal
    engine.update(packet, .7)
    assert engine.narrow is not None
    assert engine.decision['reason'] == 'oriented_narrow_passage'
    assert engine.goal == original_goal
    drive(engine, packet, start_time=.7)
    assert engine.result and engine.result['success'], (engine.phase, engine.decision)
