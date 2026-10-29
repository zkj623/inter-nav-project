from pathlib import Path
import json
import numpy as np
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    ForwardExplorationPathTracker,decision_recovery_required,navigation_body_blocked,
    body_translation_recovery_clear)
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter,smooth_recovery_action


def fixture():
    with np.load(Path(__file__).parent/'fixtures/t04_translation_handoff_v98.npz') as z:
        planner=SidePushPlanner(z['free'].copy(),z['limits'].tolist(),resolution=.05)
        planner.planning_observed=z['observed'].copy()
        boxes=[BoxState(**b) for b in json.loads(str(z['boxes_json']))]
        p=z['position'].copy();heading=float(z['heading']);target=z['target'].copy()
    planner.navigation_half_width=.22;planner.full_navigation_footprint=True
    planner.layered_local_control=True;planner.planning_scope='local';planner.control_time=0.
    tracker=ForwardExplorationPathTracker(p[:2],[(10.3,1.75)],guard_motion=False)
    tracker._buffer_active=True;tracker._buffer_target=target.copy()
    return planner,boxes,p,heading,target,tracker


def test_free_intermediate_cell_does_not_cancel_the_recorded_recovery_target():
    planner,boxes,p,heading,target,tracker=fixture()
    assert not navigation_body_blocked(planner,boxes,heading)[planner.cell(p)]
    assert planner.cell(p)!=planner.cell(target)
    assert body_translation_recovery_clear(planner,boxes,p,heading,target)
    assert decision_recovery_required(planner,boxes,p,heading,tracker)
    limiter=VelocitySlewLimiter();limiter.accept([0,0,0],0.)
    previous=np.zeros(3);before=planner.free.copy()
    for tick in range(1,81):
        planner.control_time=tick*.01
        action,state=tracker.action(p[:2],heading,planner,boxes)
        assert state['recovering_buffer'] and not state['needs_replan']
        np.testing.assert_allclose(state['target'],target)
        command=smooth_recovery_action(action,p[:2],heading,planner,boxes,limiter,planner.control_time)
        sent=np.asarray(command['move_by_speed'])
        assert np.linalg.norm(sent[:2]-previous[:2])<=.0065+1e-8
        limiter.accept(sent,planner.control_time);previous=sent
    assert np.linalg.norm(sent[:2])>=.34
    np.testing.assert_array_equal(planner.free,before)


def test_translation_releases_at_selected_free_target_cell():
    planner,boxes,p,heading,target,tracker=fixture()
    action,state=tracker.action(target,heading,planner,boxes)
    assert not state['recovering_buffer'] and state['buffer_cleared_replan']
    assert not tracker._buffer_active
    np.testing.assert_array_equal(action['move_by_speed'],[0,0,0])
    assert not decision_recovery_required(planner,boxes,target,heading,tracker)


def test_new_occupied_target_cannot_complete_translation_recovery():
    planner,boxes,p,heading,target,tracker=fixture()
    planner.free[:]=False
    action,state=tracker.action(p[:2],heading,planner,boxes)
    assert state['needs_replan']
    assert not state.get('buffer_cleared_replan',False)
    np.testing.assert_array_equal(action['move_by_speed'],[0,0,0])
