"""Recovery regression from an observed T05 map; never loaded by the runner."""
from pathlib import Path
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    ForwardExplorationPathTracker,calibrated_buffer_translation,
    body_translation_recovery_clear,navigation_body_blocked)


def test_observed_shallow_overlap_uses_short_executable_translation():
    with np.load(Path(__file__).parent/'fixtures/t05_observed_buffer_v48.npz') as fixture:
        free=fixture['free'].copy();observed=fixture['observed'].copy()
        p=SidePushPlanner(free,fixture['limits'],resolution=.05)
        position=fixture['position'].copy();heading=float(fixture['heading']);goal=fixture['goal'].copy()
    p.navigation_half_width=.22;p.layered_local_control=True;p.full_navigation_footprint=True
    p.planning_observed=observed;p.navigation_route_radius=.30;p.control_time=1.
    assert navigation_body_blocked(p,[],heading)[p.cell(position)]
    before=p.free.copy()
    tracker=ForwardExplorationPathTracker(position[:2],[(*goal,0.)],guard_motion=False)
    action,state=tracker.action(position[:2],heading,p,[])
    assert state['controller']=='calibrated_navigation_buffer_exit'
    assert state['calibrated_buffer_motion'] and not state['needs_replan']
    target=np.asarray(state['target']);cmd=np.asarray(action['move_by_speed'])
    assert np.linalg.norm(target-position[:2])<=.20
    assert cmd[0]<=-.30 and abs(cmd[1])<.01 and cmd[2]==0.
    assert body_translation_recovery_clear(p,[],position,heading,target)
    speed=np.linalg.norm(cmd[:2]);distance=.60*speed+speed**2/(2*.65)
    unit=(target-position[:2])/np.linalg.norm(target-position[:2])
    assert body_translation_recovery_clear(p,[],position,heading,position[:2]+unit*distance,require_clear_end=False)
    np.testing.assert_array_equal(p.free,before)
    assert not (p.free & ~observed).any()


def test_close_free_target_is_rejected_if_effective_stride_cannot_brake():
    p=SidePushPlanner(np.ones((100,100),bool),[[-2.5,2.5],[-2.5,2.5]],resolution=.05)
    p.navigation_half_width=.22
    p.free[:,p.cell([.65,0.])[1]:]=False
    position=np.array([0.,0.,.4]);target=np.array([.05,0.])
    assert body_translation_recovery_clear(p,[],position,0.,target)
    assert calibrated_buffer_translation(p,[],position,0.,target) is None
