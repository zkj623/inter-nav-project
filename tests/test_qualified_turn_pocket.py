from pathlib import Path
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.local_turn_maneuver import LocalTurnManeuver,turn_placement_clear
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_translation_recovery_clear
from grutopia_extension.interactive_navigation.go2_turn_response import stepping_arc_clear

def test_actual_t02_small_turning_pocket_is_found_without_relaxing_geometry():
    with np.load(Path(__file__).parent/'fixtures/go2_qualified_t02_turn_pocket.npz') as f:a={k:f[k] for k in f.files}
    p=SidePushPlanner(a['free'],a['limits'],resolution=.05)
    p.box_free=a['box_free'];p.planning_observed=a['observed'];p.planning_scope='local'
    p.navigation_route_radius=.30;p.navigation_half_width=.22;p.full_navigation_footprint=True;p.layered_local_control=True
    position=a['position'];heading=float(a['heading']);goal=float(a['goal_heading'])
    selector=LocalTurnManeuver(goal)
    target=selector._select_stage(p,[],position,heading,require_stepping_arc=True,require_stable_placement=True)
    assert target is not None, 'Recorded T02 has a reachable 10cm turning pocket'
    assert np.linalg.norm(target-position)<=.8 and selector.stage_margin>=.025
    assert turn_placement_clear(p,[],target,heading,goal,selector.stage_margin)
    assert body_translation_recovery_clear(p,[],position,heading,target)
    assert stepping_arc_clear(p,[],target,heading,1.)
    assert np.array_equal(p.free,a['free']) and np.array_equal(p.planning_observed,a['observed'])
