"""The recorded map must admit the effective request after real slew checking."""
from pathlib import Path
from dataclasses import fields
import json
import numpy as np
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.local_staging_control import LocalStagingControl
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter,SmoothExplorationTracker


def test_actual_recorded_staging_geometry_admits_checked_effective_reverse():
    with np.load(Path(__file__).parent/'fixtures/t04_native_staging_v204.npz') as f:a={k:f[k] for k in f.files}
    planner=SidePushPlanner(a['free'].copy(),a['limits'].tolist(),resolution=.05)
    planner.box_free=a['box_free'];planner.planning_scope='local';planner.planning_observed=a['observed']
    planner.navigation_route_radius=.30;planner.navigation_half_width=.22
    planner.full_navigation_footprint=True;planner.layered_local_control=True
    names={f.name for f in fields(BoxState)}
    boxes=[BoxState(**{k:v for k,v in b.items() if k in names}) for b in json.loads(str(a['boxes_json']))]
    position=a['position'];heading=float(a['heading'])
    limiter=VelocitySlewLimiter();limiter.accept(a['command'],0.)
    limiter.policy_command=a['command'].copy();limiter.observe(a['measured'],0.)
    limiter.staging=LocalStagingControl(position,heading,float(a['goal_heading']),planner,boxes,0.)
    tracker=SmoothExplorationTracker(position[:2],a['path'],limiter);tracker._waypoint_index=int(a['waypoint_index'])
    before=planner.free.copy()
    for tick in range(1,31):
        previous=limiter.command.copy();planner.control_time=tick*.008
        action,details=tracker.action(position[:2],heading,planner,boxes)
        command=np.array(action['move_by_speed'])
        assert np.linalg.norm(command[:2]-previous[:2])<=.65*.008+1e-8
        assert details['body_motion_safe'] and not details['needs_replan']
    assert command[0]<-.44
    assert np.linalg.norm(command[:2])<=.50
    np.testing.assert_array_equal(planner.free,before)
