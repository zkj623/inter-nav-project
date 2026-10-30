from pathlib import Path
import math
import numpy as np

from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.local_staging_control import LocalStagingControl
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_rotation_clear


def observed_stage():
    with np.load(Path(__file__).parent/'data/t04_staging_cell_boundary_v137.npz') as data:
        planner=SidePushPlanner(data['free'].copy(),data['limits'].tolist(),resolution=.05)
        position=data['position'][:2].copy();q=data['orientation'].copy()
        goal=float(data['goal_heading'])
    planner.navigation_route_radius=.30;planner.navigation_half_width=.22;planner.full_navigation_footprint=True
    heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
    return planner,position,heading,goal


def test_millimeter_cell_entry_does_not_cancel_the_staging_stride():
    planner,position,heading,goal=observed_stage()
    stage=LocalStagingControl(position,heading,goal,planner,[],0.)
    across=position+np.array([.003,0.])
    assert body_rotation_clear(planner,[],across,heading,goal,margin=stage.selector.stage_margin)
    assert not body_rotation_clear(planner,[],across-[.025,0.],heading,goal,margin=stage.selector.stage_margin)
    original=planner.free.copy()
    command,state=stage.update(across,heading,planner,[],VelocitySlewLimiter(),.01)
    assert state['staging_phase']=='staging_move'
    assert np.linalg.norm(command[:2])>=.20
    np.testing.assert_array_equal(planner.free,original)


def test_small_pose_oscillation_does_not_alternate_move_and_brake():
    planner,position,heading,goal=observed_stage()
    stage=LocalStagingControl(position,heading,goal,planner,[],0.)
    limiter=VelocitySlewLimiter()
    for i,x in enumerate([8.2996,8.3004,8.2991,8.3005,8.2997]):
        command,state=stage.update(np.array([x,position[1]]),heading,planner,[],limiter,.01*(i+1))
        assert state['staging_phase']=='staging_move'
        assert np.linalg.norm(command[:2])>=.20


def test_interior_turn_pose_can_still_begin_braking():
    planner,position,heading,goal=observed_stage()
    stage=LocalStagingControl(position,heading,goal,planner,[],0.)
    interior=position+np.array([.075,0.])
    for offset in [(-.025,-.025),(-.025,.025),(.025,-.025),(.025,.025)]:
        assert body_rotation_clear(planner,[],interior+offset,heading,goal,margin=stage.selector.stage_margin)
    command,state=stage.update(interior,heading,planner,[],VelocitySlewLimiter(),.01)
    assert state['staging_phase']=='staging_brake'
    np.testing.assert_array_equal(command,np.zeros(3))
