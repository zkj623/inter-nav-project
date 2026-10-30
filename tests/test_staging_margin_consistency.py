from pathlib import Path
import math
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.local_staging_control import LocalStagingControl
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_rotation_clear


def observed_boundary(goal):
    with np.load(Path(__file__).parent/'data/t05_staging_boundary_v119.npz') as data:
        planner=SidePushPlanner(data['free'].copy(),data['limits'].tolist(),resolution=.05)
        position=data['position'][:2].copy();q=data['orientation'].copy()
    planner.navigation_route_radius=.30;planner.navigation_half_width=.22;planner.full_navigation_footprint=True
    heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
    return planner,position,heading,LocalStagingControl(position,heading,goal,planner,[],0.)


@pytest.mark.parametrize('goal',[.7064224145543709,-1.0864211847023988])
def test_measured_five_mm_cell_crossing_does_not_cancel_clearance_translation(goal):
    planner,position,heading,stage=observed_boundary(goal)
    assert stage.selector.stage_margin==.05
    end=heading+(goal-heading+math.pi)%(2*math.pi)-math.pi
    across=position+np.array([.005,0.])
    assert body_rotation_clear(planner,[],across,heading,end,margin=.025)
    assert not body_rotation_clear(planner,[],across,heading,end,margin=.05)
    before=planner.free.copy();limiter=VelocitySlewLimiter()
    command,state=stage.update(across,heading,planner,[],limiter,.01)
    assert state['staging_phase']=='staging_move'
    assert np.linalg.norm(command[:2])>=.20
    np.testing.assert_array_equal(planner.free,before)


@pytest.mark.parametrize('goal',[.7064224145543709,-1.0864211847023988])
def test_braking_begins_after_selected_clearance_is_observed(goal):
    planner,position,heading,stage=observed_boundary(goal)
    inside=stage.target.copy();limiter=VelocitySlewLimiter()
    command,state=stage.update(inside,heading,planner,[],limiter,.01)
    assert state['staging_phase']=='staging_brake'
    np.testing.assert_array_equal(command,np.zeros(3))


def test_explicit_narrow_stage_keeps_its_selected_margin():
    planner,position,heading,stage=observed_boundary(.7064224145543709)
    # A narrow fallback is allowed by the existing stage search. Applying
    # a universal larger margin here would contradict that selection.
    stage.selector.stage_margin=.025
    # This pose has half-cell placement slack with the selected .025m
    # margin, while its lower-x neighbor still fails the .05m sweep.
    inside=position+np.array([.05,0.])
    goal=.7064224145543709
    end=heading+(goal-heading+math.pi)%(2*math.pi)-math.pi
    for off in [(-.025,-.025),(-.025,.025),(.025,-.025),(.025,.025)]:
        assert body_rotation_clear(planner,[],inside+off,heading,end,margin=.025)
    assert not body_rotation_clear(planner,[],inside-[.025,0.],heading,end,margin=.05)
    command,state=stage.update(inside,heading,planner,[],VelocitySlewLimiter(),.01)
    assert state['staging_phase']=='staging_brake'
    np.testing.assert_array_equal(command,np.zeros(3))


@pytest.mark.parametrize('goal',[.7064224145543709,-1.0864211847023988])
def test_pose_clearance_alone_is_not_a_stable_stopping_region(goal):
    planner,position,heading,stage=observed_boundary(goal)
    near=position+np.array([.075,0.])
    end=heading+(goal-heading+math.pi)%(2*math.pi)-math.pi
    assert body_rotation_clear(planner,[],near,heading,end,margin=.05)
    assert not body_rotation_clear(planner,[],near-[.025,0.],heading,end,margin=.05)
    command,state=stage.update(near,heading,planner,[],VelocitySlewLimiter(),.01)
    assert state['staging_phase']=='staging_move'
    assert np.linalg.norm(command[:2])>=.20
