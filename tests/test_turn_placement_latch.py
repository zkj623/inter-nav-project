import numpy as np

from test_turn_placement_slack import observed_stage
from grutopia_extension.interactive_navigation.local_staging_control import LocalStagingControl
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_rotation_clear


def test_braking_drift_within_valid_turn_space_does_not_restart_translation():
    planner,position,heading,goal=observed_stage()
    stage=LocalStagingControl(position,heading,goal,planner,[],0.)
    limiter=VelocitySlewLimiter()
    interior=position+np.array([.075,0.]);edge=position+np.array([.003,0.])
    assert body_rotation_clear(planner,[],edge,heading,goal,margin=stage.selector.stage_margin)
    _,state=stage.update(interior,heading,planner,[],limiter,.01)
    assert state['staging_phase']=='staging_brake'
    stage.update(edge,heading,planner,[],limiter,.10)
    _,state=stage.update(edge,heading,planner,[],limiter,.24)
    assert state['staging_phase']=='staging_turn'


def test_forced_initial_braking_cannot_skip_interior_admission():
    planner,position,heading,goal=observed_stage()
    stage=LocalStagingControl(position,heading,goal,planner,[],0.)
    limiter=VelocitySlewLimiter();stage.phase='staging_brake'
    edge=position+np.array([.003,0.])
    stage.update(edge,heading,planner,[],limiter,.01)
    _,state=stage.update(edge,heading,planner,[],limiter,.24)
    assert state['staging_phase']=='staging_move'


def test_losing_actual_turn_clearance_revokes_the_previous_admission():
    planner,position,heading,goal=observed_stage()
    stage=LocalStagingControl(position,heading,goal,planner,[],0.)
    limiter=VelocitySlewLimiter()
    interior=position+np.array([.075,0.]);edge=position+np.array([.003,0.])
    assert not body_rotation_clear(planner,[],position,heading,goal,margin=stage.selector.stage_margin)
    stage.update(interior,heading,planner,[],limiter,.01)
    stage.update(position,heading,planner,[],limiter,.10)
    _,state=stage.update(edge,heading,planner,[],limiter,.24)
    assert state['staging_phase']=='staging_move'
