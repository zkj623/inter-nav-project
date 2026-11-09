from pathlib import Path
import math
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.local_staging_control import LocalStagingControl
from grutopia_extension.interactive_navigation.local_turn_maneuver import turn_placement_clear
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_translation_recovery_clear
from grutopia_extension.interactive_navigation.go2_turn_response import stepping_arc_clear
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter

def recorded_case():
    with np.load(Path(__file__).parent/'fixtures/go2_qualified_t02_prepare_turn.npz') as f:a={k:f[k] for k in f.files}
    p=SidePushPlanner(a['free'],a['limits'],resolution=.05);p.box_free=a['box_free'];p.planning_observed=a['observed'];p.planning_scope='local'
    p.navigation_route_radius=.30;p.navigation_half_width=.22;p.full_navigation_footprint=True;p.layered_local_control=True
    position=a['position'];heading=float(a['heading']);goal=float(a['goal_heading'])
    return a,p,position,heading,goal

def test_actual_t02_can_prepare_heading_before_translation_without_map_changes():
    a,p,pos,heading,goal=recorded_case();stage=LocalStagingControl(pos,heading,goal,p,[],10.)
    assert not stage.failed, 'A safe preparation turn unlocks the recorded narrow-space maneuver'
    plan=stage.escape_plan;assert plan is not None
    assert np.allclose(stage.target,pos) and np.linalg.norm(plan['target']-pos)<=.8 and plan['margin']>=.025
    assert turn_placement_clear(p,[],pos,heading,stage.goal_heading,plan['margin'])
    assert stepping_arc_clear(p,[],pos,heading,math.copysign(1.,stage.goal_heading-heading))
    for offset in [-.12,0.,.12]:
        actual=stage.goal_heading+offset;end=actual+(goal-actual+math.pi)%(2*math.pi)-math.pi
        assert body_translation_recovery_clear(p,[],pos,actual,plan['target'])
        assert turn_placement_clear(p,[],plan['target'],actual,end,plan['margin'])
    assert np.array_equal(p.free,a['free']) and np.array_equal(p.planning_observed,a['observed'])

def test_preparation_handoff_waits_for_stop_and_keeps_original_deadline():
    a,p,pos,heading,goal=recorded_case();stage=LocalStagingControl(pos,heading,goal,p,[],10.)
    assert not stage.failed
    prepared=stage.goal_heading;stage.phase='staging_turn_brake'
    limiter=VelocitySlewLimiter();limiter.policy_command=np.zeros(3);limiter.measured=np.array([.10,0.,0.])
    stage.update(pos,prepared,p,[],limiter,11.);assert stage.escape_plan is not None
    limiter.measured[:]=0.;stage.update(pos,prepared,p,[],limiter,11.1)
    command,state=stage.update(pos,prepared,p,[],limiter,11.31)
    assert state['preparatory_turn_completed'] and stage.escape_plan is None and stage.phase=='staging_move'
    assert abs((stage.goal_heading-goal+math.pi)%(2*math.pi)-math.pi)<1e-9
    assert stage.started==10. and np.array_equal(stage.origin,pos)
    assert body_translation_recovery_clear(p,[],pos,prepared,stage.target)
    command,state=stage.update(pos,prepared,p,[],limiter,30.01)
    assert stage.failed and state['staging_failure_reason']=='stage_deadline' and not command.any()
