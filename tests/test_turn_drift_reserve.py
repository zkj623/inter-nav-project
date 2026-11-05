import math
import numpy as np

from grutopia_extension.interactive_navigation.local_staging_control import LocalStagingControl
from grutopia_extension.interactive_navigation.local_turn_maneuver import turn_placement_clear
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_rotation_clear
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter


def setup_turn(native=True):
    planner=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05)
    planner.navigation_half_width=.22
    planner.free[73:,:]=False
    limiter=VelocitySlewLimiter();limiter.accept([0,0,0],0.)
    if native:limiter.policy_command=np.zeros(3)
    control=LocalStagingControl(np.zeros(2),0.,math.pi/2,planner,[],0.)
    return planner,limiter,control


def test_native_large_turn_requires_drift_reserve_before_start():
    planner,limiter,control=setup_turn()
    assert body_rotation_clear(planner,[],[0,0],0.,math.pi/2,margin=control.selector.stage_margin)
    command,state=control.update(np.zeros(2),0.,planner,[],limiter,.01)
    assert state['turn_drift_reserve_m']==.08
    assert state['staging_phase']=='staging_move'
    assert not state['turn_placement_latched']
    assert turn_placement_clear(planner,[],control.target,0.,math.pi/2,control.selector.stage_margin+.08)


def test_reserved_turn_keeps_permission_during_small_clear_drift():
    planner,limiter,control=setup_turn()
    control.update(np.zeros(2),0.,planner,[],limiter,.01)
    point=control.target.copy()
    control.update(point,0.,planner,[],limiter,.02)
    control.update(point,0.,planner,[],limiter,.24)
    assert control.placement_latched
    control.phase='staging_turn'
    moved=point+np.array([0.,.04])
    assert body_rotation_clear(planner,[],moved,0.,math.pi/2,margin=control.selector.stage_margin)
    _,state=control.update(moved,0.,planner,[],limiter,.25)
    assert state['staging_phase']=='staging_turn'
    assert state['turn_placement_latched']


def test_new_wall_still_revokes_a_reserved_turn():
    planner,limiter,control=setup_turn()
    control.update(np.zeros(2),0.,planner,[],limiter,.01)
    point=control.target.copy();control.update(point,0.,planner,[],limiter,.02)
    control.phase='staging_turn';planner.free[planner.cell(point)[0]+4:,:]=False
    command,state=control.update(point,0.,planner,[],limiter,.25)
    assert state['staging_phase']=='staging_brake'
    assert not state['turn_placement_latched']
    np.testing.assert_array_equal(command,[0,0,0])


def test_legacy_profile_does_not_change_placement_requirement():
    planner,limiter,control=setup_turn(native=False)
    _,state=control.update(np.zeros(2),0.,planner,[],limiter,.01)
    assert state['turn_drift_reserve_m']==0.


def test_staging_translation_counters_unrequested_heading_drift():
    planner,limiter,control=setup_turn()
    control.update(np.zeros(2),0.,planner,[],limiter,.01)
    command,state=control.update(np.zeros(2),.08,planner,[],limiter,.02)
    assert state['staging_phase']=='staging_move'
    assert -.25<=command[2]<0.
    assert state['staging_heading_hold_error_rad']<0.


def test_extra_reserve_falls_back_without_reducing_base_execution_margin(monkeypatch):
    planner,limiter,control=setup_turn()
    base=control.selector.stage_margin;seen=[]
    def select(*args,**kwargs):
        reserve=control.selector.reserve_margin;seen.append(reserve)
        return np.array([0.,-.3]) if reserve<=.025 else None
    monkeypatch.setattr(control.selector,'_select_stage',select)
    control.selector.reserve_margin=.08
    result=control._select_target(planner,[],np.zeros(2),0.)
    assert result is not None and seen==[.08,.05,.025]
    assert control.selector.stage_margin==base


def test_missing_reachable_pose_is_reported_separately_from_deadline():
    planner,limiter,control=setup_turn()
    control.update(np.zeros(2),0.,planner,[],limiter,.01)
    planner.free[:]=False
    command,state=control.update(np.zeros(2),0.,planner,[],limiter,.02)
    assert state['staging_failure_reason']=='no_reachable_turn_pose'
    assert control.failed
    np.testing.assert_array_equal(command,[0,0,0])
