import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation import go2_voronoi_navigation as nav


def setup():
    planner=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05)
    planner.layered_local_control=True;planner.navigation_half_width=.22
    planner.free[planner.cell((.4,0))]=False
    planner.control_time=0.
    return planner,nav.ForwardExplorationPathTracker([0,0],[[0,2]],guard_motion=False)


def test_recovery_turn_is_not_cancelled_when_a_backing_candidate_appears(monkeypatch):
    p,tracker=setup()
    monkeypatch.setattr(nav,'body_translation_recovery_clear',lambda *a,**kw:False)
    monkeypatch.setattr(nav,'turn_reduces_overlap',lambda *a,**kw:True)
    first,state=tracker.action(np.zeros(2),0,p,[])
    assert state['controller']=='body_safe_recovery_turn'
    direction=np.sign(first['move_by_speed'][2]);target=state['target_heading_rad']
    monkeypatch.setattr(nav,'body_translation_recovery_clear',lambda *a,**kw:True)
    for i in range(60):
        p.control_time=(i+1)*.008
        action,state=tracker.action(np.zeros(2),direction*.01,p,[])
        assert action['move_by_speed'][:2]==[0.,0.]
        assert np.sign(action['move_by_speed'][2])==direction
        assert state['target_heading_rad']==target
    assert abs(action['move_by_speed'][2])==.8


def test_new_obstacle_stops_latched_turn_and_requests_observation_before_reselection(monkeypatch):
    p,tracker=setup();tracker._buffer_turn_target=.4
    monkeypatch.setattr(nav,'turn_reduces_overlap',lambda *a,**kw:False)
    action,state=tracker.action(np.zeros(2),0,p,[])
    assert action['move_by_speed']==[0.,0.,0.]
    assert not state['needs_replan']
    p.control_time=1.1
    action,state=tracker.action(np.zeros(2),0,p,[])
    assert action['move_by_speed']==[0.,0.,0.]
    assert state['needs_replan'] and tracker._buffer_turn_target is None
