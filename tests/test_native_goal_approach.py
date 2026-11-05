import numpy as np
from grutopia_extension.interactive_navigation.go2_interaction_motion import goal_within_tolerance,goal_braking_entry_ready
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import SmoothExplorationTracker,VelocitySlewLimiter


def setup(goal=(.26,0.),marked=True,native=True):
    p=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05)
    p.navigation_half_width=.22;p.full_navigation_footprint=True;p.layered_local_control=True
    l=VelocitySlewLimiter();l.accept([0,0,0],0.)
    if native:l.policy_command=np.zeros(3)
    t=SmoothExplorationTracker([0,0],[goal],l,final_task_goal=goal if marked else None)
    return p,l,t


def test_native_stop_starts_inside_the_unchanged_acceptance_boundary():
    assert goal_within_tolerance(.249) and not goal_within_tolerance(.25)
    assert not goal_braking_entry_ready(.249,native=True)
    assert goal_braking_entry_ready(.199,native=True)
    assert goal_braking_entry_ready(.249,native=False)


def test_final_native_approach_reaches_effective_input_with_original_slew():
    p,l,t=setup()
    for i in range(1,81):
        before=l.command.copy();p.control_time=i*.01
        action,state=t.action(np.array([0.,0.]),0.,p,[])
        assert np.linalg.norm(np.asarray(action['move_by_speed'])[:2]-before[:2])<=.0065+1e-8
        l.policy_command=np.asarray(action['move_by_speed'])
    assert state['native_final_goal_approach']
    assert action['move_by_speed'][0]>=.35-1e-8


def test_unmarked_waypoints_and_legacy_policy_keep_their_speed_law():
    for marked,native in [(False,True),(True,False)]:
        p,l,t=setup(marked=marked,native=native)
        for i in range(1,81):p.control_time=i*.01;action,state=t.action(np.array([0.,0.]),0.,p,[])
        assert not state.get('native_final_goal_approach')
        assert action['move_by_speed'][0]<.3


def test_final_approach_does_not_raise_velocity_after_collision_checks():
    p,l,t=setup()
    p.free[:,p.cell([.60,0])[1]:]=False;before=p.free.copy()
    p.control_time=.01;action,state=t.action(np.array([0.,0.]),0.,p,[])
    assert np.linalg.norm(action['move_by_speed'][:2])<.35
    np.testing.assert_array_equal(p.free,before)
