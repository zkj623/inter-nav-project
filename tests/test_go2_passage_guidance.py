import numpy as np
from grutopia_extension.interactive_navigation.go2_passage_guidance import PassageEntryGuide


class Planner:
    def __init__(self):self.calls=[]
    def path(self,start,goal,boxes):
        self.calls.append(tuple(goal));return [tuple(goal)]


def test_post_push_goal_is_verified_entry_until_robot_reaches_it():
    entry=[2.93266,-.72727];exit=[3.29174,.93438];goal=[3.3,1.4]
    guide=PassageEntryGuide({'corridor_start':entry,'corridor_end':exit},goal)
    p=Planner()
    path=guide.path(p,[2.725,.352],[])
    assert p.calls==[tuple(entry)] and path==[tuple(entry)]
    assert not guide.advance([3.24,.42])  # old failed pose must not skip entry
    np.testing.assert_allclose(guide.target,entry)
    assert guide.advance(entry)
    assert guide.path(p,entry,[])==[tuple(exit),tuple(goal)]
    np.testing.assert_allclose(guide.target,goal)


def test_lost_connection_cannot_be_replaced_by_an_unverified_goal_shortcut():
    guide=PassageEntryGuide({'corridor_start':[0,-1],'corridor_end':[0,1]},[0,2])
    class Blocked:
        def path(self,start,goal,boxes):return []
    assert guide.path(Blocked(),[1,0],[])==[]


def test_explicit_guide_keys_and_already_passed_exit_do_not_backtrack():
    guide=PassageEntryGuide({'guide_entry':[0,-1],'guide_exit':[0,1]},[0,2])
    guide.advance([0,-1]);planner=Planner()
    assert guide.path(planner,[0,1.1],[])==[(0.,2.)]


def test_entry_inside_radius_waits_for_measured_stop_before_switching_route():
    guide=PassageEntryGuide({'corridor_start':[0.,0.],'corridor_end':[0.,1.]},[0.,2.])
    detail=guide.observe_entry([-.1,0.,.4],[.4,0.,0.],[0.,0.,0.],0.,0.,heading=0.,command=[.4,0.,0.])
    assert detail['subgoal_braking'] and not detail['entry_advanced'] and not guide.entry_reached
    guide.observe_entry([-.1,0.,.4],[0.,0.,0.],[0.,0.,0.],0.,1.,heading=0.,command=[0.,0.,0.])
    detail=guide.observe_entry([-.1,0.,.4],[0.,0.,0.],[0.,0.,0.],0.,1.5,heading=0.,command=[0.,0.,0.])
    assert detail['entry_advanced'] and guide.entry_reached


def test_predictive_entry_brake_does_not_count_stopping_short_as_arrival():
    guide=PassageEntryGuide({'corridor_start':[0.,0.],'corridor_end':[0.,1.]},[0.,2.])
    detail=guide.observe_entry([-.3,0.,.4],[.4,0.,0.],[0.,0.,0.],0.,0.,heading=0.,command=[.5,0.,0.])
    assert detail['subgoal_braking'] and not guide.entry_reached
    guide.observe_entry([-.3,0.,.4],[0.,0.,0.],[0.,0.,0.],0.,1.,heading=0.,command=[0.,0.,0.])
    detail=guide.observe_entry([-.3,0.,.4],[0.,0.,0.],[0.,0.,0.],0.,1.5,heading=0.,command=[0.,0.,0.])
    assert detail['replan'] and not detail['entry_advanced'] and not guide.entry_reached
