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
