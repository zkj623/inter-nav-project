import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import SmoothExplorationTracker,VelocitySlewLimiter

def test_t02_lookahead_keeps_pointing_south_after_passing_the_corner():
    p=SidePushPlanner(np.ones((200,200),bool),[[5,15],[-5,5]],resolution=.05)
    start=[10.136238,-.7144765]
    path=[[9.875,-.675],[9.625,-.825],[9.775,-1.075],[9.875,-1.375],[9.875,-1.675],[9.875,-1.975],[9.875,-2.275],[10.025,-2.575]]
    tracker=SmoothExplorationTracker(start,path,VelocitySlewLimiter())
    tracker._planner=p;tracker._route_blocked=np.zeros_like(p.free)
    for position in [start,[9.821,-1.074],[9.906,-1.556]]:
        tracker.action_for(None,{'position':position})
    target=tracker._lookahead_target(np.asarray(position))
    assert target[1]<position[1]-.3, 'A passed waypoint must not cause a northward U-turn'

