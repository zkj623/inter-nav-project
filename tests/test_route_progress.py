import numpy as np
from grutopia_extension.interactive_navigation.route_progress import RouteProgress
from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner

def test_recorded_t02_passed_corner_does_not_pull_robot_back_north():
    start=[10.136238,-.7144765]
    path=[[9.875,-.675],[9.625,-.825],[9.775,-1.075],[9.875,-1.375],[9.875,-1.675],[9.875,-1.975],[9.875,-2.275],[10.025,-2.575]]
    route=RouteProgress(start,path);index=0
    for position in [start,[9.821,-1.074],[9.906,-1.556]]:
        index=route.advance(position,index,lambda a,b:True)
    target=route.target(position,index,.55)
    assert target[1]<position[1]-.3 and index>=4

def test_projection_cannot_advance_through_a_wall_to_parallel_branch():
    # Two close parallel legs separated by a wall; a nearest-segment jump is unsafe.
    blocked=np.zeros((80,80),bool);blocked[10:51,39:41]=True
    cell=lambda p:(int(p[1]/.05),int(p[0]/.05))
    clear=lambda a,b:AStarMapPlanner.segment_clear(cell(a),cell(b),blocked)
    route=RouteProgress([1.85,1.0],[[1.85,2.6],[2.15,2.6],[2.15,1.0]])
    index=route.advance([1.85,1.5],0,clear,lookahead=3.)
    assert index==0 and route.progress<1.

def test_progress_does_not_rewind_when_a_staging_retreat_moves_backward():
    route=RouteProgress([0,0],[[0,.3],[0,.6],[0,.9],[0,1.2],[0,1.5]])
    index=route.advance([0,.75],0,lambda a,b:True)
    progress=route.progress
    index=route.advance([0,.4],index,lambda a,b:True)
    assert route.progress==progress and route.target([0,.4],index,.55)[1]>=1.25
