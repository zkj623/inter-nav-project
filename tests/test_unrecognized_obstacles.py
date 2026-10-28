import numpy as np
from grutopia_extension.interactive_navigation.box_push_navigation import BoxPushPlanner,BoxState


def test_unrecognized_box_shaped_obstacle_is_not_removed_from_optimistic_map():
    grid=np.ones((40,160),bool)
    planner=BoxPushPlanner(grid,[[-4,4],[-1,1]],resolution=.05,robot_radius=.25)
    box=BoxState('B06',(0.,0.),(.8,1.8,.8),0.,6.)
    assert planner.path((-3,0),(3,0),[])
    assert not planner.path((-3,0),(3,0),[box])
    planner.interactable_ids=frozenset()
    result=planner.decide((-3,0),(3,0),[box])
    assert result.kind!='push'
    assert not result.explanation['without_boxes_path']
    assert not planner.path((-3,0),(3,0),[box])
