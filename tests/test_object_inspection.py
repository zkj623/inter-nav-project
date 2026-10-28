from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.interaction_catalog import RecognitionStore
from grutopia_extension.interactive_navigation.object_inspection import ObjectInspectionPlanner


def test_unknown_goal_can_trigger_observation_without_authorizing_interaction():
    row,col=np.indices((160,160));x=-4+(col+.5)*.05
    free=x<1.
    planner=SidePushPlanner(free,[[-4,4],[-4,4]],resolution=.05)
    planner.interactable_ids=frozenset();planner.planning_scope='global';planner.planning_observed=free.copy()
    box=BoxState('B1',(0.,0.),(.85,.55,.7))
    @contextmanager
    def global_planning():yield [box]
    layers=SimpleNamespace(planner=planner,global_planning=global_planning)
    view=ObjectInspectionPlanner().propose(layers,(-2.5,0.),0.,(2.5,0.),RecognitionStore(),0.)
    assert view is not None and view.object_id=='B1'
    assert planner.interactable_ids==frozenset()
    assert planner.decide((-2.5,0.),(2.5,0.),[box]).kind=='failed'
    np.testing.assert_array_equal(planner.free,free)


def test_negative_passive_classification_needs_two_active_views_but_never_grants_permission():
    from types import SimpleNamespace
    planner=ObjectInspectionPlanner();store=SimpleNamespace(confirmed=lambda key,now:SimpleNamespace(category='cabinet'))
    assert planner.needs_inspection('B1',store,set(),1.)
    planner.review('B1','cabinet',2.)
    assert planner.needs_inspection('B1',store,set(),3.)
    planner.review('B1','cabinet',4.)
    assert not planner.needs_inspection('B1',store,set(),5.)
    assert planner.needs_inspection('B1',store,set(),70.)
    assert not planner.needs_inspection('B1',store,{'B1'},70.)
