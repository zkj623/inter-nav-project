import numpy as np
from grutopia_extension.interactive_navigation.navigation_refresh import NavigationProgressWatch
from grutopia_extension.interactive_navigation.online_navigation_decision import OnlineGoalSelector

def test_repeated_staging_without_forward_progress_requests_fresh_goal():
    watch=NavigationProgressWatch();watch.set_route([0,0],[[0,1],[0,2]],[0,2],0.)
    watch.observe_stage(object());assert watch.check([0,0],7.,lambda a,b:True,True) is None
    watch.observe_stage(object());result=watch.check([0,.01],8.,lambda a,b:True,True)
    assert result['reason']=='no_forward_route_progress' and result['staging_episodes_since_progress']==2

def test_one_bounded_preparation_is_not_aborted_as_repeated_spinning():
    watch=NavigationProgressWatch();watch.set_route([0,0],[[0,1],[0,2]],[0,2],0.)
    stage=object()
    for t in [1.,5.,9.,12.]:
        watch.observe_stage(stage);assert watch.check([0,0],t,lambda a,b:True,True) is None

def test_actual_forward_progress_resets_stall_but_retreat_does_not():
    watch=NavigationProgressWatch();watch.set_route([0,0],[[0,.5],[0,1],[0,2]],[0,2],0.)
    assert watch.check([0,.3],5.,lambda a,b:True) is None
    assert watch.check([0,.0],10.,lambda a,b:True) is None
    assert watch.check([0,.0],11.1,lambda a,b:True)['without_progress_s']>6.

def test_same_goal_replanning_does_not_renew_progress_deadline():
    watch=NavigationProgressWatch();watch.set_route([0,0],[[0,1],[0,2]],[0,2],0.)
    watch.set_route([0,0],[[.1,1],[0,2]],[0,2],5.)
    assert watch.check([0,0],6.1,lambda a,b:True) is not None

def test_frontier_cooldown_expires_and_does_not_mark_goal_visited():
    selector=OnlineGoalSelector();selector.defer_frontier([1.,2.],10.)
    point,until=selector.deferred[selector.key([1.,2.])]
    assert np.array_equal(point,[1.,2.]) and until==16. and not selector.visited

def reference_case(monkeypatch,candidate,endpoint):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import grutopia_extension.interactive_navigation.online_navigation_decision as module
    monkeypatch.setattr(module,'local_unknown',lambda *args,**kw:0.)
    planner=SimpleNamespace(resolution=.05,path=lambda start,end,boxes:[[*end,0.]],
        length=lambda start,path:float(np.linalg.norm(np.asarray(start)-path[-1][:2])))
    @contextmanager
    def view():yield []
    layers=SimpleNamespace(planner=planner,global_planning=view,frontier_grid=lambda:None,
        decision={'kind':'explore','decision_time_s':1.,'before_path':[[99,99,0]]})
    selector=OnlineGoalSelector();selector.decide=lambda *args:(candidate,tuple(endpoint))
    return selector,layers

def test_reference_refresh_uses_current_pose_without_resetting_goal_age(monkeypatch):
    from grutopia_extension.interactive_navigation.box_push_navigation import BoxDecision
    candidate=BoxDecision('explore','test',explanation={'before_length_m':9.,'information_gain_m2':0.})
    selector,layers=reference_case(monkeypatch,candidate,[0,10])
    assert selector.reconsider(layers,[2,0],[10,10],[0,10],42.) is None
    assert layers.decision['start']==[2.,0.] and layers.decision['before_path']==[[0,10,0.]]
    assert layers.decision['reference_updated_time_s']==42. and layers.decision['decision_time_s']==1.

def test_materially_better_frontier_is_proposed(monkeypatch):
    from grutopia_extension.interactive_navigation.box_push_navigation import BoxDecision
    candidate=BoxDecision('explore','test',explanation={'before_length_m':11.,'information_gain_m2':0.})
    selector,layers=reference_case(monkeypatch,candidate,[10,5])
    result=selector.reconsider(layers,[0,0],[10,10],[0,10],42.)
    assert result['reason']=='better_observed_frontier' and result['candidate_endpoint']==[10,5]

def test_newly_reachable_task_goal_has_priority(monkeypatch):
    from grutopia_extension.interactive_navigation.box_push_navigation import BoxDecision
    selector,layers=reference_case(monkeypatch,BoxDecision('navigate','ordinary_path_available'),[10,10])
    assert selector.reconsider(layers,[0,0],[10,10],[0,10],42.)['reason']=='task_route_became_available'

def test_final_goal_reference_refresh_does_not_reselect_target(monkeypatch):
    from grutopia_extension.interactive_navigation.box_push_navigation import BoxDecision
    selector,layers=reference_case(monkeypatch,BoxDecision('navigate','test'),[10,10])
    layers.decision={'kind':'navigate','decision_time_s':10.};original=layers.decision
    def unwanted(*args):raise AssertionError('Reference refresh must not make a control decision')
    selector.decide=unwanted
    assert selector.refresh_reference(layers,[10,9],[10,10],42.)==1.
    assert layers.decision['reference_updated_time_s']==42. and original=={'kind':'navigate','decision_time_s':10.}

def test_rolling_refresh_only_for_forward_motion_and_nonstaging_opportunity():
    from grutopia_extension.interactive_navigation.navigation_refresh import rolling_refresh_allowed
    assert rolling_refresh_allowed('better_observed_frontier',False,[.4,0,.05])
    assert not rolling_refresh_allowed('better_observed_frontier',True,[.4,0,.05])
    assert not rolling_refresh_allowed('no_forward_route_progress',False,[.4,0,.05])
    assert not rolling_refresh_allowed('task_route_became_available',False,[0.,0,.8])
