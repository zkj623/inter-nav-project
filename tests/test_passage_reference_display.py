import copy
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np

from grutopia_extension.interactive_navigation.online_navigation_decision import OnlineGoalSelector
from grutopia_extension.interactive_navigation.navigation_refresh import passage_reference_target
from grutopia_extension.interactive_navigation.task_video_recording import reference_overlay_state


def reference_layers():
    calls=[]
    def path(start,end,boxes):calls.append((list(start),list(end)));return [[*end,0.]]
    planner=SimpleNamespace(path=path,length=lambda start,path:float(np.linalg.norm(np.asarray(start)-path[-1][:2])))
    @contextmanager
    def global_planning():yield []
    before={'kind':'push','start':[0.,0.],'before_path':[],'before_length_m':None,
            'after_path':[[1.,1.,0.]],'after_length_m':2.,'decision_time_s':10.,'reference_updated_time_s':10.,
            'reference_role':'interaction_history','box_id':'B07'}
    return SimpleNamespace(planner=planner,global_planning=global_planning,decision=before),calls


def test_passage_reference_refresh_preserves_original_interaction_comparison():
    layers,calls=reference_layers();before=copy.deepcopy(layers.decision);selector=OnlineGoalSelector()
    selector.refresh_passage_reference(layers,[.5,0.],[1.,0.],50.)
    assert layers.decision['reference_role']=='current_navigation' and layers.decision['reference_updated_time_s']==50.
    assert layers.decision['reference_start']==[.5,0.] and layers.decision['reference_path']==[[1.,0.,0.]]
    assert layers.decision['start']==before['start'] and layers.decision['before_length_m'] is None
    comparison=copy.deepcopy(layers.decision['interaction_comparison'])
    selector.refresh_passage_reference(layers,[.7,0.],[1.,1.],51.)
    assert layers.decision['interaction_comparison']==comparison
    assert layers.decision['after_path']==before['after_path'] and layers.decision['decision_time_s']==10.
    assert calls==[([.5,0.],[1.,0.]),([.7,0.],[1.,1.])]


def test_reference_target_covers_pose_turns_and_ordinary_guide_fallbacks():
    guide=SimpleNamespace(target=np.array([2.,3.]))
    connection=SimpleNamespace(index=0,plan={'operations':[{'kind':'turn','heading':1.},
        {'kind':'move','target':[1.,2.],'heading':1.}]})
    assert passage_reference_target(connection,guide,[9.,9.])==[1.,2.]
    np.testing.assert_equal(passage_reference_target(None,guide,[9.,9.]),[2.,3.])
    connection.index=2;assert passage_reference_target(connection,guide,[9.,9.])==[9.,9.]
    assert passage_reference_target(None,None,[9.,9.]) is None


def test_historical_push_overlay_does_not_claim_to_be_an_aged_current_reference():
    layers,_=reference_layers();row=reference_overlay_state(layers.decision,[0.,0.],70.)
    assert row['historical'] and row['age_s'] is None and row['evaluated'] and row['length_m'] is None


def test_active_passage_overlay_uses_current_pose_path_age_and_separate_history():
    layers,_=reference_layers();OnlineGoalSelector().refresh_passage_reference(layers,[.5,0.],[1.,0.],50.)
    row=reference_overlay_state(layers.decision,[0.,0.],50.5)
    assert not row['historical'] and row['age_s']==.5 and row['length_m']==.5
    assert row['origin']==[.5,0.] and row['path']==[[1.,0.,0.]]
    assert row['comparison']['before_length_m'] is None


def test_newly_blocked_reference_clears_old_line_without_erasing_comparison():
    layers,_=reference_layers();selector=OnlineGoalSelector()
    selector.refresh_passage_reference(layers,[.5,0.],[1.,0.],50.)
    layers.planner.path=lambda *args:[]
    selector.refresh_passage_reference(layers,[.6,0.],[1.,0.],51.)
    row=reference_overlay_state(layers.decision,[0.,0.],51.5)
    assert not row['path'] and row['length_m'] is None and row['age_s']==.5
    assert row['comparison']['after_length_m']==2. and not layers.decision['reference_stale']
