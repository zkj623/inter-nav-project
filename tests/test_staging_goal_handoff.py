from grutopia_extension.interactive_navigation.navigation_refresh import reconsider_frontier
from grutopia_extension.interactive_navigation.navigation_refresh import NavigationProgressWatch
from types import SimpleNamespace


class Selector:
    def __init__(self):self.events=[]
    def defer_frontier(self,endpoint,now):self.events.append(('defer',endpoint,now))
    def refresh_reference(self,layers,position,endpoint,now):self.events.append(('reference',endpoint,now))
    def reconsider(self,layers,position,goal,endpoint,now):
        self.events.append(('select',endpoint,now));return {'reason':'better_observed_frontier'}


def test_active_stage_keeps_goal_but_refreshes_display_then_reconsiders_afterward():
    selector=Selector();endpoint=(9.,1.4)
    assert reconsider_frontier(None,selector,None,(8.7,0.),(10.3,1.4),endpoint,26.,True) is None
    assert selector.events==[('reference',endpoint,26.)]
    request=reconsider_frontier(None,selector,None,(8.7,-.1),(10.3,1.4),endpoint,27.,False)
    assert request['reason']=='better_observed_frontier' and selector.events[-1][0]=='select'


def test_no_progress_watchdog_still_escapes_repeated_staging():
    selector=Selector();pending={'reason':'no_forward_route_progress','without_progress_s':8.}
    request=reconsider_frontier(pending,selector,None,(8.7,0.),(10.3,1.4),(11.3,0.),34.,True)
    assert request is pending and selector.events==[('defer',(11.3,0.),34.)]


def test_completed_initial_turn_gets_bounded_translation_opportunity_without_clock_reset():
    watch=NavigationProgressWatch();watch.set_route((0.,0.),[(2.,0.)],(2.,0.),0.)
    watch.observe_stage(SimpleNamespace(failed=False),1.);watch.observe_stage(None,8.)
    assert watch.check((0.,0.),8.1,lambda a,b:True,False) is None
    assert watch.last_progress==0.
    assert watch.check((0.,0.),10.1,lambda a,b:True,False)['without_progress_s']==10.1


def test_same_target_replan_and_second_stage_cannot_renew_translation_credit():
    watch=NavigationProgressWatch();watch.set_route((0.,0.),[(2.,0.)],(2.,0.),0.)
    watch.observe_stage(SimpleNamespace(failed=False),1.);watch.observe_stage(None,8.)
    watch.set_route((0.,0.),[(2.,0.)],(2.,0.),8.5)
    watch.observe_stage(SimpleNamespace(failed=False),8.6);watch.observe_stage(None,9.5)
    assert watch.translation_until==10.
    assert watch.check((0.,0.),10.1,lambda a,b:True,False)['reason']=='no_forward_route_progress'


def test_translation_credit_prevents_opportunistic_reversal_until_real_progress():
    watch=NavigationProgressWatch();watch.set_route((0.,0.),[(2.,0.)],(2.,0.),0.)
    watch.observe_stage(SimpleNamespace(failed=False),1.);watch.observe_stage(None,8.)
    selector=Selector()
    assert reconsider_frontier(None,selector,None,(0.,0.),(3.,0.),(2.,0.),8.1,False,watch.translation_grace(8.1)) is None
    assert selector.events[-1][0]=='reference'
    watch.check((.3,0.),8.5,lambda a,b:True,False)
    assert not watch.translation_grace(8.5) and watch.last_progress==8.5
