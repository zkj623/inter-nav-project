"""Actual map/footprint certificates must precede effect-based skill completion."""
from contextlib import contextmanager
from types import SimpleNamespace as NS
import math
import numpy as np
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
from grutopia_extension.interactive_navigation.interaction_skill import SkillContext
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill


def scenario(state='settle_after_retreat'):
    free=np.ones((240,240),bool);coordinates=-3+(np.arange(240)+.5)*.025
    free[np.abs(coordinates)<.05,:]=False
    free[:,np.abs(coordinates)<.55]=True
    p=SidePushPlanner(free,[[-3,3],[-3,3]],resolution=.025)
    p.navigation_half_width=.22;p.navigation_route_radius=.30;p.full_navigation_footprint=True
    p.planning_scope='local';p.planning_observed=np.ones_like(free)
    box=BoxState('B1',(-.76,-.45),(.85,.55,.95),mass_kg=6.)
    original=BoxState('B1',(0.,-.45),box.dimensions,mass_kg=6.)
    p.passages['B1']={'center':[0.,-.45],'axis':[0.,1.],'boundary':-.45}
    p.active_passages['B1']=dict(p.passages['B1'])
    class Layers:
        planner=p;current_boxes=[box];visible_box_ids=['B1'];last_local=10.
        global_free=free.copy()
        def refresh_global(self,reason):self.reason=reason
        @contextmanager
        def global_planning(self):
            saved=p.free,p.planning_scope,p.planning_observed
            p.free=self.global_free;p.planning_scope='global';p.planning_observed=np.ones_like(free)
            try:yield self.current_boxes
            finally:p.free,p.planning_scope,p.planning_observed=saved
    layers=Layers()
    permission=InteractionPermission('B1','box','push_aside',(),(),0.,source='ground_truth_layout',source_reference='fixture',source_digest='a'*64)
    skill=PushAsideSkill(permission,NS(box_id='B1',direction=(-1.,0.),push_distance=.85),(0.,-1.5),(0.,1.2))
    skill.state=state;skill.started=0.;skill.state_since=0.;skill.initial_box=np.asarray(original.xy)
    skill.initial_box_geometry=original;skill.engage_box_position=np.asarray(original.xy);skill.last_box_motion_time=5.;skill.retries=1
    ctx=SkillContext(10.,np.array([1.,-1.2,.41]),math.pi,np.zeros(3),0.,0.,True,box,[box],p,layers,NS(command=np.zeros(3)))
    return skill,ctx


def settle(skill,ctx):
    first=skill.step(ctx);assert first.status=='running'
    ctx.now=10.5;ctx.layers.last_local=10.5
    return skill.step(ctx)


def test_actual_observed_effect_can_finish_before_nominal_push_distance():
    for state in ['settle_after_retreat','settle_at_stage','settle_alignment']:
        skill,ctx=scenario(state);before=ctx.planner.free.copy();observed=ctx.planner.planning_observed.copy()
        result=settle(skill,ctx)
        assert result.status=='succeeded'
        proof=result.certificate
        assert proof['original_box_would_block_corridor'] is True
        assert proof['completion_rule']=='current_global_route_and_full_body_local_corridor'
        assert proof['checked_pose_count']>50
        assert proof['body_margin_m']==.04
        assert proof['confirmation_stage']==state
        assert proof['actual_push_distance_m']==.76 and proof['nominal_planned_push_m']==.85
        np.testing.assert_array_equal(ctx.planner.free,before)
        np.testing.assert_array_equal(ctx.planner.planning_observed,observed)


def test_missing_global_connection_cannot_finish_the_skill():
    skill,ctx=scenario();ctx.layers.global_free=np.zeros_like(ctx.planner.free)
    result=settle(skill,ctx)
    assert result.status=='running' and result.state=='prepare'
    assert skill.certificate is None


def test_a_corridor_not_blocked_by_the_original_box_is_not_a_push_effect():
    skill,ctx=scenario();skill.initial_box_geometry=ctx.box
    original_passage=ctx.planner.passages['B1'];original_active=ctx.planner.active_passages['B1']
    result=settle(skill,ctx)
    assert result.status=='running' and result.state=='prepare'
    assert ctx.planner.passages['B1'] is original_passage
    assert ctx.planner.active_passages['B1'] is original_active


def test_new_scan_and_current_box_visibility_are_required():
    for missing in ['scan','visibility']:
        skill,ctx=scenario()
        if missing=='scan':skill.last_box_motion_time=20.
        else:ctx.layers.visible_box_ids=[]
        result=settle(skill,ctx)
        assert result.status=='running' and skill.certificate is None


def test_released_corridor_brakes_and_rechecks_before_reporting_success():
    skill,ctx=scenario('retreat')
    ctx.position=np.array([.25,-.45,.41]);ctx.body_velocity=np.array([-.1,0.,0.])
    ctx.limiter.command=np.array([-.2,0.,0.]);skill.release_origin=np.array([.10,-.45])
    original_passage=ctx.planner.passages['B1']
    first=skill.step(ctx)
    assert first.status=='running' and first.state=='settle_released_corridor'
    assert first.velocity==(0.,0.,0.) and skill.certificate is None
    assert ctx.planner.passages['B1'] is original_passage
    ctx.now=10.2
    assert skill.step(ctx).status=='running'
    ctx.body_velocity=np.zeros(3);ctx.limiter.command=np.zeros(3)
    ctx.now=10.3;skill.step(ctx)
    ctx.now=10.8;ctx.layers.last_local=10.8
    assert skill.step(ctx).status=='succeeded'
    assert skill.certificate['confirmation_stage']=='settle_released_corridor'


def test_changed_corridor_during_release_is_not_accepted():
    skill,ctx=scenario('retreat');ctx.position=np.array([.25,-.45,.41])
    skill.release_origin=np.array([.10,-.45])
    assert skill.step(ctx).state=='settle_released_corridor'
    ctx.layers.global_free=np.zeros_like(ctx.planner.free)
    ctx.now=10.1;skill.step(ctx)
    ctx.now=10.6
    result=skill.step(ctx)
    assert result.status=='running' and result.state=='retreat' and result.certificate is None
