"""Geometry regression from the failed v105 withdrawal; not a dynamics replay."""
from pathlib import Path
from types import SimpleNamespace as NS
import json
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState,PushPlan
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
from grutopia_extension.interactive_navigation.interaction_skill import SkillContext
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_translation_recovery_clear,body_motion_clear
from grutopia_extension.interactive_navigation.go2_contact_control import observed_contact_retreat


def scenario():
    with np.load(Path(__file__).parent/'fixtures/t04_short_push_blocked_retreat_v105.npz',allow_pickle=False) as z:
        planner=SidePushPlanner(z['free'].copy(),z['limits'].tolist(),resolution=.05)
        planner.box_free=z['box_free'].copy();planner.planning_observed=z['observed'].copy()
        position=z['position'].copy();heading=float(z['heading'])
        boxes=[BoxState(**b) for b in json.loads(str(z['boxes_json']))]
        permission=InteractionPermission(**json.loads(str(z['permission_json'])))
        plan=PushPlan(**json.loads(str(z['plan_json'])))
    planner.navigation_half_width=.22;planner.full_navigation_footprint=True
    planner.layered_local_control=True;planner.planning_scope='local';planner.navigation_route_radius=.30
    box=next(b for b in boxes if b.id=='B06')
    skill=PushAsideSkill(permission,plan,[2.11,-.78],[10.3,1.75])
    skill.state='retreat';skill.started=0.;skill.state_since=9.;skill.last_release_probe=10.
    skill.initial_box=np.asarray(plan.initial_xy);skill.engage_box_position=np.asarray(box.xy)
    skill.initial_box_geometry=BoxState('B06',tuple(skill.initial_box),box.dimensions)
    skill.last_box_motion_time=9.;skill.release_origin=position[:2].copy();skill.release_reason='contact_attitude'
    limiter=VelocitySlewLimiter();limiter.observe([0.,0.,0.],10.);limiter.accept([0.,0.,0.],10.)
    layers=NS(planner=planner,current_boxes=boxes,visible_box_ids=[b.id for b in boxes],last_local=10.,local_period=.1)
    ctx=SkillContext(10.,position,heading,np.zeros(3),0.,1.3,True,box,boxes,planner,layers,limiter,body_angular_velocity=np.zeros(3))
    return skill,ctx


def test_small_push_can_escape_observed_overlap_after_a_measured_stop():
    skill,ctx=scenario();free=ctx.planner.free.copy();observed=ctx.planner.planning_observed.copy()
    along=float(np.dot(np.asarray(ctx.box.xy)-skill.initial_box,skill.direction))
    assert .02<along<.03
    assert not body_motion_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,0.,0.,horizon=0.)
    assert observed_contact_retreat(ctx.planner,ctx.boxes,ctx.position,ctx.heading,skill.direction,ctx.limiter,ctx.now) is None
    assert skill.step(ctx).state=='retreat'
    ctx.now+=.41;ctx.layers.last_local=ctx.now
    command=skill.step(ctx)
    assert command.state=='recover_release_pose' and command.status=='running' and command.certificate is None
    assert command.events[-1]['reason']=='blocked_contact_withdrawal'
    assert body_translation_recovery_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,skill.release_recovery_target)
    np.testing.assert_array_equal(ctx.planner.free,free)
    np.testing.assert_array_equal(ctx.planner.planning_observed,observed)


@pytest.mark.parametrize('missing',['stop','visible','new_scan','fresh_scan','engage','motion','budget','safe_exit'])
def test_short_push_does_not_bypass_the_other_contact_recovery_gates(missing):
    skill,ctx=scenario()
    if missing=='stop':ctx.body_velocity[0]=.10
    if missing=='visible':ctx.layers.visible_box_ids=[]
    if missing=='new_scan':skill.last_box_motion_time=11.
    if missing=='fresh_scan':ctx.layers.last_local=9.
    if missing=='engage':skill.engage_box_position=None
    if missing=='motion':skill.last_box_motion_time=None
    if missing=='budget':skill.release_pose_recoveries=skill.config.max_retries
    if missing=='safe_exit':ctx.planner.free[:]=False;ctx.planner.box_free[:]=False
    assert skill.step(ctx).state=='retreat'
    ctx.now+=.41
    if missing!='fresh_scan':ctx.layers.last_local=ctx.now
    assert skill.step(ctx).state=='retreat'


def test_short_push_neither_certifies_a_passage_nor_enters_normal_post_push_release():
    skill,ctx=scenario();along=.022110939
    assert not skill._begin_release_pose_recovery(ctx,along,blocked_contact=False)
    assert not skill._observed_passage_effect(ctx,along)
    assert skill.certificate is None and skill.status=='running'
