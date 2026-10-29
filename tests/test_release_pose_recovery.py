"""Use the actual online T04 release snapshot; no occupancy fixture enters navigation."""
from pathlib import Path
from types import SimpleNamespace as NS
import json,math
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
from grutopia_extension.interactive_navigation.interaction_skill import SkillContext
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_motion_clear,body_translation_recovery_clear


def recorded_v85_scenario():
    with np.load(Path(__file__).parent/'fixtures/t04_release_body_overlap_v85.npz',allow_pickle=False) as z:
        p=SidePushPlanner(z['local_free'].copy(),z['limits'].tolist(),resolution=.05)
        p.box_free=z['box_free'].copy();p.planning_observed=z['observed'].copy()
        position=z['position'].copy();heading=float(z['heading']);boxes=[BoxState(**b) for b in json.loads(str(z['boxes_json']))]
    p.navigation_half_width=.22;p.full_navigation_footprint=True;p.layered_local_control=True;p.planning_scope='local';p.navigation_route_radius=.30
    box=next(b for b in boxes if b.id=='B06')
    permission=InteractionPermission('B06','box','push_aside',(),(),0.,source='ground_truth_layout',source_reference='recorded_fixture',source_digest='a'*64)
    skill=PushAsideSkill(permission,NS(box_id='B06',direction=(-1.,0.),push_distance=.85),(2.11,-.78),(10.3,1.75))
    skill.state='settle_released_corridor';skill.started=0.;skill.state_since=9.
    skill.initial_box=np.asarray(box.xy)+np.array([.620680809,0.]);skill.engage_box_position=skill.initial_box.copy()
    skill.initial_box_geometry=BoxState('B06',tuple(skill.initial_box),box.dimensions)
    skill.last_box_motion_time=5.;skill.release_origin=np.array([10.37,.07]);skill.release_reason='contact_alignment_drift'
    limiter=VelocitySlewLimiter();limiter.observe([0.,0.,0.],10.);limiter.accept([0.,0.,0.],10.)
    layers=NS(planner=p,current_boxes=boxes,visible_box_ids=[b.id for b in boxes],last_local=10.)
    ctx=SkillContext(10.,position,heading,np.zeros(3),0.,5.9,True,box,boxes,p,layers,limiter,body_angular_velocity=np.zeros(3))
    return skill,ctx


def scenario():
    # State-contract fixture with a single known corner contact and a genuinely
    # clear withdrawal. It is not physical evidence for the old T04v85 pose.
    skill,ctx=recorded_v85_scenario()
    p=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05)
    p.navigation_half_width=.22;p.full_navigation_footprint=True;p.layered_local_control=True;p.planning_scope='local';p.navigation_route_radius=.30
    p.free[p.cell([1.30,.20])]=False;p.box_free=p.free.copy();p.planning_observed=np.ones_like(p.free)
    box=BoxState('B06',(-.8,0.),(.85,.55,.95),mass_kg=6.)
    ctx.planner=p;ctx.box=box;ctx.boxes=[box];ctx.position=np.array([1.,0.,.42]);ctx.heading=math.pi
    ctx.layers=NS(planner=p,current_boxes=[box],visible_box_ids=['B06'],last_local=10.)
    skill.initial_box=np.array([0.,0.]);skill.engage_box_position=skill.initial_box.copy()
    skill.initial_box_geometry=BoxState('B06',(0.,0.),box.dimensions)
    skill.task_start=np.array([0.,-2.]);skill.task_goal=np.array([0.,2.]);skill.release_origin=np.array([.9,0.])
    return skill,ctx


def test_old_recorded_v85_exit_is_rejected_after_intermediate_cells_are_checked():
    skill,ctx=recorded_v85_scenario();free=ctx.planner.free.copy();observed=ctx.planner.planning_observed.copy()
    assert not body_translation_recovery_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,[10.625,.025])
    assert skill.step(ctx).state=='settle_released_corridor'
    ctx.now=10.5;ctx.layers.last_local=10.5
    result=skill.step(ctx)
    assert result.state=='retreat' and result.certificate is None
    np.testing.assert_array_equal(ctx.planner.free,free);np.testing.assert_array_equal(ctx.planner.planning_observed,observed)


def begin(skill,ctx):
    assert skill.step(ctx).state=='settle_released_corridor'
    ctx.now=10.5;ctx.layers.last_local=10.5
    result=skill.step(ctx)
    assert result.state=='recover_release_pose' and result.certificate is None
    return result


def test_known_release_overlap_enters_existing_checked_recovery():
    skill,ctx=scenario();free=ctx.planner.free.copy();observed=ctx.planner.planning_observed.copy()
    assert not body_motion_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,0.,0.,horizon=0.)
    begin(skill,ctx)
    assert body_translation_recovery_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,skill.release_recovery_target)
    first_motion=None
    for _ in range(120):
        ctx.now+=.01;result=skill.step(ctx)
        ctx.limiter.accept(result.velocity,ctx.now)
        if np.linalg.norm(result.velocity)>.01:first_motion=np.array(result.velocity);break
    assert first_motion is not None and np.linalg.norm(first_motion[:2])>0. and first_motion[2]==0.
    c,s=math.cos(ctx.heading),math.sin(ctx.heading);world=np.array([[c,-s],[s,c]])@first_motion[:2]
    assert body_translation_recovery_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,ctx.position[:2]+world*.5,require_clear_end=False)
    assert result.status=='running' and result.certificate is None
    np.testing.assert_array_equal(ctx.planner.free,free);np.testing.assert_array_equal(ctx.planner.planning_observed,observed)


@pytest.mark.parametrize('missing',['push','engage','release_gap','safe_exit','budget'])
def test_release_recovery_requires_a_real_push_clear_contact_and_safe_exit(missing):
    skill,ctx=scenario();along=.62068
    if missing=='push':along=.01
    if missing=='engage':skill.engage_box_position=None
    if missing=='release_gap':ctx.position[:2]+=skill.direction*(skill._box_frame(ctx)[0]-.10)
    if missing=='safe_exit':ctx.planner.free[:]=False;ctx.planner.box_free[:]=False
    if missing=='budget':skill.release_pose_recoveries=skill.config.max_retries
    assert not skill._begin_release_pose_recovery(ctx,along)
    assert skill.state=='settle_released_corridor' and skill.certificate is None


def test_recovery_requires_new_settling_and_effect_confirmation():
    skill,ctx=scenario();begin(skill,ctx)
    ctx.position=np.array([*skill.release_recovery_target,.423]);ctx.now+=.1
    assert body_motion_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,0.,0.,horizon=0.)
    result=skill.step(ctx)
    assert result.state=='settle_released_corridor' and result.status=='running'
    assert result.velocity==(0.,0.,0.) and result.certificate is None
    ctx.now+=.01
    assert skill.step(ctx).status=='running'


def test_free_pose_with_blocked_measured_stopping_motion_is_not_released():
    skill,ctx=scenario();blocked_position=ctx.position[:2].copy();begin(skill,ctx)
    ctx.position=np.array([*skill.release_recovery_target,.423]);ctx.now+=.1
    c,s=math.cos(ctx.heading),math.sin(ctx.heading);rotation=np.array([[c,-s],[s,c]])
    ctx.limiter.measured=np.array([*(rotation.T@(blocked_position-ctx.position[:2])/.8),0.])
    result=skill.step(ctx)
    assert result.state=='recover_release_pose' and result.velocity==(0.,0.,0.) and result.certificate is None


def test_release_recovery_has_a_bounded_duration():
    skill,ctx=scenario();begin(skill,ctx)
    ctx.now+=skill.config.max_preparation_recovery_seconds+.01
    result=skill.step(ctx)
    assert result.status=='failed' and result.reason=='release_pose_recovery_timeout'
