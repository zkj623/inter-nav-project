from pathlib import Path
from types import SimpleNamespace as NS
from dataclasses import fields
import json,math
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_axis_recovery_target,body_motion_clear,ForwardExplorationPathTracker
from grutopia_extension.interactive_navigation.go2_contact_control import observed_release_braking_clear
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
from grutopia_extension.interactive_navigation.interaction_skill import SkillContext
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter

def scenario():
    with np.load(Path(__file__).parent/'fixtures/t04_stalled_gait_axes_v187.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    p=SidePushPlanner(d['free'].copy(),d['limits'].tolist(),resolution=.05)
    p.navigation_half_width=.22;p.navigation_route_radius=.30;p.full_navigation_footprint=True;p.layered_local_control=True;p.planning_scope='local'
    p.box_free=d['box_free'].copy();p.planning_observed=d['observed'].copy()
    keys={f.name for f in fields(BoxState)};convert=lambda b:BoxState(**{k:v for k,v in b.items() if k in keys})
    boxes=[convert(b) for b in json.loads(str(d['boxes_json']))];box=next(b for b in boxes if b.id=='B06')
    q=d['orientation'];w,x,y,z=q/np.linalg.norm(q);h=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z));now=140.
    skill=PushAsideSkill(InteractionPermission(**json.loads(str(d['permission_json']))),NS(**json.loads(str(d['plan_json']))),[2.11,-.78],[10.3,1.75])
    skill.state='recover_release_pose';skill.started=now-20.;skill.state_since=now-4.
    skill.initial_box=np.asarray(skill.plan.initial_xy);skill.initial_box_geometry=convert(json.loads(str(d['initial_box_json'])))
    skill.engage_box_position=np.asarray(box.xy);skill.release_origin=np.array([10.052248,.37358]);skill.release_reason='side_parked'
    skill.last_box_motion_time=now-3.;skill.release_pose_recoveries=1;skill.progress_mark=d['position'][:2].copy();skill.progress_since=now-1.6
    tracker=ForwardExplorationPathTracker(d['position'][:2],[(10.3,1.75)],guard_motion=False)
    tracker._buffer_active=True;tracker._buffer_target=np.array([10.375,.375]);skill.release_recovery_tracker=tracker
    limiter=VelocitySlewLimiter();limiter.accept([-.35,-.15,0.],now-.01);limiter.observe([0.,0.,0.],now)
    layers=NS(current_boxes=boxes,visible_box_ids=[b.id for b in boxes],last_local=now,local_period=.1)
    ctx=SkillContext(now,d['position'].copy(),h,np.zeros(3),0.,.5,True,box,boxes,p,layers,limiter,
        body_angular_velocity=np.zeros(3),orientation=q,policy_command={'profile':'unitree_go2_45'})
    return skill,ctx

def test_recorded_stall_has_checked_pure_backward_axis_with_reserve():
    skill,ctx=scenario();free=ctx.planner.free.copy()
    chosen=body_axis_recovery_target(ctx.planner,ctx.boxes,ctx.position,ctx.heading)
    assert chosen['axis']=='backward' and chosen['distance_m']==.2
    np.testing.assert_allclose(chosen['command'],[-.35,0.,0.],atol=1e-12)
    assert observed_release_braking_clear(ctx.planner,ctx.boxes,chosen['target'],ctx.heading)
    assert body_axis_recovery_target(ctx.planner,ctx.boxes,ctx.position,ctx.heading,excluded=['backward']) is None
    assert body_axis_recovery_target(ctx.planner,ctx.boxes,ctx.position,ctx.heading,maximum_distance=.19) is None
    np.testing.assert_array_equal(ctx.planner.free,free)
    ctx.planner.free[:]=False
    assert body_axis_recovery_target(ctx.planner,ctx.boxes,ctx.position,ctx.heading) is None

def test_actual_stall_waits_for_stop_and_changes_primitive_without_resetting_deadline():
    skill,ctx=scenario();started=skill.state_since;position=ctx.position.copy()
    result=skill.step(ctx)
    assert result.state=='recover_release_pose' and result.velocity==(0.,0.,0.) and skill.release_axis_pending
    assert result.events[-1]['reason']=='recovery_translation_stalled'
    ctx.now+=.1
    assert skill.step(ctx).velocity==(0.,0.,0.) and skill.release_pose_recoveries==1
    ctx.limiter.accept([0.,0.,0.],ctx.now)
    skill.step(ctx);ctx.now+=.41;ctx.layers.last_local=ctx.now
    result=skill.step(ctx)
    assert result.events[-1]['reason']=='measured_stall_select_body_axis'
    assert result.events[-1]['axis']=='backward' and skill.release_pose_recoveries==2
    assert skill.release_axis_attempts==['backward'] and skill.state_since==started
    assert result.certificate is None and result.status=='running'
    np.testing.assert_array_equal(ctx.position,position)

def test_current_free_cell_does_not_cancel_latched_exit_target():
    skill,ctx=scenario();choice=body_axis_recovery_target(ctx.planner,ctx.boxes,ctx.position,ctx.heading)
    tracker=ForwardExplorationPathTracker(ctx.position[:2],[(10.3,1.75)],guard_motion=False)
    assert tracker.set_checked_buffer_target(ctx.position,ctx.heading,ctx.planner,ctx.boxes,choice['target'])
    skill.release_recovery_tracker=tracker
    direction=(choice['target']-ctx.position[:2])/.2
    ctx.position[:2]+=.15*direction
    assert body_motion_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,0.,0.,horizon=0.)
    assert ctx.planner.cell(ctx.position)!=ctx.planner.cell(choice['target'])
    skill.progress_mark=ctx.position[:2].copy();skill.progress_since=ctx.now;ctx.limiter.accept([0.,0.,0.],ctx.now-.01)
    result=skill.step(ctx)
    assert result.state=='recover_release_pose' and result.velocity[0]<0.
    assert result.certificate is None

@pytest.mark.parametrize('condition',['legacy','moving','recent_progress','budget'])
def test_axis_fallback_requires_native_stall_and_budget(condition):
    skill,ctx=scenario()
    if condition=='legacy':ctx.policy_command=None
    if condition=='moving':ctx.body_velocity[0]=-.1
    if condition=='recent_progress':skill.progress_since=ctx.now
    if condition=='budget':skill.release_pose_recoveries=skill.config.max_retries
    skill.step(ctx)
    assert not skill.release_axis_pending and not skill.release_axis_attempts

def test_pending_axis_stop_cannot_extend_original_recovery_timeout():
    skill,ctx=scenario();skill.release_axis_pending=True;ctx.now=skill.state_since+skill.config.max_preparation_recovery_seconds+.01
    result=skill.step(ctx)
    assert result.status=='failed' and result.reason=='release_pose_recovery_timeout'
