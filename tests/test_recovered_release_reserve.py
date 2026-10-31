"""Keep the observed braking reserve after the real v182 overlap recovery."""
from pathlib import Path
from types import SimpleNamespace as NS
from dataclasses import fields
import json,math
import numpy as np
import pytest
import grutopia_extension.interactive_navigation.push_aside_skill as module
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
from grutopia_extension.interactive_navigation.interaction_skill import SkillContext
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.go2_contact_control import observed_release_braking_clear
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_motion_clear,body_translation_recovery_clear

def scenario():
    with np.load(Path(__file__).parent/'fixtures/t04_recovered_release_reserve_v182.npz',allow_pickle=False) as f:
        p=SidePushPlanner(f['free'].copy(),f['limits'].tolist(),resolution=.05)
        p.box_free=f['box_free'].copy();p.planning_observed=f['observed'].copy()
        position=f['position'].copy();orientation=f['orientation'].copy();now=float(f['time_s'])
        linear=f['body_velocity'].copy();angular=f['angular_velocity'].copy()
        keys={field.name for field in fields(BoxState)}
        convert=lambda item:BoxState(**{k:v for k,v in item.items() if k in keys})
        boxes=[convert(item) for item in json.loads(str(f['boxes_json']))]
        original=convert(json.loads(str(f['initial_box_json'])))
        permission=InteractionPermission(**json.loads(str(f['permission_json'])))
        plan=NS(**json.loads(str(f['plan_json'])))
        engage=f['engage_box_position'].copy();release=f['release_origin'].copy();last_motion=float(f['last_box_motion_time'])
    p.navigation_half_width=.22;p.navigation_route_radius=.30;p.full_navigation_footprint=True
    p.layered_local_control=True;p.planning_scope='local'
    w,x,y,z=orientation/np.linalg.norm(orientation)
    heading=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    box=next(b for b in boxes if b.id=='B06')
    skill=PushAsideSkill(permission,plan,[2.11,-.78],[10.3,1.75])
    skill.state='settle_released_corridor';skill.started=now-10.;skill.state_since=now
    skill.initial_box=np.asarray(plan.initial_xy);skill.initial_box_geometry=original
    skill.engage_box_position=engage;skill.release_origin=release;skill.release_reason='side_parked'
    skill.last_box_motion_time=last_motion;skill.last_release_probe=now
    limiter=VelocitySlewLimiter();limiter.observe([*linear[:2],angular[2]],now);limiter.accept([0.,0.,0.],now)
    layers=NS(planner=p,current_boxes=boxes,visible_box_ids=[b.id for b in boxes],last_local=now,local_period=.1)
    ctx=SkillContext(now,position,heading,linear,float(angular[2]),.5,True,box,boxes,p,layers,limiter,
        body_angular_velocity=angular,orientation=orientation,policy_command={'profile':'unitree_go2_45'})
    return skill,ctx

def test_recorded_free_pose_reenters_checked_retreat_for_missing_reserve():
    skill,ctx=scenario();free=ctx.planner.free.copy()
    assert body_motion_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,0.,0.,horizon=0.)
    assert not observed_release_braking_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading)
    short=ctx.position+np.array([.025,0,0]);enough=ctx.position+np.array([.05,0,0])
    assert not observed_release_braking_clear(ctx.planner,ctx.boxes,short,ctx.heading)
    assert observed_release_braking_clear(ctx.planner,ctx.boxes,enough,ctx.heading)
    assert body_translation_recovery_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,enough)
    assert skill.step(ctx).state=='settle_released_corridor'
    ctx.now+=.41;ctx.layers.last_local=ctx.now
    result=skill.step(ctx)
    assert result.state=='retreat' and result.status=='running' and result.certificate is None
    assert result.events[-1]['reason']=='stopped_release_pose_needs_braking_reserve'
    ctx.now+=.01;ctx.limiter.accept([0.,0.,0.],ctx.now-.005)
    result=skill.step(ctx)
    assert result.state=='retreat' and result.velocity[0]<0.
    np.testing.assert_array_equal(ctx.planner.free,free)

@pytest.mark.parametrize('native',[False,True])
def test_recovered_effect_cannot_bypass_native_reserve_guard(monkeypatch,native):
    skill,ctx=scenario();ctx.policy_command={} if native else None
    calls=[]
    def observed(*args,**kwargs):
        calls.append(True);return {'confirmation_stage':'settle_released_corridor'}
    monkeypatch.setattr(skill,'_observed_passage_effect',observed)
    result=skill._recovered_passage_effect(ctx,.899)
    assert result is (not native)
    assert bool(calls) is (not native)
    if native:assert skill.status=='running' and skill.certificate is None

def test_settled_pose_with_reserve_can_finish_without_extra_withdrawal(monkeypatch):
    skill,ctx=scenario();ctx.position[0]+=.05
    assert observed_release_braking_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading)
    monkeypatch.setattr(skill,'_observed_passage_effect',lambda *a,**k:{'confirmation_stage':'settle_released_corridor'})
    assert skill.step(ctx).state=='settle_released_corridor'
    ctx.now+=.41;ctx.layers.last_local=ctx.now
    result=skill.step(ctx)
    assert result.state=='succeeded' and result.status=='succeeded'
