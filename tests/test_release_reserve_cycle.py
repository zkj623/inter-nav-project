"""A failed stopped reserve must not re-trigger braking on gait oscillation."""
from pathlib import Path
from types import SimpleNamespace as NS
from dataclasses import fields
import json, math
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_contact_control import observed_contact_retreat, observed_release_braking_clear
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
from grutopia_extension.interactive_navigation.interaction_skill import SkillContext
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter


def scenario():
    with np.load(Path(__file__).parent/'fixtures/t10_release_cycle_qv22.npz',allow_pickle=False) as z:
        d={k:z[k] for k in z.files}
    p=SidePushPlanner(d['free'].copy(),d['limits'].tolist(),resolution=.05)
    p.box_free=d['free'].copy();p.planning_observed=d['observed'].copy()
    p.navigation_half_width=.22;p.navigation_route_radius=.30;p.full_navigation_footprint=True
    p.layered_local_control=True;p.planning_scope='local'
    keys={f.name for f in fields(BoxState)}
    boxes=[BoxState(**{k:v for k,v in b.items() if k in keys}) for b in json.loads(str(d['boxes_json']))]
    box=next(b for b in boxes if b.id=='B06')
    start=json.loads(str(d['start_json']));rows=json.loads(str(d['sample_rows_json']))
    skill=PushAsideSkill(InteractionPermission(**start['permission']),NS(**start['plan']),[4.6875,2.9125],[10.3,1.75])
    skill.state='settle_released_corridor';skill.started=start['time_s'];skill.state_since=84.
    skill.initial_box=np.asarray(skill.plan.initial_xy);skill.engage_box_position=np.asarray(box.xy)
    skill.release_origin=np.array([10.08,.33]);skill.release_reason='side_parked';skill.last_box_motion_time=78.
    w,x,y,z=d['orientation'];heading=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    limiter=VelocitySlewLimiter();limiter.observe([0,0,0],84.);limiter.accept([0,0,0],84.)
    layers=NS(last_local=84.,local_period=.1,visible_box_ids=['B04','B06'])
    ctx=SkillContext(84.,d['position'].copy(),heading,np.zeros(3),0.,.5,True,box,boxes,p,layers,limiter,
                     orientation=d['orientation'],body_angular_velocity=np.zeros(3),policy_command=np.zeros(3))
    return skill,ctx,rows


def failed_stop(skill,ctx):
    assert not observed_release_braking_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading)
    skill.step(ctx);ctx.now+=.41;ctx.layers.last_local=ctx.now
    result=skill.step(ctx)
    assert result.state=='retreat' and result.certificate is None
    return skill.release_reserve_since


def test_recorded_release_cycle_keeps_checked_retreat_through_false_free_probes(monkeypatch):
    skill,ctx,rows=scenario();free=ctx.planner.free.copy()
    failed_stop(skill,ctx)
    # Old recorded moving/stopped poses oscillate by centimetres. Even a
    # temporarily valid map/corridor cannot cancel the newly required retreat.
    monkeypatch.setattr(skill,'_observed_passage_effect',lambda *a,**k:True)
    import grutopia_extension.interactive_navigation.push_aside_skill as module
    monkeypatch.setattr(module,'observed_release_braking_clear',lambda *a:True)
    for r in rows[1:]:
        ctx.now=r['time_s'];ctx.position=np.asarray(r['position']);ctx.layers.last_local=ctx.now
        ctx.limiter.accept([0.,0.,0.],ctx.now-.1)
        result=skill.step(ctx)
        assert result.state=='retreat' and result.velocity[0]<0. and result.certificate is None
    np.testing.assert_array_equal(ctx.planner.free,free)


def test_recorded_map_admits_the_extra_retreat_with_existing_full_motion_checks():
    skill,ctx,_=scenario()
    answer=observed_contact_retreat(ctx.planner,ctx.boxes,ctx.position,ctx.heading,skill.direction,ctx.limiter,ctx.now+.1,speed=.35)
    assert answer is not None and answer[0][0]<0.
    target=ctx.position.copy();target[:2]-=skill.direction*.08
    assert observed_release_braking_clear(ctx.planner,ctx.boxes,target,ctx.heading)
    ctx.planner.free[:]=False;ctx.planner.box_free[:]=False;ctx.planner.planning_observed[:]=False
    assert observed_contact_retreat(ctx.planner,ctx.boxes,ctx.position,ctx.heading,skill.direction,ctx.limiter,ctx.now+.1,speed=.35) is None


def test_net_progress_then_measured_stop_is_still_required(monkeypatch):
    skill,ctx,_=scenario();failed_stop(skill,ctx)
    monkeypatch.setattr(skill,'_observed_passage_effect',lambda *a,**k:True)
    ctx.position[:2]-=skill.direction*.081;ctx.now+=.3;ctx.layers.last_local=ctx.now
    result=skill.step(ctx)
    assert result.state=='settle_released_corridor' and result.status=='running' and result.certificate is None


def test_sideways_motion_does_not_count_as_withdrawal(monkeypatch):
    skill,ctx,_=scenario();failed_stop(skill,ctx)
    import grutopia_extension.interactive_navigation.push_aside_skill as module
    monkeypatch.setattr(module,'observed_release_braking_clear',lambda *a:True)
    monkeypatch.setattr(skill,'_observed_passage_effect',lambda *a,**k:True)
    ctx.position[:2]+=skill.tangent*.081;ctx.now+=.3;ctx.layers.last_local=ctx.now
    result=skill.step(ctx)
    assert result.state!='settle_released_corridor' and result.certificate is None


def test_failed_reserve_cycle_has_bounded_total_time():
    skill,ctx,_=scenario();since=failed_stop(skill,ctx)
    ctx.now=since+skill.config.max_preparation_recovery_seconds+.01
    result=skill.step(ctx)
    assert result.reason=='release_braking_reserve_timeout' and result.velocity==(0.,0.,0.)


def test_subsequent_failed_stop_does_not_renew_deadline():
    skill,ctx,_=scenario();since=failed_stop(skill,ctx)
    skill.state='settle_released_corridor';skill.settled_since=None;ctx.now+=2.
    failed_stop(skill,ctx)
    assert skill.release_reserve_since==since
