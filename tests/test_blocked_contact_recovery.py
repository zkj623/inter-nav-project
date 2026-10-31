from pathlib import Path
from types import SimpleNamespace as NS
import json,math
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
from grutopia_extension.interactive_navigation.interaction_skill import SkillContext
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import turn_reduces_overlap,body_motion_clear


def scenario():
    with np.load(Path(__file__).parent/'fixtures/t04_blocked_contact_withdrawal_v89.npz',allow_pickle=False) as z:
        p=SidePushPlanner(z['local_free'].copy(),z['limits'].tolist(),resolution=.05)
        p.box_free=z['box_free'].copy();p.planning_observed=z['observed'].copy()
        position=z['position'].copy();heading=float(z['heading']);boxes=[BoxState(**b) for b in json.loads(str(z['boxes_json']))]
    p.navigation_half_width=.22;p.full_navigation_footprint=True;p.layered_local_control=True;p.planning_scope='local';p.navigation_route_radius=.30
    box=next(b for b in boxes if b.id=='B06')
    permission=InteractionPermission('B06','box','push_aside',(),(),0.,source='ground_truth_layout',source_reference='recorded_fixture',source_digest='a'*64)
    skill=PushAsideSkill(permission,NS(box_id='B06',direction=(-1.,0.),push_distance=.85),(2.11,-.78),(10.3,1.75))
    skill.state='retreat';skill.started=0.;skill.state_since=9.
    skill.initial_box=np.array([10.1799993515,.2774845362]);skill.engage_box_position=skill.initial_box.copy()
    skill.initial_box_geometry=BoxState('B06',tuple(skill.initial_box),(.85,.55,.95))
    skill.last_box_motion_time=5.;skill.last_release_probe=10.;skill.release_origin=np.array([10.2,.30]);skill.release_reason='contact_attitude'
    limiter=VelocitySlewLimiter();limiter.observe([0.,0.,0.],10.);limiter.accept([0.,0.,0.],10.)
    layers=NS(planner=p,current_boxes=boxes,visible_box_ids=[b.id for b in boxes],last_local=10.,local_period=.1)
    ctx=SkillContext(10.,position,heading,np.zeros(3),0.,1.1,True,box,boxes,p,layers,limiter,body_angular_velocity=np.zeros(3))
    return skill,ctx


def test_actual_blocked_contact_pose_can_use_existing_checked_turn_after_stopping():
    skill,ctx=scenario();free=ctx.planner.free.copy();observed=ctx.planner.planning_observed.copy()
    assert skill._box_frame(ctx)[0]<skill.config.front_contact_reach+.02
    assert skill.step(ctx).state=='settle_blocked_retreat'
    ctx.now+=.41;ctx.layers.last_local=ctx.now
    command=skill.step(ctx)
    assert command.state=='recover_release_pose' and command.certificate is None
    assert command.events[-1]['reason']=='blocked_contact_withdrawal'
    target=skill.release_recovery_tracker._buffer_turn_target
    assert turn_reduces_overlap(ctx.planner,ctx.boxes,ctx.position,ctx.heading,target)
    ctx.now+=.01;ctx.planner.control_time=ctx.now
    command=skill.step(ctx)
    assert command.velocity[0]==command.velocity[1]==0. and command.velocity[2]<0.
    np.testing.assert_array_equal(ctx.planner.free,free);np.testing.assert_array_equal(ctx.planner.planning_observed,observed)


@pytest.mark.parametrize('missing',['stop','visible','new_scan','fresh_scan'])
def test_blocked_contact_recovery_requires_stop_and_fresh_visible_geometry(missing):
    skill,ctx=scenario()
    if missing=='stop':ctx.body_velocity[0]=.10
    if missing=='visible':ctx.layers.visible_box_ids=[]
    if missing=='new_scan':skill.last_box_motion_time=11.
    if missing=='fresh_scan':ctx.layers.last_local=9.
    assert skill.step(ctx).state=='settle_blocked_retreat'
    ctx.now+=.41
    if missing!='fresh_scan':ctx.layers.last_local=ctx.now
    assert skill.step(ctx).state=='settle_blocked_retreat'


def test_turning_out_of_overlap_still_requires_new_settling_and_passage_proof():
    skill,ctx=scenario();skill.step(ctx);ctx.now+=.41;ctx.layers.last_local=ctx.now
    assert skill.step(ctx).state=='recover_release_pose'
    ctx.heading=skill.release_recovery_tracker._buffer_turn_target;ctx.now+=.1
    assert body_motion_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,0.,0.,horizon=0.)
    result=skill.step(ctx)
    # The tracker first acknowledges the completed latched turn.
    assert result.state=='recover_release_pose' and result.velocity==(0.,0.,0.)
    ctx.now+=.01;result=skill.step(ctx)
    assert result.state=='settle_released_corridor' and result.status=='running' and result.certificate is None
