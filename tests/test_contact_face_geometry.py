"""Actual Go2 head-envelope and observed approach-corridor invariants."""
import itertools
import math
from types import SimpleNamespace as NS
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.go2_contact_control import contact_face_lateral_allowance
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
from grutopia_extension.interactive_navigation.interaction_skill import SkillContext
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_motion_clear


def scenario(gap=.47, lateral=.0867, heading=.0763, width=.65):
    box=BoxState('B07',(1.3,0.),(.85,width,.70),mass_kg=5.)
    planner=SidePushPlanner(np.ones((160,160),bool),[[-4,4],[-4,4]],resolution=.05)
    planner.navigation_half_width=.22;planner.full_navigation_footprint=True
    planner.planning_observed=np.ones_like(planner.free);planner.box_free=planner.free.copy()
    permission=InteractionPermission('B07','box','push_aside',(),(),0.,source='ground_truth_layout',
        source_reference='geometry_test',source_digest='a'*64)
    skill=PushAsideSkill(permission,NS(box_id='B07',direction=(1.,0.),push_distance=.85),(0,0),(2,0))
    skill.state='align_lateral';skill.started=0.;skill.state_since=0.;skill.initial_box=np.array(box.xy)
    skill.anchor=np.array([box.xy[0]-.425-.33-.35,0.])
    position=np.array([box.xy[0]-.425-gap-.33*math.cos(heading),-lateral-.33*math.sin(heading),.42])
    limiter=VelocitySlewLimiter();limiter.observe([0.,0.,0.],1.);limiter.accept([0.,0.,0.],1.)
    ctx=SkillContext(1.,position,heading,np.zeros(3),0.,5.3,True,box,[box],planner,None,limiter,body_angular_velocity=np.zeros(3))
    return skill,ctx


def test_retreat_sized_offset_uses_a_safe_face_without_a_tiny_lateral_stride():
    skill,ctx=scenario();free=ctx.planner.free.copy();observed=ctx.planner.planning_observed.copy()
    assert skill._contact_ready(ctx)
    command=skill.step(ctx)
    assert command.state=='settle_contact' and command.velocity==(0.,0.,0.) and command.certificate is None
    assert command.events[-1]['contact_face_margin_m']==.04
    np.testing.assert_array_equal(ctx.planner.free,free);np.testing.assert_array_equal(ctx.planner.planning_observed,observed)


def test_geometric_readiness_still_requires_measured_settling_before_approach():
    skill,ctx=scenario();skill.step(ctx)
    ctx.now=1.1;ctx.body_velocity[0]=.12
    assert skill.step(ctx).state=='settle_contact'
    ctx.now=1.2;ctx.body_velocity[:]=0
    assert skill.step(ctx).state=='settle_contact'
    ctx.now=1.61
    assert skill.step(ctx).state=='engage'


@pytest.mark.parametrize('gap,lateral,heading,width',[
    (.61,.02,0.,.65),(.19,.02,0.,.65),(.35,.101,0.,.65),(.35,.02,.101,.65),(.35,.06,0.,.28),
])
def test_face_contact_does_not_bypass_gap_heading_or_object_size(gap,lateral,heading,width):
    skill,ctx=scenario(gap,lateral,heading,width)
    assert not skill._contact_ready(ctx)


def test_unobserved_forward_corridor_is_rejected_with_the_target_box_retained():
    skill,ctx=scenario(gap=.55,lateral=0.,heading=0.)
    cell=ctx.planner.cell([.70,0.]);ctx.planner.free[cell]=False
    ctx.planner.box_free[cell]=False;ctx.planner.planning_observed[cell]=False
    assert body_motion_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,0.,0.,horizon=0.)
    free=ctx.planner.free.copy()
    assert not skill._contact_ready(ctx)
    np.testing.assert_array_equal(ctx.planner.free,free)


def test_the_declared_four_centimetre_margin_is_not_consumed_by_head_width():
    allowance=contact_face_lateral_allowance((.85,.28,.7),(1.,0.),0.,0.)
    assert allowance==pytest.approx(.14-.051-.04)
    assert allowance<.055
    assert contact_face_lateral_allowance((.85,.18,.7),(1.,0.),0.,0.)==0.


def test_projection_bound_covers_all_head_corners_for_roll_pitch_and_yaw():
    corners=np.array(list(itertools.product([.239,.341],[-.051,.051],[-.108,.061]))).T
    for roll,pitch,heading in itertools.product([-.22,-.11,0.,.11,.22],[-.22,0.,.22],[-.18,0.,.18]):
        sr,cr=math.sin(roll),math.cos(roll);sp,cp=math.sin(pitch),math.cos(pitch);sy,cy=math.sin(heading),math.cos(heading)
        rotation=np.array([[cy,-sy,0.],[sy,cy,0.],[0.,0.,1.]])@np.array([[cp,0.,sp],[0.,1.,0.],[-sp,0.,cp]])@np.array([[1.,0.,0.],[0.,cr,-sr],[0.,sr,cr]])
        tilt=math.degrees(math.acos(cr*cp))
        actual_offsets=(rotation@corners)[1]-.33*sy
        allowance=contact_face_lateral_allowance((2.,2.,1.),(1.,0.),-heading,tilt,maximum=2.)
        bound=1.-.04-allowance
        assert max(abs(actual_offsets))<=bound+1e-12


def test_axis_equivalence_and_unsupported_diagonal_are_explicit():
    a=contact_face_lateral_allowance((.85,.65,.7),(1.,0.),.08,10.)
    assert contact_face_lateral_allowance((.65,.85,.7),(0.,-1.),.08,10.)==pytest.approx(a)
    assert contact_face_lateral_allowance((.85,.65,.7),(1.,1.),.08,10.)==0.


def test_pushing_can_center_within_a_full_face_but_must_not_leave_it():
    for lateral,expected in [(.12,'push'),(.151,'brake_for_recovery')]:
        skill,ctx=scenario(gap=.02,lateral=lateral,heading=0.)
        skill.state='push';skill.state_since=.5;skill.push_progress=.03;skill.push_progress_time=1.
        skill.initial_box=np.asarray(ctx.box.xy)-np.array([.03,0.]);skill.engage_box_position=skill.initial_box.copy()
        ctx.planner=NS(clearance=lambda box:{'side_parked':False})
        result=skill.step(ctx)
        assert result.state==expected and result.certificate is None


def test_skill_uses_observed_face_pose_instead_of_the_expanded_world_box():
    from dataclasses import replace
    # Use the world centerline: an offset towards the rotated face would
    # legitimately cancel its rotation and would not be a rejection case.
    skill,ctx=scenario(lateral=0.)
    ctx.orientation=(math.cos(ctx.heading/2),0.,0.,math.sin(ctx.heading/2))
    original=ctx.box
    ctx.box=replace(original,local_dimensions=original.dimensions,orientation=(math.cos(.1),0.,0.,math.sin(.1)),center_z=.36)
    ctx.boxes=[ctx.box]
    assert not skill._contact_ready(ctx)
    ctx.box=replace(ctx.box,orientation=(1.,0.,0.,0.));ctx.boxes=[ctx.box]
    assert skill._contact_ready(ctx)


def test_partial_observed_pose_does_not_fall_back_to_axis_aligned_acceptance():
    from dataclasses import replace
    skill,ctx=scenario()
    ctx.box=replace(ctx.box,local_dimensions=ctx.box.dimensions,orientation=(1.,0.,0.,0.),center_z=.36)
    ctx.boxes=[ctx.box]
    assert not skill._contact_ready(ctx)
