from pathlib import Path
from dataclasses import replace
import math
import numpy as np
import pytest
from test_push_aside_skill import alignment_context
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.go2_contact_control import contact_heading_hold
from grutopia_extension.interactive_navigation.go2_contact_turn import checked_contact_turn_command,checked_contact_step_command


def ready_brake(profile=True):
    skill,ctx=alignment_context('align_lateral',gap=.35,lateral=.14,heading=0.)
    skill.progress_mark=ctx.position[:2].copy();skill.progress_since=0.;skill.translation_braking=True
    ctx.policy_command=np.zeros(3) if profile else None
    ctx.now=1.6;assert skill.step(ctx).velocity==(0.,0.,0.)
    return skill,ctx


def test_initial_plateau_arms_bounded_step_only_after_measured_brake_without_resetting_deadline():
    skill,ctx=ready_brake();deadline=skill.state_since
    assert skill.lateral_step is None
    ctx.now=1.91;command=skill.step(ctx)
    assert skill.lateral_step is not None and command.state=='align_lateral'
    assert command.velocity==(0.,0.,0.) and skill.state_since==deadline
    assert command.events[-1]['reason']=='settled_initial_lateral_plateau'
    ctx.now=20.01
    assert skill.step(ctx).state=='brake_for_recovery' and skill.retries==1


@pytest.mark.parametrize('kind',['legacy','progress','moving'])
def test_no_initial_step_for_legacy_real_progress_or_unfinished_braking(kind):
    skill,ctx=ready_brake(profile=kind!='legacy');ctx.now=1.91
    if kind=='progress':ctx.position[1]+=.03
    if kind=='moving':ctx.body_velocity[1]=.10
    skill.step(ctx);ctx.now=2.22;skill.step(ctx)
    assert skill.lateral_step is None


def real_geometry():
    with np.load(Path(__file__).parent/'data/t04_lateral_plateau_v175.npz') as f:a={k:f[k] for k in f.files}
    planner=SidePushPlanner(a['free'].copy(),a['limits'].tolist(),resolution=.05);planner.navigation_half_width=.22
    boxes=[BoxState(str(name),tuple(xy),tuple(size),float(yaw)) for name,xy,size,yaw in
        zip(a['box_ids'],a['box_xy'],a['box_dimensions'],a['box_yaw'])]
    q=a['orientation']/np.linalg.norm(a['orientation']);heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
    c,s=math.cos(heading),math.sin(heading);rot=np.array([[c,-s],[s,c]])
    position=a['position'];box=next(b for b in boxes if b.id=='B06');target=np.array([position[0],box.xy[1]])
    error=(math.pi-heading+math.pi)%(2*math.pi)-math.pi
    request=np.array([*(rot.T@np.array([0.,.35])),contact_heading_hold(error,0.)])
    limiter=VelocitySlewLimiter();limiter.accept(request,0.)
    return planner,boxes,position,heading,rot,target,request,limiter


def test_actual_t04_short_step_is_checked_to_goal_without_removing_box_or_unknown_cells():
    p,boxes,pos,h,rot,target,request,limiter=real_geometry();saved=p.free.copy()
    assert not checked_contact_turn_command(request,pos,h,p,boxes,limiter,.01)[1]
    command,safe,mode=checked_contact_step_command(request,pos,h,p,boxes,limiter,.01,target_position=target)
    assert safe and mode=='finite_target' and np.linalg.norm(command[:2])>0.
    np.testing.assert_array_equal(p.free,saved)
    # An unknown cell newly entering the actual finite sweep must veto it.
    p.free[p.cell(pos[:2]+[0.,.32])]=False
    command,safe,mode=checked_contact_step_command(request,pos,h,p,boxes,limiter,.01,target_position=target)
    assert not safe and mode=='blocked';np.testing.assert_array_equal(command,np.zeros(3))


def test_actual_momentum_and_selected_box_still_veto_finite_correction():
    p,boxes,pos,h,rot,target,request,limiter=real_geometry()
    limiter.measured=np.array([*(rot.T@np.array([0.,.6])),0.])
    command,safe,_=checked_contact_step_command(request,pos,h,p,boxes,limiter,.01,target_position=target)
    assert not safe;np.testing.assert_array_equal(command,np.zeros(3))
    limiter.measured[:]=0.
    moved=[replace(b,xy=tuple(pos[:2]+[0.,.4])) if b.id=='B06' else b for b in boxes]
    command,safe,_=checked_contact_step_command(request,pos,h,p,moved,limiter,.01,target_position=target)
    assert not safe;np.testing.assert_array_equal(command,np.zeros(3))


def test_large_yaw_cannot_gain_the_finite_heading_hold_exception():
    p=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05);p.navigation_half_width=.22
    p.free[p.cell([0.,.45])[0]:,:]=False
    limiter=VelocitySlewLimiter();limiter.accept([0.,.35,1.],0.)
    command,safe,_=checked_contact_step_command([0.,.35,1.],[0.,0.],0.,p,[],limiter,.01,target_position=[0.,.15])
    assert not safe;np.testing.assert_array_equal(command,np.zeros(3))
