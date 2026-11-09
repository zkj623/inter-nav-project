"""Recorded-map and execution checks for the short post-push connection."""
from dataclasses import fields
import json
from pathlib import Path
import numpy as np
import pytest

from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.go2_pose_connection import plan_pose_connection,PoseConnectionControl,PassageConnectionAdmission,_executable_translation_clear
from grutopia_extension.interactive_navigation.go2_passage_guidance import _straight_body_connector_clear
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter


def snapshot(task):
    with np.load(Path(__file__).parent/'fixtures'/f'{task}_pose_connection_v197.npz') as data:
        arrays={key:data[key].copy() for key in data.files}
    planner=SidePushPlanner(arrays['free'],arrays['limits'].tolist(),resolution=float(arrays['resolution']))
    planner.navigation_half_width=.22
    names={f.name for f in fields(BoxState)}
    boxes=[BoxState(**{k:v for k,v in box.items() if k in names}) for box in json.loads(str(arrays['boxes_json']))]
    return planner,boxes,arrays['position'],float(arrays['heading']),json.loads(str(arrays['certificate_json']))


@pytest.mark.parametrize('task',['t04','t05'])
def test_recorded_release_has_a_shorter_full_body_pose_connection(task):
    planner,boxes,p,h,certificate=snapshot(task);before=planner.free.copy()
    plan=plan_pose_connection(p,h,certificate,planner,boxes)
    assert plan is not None and plan['body_margin_m']==.04
    entry=np.asarray(certificate.get('guide_entry',certificate['corridor_start']))
    end=np.asarray(certificate.get('guide_exit',certificate['corridor_end']))
    via_entry=np.linalg.norm(entry-p[:2])+np.linalg.norm(end-entry)
    assert plan['length_m']<via_entry-.5
    np.testing.assert_allclose(plan['operations'][-1]['target'],plan['exit'])
    axis=np.asarray(certificate['axis']);axis/=np.linalg.norm(axis)
    assert np.dot(np.asarray(plan['exit'])-certificate['center'],axis)>.5
    assert plan['full_stride_and_braking_checked']
    if plan['operations'][0]['kind']=='turn':
        error=(plan['operations'][0]['heading']-h+np.pi)%(2*np.pi)-np.pi
        assert abs(error)>=.15-1e-6
    np.testing.assert_array_equal(planner.free,before)


def test_no_pose_path_through_unknown_wall_across_door():
    planner,boxes,p,h,certificate=snapshot('t04')
    planner.free[planner.cell(certificate['corridor_end'])[0]-2,:]=False
    before=planner.free.copy()
    assert plan_pose_connection(p,h,certificate,planner,boxes) is None
    np.testing.assert_array_equal(planner.free,before)


def open_control(target=(.12,0.)):
    p=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05)
    p.navigation_half_width=.22
    limiter=VelocitySlewLimiter();limiter.accept([0,0,0],0.);limiter.policy_command=np.zeros(3)
    plan={'operations':[{'kind':'move','target':list(target),'heading':0.}]}
    return PoseConnectionControl(plan,0.),p,limiter


def test_short_move_delivers_effective_request_before_command_only_braking():
    control,planner,limiter=open_control()
    for i in range(1,101):
        now=i*.01;previous=limiter.command.copy()
        command,state=control.update(np.array([0,0,.34]),0.,planner,[],limiter,now,[0,0,0],[0,0,0],0.)
        assert np.linalg.norm(command[:2]-previous[:2])<=.65*.01+1e-8
        limiter.accept(command,now);limiter.policy_command=command.copy()
    assert command[0]==pytest.approx(.35)
    assert not control.braking and not control.done


def test_unexecuted_effective_move_times_out_without_advancing():
    control,planner,limiter=open_control()
    control.update(np.array([0,0,.34]),0.,planner,[],limiter,.01,[0,0,0],[0,0,0],0.)
    command,_=control.update(np.array([0,0,.34]),0.,planner,[],limiter,3.1,[0,0,0],[0,0,0],0.)
    assert control.braking and not control.done
    assert control.pending_failure=='pose_connection_translation_stalled'


def test_arriving_while_moving_does_not_finish_pose_sequence():
    control,planner,limiter=open_control()
    command,_=control.update(np.array([.10,0,.34]),0.,planner,[],limiter,.01,[.2,0,0],[0,0,0],0.)
    assert control.braking and not control.done
    limiter.accept([0,0,0],.1)
    control.update(np.array([.12,0,.34]),0.,planner,[],limiter,1.,[0,0,0],[0,0,0],0.)
    control.update(np.array([.12,0,.34]),0.,planner,[],limiter,1.5,[0,0,0],[0,0,0],0.)
    assert control.done and control.failed is None


def test_new_obstacle_stops_sequence_without_removing_occupancy():
    control,planner,limiter=open_control((.6,0.))
    planner.free[:,planner.cell([.7,0])[1]]=False
    before=planner.free.copy()
    command,state=control.update(np.array([0,0,.34]),0.,planner,[],limiter,.01,[0,0,0],[0,0,0],0.)
    assert control.pending_failure=='pose_connection_segment_changed'
    assert not control.done
    np.testing.assert_array_equal(command,[0,0,0])
    np.testing.assert_array_equal(planner.free,before)


def test_recorded_release_just_above_quiet_threshold_waits_then_admits():
    gate=PassageConnectionAdmission(0.)
    velocity=[-.05880128592252731,-.01305613573640585,-.003948237746953964]
    angular=[.019047105684876442,-.006385680288076401,.00756140798330307]
    first=gate.update([0,0,.34],velocity,angular,1.,0.)
    assert not first['ready'] and not first['quiet'] and not first['timed_out']
    assert not gate.update([.005,0,.34],[0,0,0],[0,0,0],1.,.1)['ready']
    assert gate.update([.005,0,.34],[0,0,0],[0,0,0],1.,.51)['ready']


def test_transient_angular_spike_restarts_connection_hold():
    gate=PassageConnectionAdmission(0.)
    gate.update([0,0,.34],[0,0,0],[0,0,0],1.,0.)
    assert not gate.update([0,0,.34],[0,0,0],[.11,0,0],1.,.3)['ready']
    assert not gate.update([0,0,.34],[0,0,0],[0,0,0],1.,.5)['ready']
    assert gate.update([0,0,.34],[0,0,0],[0,0,0],1.,.91)['ready']


def test_unsettled_connection_admission_is_bounded():
    gate=PassageConnectionAdmission(0.)
    result=gate.update([0,0,.34],[.07,0,0],[0,0,0],1.,6.01)
    assert result['timed_out'] and not result['ready']


def test_free_short_endpoint_does_not_prove_effective_stride_and_braking_space():
    planner=SidePushPlanner(np.ones((120,120),bool),[[-3,3],[-3,3]],resolution=.05)
    planner.navigation_half_width=.22
    planner.free[:,planner.cell([.70,0])[1]:]=False
    assert _straight_body_connector_clear(planner,[],[0,0],0.,[.12,0])
    assert not _executable_translation_clear(planner,[],[0,0],0.,[.12,0])


def test_private_search_map_still_contains_every_current_box():
    planner,boxes,p,h,certificate=snapshot('t04')
    before=planner.free.copy()
    center=np.asarray(certificate['center'])+.4*np.asarray(certificate['axis'])
    boxes.append(BoxState('test_blocker',tuple(center),(4.,.08,.5)))
    assert plan_pose_connection(p,h,certificate,planner,boxes) is None
    np.testing.assert_array_equal(planner.free,before)


def test_initial_body_motion_is_not_braked_before_effective_progress():
    control,planner,limiter=open_control()
    control.update(np.array([0,0,.34]),0.,planner,[],limiter,.01,[0,0,0],[0,0,0],0.)
    limiter.accept([.35,0,0],.50)
    command,state=control.update(np.array([.005,0,.34]),0.,planner,[],limiter,.51,[.05,0,0],[0,0,0],0.)
    assert not control.braking and command[0]>.3
    assert state['progress_since_stop_m']<state['minimum_progress_before_goal_brake_m']
    control.update(np.array([.045,0,.34]),0.,planner,[],limiter,.6,[.2,0,0],[0,0,0],0.)
    assert control.braking


def region_control():
    control,planner,limiter=open_control()
    control.plan={'operations':[{'kind':'move','target':[.1,0.],'heading':0.},
        {'kind':'turn','heading':1.}, {'kind':'move','target':[.1,1.],'heading':1.}]}
    control._brake()
    return control,planner,limiter


def test_checked_execution_region_avoids_chasing_an_unneeded_point():
    control,planner,limiter=region_control()
    p=np.array([.04,0,.34])
    control.update(p,0.,planner,[],limiter,1.,[0,0,0],[0,0,0],0.)
    _,state=control.update(p,0.,planner,[],limiter,1.5,[0,0,0],[0,0,0],0.)
    assert state['accepted_execution_region'] and control.index==1
    assert not control.done


def test_nearby_point_does_not_advance_when_next_motion_is_blocked():
    control,planner,limiter=region_control()
    planner.free[planner.cell([0,.7])[0],:]=False
    p=np.array([.04,0,.34])
    control.update(p,0.,planner,[],limiter,1.,[0,0,0],[0,0,0],0.)
    control.update(p,0.,planner,[],limiter,1.5,[0,0,0],[0,0,0],0.)
    assert control.index==0 and not control.done


def test_checked_region_does_not_override_total_deadline():
    control,planner,limiter=region_control();p=np.array([.04,0,.34])
    control.update(p,0.,planner,[],limiter,46.,[0,0,0],[0,0,0],0.)
    control.update(p,0.,planner,[],limiter,46.5,[0,0,0],[0,0,0],0.)
    assert control.failed=='pose_connection_deadline' and control.index==0


def test_intentional_turn_settling_is_not_counted_as_a_new_gait_stall():
    control,planner,limiter=open_control()
    # A small in-place correction fits; neither 0.2m/s stepping arc fits.
    planner.free[:]=False
    lo=planner.cell([-.45,-.35]);hi=planner.cell([.65,.35])
    planner.free[lo[0]:hi[0]+1,lo[1]:hi[1]+1]=True
    control.plan={'operations':[{'kind':'turn','heading':.11}]}
    control.phase='execute'
    control.assist.sign=1.;control.assist.heading=0.;control.assist.last_progress=0.
    control._brake()
    p=np.array([0,0,.34])
    control.update(p,0.,planner,[],limiter,1.,[0,0,0],[0,0,0],0.)
    control.update(p,0.,planner,[],limiter,1.5,[0,0,0],[0,0,0],0.)
    assert not control.braking and control.index==0 and control.stop_attempts==1
    command,state=control.update(p,0.,planner,[],limiter,1.51,[0,0,0],[0,0,0],0.)
    assert control.pending_failure is None and state['phase'] in ['continuous_turn','acknowledged_heading_step']
    assert command[2]>0
    assert control.started==0. and control.operation_started==0.


def test_resetting_response_window_does_not_clear_an_actual_safety_failure():
    control,planner,limiter=open_control()
    control._brake('pose_connection_attitude')
    p=np.array([.12,0,.34])
    control.update(p,0.,planner,[],limiter,1.,[0,0,0],[0,0,0],0.)
    control.update(p,0.,planner,[],limiter,1.5,[0,0,0],[0,0,0],0.)
    assert control.failed=='pose_connection_attitude' and not control.done


def test_fine_heading_step_requires_actual_input_before_stopping_on_body_twist():
    control,planner,limiter=open_control()
    control.plan={'operations':[{'kind':'turn','heading':.11}]}
    p=np.array([0,0,.34])
    command,_=control.update(p,0.,planner,[],limiter,.01,[0,0,0],[0,0,0],0.)
    limiter.accept(command,.01);limiter.policy_command=command.copy()
    command,state=control.update(p,.10,planner,[],limiter,.02,[0,0,0],[0,0,0],0.)
    assert not control.braking and not control.done
    assert state['desired_command'][2]==pytest.approx(.55)
    assert state['yaw_step']['input_complete'] is False
    limiter.accept([0,0,.55],.5);limiter.policy_command=np.array([0,0,.55])
    control.update(p,.10,planner,[],limiter,.51,[0,0,0],[0,0,0],0.)
    _,state=control.update(p,.10,planner,[],limiter,.56,[0,0,0],[0,0,0],0.)
    assert state['yaw_step']['input_complete'] and control.braking and not control.done
    limiter.accept([0,0,0],1.)
    control.update(p,.11,planner,[],limiter,1.1,[0,0,0],[0,0,0],0.)
    control.update(p,.11,planner,[],limiter,1.6,[0,0,0],[0,0,0],0.)
    assert control.done


def test_fine_heading_step_retains_actual_motion_checks():
    control,planner,limiter=open_control()
    control.plan={'operations':[{'kind':'turn','heading':.11}]}
    p=np.array([0,0,.34]);control.update(p,0.,planner,[],limiter,.01,[0,0,0],[0,0,0],0.)
    planner.free[:]=False
    command,_=control.update(p,0.,planner,[],limiter,.02,[0,0,0],[0,0,0],0.)
    assert control.braking and control.pending_failure=='pose_connection_motion_blocked'
    np.testing.assert_array_equal(command,[0,0,0])
    assert not control.done


def test_fine_turn_keeps_one_position_anchor_across_normal_stops():
    control,planner,limiter=open_control()
    control.plan={'operations':[{'kind':'turn','heading':.15}]}
    p=np.array([0,0,.34])
    command,_=control.update(p,0.,planner,[],limiter,.01,[0,0,0],[0,0,0],0.)
    limiter.accept(command,.01)
    drifted=np.array([-.03,.02,.34])
    _,state=control.update(drifted,0.,planner,[],limiter,.02,[0,0,0],[0,0,0],0.)
    assert state['desired_command'][0]>.05 and state['desired_command'][1]<-.03
    np.testing.assert_array_equal(control.turn_position_hold.anchor,[0.,0.])
    control._brake()
    limiter.accept([0,0,0],.5)
    _,state=control.update(drifted,0.,planner,[],limiter,.6,[0,0,0],[0,0,0],0.)
    np.testing.assert_array_equal(state['desired_command'],[0.,0.,0.])
    np.testing.assert_array_equal(control.turn_position_hold.anchor,[0.,0.])


def test_actual_turn_safety_failure_clears_position_hold_anchor():
    control,planner,limiter=open_control()
    control.turn_position_hold.apply([0,0,1],[0,0],0.,[0,0,0])
    control._brake('pose_connection_motion_blocked')
    assert control.turn_position_hold.anchor is None
    assert control.pending_failure=='pose_connection_motion_blocked'


@pytest.mark.parametrize('height,target,expected',[(.90,1.,True),(.25,.35,False)])
def test_exit_region_requires_the_actual_rear_to_cross(height,target,expected):
    control,planner,limiter=open_control((0.,target))
    control.plan['operations'][0]['heading']=np.pi/2
    control.plan['reference_passage']={'axis':[0.,1.],'center':[0.,0.]}
    control._brake();p=np.array([0,height,.34])
    control.update(p,np.pi/2,planner,[],limiter,1.,[0,0,0],[0,0,0],0.)
    control.update(p,np.pi/2,planner,[],limiter,1.5,[0,0,0],[0,0,0],0.)
    assert control.done is expected
