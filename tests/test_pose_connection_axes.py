import copy
import json
import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from test_pose_connection import open_control
import grutopia_extension.interactive_navigation.go2_pose_connection as module
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner


def test_reverse_diagonal_uses_two_checked_body_axes_at_any_heading():
    _,planner,_=open_control()
    for heading in [0.,math.pi/3]:
        c,s=math.cos(heading),math.sin(heading);rotation=np.array([[c,-s],[s,c]])
        target=rotation@[-.15,-.18]
        path=module._translation_waypoints(planner,[],[0,0],heading,target)
        assert len(path)==2
        np.testing.assert_allclose(rotation.T@path[0],[0.,-.18],atol=1e-10)
        np.testing.assert_allclose(path[-1],target)
        assert all(module._executable_translation_clear(planner,[],a,heading,b) for a,b in zip(([0,0],path[0]),path))


def test_forward_diagonal_and_minor_reverse_drift_are_preserved():
    _,planner,_=open_control()
    for target in [[.15,-.18],[-.02,.25]]:
        path=module._translation_waypoints(planner,[],[0,0],0.,target)
        assert len(path)==1
        np.testing.assert_allclose(path[0],target)


def test_clear_diagonal_does_not_admit_blocked_axis_corners(monkeypatch):
    _,planner,_=open_control();start=np.array([0.,0.]);target=np.array([-.15,-.18]);before=planner.free.copy()
    monkeypatch.setattr(module,'_executable_translation_clear',lambda p,b,a,h,z,**kw:np.allclose(a,start) and np.allclose(z,target))
    assert module._translation_waypoints(planner,[],start,0.,target) is None
    np.testing.assert_array_equal(planner.free,before)


def test_actual_pose_deviation_is_split_once_without_mutating_supplied_plan():
    _,planner,limiter=open_control();plan={'operations':[{'kind':'move','heading':0.,'target':[0.,.3]}]};before=copy.deepcopy(plan)
    control=module.PoseConnectionControl(plan,0.);position=np.array([.15,0.,.34])
    command,state=control.update(position,0.,planner,[],limiter,.01,[0,0,0],[0,0,0],0.)
    assert plan==before and len(control.plan['operations'])==2 and control.axis_splits==1
    assert abs(state['desired_command'][0])<1e-8 and state['desired_command'][1]>0.
    limiter.accept(command,.01);limiter.policy_command=command
    control.update(position,0.,planner,[],limiter,.02,[0,0,0],[0,0,0],0.)
    assert control.axis_splits==1 and control.started==0.


def test_runtime_axis_sequence_rejection_brakes_instead_of_using_original_diagonal(monkeypatch):
    control,planner,limiter=open_control((-.15,-.18))
    monkeypatch.setattr(module,'_translation_waypoints',lambda *a,**kw:None)
    command,_=control.update(np.array([0,0,.34]),0.,planner,[],limiter,.01,[0,0,0],[0,0,0],0.)
    assert control.pending_failure=='pose_connection_axis_sequence_blocked' and control.index==0
    np.testing.assert_array_equal(command,[0,0,0])


def test_axis_refinement_has_a_fixed_budget_and_does_not_reset_total_deadline():
    control,planner,limiter=open_control((-.15,-.18));control.axis_splits=4
    command,_=control.update(np.array([0,0,.34]),0.,planner,[],limiter,1.,[0,0,0],[0,0,0],0.)
    assert control.pending_failure=='pose_connection_axis_split_limit' and control.started==0.
    np.testing.assert_array_equal(command,[0,0,0])


def test_fore_aft_gait_oscillation_cannot_renew_passage_progress_deadline():
    control,planner,limiter=open_control((.6,0.))
    control.update(np.array([0,0,.34]),0.,planner,[],limiter,.01,[0,0,0],[0,0,0],0.)
    for i in range(1,66):
        now=i*.05;position=np.array([.03 if i%2 else -.03,0.,.34])
        command,_=control.update(position,0.,planner,[],limiter,now,[0,0,0],[0,0,0],0.)
        limiter.accept(command,now);limiter.policy_command=command.copy()
    assert control.braking and control.pending_failure=='pose_connection_translation_stalled'
    assert control.progress_time==pytest.approx(.05)


def test_recorded_t06_map_has_a_checked_connection_without_reverse_diagonal_commands():
    f=Path(__file__).parent/'fixtures/t06_pose_connection_qv19.npz'
    with np.load(f) as z:a={k:z[k] for k in z.files}
    p=SidePushPlanner(a['free'],a['limits'].tolist(),resolution=float(a['resolution']));p.navigation_half_width=.22
    names={f.name for f in fields(BoxState)};boxes=[BoxState(**{k:v for k,v in b.items() if k in names}) for b in json.loads(str(a['boxes_json']))]
    heading=float(a['heading']);start=a['position'];before=p.free.copy()
    assert module._reverse_diagonal_shift(start,heading,a['initial_request_target'])
    plan=module.plan_pose_connection(start,heading,json.loads(str(a['certificate_json'])),p,boxes)
    assert plan is not None and plan['body_margin_m']==.04 and plan['full_stride_and_braking_checked']
    position=start[:2]
    for op in plan['operations']:
        if op['kind']=='move':
            assert not module._reverse_diagonal_shift(position,op['heading'],op['target'])
            assert module._executable_translation_clear(p,boxes,position,op['heading'],op['target'])
            position=np.array(op['target'])
    np.testing.assert_array_equal(p.free,before)
