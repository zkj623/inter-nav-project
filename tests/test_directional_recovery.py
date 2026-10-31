from pathlib import Path
from dataclasses import fields
import json,math
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import calibrated_buffer_translation,body_translation_recovery_clear
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter,smooth_recovery_action

def recorded_case():
    with np.load(Path(__file__).parent/'fixtures/t04_directional_recovery_v185.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    p=SidePushPlanner(d['free'].copy(),d['limits'].tolist(),resolution=.05)
    p.navigation_half_width=.22;p.navigation_route_radius=.30;p.full_navigation_footprint=True;p.layered_local_control=True;p.planning_scope='local'
    p.box_free=d['box_free'].copy();p.planning_observed=d['observed'].copy()
    keys={f.name for f in fields(BoxState)};boxes=[BoxState(**{k:v for k,v in b.items() if k in keys}) for b in json.loads(str(d['boxes_json']))]
    limiter=VelocitySlewLimiter();limiter.accept(d['previous_command'],float(d['previous_command_time_s']))
    limiter.recovery_gate.mode='translation';limiter.recovery_gate.direction=d['previous_command'][:2]/np.linalg.norm(d['previous_command'][:2])
    h=float(d['heading']);raw=calibrated_buffer_translation(p,boxes,d['position'],h,d['target'])
    return d,p,boxes,limiter,h,raw

def test_recorded_acceleration_preview_preserves_safe_checked_ray():
    d,p,boxes,limiter,h,raw=recorded_case();free=p.free.copy();before=limiter.command.copy()
    old=limiter.preview(raw,float(d['time_s']));new,detail=limiter.preview_directional_translation(raw,float(d['time_s']))
    c,s=math.cos(h),math.sin(h);rot=np.array([[c,-s],[s,c]])
    safe=lambda cmd:body_translation_recovery_clear(p,boxes,d['position'],h,d['position'][:2]+rot@cmd[:2]*.5,require_clear_end=False)
    assert not safe(old) and safe(new) and detail['direction_preserved']
    assert abs(new[0]*raw[1]-new[1]*raw[0])<1e-12
    assert np.linalg.norm(new[:2]-before[:2])<=.65*(float(d['time_s'])-float(d['previous_command_time_s']))+1e-10
    np.testing.assert_array_equal(limiter.command,before);np.testing.assert_array_equal(p.free,free)

def test_actual_recovery_uses_checked_direction_without_bypassing_geometry():
    d,p,boxes,limiter,h,raw=recorded_case()
    old=smooth_recovery_action({'move_by_speed':raw},d['position'],h,p,boxes,limiter,float(d['time_s']))
    assert old['move_by_speed']==[0.,0.,0.]
    d,p,boxes,limiter,h,raw=recorded_case()
    new=smooth_recovery_action({'move_by_speed':raw},d['position'],h,p,boxes,limiter,float(d['time_s']),directional_translation=True)
    assert new['move_by_speed'][0]<-.28 and limiter.recovery_diagnostic['sweep_safe']
    p.free[:]=False
    blocked=smooth_recovery_action({'move_by_speed':raw},d['position'],h,p,boxes,limiter,float(d['time_s']),directional_translation=True)
    assert blocked['move_by_speed']==[0.,0.,0.] and not limiter.recovery_diagnostic['sweep_safe']

@pytest.mark.parametrize('target',[[-.35,-.0128,0.],[.3,.1,0.]])
def test_directional_start_keeps_original_acceleration_from_rest(target):
    limiter=VelocitySlewLimiter();limiter.accept([0.,0.,0.],1.)
    new,detail=limiter.preview_directional_translation(target,1.01)
    np.testing.assert_allclose(new,limiter.preview(target,1.01),atol=1e-14)
    assert detail['direction_preserved'] and np.linalg.norm(new[:2])<=.0065+1e-12

def test_unreachable_direction_cannot_jump_outside_slew_ball():
    limiter=VelocitySlewLimiter();limiter.accept([.2,0.,0.],1.)
    new,detail=limiter.preview_directional_translation([0.,.35,0.],1.005)
    assert not detail['direction_preserved']
    np.testing.assert_array_equal(new,limiter.preview([0.,.35,0.],1.005))
    assert np.linalg.norm(new[:2]-limiter.command[:2])<=.00325+1e-12

def test_directional_braking_respects_deceleration_limit():
    limiter=VelocitySlewLimiter();limiter.accept([-.35,0.,0.],1.)
    new,detail=limiter.preview_directional_translation([-.1,0.,0.],1.01)
    assert detail['direction_preserved'] and new[0]==pytest.approx(-.3435)
    assert new[1]==0.

def test_no_translation_uses_existing_zero_or_turn_slew():
    limiter=VelocitySlewLimiter();limiter.accept([.2,0.,.1],1.)
    new,detail=limiter.preview_directional_translation([0.,0.,0.],1.01)
    assert not detail['direction_preserved'];np.testing.assert_array_equal(new,limiter.preview([0.,0.,0.],1.01))
