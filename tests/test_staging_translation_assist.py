from pathlib import Path
import math
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.local_staging_control import LocalStagingControl,StagingTranslationAssist
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter
from grutopia_extension.interactive_navigation.go2_contact_turn import checked_contact_turn_command


def recorded_stage():
    with np.load(Path(__file__).parent/'data/t05_post_push_staging_v170.npz') as f:a={key:f[key] for key in f.files}
    p=SidePushPlanner(a['free'].copy(),a['limits'].tolist(),resolution=.05)
    p.navigation_half_width=.22;p.navigation_route_radius=.30;p.full_navigation_footprint=True
    p.planning_observed=a['observed'];p.layered_local_control=True;p.planning_scope='local'
    boxes=[BoxState(str(name),tuple(xy),tuple(dims),float(yaw)) for name,xy,dims,yaw in
           zip(a['box_ids'],a['box_xy'],a['box_dimensions'],a['box_yaw'])]
    q=a['orientation'];h=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
    stage=LocalStagingControl(a['position'],h,float(a['goal_heading']),p,boxes,0.)
    # Restore the controller's selected target from the same real checkpoint.
    stage.target=a['recorded_target'].copy()
    return p,boxes,a['position'],h,stage


def test_recorded_diagonal_dead_zone_gets_assistance_only_after_delivered_stall():
    p,boxes,position,h,stage=recorded_stage();limiter=VelocitySlewLimiter()
    assert not stage.failed
    original=p.free.copy();limiter.policy_command=np.array([-.214,-.210,0.])
    first,state=stage.update(position,h,p,boxes,limiter,0.)
    assert state['staging_phase']=='staging_move'
    assert np.linalg.norm(first[:2])==pytest.approx(.30)
    before,_=stage.update(position,h,p,boxes,limiter,1.49)
    np.testing.assert_allclose(before,first)
    request,state=stage.update(position,h,p,boxes,limiter,1.51)
    assert state['staging_translation_assist_active']
    assert max(abs(request[:2]))==pytest.approx(.35)
    assert np.linalg.norm(request[:2])<=.50 and request[2]==0.
    assert np.dot(request[:2],first[:2])>0.
    # The stronger request still goes through the ordinary finite-motion check.
    limiter.accept(request,1.5)
    applied,safe=checked_contact_turn_command(request,position,h,p,boxes,limiter,1.51,target_position=stage.target)
    assert safe and np.linalg.norm(applied[:2])>0.
    np.testing.assert_array_equal(p.free,original)
    p.free[:]=False
    applied,safe=checked_contact_turn_command(request,position,h,p,boxes,limiter,1.52,target_position=stage.target)
    assert not safe
    np.testing.assert_array_equal(applied,np.zeros(3))


@pytest.mark.parametrize('feedback,measured,positions',[
    (None,[0,0,0],[[0,0],[0,0]]),
    ([0,0,0],[0,0,0],[[0,0],[0,0]]),
    ([-.21,-.21,0],[.03,0,0],[[0,0],[0,0]]),
    ([-.21,-.21,0],[0,0,0],[[0,0],[.011,0]])])
def test_legacy_holds_measured_motion_or_real_progress_do_not_trigger(feedback,measured,positions):
    limiter=VelocitySlewLimiter();limiter.policy_command=feedback;limiter.measured=np.asarray(measured,float)
    assist=StagingTranslationAssist();direction=np.array([-1.,-1.])/math.sqrt(2)
    assert assist.speed(.30,direction,positions[0],limiter,0.)==.30
    assert assist.speed(.30,direction,positions[1],limiter,2.)==.30
    assert not assist.active


def test_assistance_keeps_predictive_braking_zero_and_original_stage_budget():
    p,boxes,position,h,stage=recorded_stage();limiter=VelocitySlewLimiter()
    limiter.policy_command=np.array([-.214,-.21,0.])
    stage.update(position,h,p,boxes,limiter,0.)
    stage.update(position,h,p,boxes,limiter,1.6)
    assert stage.translation_assist.active
    assert stage.translation_assist.speed(0.,np.array([1.,0.]),position,limiter,1.7)==0.
    command,state=stage.update(position,h,p,boxes,limiter,20.01)
    assert state['staging_phase']=='staging_failed' and stage.failed
    np.testing.assert_array_equal(command,np.zeros(3))
