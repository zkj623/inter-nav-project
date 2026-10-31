"""Post-stall gait admission must reserve the larger physical stopping sweep."""
import math
import numpy as np
import pytest
from test_push_axis_recovery import stopped_case
from test_gait_axis_recovery import scenario
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    body_axis_recovery_target,calibrated_buffer_translation,ForwardExplorationPathTracker,
    navigation_footprint_samples)
from grutopia_extension.interactive_navigation.go2_contact_control import observed_release_braking_clear

def test_recorded_axis_keeps_target_but_admits_effective_post_stall_speed():
    d,p,boxes=stopped_case();h=float(d['heading']);pos=d['position']
    normal=body_axis_recovery_target(p,boxes,pos,h,away_direction=[1.,0.])
    adapted=body_axis_recovery_target(p,boxes,pos,h,away_direction=[1.,0.],reverse_speed=.45)
    assert normal['command'][0]==-.35 and adapted['command'][0]==-.45
    np.testing.assert_array_equal(normal['target'],adapted['target'])
    assert adapted['axis']=='push_axis' and abs(adapted['command'][1])<=.25*abs(adapted['command'][0])
    assert observed_release_braking_clear(p,boxes,adapted['target'],h)

def test_clear_target_does_not_authorize_larger_stride_into_unknown():
    d,p,boxes=stopped_case();h=float(d['heading']);pos=d['position'];target=pos[:2]+[.2,0.]
    low=calibrated_buffer_translation(p,boxes,pos,h,target)
    high=calibrated_buffer_translation(p,boxes,pos,h,target,reverse_speed=.45)
    assert low is not None and high is not None
    xx,yy=navigation_footprint_samples(p)
    furthest=float(np.rint((math.cos(h)*xx-math.sin(h)*yy)/p.resolution).max()*p.resolution)
    travel=lambda command:.60*np.linalg.norm(command[:2])+np.linalg.norm(command[:2])**2/(2*.65)
    boundary=pos[0]+furthest+(travel(low)+travel(high))/2
    column=p.cell([boundary,pos[1]])[1]
    p.free[:,column:]=False;p.box_free[:,column:]=False;p.planning_observed[:,column:]=False
    assert observed_release_braking_clear(p,boxes,target,h)
    assert calibrated_buffer_translation(p,boxes,pos,h,target) is not None
    assert calibrated_buffer_translation(p,boxes,pos,h,target,reverse_speed=.45) is None
    tracker=ForwardExplorationPathTracker(pos[:2],[(10.3,1.75)],guard_motion=False,buffer_reverse_speed=.45)
    assert not tracker.set_checked_buffer_target(pos,h,p,boxes,target)

def test_latched_tracker_rechecks_at_the_admitted_speed():
    d,p,boxes=stopped_case();h=float(d['heading']);pos=d['position'];target=pos[:2]+[.2,0.]
    tracker=ForwardExplorationPathTracker(pos[:2],[(10.3,1.75)],guard_motion=False,buffer_reverse_speed=.45)
    assert tracker.set_checked_buffer_target(pos,h,p,boxes,target)
    for offset in [0.,.025,.05]:
        action,state=tracker.action(pos[:2]+[offset,0.],h,p,boxes)
        assert state['recovering_buffer'] and not state['needs_replan']
        assert action['move_by_speed'][0]==-.45 and state['buffer_reverse_speed_mps']==.45

def test_native_stall_uses_higher_admission_after_quiet_without_deadline_reset():
    skill,ctx=scenario();deadline_origin=skill.state_since
    result=skill.step(ctx)
    assert skill.release_axis_pending and result.velocity==(0.,0.,0.)
    assert skill.release_recovery_tracker.buffer_reverse_speed==.35
    ctx.now+=.1;ctx.limiter.accept([0.,0.,0.],ctx.now);skill.step(ctx)
    ctx.now+=.41;ctx.layers.last_local=ctx.now
    result=skill.step(ctx)
    assert result.events[-1]['reason']=='measured_stall_select_body_axis'
    assert result.events[-1]['reverse_speed_mps']==.45
    assert result.events[-1]['checked_initial_command'][0]==-.45
    assert skill.release_recovery_tracker.buffer_reverse_speed==.45
    assert skill.state_since==deadline_origin and skill.release_pose_recoveries==2
    ctx.limiter.accept(result.velocity,ctx.now);ctx.now+=.01;ctx.layers.last_local=ctx.now
    result=skill.step(ctx)
    assert 0.<np.linalg.norm(result.velocity[:2])<=.65*.01+1e-8
    assert result.certificate is None

@pytest.mark.parametrize('invalid',[float('nan'),float('inf'),.34,.46])
def test_recovery_speed_has_a_fixed_validated_upper_bound(invalid):
    with pytest.raises(ValueError):ForwardExplorationPathTracker([0,0],[(1,0)],buffer_reverse_speed=invalid)

@pytest.mark.parametrize('target',[[.3,0.],[0.,.3],[0.,-.3]])
def test_reverse_adaptation_does_not_increase_forward_or_lateral_speed(target):
    p=SidePushPlanner(np.ones((100,100),bool),[(-2.5,2.5),(-2.5,2.5)],resolution=.05)
    normal=calibrated_buffer_translation(p,[],[0.,0.],0.,target)
    adapted=calibrated_buffer_translation(p,[],[0.,0.],0.,target,reverse_speed=.45)
    np.testing.assert_array_equal(normal,adapted)
