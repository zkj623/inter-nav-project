import numpy as np
from grutopia_extension.interactive_navigation.go2_heading_trim import HeadingTrimControl
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter


def fixture():
    p=SidePushPlanner(np.ones((100,100),bool),[[-2.5,2.5],[-2.5,2.5]],resolution=.05)
    p.navigation_half_width=.22
    return p,VelocitySlewLimiter()


def test_heading_brake_does_not_restart_before_measured_settling():
    p,l=fixture();control=HeadingTrimControl(0.)
    command,done,_=control.update([0,0,.4],-.03,p,[],l,0.,np.zeros(3),np.array([0,0,.2]),0.)
    assert control.braking and not done and np.all(command==0.)
    command,done,_=control.update([0,0,.4],-.08,p,[],l,.1,np.zeros(3),np.array([0,0,.1]),0.)
    assert control.braking and not done and np.all(command==0.)
    control.update([0,0,.4],-.08,p,[],l,.5,np.zeros(3),np.zeros(3),0.)
    command,done,_=control.update([0,0,.4],-.08,p,[],l,.81,np.zeros(3),np.zeros(3),0.)
    assert control.braking and not done and np.all(command==0.)
    command,done,_=control.update([0,0,.4],-.08,p,[],l,1.71,np.zeros(3),np.zeros(3),0.)
    assert not control.braking and not done and np.all(command==0.)


def test_stalled_yaw_uses_checked_stepping_arc_when_room_exists():
    p,l=fixture();control=HeadingTrimControl(.12)
    control.update([0,0,.4],0.,p,[],l,0.,np.zeros(3),np.zeros(3),0.)
    control.update([0,0,.4],0.,p,[],l,1.21,np.zeros(3),np.zeros(3),0.)
    command,done,detail=control.update([0,0,.4],0.,p,[],l,3.,np.zeros(3),np.zeros(3),0.)
    assert not done and detail['mode']=='prime_heading_step'
    assert detail['body_motion_safe'] and command[0]<0. and command[2]==0.
    l.accept([-.35,0.,0.],3.1);l.observe([-.12,0.,0.],3.1)
    control.update([-.04,0,.4],0.,p,[],l,3.1,[-.12,0.,0.],np.zeros(3),0.)
    command,done,detail=control.update([-.07,0,.4],0.,p,[],l,3.31,[-.12,0.,0.],np.zeros(3),0.)
    assert detail['mode']=='checked_heading_step' and command[0]<0. and command[2]>0.


def test_trim_does_not_force_arc_through_a_tight_observed_boundary():
    p,l=fixture();p.free[:,:p.cell([-.36,0])[1]+1]=False;p.free[:,p.cell([.55,0])[1]:]=False
    control=HeadingTrimControl(.12)
    control.update([0,0,.4],0.,p,[],l,0.,np.zeros(3),np.zeros(3),0.)
    control.update([0,0,.4],0.,p,[],l,1.21,np.zeros(3),np.zeros(3),0.)
    command,done,detail=control.update([0,0,.4],0.,p,[],l,3.,np.zeros(3),np.zeros(3),0.)
    assert detail.get('arc_unavailable') and command[0]==command[1]==0. and not done


def test_heading_trim_has_a_bounded_translation_region():
    p,l=fixture();control=HeadingTrimControl(.12)
    control.update([0,0,.4],0.,p,[],l,0.,np.zeros(3),np.zeros(3),0.)
    _,done,_=control.update([.3,0,.4],0.,p,[],l,3.,np.zeros(3),np.zeros(3),0.)
    assert control.travel_budget_exhausted and control.braking and not done
    command,done,_=control.update([.3,0,.4],0.,p,[],l,4.21,np.zeros(3),np.zeros(3),0.)
    assert control.failed=='heading_trim_translation_limit' and not done and np.all(command==0.)


def test_settled_band_tolerates_small_rebound_after_aiming_but_not_large_error():
    p,l=fixture();control=HeadingTrimControl(0.,tolerance=.05,acceptance_tolerance=.08)
    # The smaller aiming band was actually entered first.
    control.update([0,0,.4],-.03,p,[],l,0.,np.zeros(3),np.zeros(3),0.)
    _,done,_=control.update([0,0,.4],-.07,p,[],l,1.21,np.zeros(3),np.zeros(3),0.)
    assert not done  # pose changed even though the supplied gyro was zero
    _,done,_=control.update([0,0,.4],-.07,p,[],l,2.42,np.zeros(3),np.zeros(3),0.)
    assert done
    control=HeadingTrimControl(0.,tolerance=.05,acceptance_tolerance=.08)
    control.update([0,0,.4],-.03,p,[],l,0.,np.zeros(3),np.zeros(3),0.)
    _,done,_=control.update([0,0,.4],-.09,p,[],l,1.21,np.zeros(3),np.zeros(3),0.)
    assert not done


def test_residual_yaw_rate_cannot_carry_handoff_outside_contact_band():
    p,l=fixture();control=HeadingTrimControl(0.,tolerance=.05,acceptance_tolerance=.08)
    control.update([0,0,.4],-.03,p,[],l,0.,np.zeros(3),np.zeros(3),0.)
    control.update([0,0,.4],-.075,p,[],l,.1,np.zeros(3),np.array([0.,0.,-.02]),0.)
    _,done,detail=control.update([0,0,.4],-.075,p,[],l,1.31,np.zeros(3),np.array([0.,0.,-.02]),0.)
    assert detail['predicted_outward_error_rad']>.08 and not done
