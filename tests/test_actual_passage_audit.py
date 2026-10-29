from tools.audit_actual_passage import crossing_evidence,final_stop_evidence
import math
import numpy as np


def trace(x=0.,ys=None):
    ys=np.linspace(-1.,1.4,301) if ys is None else ys
    return [{'time_s':i*.01,'position':[x,float(y),.42],'orientation':[math.sqrt(.5),0.,0.,math.sqrt(.5)],
             'body_velocity':[0.,0.,0.],'body_angular_velocity':[0.,0.,0.]} for i,y in enumerate(ys)]


def audit(rows,xy=(-1.,0.),observations=None):
    box={'xy':[0.,0.],'dimensions':[.85,.55,.95],'yaw':0.}
    obs=observations or [{'time_s':i*.01,'xy':list(xy)} for i in range(len(rows)+1)]
    return crossing_evidence(rows,{'axis':[0.,1.],'center':[0.,0.]},box,obs)


def test_actual_full_body_crossing_uses_vacated_box_volume():
    result=audit(trace());assert result['valid']
    item=next(c for c in result['crossings'] if c['valid'])
    assert item['rear_beyond_boundary_m']>.02
    assert item['original_box_would_block_body_clearance']
    assert item['observed_side_displacement_m']==1.


def test_reaching_other_side_via_a_distant_detour_does_not_count():
    result=audit(trace(x=2.));assert not result['valid'] and result['crossings']
    assert not result['crossings'][0]['original_box_would_block_body_clearance']


def test_push_can_create_required_clearance_beside_the_old_box():
    # Physical body misses the original box by1.5cm, below the4cm requirement.
    assert audit(trace(x=.69))['valid']


def test_unmoved_box_or_box_still_occupying_the_robot_cannot_count():
    assert not audit(trace(),xy=(0.,0.))['valid']
    assert not audit(trace(),xy=(-.4,0.))['valid']


def test_crossing_requires_a_before_side_and_continuous_pose_trace():
    assert not audit(trace(ys=np.linspace(.6,1.4,100)))['valid']
    ys=np.r_[np.linspace(-1.,.15,101),np.linspace(.5,1.4,100)]
    result=audit(trace(ys=ys));assert not result['valid']
    assert not result['crossings'][0]['continuous_local_trace']


def test_missing_or_moving_box_observations_cannot_certify_traversal():
    rows=trace()
    assert not audit(rows,observations=[{'time_s':0.,'xy':[-1.,0.]},{'time_s':10.,'xy':[-1.,0.]}])['valid']
    obs=[{'time_s':i*.01,'xy':[-1.-.03*(i%2),0.]} for i in range(len(rows)+1)]
    assert not audit(rows,observations=obs)['valid']


def stopped_rows():return trace(ys=np.full(61,1.4))


def test_final_pose_alone_does_not_prove_measured_stop():
    rows=stopped_rows();assert final_stop_evidence(rows,[0.,1.4])['valid']
    rows[-5]['body_velocity'][0]=.09
    assert not final_stop_evidence(rows,[0.,1.4])['valid']


def test_yaw_drift_moving_pose_short_hold_and_goal_error_reject_success():
    for failure in ['yaw','pose','short','goal']:
        rows=stopped_rows();goal=[0.,1.4]
        if failure=='yaw':rows[-2]['body_angular_velocity'][2]=.11
        if failure=='pose':rows[-3]['position'][0]=.03
        if failure=='short':rows=rows[-30:]
        if failure=='goal':goal=[.3,1.4]
        assert not final_stop_evidence(rows,goal)['valid']
