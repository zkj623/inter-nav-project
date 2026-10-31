from pathlib import Path
from dataclasses import fields
import json,math
import numpy as np
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_axis_recovery_target,ForwardExplorationPathTracker
from grutopia_extension.interactive_navigation.go2_contact_control import observed_release_braking_clear
from test_gait_axis_recovery import scenario

def stopped_case():
    with np.load(Path(__file__).parent/'fixtures/t04_stopped_push_axis_v189.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    p=SidePushPlanner(d['free'].copy(),d['limits'].tolist(),resolution=.05);p.navigation_half_width=.22;p.navigation_route_radius=.30
    p.full_navigation_footprint=True;p.layered_local_control=True;p.planning_scope='local';p.box_free=d['box_free'].copy();p.planning_observed=d['observed'].copy()
    keys={f.name for f in fields(BoxState)};boxes=[BoxState(**{k:v for k,v in b.items() if k in keys}) for b in json.loads(str(d['boxes_json']))]
    assert float(d['map_time_s'])<=float(d['pose_time_s'])
    return d,p,boxes

def test_stopped_body_axes_fail_but_reverse_push_axis_is_checked():
    d,p,boxes=stopped_case();free=p.free.copy();h=float(d['heading'])
    assert body_axis_recovery_target(p,boxes,d['position'],h) is None
    choice=body_axis_recovery_target(p,boxes,d['position'],h,away_direction=[1.,0.])
    assert choice['axis']=='push_axis' and choice['distance_m']==.2
    np.testing.assert_allclose(choice['target'],d['position'][:2]+[.2,0.],atol=1e-12)
    assert choice['command'][0]==-.35 and abs(choice['command'][1])<.05
    assert observed_release_braking_clear(p,boxes,choice['target'],h)
    assert body_axis_recovery_target(p,boxes,d['position'],h,away_direction=[1.,0.],excluded=['push_axis']) is None
    np.testing.assert_array_equal(p.free,free)

def test_push_axis_does_not_authorize_an_unsafe_map():
    d,p,boxes=stopped_case();p.free[:]=False
    assert body_axis_recovery_target(p,boxes,d['position'],float(d['heading']),away_direction=[1.,0.]) is None

def test_push_axis_secondary_command_is_bounded():
    p=SidePushPlanner(np.ones((100,100),bool),[(-2.5,2.5),(-2.5,2.5)],resolution=.05)
    p.navigation_half_width=.22;p.full_navigation_footprint=True;p.layered_local_control=True;p.planning_scope='local'
    kwargs={'excluded':['backward','left','right']}
    assert body_axis_recovery_target(p,[],[0,0],0.,away_direction=[-math.cos(.4),-math.sin(.4)],**kwargs) is None
    assert body_axis_recovery_target(p,[],[0,0],0.,away_direction=[-math.cos(.1),-math.sin(.1)],**kwargs)['axis']=='push_axis'

def test_actual_reserved_pose_can_brake_before_intermediate_target():
    skill,ctx=scenario();back=-np.array([math.cos(ctx.heading),math.sin(ctx.heading)])
    target=ctx.position[:2]+.30*back
    tracker=ForwardExplorationPathTracker(ctx.position[:2],[(10.3,1.75)],guard_motion=False)
    assert tracker.set_checked_buffer_target(ctx.position,ctx.heading,ctx.planner,ctx.boxes,target)
    skill.release_recovery_tracker=tracker;ctx.position[:2]+=.20*back
    skill.progress_mark=ctx.position[:2].copy();skill.progress_since=ctx.now
    assert ctx.planner.cell(ctx.position)!=ctx.planner.cell(target)
    assert observed_release_braking_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading)
    result=skill.step(ctx)
    assert result.state=='settle_released_corridor' and result.status=='running' and result.certificate is None
