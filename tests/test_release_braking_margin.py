import numpy as np
import pytest
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_contact_control import observed_release_braking_clear
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_motion_clear
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from test_push_aside_skill import alignment_context
import grutopia_extension.interactive_navigation.push_aside_skill as module


def test_recorded_box_edge_cell_requires_reserve_before_braking():
    free=np.ones((200,300),bool);free[133,63]=False
    planner=SidePushPlanner(free,[[.5,15.5],[-6.2,3.8]],resolution=.05)
    planner.navigation_half_width=.22
    boxes=[BoxState('B07',(4.141192913,.250015348),(.894519239,.709411726,.7))]
    initial=planner.free.copy()
    position=[3.199,.227];heading=-.0106
    assert body_motion_clear(planner,boxes,position,heading,0.,0.,horizon=0.)
    assert not observed_release_braking_clear(planner,boxes,position,heading)
    assert observed_release_braking_clear(planner,boxes,[3.05,.22565],.02986)
    # Replay's stopped pose meets neither the actual footprint nor reserve.
    assert not body_motion_clear(planner,boxes,[3.152864,.22565],.02986,0.,0.,horizon=0.)
    np.testing.assert_array_equal(planner.free,initial)


@pytest.mark.parametrize('feedback,reserve,expected',[
    (np.zeros(3),False,'retreat'),(np.zeros(3),True,'settle_released_corridor'),
    (None,False,'settle_released_corridor')])
def test_effect_probe_does_not_brake_verified_gait_without_reserve(monkeypatch,feedback,reserve,expected):
    skill,ctx=alignment_context('retreat',gap=.01,lateral=.01,heading=0.)
    skill.release_origin=ctx.position[:2].copy();skill.release_reason='side_parked';ctx.policy_command=feedback
    monkeypatch.setattr(skill,'_observed_passage_effect',lambda *args,**kwargs:True)
    monkeypatch.setattr(module,'observed_release_braking_clear',lambda *args:reserve)
    monkeypatch.setattr(module,'observed_contact_retreat',lambda *args,**kwargs:(np.array([-.35,0.,0.]),ctx.position[:2]-[.2,0.]))
    command=skill.step(ctx)
    assert command.state==expected
    if expected=='retreat':assert command.velocity[0]<0.
