import math
import numpy as np
import pytest

import test_online_contact_retreat as retreat_fixtures
from test_push_aside_skill import alignment_context
from grutopia_extension.interactive_navigation.go2_contact_control import observed_contact_retreat
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_translation_recovery_clear
import grutopia_extension.interactive_navigation.push_aside_skill as skill_module


def test_faster_retreat_rechecks_its_longer_observed_corridor():
    planner,box,limiter=retreat_fixtures.OnlineContactRetreatTests().make(boundary=.55)
    assert observed_contact_retreat(planner,[box],[0.,0.],math.pi,[-1.,0.],limiter,.02) is not None
    assert observed_contact_retreat(planner,[box],[0.,0.],math.pi,[-1.,0.],limiter,.02,speed=.35) is None


def test_configured_reverse_speed_retains_overlap_and_acceleration_checks():
    planner,box,limiter=retreat_fixtures.OnlineContactRetreatTests().make(boundary=.90)
    original=planner.free.copy();limiter.accept([-.20,0.,0.],0.)
    command,target=observed_contact_retreat(planner,[box],[0.,0.],math.pi,[-1.,0.],limiter,.1,speed=.35)
    assert -.35<command[0]<-.20
    assert np.linalg.norm(target)==pytest.approx(.35*.9)
    assert body_translation_recovery_clear(planner,[box],[0.,0.],math.pi,target,require_clear_end=False)
    np.testing.assert_array_equal(planner.free,original)


@pytest.mark.parametrize('speed',[0.,-.20,float('nan'),float('inf')])
def test_invalid_retreat_speed_is_rejected(speed):
    planner,box,limiter=retreat_fixtures.OnlineContactRetreatTests().make(boundary=.9)
    with pytest.raises(ValueError,match='finite and positive'):
        observed_contact_retreat(planner,[box],[0.,0.],math.pi,[-1.,0.],limiter,.02,speed=speed)


@pytest.mark.parametrize('feedback,expected',[(None,.20),(np.zeros(3),.35)])
def test_skill_uses_configured_speed_only_with_validated_profile_feedback(monkeypatch,feedback,expected):
    skill,ctx=alignment_context('retreat',gap=.01,lateral=.01,heading=0.)
    skill.release_origin=ctx.position[:2].copy();skill.release_reason='side_parked';ctx.policy_command=feedback
    chosen=[]
    def checked(*args,**kwargs):
        speed=kwargs.get('speed',.20);chosen.append(speed)
        return np.array([-speed,0.,0.]),ctx.position[:2]-np.array([.2,0.])
    monkeypatch.setattr(skill_module,'observed_contact_retreat',checked)
    command=skill.step(ctx)
    assert chosen==[expected] and command.velocity==(-expected,0.,0.)
