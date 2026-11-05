import numpy as np
import pytest
from grutopia_extension.interactive_navigation.local_staging_control import StagingTranslationAssist,StagingYawRate,native_translation_speed
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter


def test_native_turn_does_not_drop_to_legacy_cruise_after_motion_starts():
    rate=StagingYawRate()
    for now in [0.,.13,.5,1.,2.]:
        assert rate.request(2.,.5,now,native=True)==1.
    assert rate.request(-2.,-.5,3.,native=True)==-1.


def test_effective_start_keeps_zero_and_requires_native_feedback():
    assist=StagingTranslationAssist();limiter=VelocitySlewLimiter();direction=np.array([1.,0.])
    assert assist.speed(.2,direction,[0,0],limiter,0.,effective_start=True)==.2
    limiter.policy_command=np.zeros(3)
    assert assist.speed(0.,direction,[0,0],limiter,.1,effective_start=True)==0.
    assert assist.speed(.2,direction,[0,0],limiter,.2,effective_start=True)==pytest.approx(.35)


def test_native_finite_movement_keeps_effective_floor_until_braking():
    assist=StagingTranslationAssist();limiter=VelocitySlewLimiter();limiter.policy_command=np.zeros(3)
    direction=np.array([1.,0.])
    assert assist.speed(.2,direction,[0,0],limiter,0.,effective_start=True)==pytest.approx(.35)
    limiter.measured=np.array([.15,0,0])
    assist.speed(.2,direction,[.01,0],limiter,.1,effective_start=True)
    assert assist.speed(.2,direction,[.04,0],limiter,.25,effective_start=True)==pytest.approx(.35)
    assert assist.start_ready and assist.starting
    assert assist.speed(0.,direction,[.05,0],limiter,.3,effective_start=True)==0.


def test_diagonal_start_preserves_direction_and_vector_cap():
    assist=StagingTranslationAssist();limiter=VelocitySlewLimiter();limiter.policy_command=np.zeros(3)
    direction=np.array([-.6,.8])
    speed=assist.speed(.2,direction,[0,0],limiter,0.,effective_start=True)
    assert speed<=.50
    assert max(abs(speed*direction))==pytest.approx(.35)


def test_predictive_braking_zero_is_preserved_after_start_boost():
    assist=StagingTranslationAssist();limiter=VelocitySlewLimiter();limiter.policy_command=np.zeros(3)
    assist.speed(.2,np.array([1.,0.]),[0,0],limiter,0.,effective_start=True)
    assert assist.starting
    assert assist.speed(0.,np.array([1.,0.]),[.1,0],limiter,.3,effective_start=True)==0.
    assert not assist.starting


def test_recorded_near_reverse_direction_gets_measured_effective_bound():
    direction=np.array([-.35,-.0632202476]);direction/=np.linalg.norm(direction)
    speed=native_translation_speed(.35,direction)
    assert speed<=.50
    assert (speed*direction)[0]==pytest.approx(-.45)
    assert native_translation_speed(0.,direction)==0.


def test_sideways_and_forward_bounds_do_not_inherit_reverse_increase():
    assert native_translation_speed(.20,np.array([1.,0.]))==pytest.approx(.35)
    assert native_translation_speed(.20,np.array([0.,-1.]))==pytest.approx(.35)
