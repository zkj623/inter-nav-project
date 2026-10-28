import math
from types import SimpleNamespace as NS
import numpy as np
import pytest
from grutopia_extension.interactive_navigation.interaction_skill import SkillRegistry,SkillContext
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill
from grutopia_extension.interactive_navigation.go2_attitude_recovery import NavigationAttitudeRecovery


def make_skill():
    permission=InteractionPermission('B1','box','push_aside',('rgb1','rgb2'),('Qwen',),0.)
    plan=NS(box_id='B1',direction=(1.,0.),push_distance=1.)
    return PushAsideSkill(permission,plan,(0.,0.),(2.,0.))


def context(box=None,now=0.):
    return SkillContext(now,np.array([0.,0.,.4]),0.,np.zeros(3),0.,0.,True,box,[],None,None,None)


def test_skill_requires_recognition_permission_and_matching_target():
    registry=SkillRegistry();registry.register(PushAsideSkill)
    with pytest.raises(ValueError):registry.create(None)
    with pytest.raises(ValueError):
        PushAsideSkill(InteractionPermission('B1','box','push_aside',('one',),('Qwen',),0.),
                       NS(box_id='B1'),(0,0),(1,0))
    with pytest.raises(ValueError):
        PushAsideSkill(InteractionPermission('B2','box','push_aside',('one','two'),('Qwen',),0.),
                       NS(box_id='B1'),(0,0),(1,0))


def test_lost_track_returns_safe_failure_before_dereferencing_box():
    skill=make_skill();command=skill.step(context())
    assert command.status=='failed' and command.reason=='target_track_lost'
    assert command.velocity==(0.,0.,0.)


def test_retry_resets_old_release_origin():
    skill=make_skill();skill.release_origin=np.array([123.,0.])
    command=skill._retry(context(),'alignment_drift')
    assert skill.release_origin is None and skill.retries==1
    assert command.velocity==(0.,0.,0.)


def test_thirty_degree_hard_health_limit_is_unchanged():
    skill=make_skill();ctx=context(NS(id='B1',xy=(1.,0.)))
    ctx.tilt_degrees=30.
    assert skill.step(ctx).reason=='robot_unstable'


def quaternion(degrees):
    a=math.radians(degrees)/2
    return [math.cos(a),0.,math.sin(a),0.]


def test_stable_moderate_tilt_can_resume_but_sliding_cannot():
    for sliding in [False,True]:
        recovery=NavigationAttitudeRecovery(minimum_hold_ticks=6,stable_ticks=3,maximum_hold_ticks=20)
        recovery.update([0,0,.4],quaternion(20.))
        for i in range(6):answer=recovery.update([i*.03 if sliding else 0,0,.4],quaternion(12.))
        assert answer['state']==('hold' if sliding else 'released')
