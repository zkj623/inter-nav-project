"""An SDK binding affects one planner/skill, never simulation globals."""
from types import SimpleNamespace as NS
import numpy as np

from deploy.interaction.motion import SdkNavigationMotion
from deploy.interaction.skill_backend import create_push_skill, SdkPushBackend
from deploy.tests.test_sdk_contact_profile import profile, permission
from grutopia_extension.interactive_navigation.push_aside_backend import PushAsideBackend
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import navigation_footprint_samples
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner


def test_measured_adapter_changes_only_the_bound_planner_and_skill():
    ordinary = SidePushPlanner(np.ones((160, 160), bool), [[-4, 4], [-4, 4]], resolution=.05)
    measured = SidePushPlanner(np.ones((160, 160), bool), [[-4, 4], [-4, 4]], resolution=.05)
    before = tuple(value.copy() for value in navigation_footprint_samples(ordinary))
    pr = profile(foot_x_min_m=-.4, foot_x_max_m=.6, foot_half_width_m=.3)
    SdkNavigationMotion(pr).bind(measured)
    xx, yy = navigation_footprint_samples(measured)
    assert (xx.min(), xx.max(), yy.min(), yy.max()) == (-.4, .6, -.3, .3)
    for original, after in zip(before, navigation_footprint_samples(ordinary)):
        np.testing.assert_array_equal(original, after)
    assert not hasattr(ordinary, 'motion_backend')
    plan = NS(box_id='B1', direction=(1., 0.), push_distance=.6)
    real = create_push_skill(permission(), plan, (0, 0), (1, 0), pr)
    native_permission = InteractionPermission('B1', 'box', 'push_aside', ('a', 'b'), ('model',), 0.)
    native = PushAsideSkill(native_permission, plan, (0, 0), (1, 0))
    assert isinstance(real.backend, SdkPushBackend)
    assert type(native.backend) is PushAsideBackend
    assert native.config.forward_speed == .25 and native.backend.nose_offset_m == .33
    assert real.config.forward_speed == pr.forward_mps and real.backend.nose_offset_m == pr.nose_offset_m


def test_sdk_explicitly_implements_every_execution_backend_operation():
    import inspect
    operations = {name for name, value in vars(PushAsideBackend).items() if inspect.isfunction(value)}
    assert operations <= set(vars(SdkPushBackend)), 'New default behavior must be explicitly adapted for SDK execution'
