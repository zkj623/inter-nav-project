import math
import unittest
from types import SimpleNamespace

import numpy as np

from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_exploration_tracking import ExplorationPathTracker
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import ForwardExplorationPathTracker
from grutopia_extension.interactive_navigation.mapping_runtime import MapNavigationRuntime


class EffectiveVelocityTargetTests(unittest.TestCase):
    def scene(self):
        p=SidePushPlanner(np.ones((241,241),bool),[[-3.,3.],[-3.,3.]])
        x=-3.+(np.arange(241)+.5)*p.resolution;y=x.copy()
        xx,yy=np.meshgrid(x,y)
        p.free[(xx>.75)&(xx<.90)&(yy>-.35)&(yy<.10)]=False
        p.layered_local_control=True;p.full_navigation_footprint=True
        p.navigation_half_width=.22;p.navigation_pose_heading=None
        return p

    def test_reported_target_is_fallback_waypoint_used_for_command(self):
        p=self.scene();tracker=ExplorationPathTracker((0.,0.),[(0.,.10),(2.,0.)],lookahead=1.3)
        action,state=tracker.action((0.,0.),0.,p,[])
        np.testing.assert_allclose(state['target'],(0.,.10))
        self.assertGreater(action['move_by_speed'][1],0.)
        self.assertGreater(action['move_by_speed'][2],1.)

    def test_guarded_speed_uses_fallback_heading_not_a_different_carrot(self):
        p=self.scene();tracker=ForwardExplorationPathTracker((0.,0.),[(0.,.10),(2.,0.)],lookahead=1.3,guard_motion=True)
        action,state=tracker.action((0.,0.),0.,p,[])
        np.testing.assert_allclose(state['target'],(0.,.10))
        self.assertAlmostEqual(state['heading_error_rad'],math.pi/2)
        self.assertLess(np.linalg.norm(action['move_by_speed'][:2]),1e-8)
        self.assertGreater(action['move_by_speed'][2],0.)

    def test_early_stop_clears_previous_effective_target(self):
        owner=SimpleNamespace(_last_velocity_target=(3.,4.),_planned_path=[(1.,0.)],
            action_for=lambda *args,**kwargs:{'move_by_speed':[0.,0.,0.]})
        MapNavigationRuntime.velocity_action_for(owner,None,{'position':(0.,0.,0.)})
        self.assertIsNone(owner._last_velocity_target)


if __name__=='__main__':unittest.main()
