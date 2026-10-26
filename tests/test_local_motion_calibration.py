import unittest
from unittest.mock import patch
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_exploration_tracking import ExplorationPathTracker
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    ForwardExplorationPathTracker,body_motion_clear,navigation_footprint_samples)


class LocalCalibrationTests(unittest.TestCase):
    def planner(self,free=None):
        p=SidePushPlanner(np.ones((201,201),bool) if free is None else free,[[-2.5,2.5],[-2.5,2.5]])
        p.layered_local_control=True;p.full_navigation_footprint=True
        return p

    def test_recovery_uses_effective_reverse_request(self):
        free=np.ones((201,201),bool);free[:,118:]=False
        p=self.planner(free);tracker=ForwardExplorationPathTracker((0,0),[(-1,0)],guard_motion=False)
        action,state=tracker.action((0,0),0,p,[])
        self.assertLessEqual(action['move_by_speed'][0],-.34)
        self.assertLessEqual(np.linalg.norm(action['move_by_speed'][:2]),.400001)
        self.assertTrue(state['calibrated_buffer_motion'])

    def test_turn_is_preferred_before_translation_consumes_corner_clearance(self):
        free=np.zeros((201,201),bool);free[90:110]=True;free[100:,90:111]=True
        p=self.planner(free);position=(-.25,0)
        self.assertFalse(body_motion_clear(p,[],position,0,.25,1.2,lateral=.20))
        self.assertTrue(body_motion_clear(p,[],position,0,0,.4))
        tracker=ForwardExplorationPathTracker(position,[(0,.55)],guard_motion=False)
        def reference(*args,**kwargs):
            return {'move_by_speed':[.25,.20,1.2]},{'target':[0,.55],'recovering_buffer':False,'needs_replan':False}
        with patch.object(ExplorationPathTracker,'action',side_effect=reference):action,state=tracker.action(position,0,p,[])
        self.assertEqual(action['move_by_speed'][:2],[0.,0.])
        self.assertTrue(state['turn_before_translation'])

    def test_stalled_small_yaw_request_can_ramp_above_dead_zone(self):
        p=self.planner();tracker=ForwardExplorationPathTracker((0,0),[(1,.05)],guard_motion=False)
        def reference(*args,**kwargs):
            return {'move_by_speed':[0.,0.,.1]},{'target':[1,.05],'recovering_buffer':False,'needs_replan':False}
        with patch.object(ExplorationPathTracker,'action',side_effect=reference):
            speeds=[tracker.action((0,0),0,p,[])[0]['move_by_speed'][2] for _ in range(65)]
        self.assertAlmostEqual(speeds[0],.4)
        self.assertAlmostEqual(speeds[-1],1.)

    def test_footprint_contains_exact_front_and_side_extrema(self):
        p=self.planner();x,y=navigation_footprint_samples(p)
        self.assertAlmostEqual(x.max(),.50);self.assertAlmostEqual(x.min(),-.32)
        self.assertAlmostEqual(y.max(),.22);self.assertAlmostEqual(y.min(),-.22)


if __name__=='__main__':unittest.main()
