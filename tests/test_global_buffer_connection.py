import math
import unittest

import numpy as np

from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.go2_exploration_tracking import decision_buffer_recovery
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    navigation_body_blocked,body_motion_clear,body_translation_recovery_clear)


class GlobalBufferConnectionTests(unittest.TestCase):
    def scene(self):
        p=SidePushPlanner(np.ones((241,241),bool),[[-3.,3.],[-3.,3.]])
        x=-3.+(np.arange(241)+.5)*p.resolution
        p.free[:,x<-.25]=False
        p.navigation_route_radius=.30;p.navigation_pose_heading=None
        p.navigation_half_width=.22;p.full_navigation_footprint=True
        p.layered_local_control=True;p.planning_scope='global'
        local=p.free.copy()
        local &= ~p.box_mask(BoxState('temporary',(.37,-.375),(.14,.15,.3)))
        return p,local

    def test_alternative_exit_has_full_body_local_connection(self):
        p,local=self.scene();position=(0.,0.);heading=math.pi/2;goal=(1.5,1.)
        self.assertTrue(p.blocked([])[p.cell(position)])
        self.assertFalse(navigation_body_blocked(p,[],heading)[p.cell(position)])
        # Straight toward the closest circular-buffer exit crosses the
        # temporary obstacle with the rear of the north-facing robot.
        global_free=p.free;p.free=local
        self.assertFalse(body_motion_clear(p,[],position,heading,0.,0.,lateral=-.25,horizon=.5))
        p.free=global_free
        recovery=decision_buffer_recovery(position,heading,p,[],goal,heading,local_free=local,local_boxes=[])
        self.assertIsNotNone(recovery)
        action,state=recovery;target=state['target']
        self.assertFalse(p.blocked([])[p.cell(target)])
        self.assertTrue(p.path(target,goal,[]))
        self.assertGreater(target[1],0.)
        p.free=local
        self.assertTrue(body_translation_recovery_clear(p,[],position,heading,target))
        vx,vy,w=action['move_by_speed']
        self.assertTrue(body_motion_clear(p,[],position,heading,vx,w,lateral=vy,horizon=.5))

    def test_valid_exit_is_kept_but_invalidated_by_local_obstacle(self):
        p,local=self.scene();global_free=p.free
        first=decision_buffer_recovery((0.,0.),math.pi/2,p,[],(1.5,1.),math.pi/2,local_free=local)
        second=decision_buffer_recovery((0.,0.),math.pi/2,p,[],(1.5,1.),math.pi/2,local_free=local)
        self.assertEqual(first[1]['target'],second[1]['target'])
        self.assertTrue(second[1]['reused_exit_target'])
        self.assertIsNone(decision_buffer_recovery((0.,0.),math.pi/2,p,[],(1.5,1.),math.pi/2,
            local_free=np.zeros_like(local)))
        self.assertIs(p.free,global_free)
        self.assertIsNone(p._global_buffer_exit)

    def test_global_free_pose_does_not_request_recovery(self):
        p,local=self.scene()
        self.assertIsNone(decision_buffer_recovery((1.,0.),math.pi/2,p,[],(1.5,1.),math.pi/2,local_free=local))

    def test_real_obstacle_at_start_is_not_erased(self):
        p,local=self.scene();p.free[p.cell((0.,0.))]=False;before=p.free.copy()
        self.assertIsNone(decision_buffer_recovery((0.,0.),math.pi/2,p,[],(1.5,1.),math.pi/2,local_free=local))
        np.testing.assert_array_equal(p.free,before)


if __name__=='__main__':unittest.main()
