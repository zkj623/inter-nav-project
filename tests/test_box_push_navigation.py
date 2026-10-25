import unittest
import numpy as np
from dataclasses import replace
from grutopia_extension.interactive_navigation.box_push_navigation import BoxPushPlanner,BoxState
from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner


class BoxPushPlanningTests(unittest.TestCase):
    def setUp(self):
        self.free=np.ones((121,241),bool)
        self.free[[0,-1],:]=False;self.free[:,[0,-1]]=False
        self.planner=BoxPushPlanner(self.free,[[0,6],[-1.5,1.5]],robot_radius=.25)

    def test_ordinary_path_does_not_trigger_unnecessary_push(self):
        box=BoxState('box',(3,1.0),(.5,.5,.5))
        decision=self.planner.decide((1,0),(5,0),[box])
        self.assertEqual(decision.kind,'navigate')
        self.assertIsNone(decision.push)

    def test_goal_occupied_by_movable_body_is_cleared_in_prediction(self):
        box=BoxState('box',(3,0),(.5,.5,.5))
        decision=self.planner.decide((1,0),(3,0),[box])
        self.assertEqual(decision.kind,'push')
        plan=decision.push
        moved=replace(box,xy=plan.destination_xy)
        direction=np.array(plan.direction)
        release=np.array(plan.destination_xy)-direction*(.25+self.planner.contact_offset+.18)
        self.assertTrue(self.planner.path(release,(3,0),[moved]))
        self.assertTrue(np.array_equal(self.planner.free,self.free))

    def test_fixed_geometry_is_not_mistaken_for_a_movable_obstacle(self):
        free=self.free.copy();free[:,110:130]=False
        planner=BoxPushPlanner(free,[[0,6],[-1.5,1.5]],robot_radius=.25)
        decision=planner.decide((1,0),(5,0),[BoxState('unrelated',(1,1),(.3,.3,.3))])
        self.assertEqual(decision.kind,'failed')
        self.assertEqual(decision.reason,'fixed_geometry_blocks_task')

    def test_infeasible_parking_does_not_produce_a_push_command(self):
        free=np.zeros_like(self.free)
        free[48:73,30:210]=True
        planner=BoxPushPlanner(free,[[0,6],[-1.5,1.5]],robot_radius=.25)
        decision=planner.decide((1.2,0),(4.8,0),[BoxState('box',(3,0),(.5,.5,.5))])
        self.assertEqual(decision.kind,'failed')
        self.assertIsNone(decision.push)

    def test_detour_shortcuts_preserve_collision_clearance(self):
        free=self.free.copy();free[36:85,110:130]=False
        planner=BoxPushPlanner(free,[[0,6],[-1.5,1.5]],robot_radius=.25)
        start=(1,0);path=planner.path(start,(5,0),[])
        self.assertTrue(path)
        self.assertLess(len(path),10)
        blocked=planner.blocked([])
        for first,last in zip([(*start,0),*path[:-1]],path):
            self.assertTrue(AStarMapPlanner.segment_clear(planner.cell(first[:2]),planner.cell(last[:2]),blocked))


if __name__=='__main__':unittest.main()
