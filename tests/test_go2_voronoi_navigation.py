import math
import json
import unittest
from dataclasses import replace
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    ExplorationVoronoiRoutes,ForwardExplorationPathTracker,update_task_cameras,route_blocked,navigation_body_blocked,
)
from grutopia_extension.interactive_navigation.visualization import following_camera_world_pose,_quaternion_matrix


class VoronoiNavigationTests(unittest.TestCase):
    def planner(self):
        free=np.ones((121,241),bool);free[:25]=False;free[96:]=False
        return SidePushPlanner(free,[[-3,3],[-1.5,1.5]],robot_radius=.15)

    def test_actual_route_uses_exploration_skeleton_instead_of_parallel_shortcut(self):
        p=self.planner();routes=ExplorationVoronoiRoutes(p)
        path=routes.plan((-2,.30),(2,.30),[])
        self.assertTrue(path)
        self.assertEqual(routes.last_plan['engine'],'exploration_voronoi')
        self.assertGreater(len(path),8)
        self.assertTrue(any(abs(point[1])<.06 for point in path))
        np.testing.assert_allclose(path[-1],(2,.30,0.))

    def test_moved_box_invalidates_cached_route_and_cannot_be_crossed(self):
        p=self.planner();routes=ExplorationVoronoiRoutes(p)
        self.assertTrue(routes.plan((-2,0),(2,0),[]))
        box=BoxState('blocker',(0,0),(.6,1.6,.7))
        self.assertFalse(routes.plan((-2,0),(2,0),[box]))
        self.assertEqual(routes.graph_builds,2)
        self.assertTrue(routes.plan((-2,0),(2,0),[]))
        self.assertEqual(routes.graph_builds,2)

    def test_nearest_disconnected_spurs_do_not_force_a_grid_fallback(self):
        p=self.planner();routes=ExplorationVoronoiRoutes(p)
        start=(-2,.30);goal=(2,.30)
        routes.plan(start,goal,[])
        engine,_=next(iter(routes._cache.values()))
        snapshot=engine.graph.cached_snapshot()
        cells=set(snapshot.skeleton_cells)|{p.cell(start),p.cell(goal)}
        engine.graph._snapshot=replace(snapshot,skeleton_cells=tuple(cells))
        self.assertTrue(routes.plan(start,goal,[]))
        self.assertEqual(routes.last_plan['engine'],'exploration_voronoi')
        self.assertTrue(routes.last_plan['shared_component_attachment'])

    def test_navigation_turns_to_a_sideways_route_before_forward_walking(self):
        p=SidePushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]])
        tracker=ForwardExplorationPathTracker((0,0),[(0,1)])
        action,state=tracker.action((0,0),0.,p,[])
        self.assertFalse(state['needs_replan'])
        self.assertEqual(action['move_by_speed'][:2],[0.,0.])
        self.assertGreater(action['move_by_speed'][2],0.)
        action,state=tracker.action((0,0),math.pi/2,p,[])
        self.assertGreater(action['move_by_speed'][0],.3)
        self.assertAlmostEqual(action['move_by_speed'][1],0.,places=7)

    def test_unsafe_body_rotation_cannot_be_commanded_by_forward_tracker(self):
        free=np.zeros((201,201),bool);free[90:111]=True
        p=SidePushPlanner(free,[[-2.5,2.5],[-2.5,2.5]])
        tracker=ForwardExplorationPathTracker((0,0),[(0,.02)])
        action,state=tracker.action((0,0),0.,p,[])
        self.assertEqual(action['move_by_speed'],[0.,0.,0.])
        self.assertTrue(state['blocked_motion_sweep'])

    def test_robot_yaw_does_not_change_global_voronoi_route(self):
        p=self.planner();routes=ExplorationVoronoiRoutes(p)
        p.robot_heading=0.;first=routes.plan((-2,.30),(2,.30),[])
        p.robot_heading=math.pi/2;second=routes.plan((-2,.30),(2,.30),[])
        self.assertEqual(first,second)
        self.assertEqual(routes.graph_builds,1)
        self.assertEqual(p.robot_heading,math.pi/2)

    def test_clearance_recovery_uses_the_validated_gait_speed(self):
        free=np.ones((101,101),bool);free[:,:20]=False
        p=SidePushPlanner(free,[[-1.25,1.25],[-1.25,1.25]])
        tracker=ForwardExplorationPathTracker((-.60,0),[(.2,0)])
        action,state=tracker.action((-.60,0),math.pi/2,p,[])
        self.assertTrue(state['recovering_buffer'])
        self.assertAlmostEqual(np.linalg.norm(action['move_by_speed'][:2]),.25)
        self.assertEqual(action['move_by_speed'][2],0.)

    def test_route_buffer_is_recovered_even_when_body_footprint_is_free(self):
        free=np.ones((101,101),bool);free[:,:20]=False
        p=SidePushPlanner(free,[[-1.25,1.25],[-1.25,1.25]])
        p.robot_heading=math.pi/2;position=(-.52,0.)
        self.assertFalse(p.blocked([])[p.cell(position)])
        action,state=ForwardExplorationPathTracker(position,[(.2,0)]).action(position,math.pi/2,p,[])
        self.assertTrue(state['recovering_buffer'])
        self.assertAlmostEqual(np.linalg.norm(action['move_by_speed'][:2]),.25)

    def test_turn_correction_preserves_the_checked_world_route_direction(self):
        p=SidePushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]])
        heading=.2
        action,state=ForwardExplorationPathTracker((0,0),[(1,0)]).action((0,0),heading,p,[])
        vx,vy,_=action['move_by_speed']
        self.assertGreater(vx,.3)
        self.assertLess(vy,0.)
        self.assertAlmostEqual(math.sin(heading)*vx+math.cos(heading)*vy,0.,places=6)

    def test_buffer_exit_keeps_walking_after_one_free_cell_until_clearance_is_restored(self):
        free=np.ones((101,101),bool);free[:,:20]=False
        p=SidePushPlanner(free,[[-1.25,1.25],[-1.25,1.25]])
        tracker=ForwardExplorationPathTracker((-.52,0),[(.2,0)])
        tracker.action((-.52,0),math.pi/2,p,[])
        action,state=tracker.action((-.48,0),math.pi/2,p,[])
        self.assertTrue(state['recovering_buffer'])
        self.assertAlmostEqual(np.linalg.norm(action['move_by_speed'][:2]),.25)
        action,state=tracker.action((-.32,0),math.pi/2,p,[])
        self.assertTrue(state['buffer_cleared_replan'])
        self.assertEqual(action['move_by_speed'],[0.,0.,0.])

    def test_in_place_rotation_requests_ramp_up_without_a_full_speed_jump(self):
        p=SidePushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]])
        tracker=ForwardExplorationPathTracker((0,0),[(0,1)])
        rotations=[abs(tracker.action((0,0),0.,p,[])[0]['move_by_speed'][2]) for _ in range(10)]
        self.assertLess(rotations[0],.5)
        self.assertGreater(rotations[-1],rotations[0])
        self.assertLess(rotations[-1],.7)

    def test_pure_mode_uses_the_original_carrot_from_the_actual_pose(self):
        p=SidePushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]])
        tracker=ForwardExplorationPathTracker((0,0),[(.3,0),(.6,0),(.9,0)],guard_motion=False)
        _,state=tracker.action((0,.1),0.,p,[])
        expected=.3+.55-math.hypot(.3,.1)
        np.testing.assert_allclose(state['target'],(expected,0.))
        self.assertEqual(state['controller'],'pure_exploration_velocity_action_for')

    def test_pure_mode_yaw_tracks_the_route_instead_of_an_initial_heading(self):
        p=SidePushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]])
        action,state=ForwardExplorationPathTracker((0,0),[(0,1)],guard_motion=False).action((0,0),0.,p,[])
        self.assertGreater(action['move_by_speed'][2],0.)
        self.assertIsNone(state['planned_heading'])

    def test_route_clearance_can_increase_without_changing_box_parking_radius(self):
        p=SidePushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]])
        box=BoxState('B',(0,0),(.6,.4,.7))
        point=(.5625,0.)
        self.assertFalse(route_blocked(p,[box])[p.cell(point)])
        p.navigation_route_radius=.30
        self.assertTrue(route_blocked(p,[box])[p.cell(point)])
        self.assertEqual(p.radius,.25)

    def test_navigation_footprint_protects_front_paws_beyond_the_torso(self):
        free=np.ones((201,201),bool);free[:,118:]=False
        p=SidePushPlanner(free,[[-2.5,2.5],[-2.5,2.5]])
        p.robot_heading=0.
        self.assertFalse(p.blocked([])[p.cell((0,0))])
        self.assertTrue(navigation_body_blocked(p,[],0.)[p.cell((0,0))])

    def test_pure_tracker_can_make_a_safe_axis_step_to_gain_turn_clearance(self):
        free=np.ones((201,201),bool)
        free[:101,:84]=False
        p=SidePushPlanner(free,[[-2.5,2.5],[-2.5,2.5]])
        tracker=ForwardExplorationPathTracker((.12,0),[(-.4,.4)],guard_motion=False)
        action,state=tracker.action((.12,0),math.pi,p,[])
        self.assertTrue(state['foot_clearance_axis_step'])
        self.assertAlmostEqual(action['move_by_speed'][0],0.)
        self.assertLess(action['move_by_speed'][1],0.)

    def test_pure_carrot_shortens_when_the_long_chord_is_blocked(self):
        p=SidePushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]])
        tracker=ForwardExplorationPathTracker((0,0),[(1,0)],guard_motion=False)
        tracker._tracking_segment_clear=lambda start,target:np.linalg.norm(np.asarray(target)-start)<=.11
        action,state=tracker.action((0,0),0.,p,[])
        self.assertFalse(state['needs_replan'])
        np.testing.assert_allclose(state['target'],(.10,0.))
        self.assertGreater(action['move_by_speed'][0],0.)
        self.assertEqual(tracker.velocity_lookahead_distance,.55)

    def test_footprint_roadmap_heading_stays_stable_while_control_yaw_changes(self):
        p=self.planner();p.navigation_pose_heading=0.
        routes=ExplorationVoronoiRoutes(p)
        p.robot_heading=0.;first=routes.plan((-2,0),(2,0),[])
        p.robot_heading=1.;second=routes.plan((-2,0),(2,0),[])
        self.assertTrue(first)
        self.assertEqual(first,second)
        self.assertEqual(routes.graph_builds,1)

    def test_narrow_lane_recovers_feasible_heading_before_translation(self):
        free=np.zeros((201,201),bool);free[90:111]=True
        p=SidePushPlanner(free,[[-2.5,2.5],[-2.5,2.5]])
        p.navigation_pose_heading=0.
        tracker=ForwardExplorationPathTracker((0,0),[(1,0)],guard_motion=False)
        action,state=tracker.action((0,0),math.pi/4,p,[])
        self.assertEqual(state['controller'],'footprint_heading_recovery')
        self.assertEqual(action['move_by_speed'][:2],[0.,0.])
        self.assertLess(action['move_by_speed'][2],0.)
        json.dumps(state)

    def test_heading_recovery_metadata_serializes_numpy_positions(self):
        free=np.zeros((201,201),bool);free[90:111]=True
        p=SidePushPlanner(free,[[-2.5,2.5],[-2.5,2.5]]);p.navigation_pose_heading=0.
        tracker=ForwardExplorationPathTracker((0,0),[(1,0)],guard_motion=False)
        _,state=tracker.action(np.array([0.,0.],dtype=np.float32),math.pi/4,p,[])
        json.dumps(state)

    def test_turning_speed_has_no_jump_at_small_heading_errors(self):
        p=SidePushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]])
        speeds=[]
        for heading in [-.249,-.251]:
            action,state=ForwardExplorationPathTracker((0,0),[(1,0)]).action((0,0),heading,p,[])
            self.assertFalse(state['needs_replan'])
            speeds.append(action['move_by_speed'][0])
        self.assertGreater(min(speeds),.35)
        self.assertLess(abs(speeds[0]-speeds[1]),.002)

    def test_camera_follows_actual_dog_yaw_and_forward_projects_up(self):
        class Camera:
            def update_pose(self,p,q):self.q=np.asarray(q)
        first,third=Camera(),Camera()
        for heading in [0.,math.pi/2,math.pi,-math.pi/2]:
            q=np.array([math.cos(heading/2),0,0,math.sin(heading/2)])
            update_task_cameras(first,third,(0,0,.41),q)
            np.testing.assert_allclose(third.q,q)
            _,orientation=following_camera_world_pose((0,0,.41),heading,(0,0,2.5),90.)
            camera_up=_quaternion_matrix(orientation)[:,2]
            np.testing.assert_allclose(camera_up,(math.cos(heading),math.sin(heading),0),atol=1e-7)


if __name__=='__main__':unittest.main()
