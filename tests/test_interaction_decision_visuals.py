import json,unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.layered_navigation_map import LayeredNavigationMap
from grutopia_extension.interactive_navigation.mapping import MappingConfig,FusedMap
from grutopia_extension.interactive_navigation.task_video_recording import TaskVideoRecorder
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import navigation_body_blocked,ForwardExplorationPathTracker


class DecisionVisualTests(unittest.TestCase):
    def test_blocked_route_explanation_accounts_for_push_cost_without_infinite_savings(self):
        p=SidePushPlanner(np.ones((161,161),bool),[[-2,2],[-2,2]],
            passages={'box':{'axis':[0,1],'center':[0,0]}},robot_radius=.30)
        box=BoxState('box',(0,0),(.85,.55,.95),0,6)
        d=p.decide((0,-1.5),(0,.1),[box]);details=d.explanation
        self.assertEqual(d.kind,'push');self.assertTrue(details['opens_connectivity'])
        self.assertIsNone(details['before_length_m']);self.assertIsNone(details['path_saving_m'])
        self.assertGreater(details['after_length_m'],0)
        self.assertAlmostEqual(details['weighted_push_cost'],2.5*d.push.push_distance)
        self.assertEqual(details['box_id'],d.push.box_id)
        json.dumps(details,allow_nan=False)

    def test_rendered_map_changes_at_old_box_location_after_move(self):
        p=SidePushPlanner(np.ones((161,161),bool),[[-2,2],[-2,2]])
        cfg=MappingConfig(x_limits=(-2,2),y_limits=(-2,2),grid_resolution=.05)
        layers=LayeredNavigationMap(p,cfg);box=BoxState('B01',(0,0),(.6,.6,.5))
        obs={'position':[0,-1,.41],'orientation':[1,0,0,0],'sensors':{}}
        layers.update(0,obs,[box],force=True)
        runtime=SimpleNamespace(map=FusedMap(cfg),navigation_layers=layers,_planned_path=())
        recorder=TaskVideoRecorder.__new__(TaskVideoRecorder)
        recorder.task={'id':'T01','start':[0,-1],'goal':[0,1]};recorder._trajectory=[]
        interaction_observation=SimpleNamespace(robot_position=tuple(obs['position']))
        before=recorder._render_map(0,'before',interaction_observation,runtime)
        x,y=recorder._world_to_pixel((0,0),runtime.map)
        layers.update(.1,obs,[replace(box,xy=(1,0))]);layers.refresh_global()
        after=recorder._render_map(1,'after',interaction_observation,runtime)
        self.assertEqual(after.shape,(720,1280,3))
        self.assertFalse(np.array_equal(before[y-5:y+5,x-5:x+5],after[y-5:y+5,x-5:x+5]))

    def test_navigation_footprint_includes_wider_foot_margin_when_configured(self):
        free=np.ones((201,201),bool);free[110:]=False
        p=SidePushPlanner(free,[[-2.5,2.5],[-2.5,2.5]])
        p.navigation_half_width=.22
        narrow=navigation_body_blocked(p,[],0.)
        p.navigation_half_width=.25
        wide=navigation_body_blocked(p,[],0.)
        self.assertGreater(wide.sum(),narrow.sum())

    def test_recovery_releases_a_free_pose_without_forcing_old_world_heading(self):
        p=SidePushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]])
        p.navigation_pose_heading=0.
        tracker=ForwardExplorationPathTracker((0,0),[(.7,-.7)],guard_motion=False)
        tracker._heading_recovery=True
        action,state=tracker.action((0,0),-.7,p,[])
        self.assertFalse(tracker._heading_recovery)
        self.assertNotEqual(state['controller'],'footprint_heading_recovery')
        self.assertGreater(action['move_by_speed'][0],0.)

    def test_global_halo_is_not_a_local_collision_when_actual_body_fits(self):
        free=np.ones((101,101),bool);free[:,:20]=False
        p=SidePushPlanner(free,[[-1.25,1.25],[-1.25,1.25]])
        p.navigation_route_radius=.30;p.layered_local_control=True;p.full_navigation_footprint=True
        position=(-.52,0.)
        self.assertFalse(navigation_body_blocked(p,[],np.pi/2)[p.cell(position)])
        # Walk along the free wall corridor. Turning toward the east here
        # requires extra rear-foot clearance and is a separate maneuver.
        tracker=ForwardExplorationPathTracker(position,[(-.52,.7)],guard_motion=False)
        action,state=tracker.action(position,np.pi/2,p,[])
        self.assertFalse(state['recovering_buffer'])
        self.assertGreater(np.linalg.norm(action['move_by_speed'][:2]),0.)


if __name__=='__main__':unittest.main()
