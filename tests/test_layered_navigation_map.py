import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np

from grutopia_extension.interactive_navigation.mapping import MappingConfig,FusedMap
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.layered_navigation_map import LayeredNavigationMap


class LayeredMapTests(unittest.TestCase):
    def make(self,free=None,**kwargs):
        p=SidePushPlanner(np.ones((201,201),bool) if free is None else free,
            [[-2.5,2.5],[-2.5,2.5]],robot_radius=.10)
        cfg=MappingConfig(x_limits=(-2.5,2.5),y_limits=(-2.5,2.5),grid_resolution=.05)
        m=LayeredNavigationMap(p,cfg,**kwargs)
        return p,m,cfg

    def observation(self,xy=(0,0)):
        return {'position':[*xy,.41],'sensors':{}}

    def test_moved_box_vacates_old_cells_but_not_wall_or_overlapping_box(self):
        free=np.ones((201,201),bool);free[100,100]=False
        p,m,_=self.make(free)
        a=BoxState('A',(0,0),(.5,.5,.5));b=BoxState('B',(.20,0),(.20,.20,.5))
        m.update(0,self.observation(),[a,b],force=True)
        m.update(.1,self.observation(),[replace(a,xy=(1,0)),b])
        self.assertFalse(m.local_occupied[p.cell((-.15,0))])
        self.assertTrue(m.local_occupied[p.cell((.2,0))])
        self.assertTrue(m.local_occupied[100,100])
        self.assertTrue(m.statistics()['immutable_prior_preserved'])

    def test_local_move_precedes_periodic_global_refresh_then_event_forces_it(self):
        p,m,_=self.make()
        a=BoxState('A',(0,0),(.4,.4,.5));m.update(0,self.observation(),[a],force=True)
        moved=replace(a,xy=(1,0));m.update(.1,self.observation(),[moved])
        self.assertTrue(m.global_occupied[p.cell((0,0))])
        self.assertFalse(m.local_occupied[p.cell((0,0))])
        rev=m.global_revision;m.refresh_global([moved],reason='passage_cleared')
        self.assertFalse(m.global_occupied[p.cell((0,0))])
        self.assertGreater(m.global_revision,rev)

    def test_new_local_obstacle_blocks_motion_without_rebuilding_global_map(self):
        p,m,_=self.make();m.update(0,self.observation(),[],force=True)
        hit=np.zeros_like(p.free);hit[p.cell((.8,0))]=True
        m.ingest_evidence(.1,hit,np.zeros_like(hit));m._bind_local()
        self.assertTrue(m.local_occupied[p.cell((.8,0))])
        self.assertFalse(m.global_occupied[p.cell((.8,0))])
        self.assertFalse(p.free[p.cell((.8,0))])
        with m.global_planning():self.assertTrue(p.free[p.cell((.8,0))])
        self.assertFalse(p.free[p.cell((.8,0))])

    def test_ray_free_evidence_clears_dynamic_hit_without_erasing_static(self):
        free=np.ones((201,201),bool);free[100,100]=False
        p,m,_=self.make(free);m.update(0,self.observation(),[],force=True)
        hit=np.zeros_like(p.free);hit[100,135]=True
        m.ingest_evidence(.1,hit,np.zeros_like(hit))
        m.ingest_evidence(.2,np.zeros_like(hit),np.ones_like(hit))
        self.assertFalse(m.local_occupied[100,135]);self.assertTrue(m.local_occupied[100,100])

    def test_transient_hit_expires_and_movable_box_is_never_promoted(self):
        p,m,_=self.make();box=BoxState('B',(0,0),(.5,.5,.5))
        m.update(0,self.observation(),[box],force=True)
        hit=np.zeros_like(p.free);hit[100,125]=True
        m.ingest_evidence(.1,hit,np.zeros_like(hit))
        m.update(1.5,self.observation(),[box]);self.assertFalse(m.local_dynamic[100,125])
        for t in [2.,3.,4.]:
            m.ingest_evidence(t,p.box_mask(box),np.zeros_like(hit));m.refresh_global([box])
        self.assertFalse(m.stable_obstacles.any())

    def test_opened_door_changes_actual_global_connectivity(self):
        free=np.ones((201,201),bool);free[95:106]=False;free[95:106,65:136]=True
        p,m,_=self.make(free)
        box=BoxState('B',(0,0),(1.75,.5,.5));m.update(0,self.observation((0,-1)),[box],force=True)
        with m.global_planning() as boxes:self.assertFalse(p.path((0,-1),(0,1),boxes))
        moved=replace(box,xy=(1.2,0))
        m.update(.1,self.observation((0,-1)),[moved]);m.refresh_global([moved])
        with m.global_planning() as boxes:self.assertTrue(p.path((0,-1),(0,1),boxes))

    def test_voxel_ghost_is_removed_only_in_vacated_nonstatic_region(self):
        p,m,cfg=self.make();a=BoxState('A',(0,0),(.5,.5,.5))
        m.update(0,self.observation(),[a],force=True)
        fused=FusedMap(cfg);fused.voxels.update(np.array([[0,0,.3],[1.5,1.5,.3]]),step=0,source='lidar')
        m.update(.1,self.observation(),[replace(a,xy=(1,0))])
        m.synchronize_display(SimpleNamespace(map=fused),1)
        self.assertEqual(len(fused.voxels.points()),1)
        np.testing.assert_allclose(fused.voxels.points()[0],[1.5,1.5,.3])

    def test_footprint_self_clear_preserves_wall_and_contacting_box(self):
        free=np.ones((201,201),bool);free[100,100]=False
        p,m,_=self.make(free);box=BoxState('B',(.30,0),(.2,.2,.5))
        m.update(0,self.observation(),[box],force=True)
        hit=np.zeros_like(p.free);hit[p.cell((-.1,.1))]=True;hit[p.cell((.8,0))]=True
        m.ingest_evidence(.1,hit,np.zeros_like(hit))
        self.assertFalse(m.local_occupied[p.cell((-.1,.1))])
        self.assertTrue(m.local_occupied[p.cell((.8,0))])
        self.assertTrue(m.local_occupied[100,100])
        self.assertTrue(m.local_occupied[p.cell((.30,0))])
        self.assertGreater(m.self_cleared_cells,0)

    def test_self_return_filter_does_not_remove_known_objects_or_mutate_sensor_data(self):
        free=np.ones((201,201),bool);free[100,100]=False
        p,m,_=self.make(free);box=BoxState('B',(.30,0),(.2,.2,.5))
        m.update(0,self.observation(),[box],force=True)
        points=np.array([[-.1,.1,.3],[.8,0,.3],[.30,0,.3],[0,0,.3]])
        observation={**self.observation(),'sensors':{'lidar':{'pointcloud':points}}}
        filtered=m.filter_robot_returns(observation,count=True)
        self.assertEqual(len(filtered['sensors']['lidar']['pointcloud']),3)
        self.assertEqual(len(observation['sensors']['lidar']['pointcloud']),4)
        self.assertEqual(m.filtered_self_points,1)


if __name__=='__main__':unittest.main()
