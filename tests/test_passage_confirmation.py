import unittest
from contextlib import contextmanager
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.go2_passage_confirmation import observed_passage_routes


class Layers:
    def __init__(self,planner,boxes):
        self.planner=planner;self.boxes=boxes
    @contextmanager
    def global_planning(self):
        p=self.planner;scope=p.planning_scope;p.planning_scope='global'
        try:yield self.boxes
        finally:p.planning_scope=scope


class PassageConfirmationTests(unittest.TestCase):
    def make(self):
        r,c=np.indices((120,120));x=-3+(c+.5)*.05;y=-3+(r+.5)*.05
        free=(y<.5)|((x>-.6)&(x<.6))
        p=SidePushPlanner(free,[[-3,3],[-3,3]],resolution=.05)
        p.navigation_half_width=.22;p.navigation_route_radius=.3
        p.planning_scope='local';p.planning_observed=np.ones_like(free)
        p.passages={'box':{'center':[0.,0.],'axis':[.0791163,.99686539]}}
        box=BoxState('box',(-.95,0),(.7,.6,.7))
        return p,box,Layers(p,[box])
    def test_valid_connected_body_corridor_can_differ_from_the_first_grid_route(self):
        p,b,layers=self.make();proof,route,full=observed_passage_routes(layers,b,[2,-1],[0,1.5],[1,-.2],[b])
        self.assertIsNotNone(proof);self.assertTrue(route);self.assertTrue(full)
        self.assertEqual(proof.get('confirmation_source'),'connected_observed_corridor')
        self.assertTrue(proof['side_parked'])
        self.assertIsNotNone(p.certify_cleared_passage(b,full,[2,-1],[b]))
    def test_wall_cannot_be_removed_to_make_a_certificate(self):
        p,b,layers=self.make();r,c=np.indices(p.free.shape);y=-3+(r+.5)*.05
        p.free[(y>.15)&(y<.3)]=False;before=p.free.copy()
        proof,_,_=observed_passage_routes(layers,b,[2,-1],[0,1.5],[1,-.2],[b])
        self.assertIsNone(proof);np.testing.assert_array_equal(p.free,before)
    def test_unknown_corridor_remains_uncertified(self):
        p,b,layers=self.make();r,c=np.indices(p.free.shape);y=-3+(r+.5)*.05
        band=(y>.15)&(y<.3);p.free[band]=False;p.planning_observed[band]=False
        proof,_,_=observed_passage_routes(layers,b,[2,-1],[0,1.5],[1,-.2],[b])
        self.assertIsNone(proof)

if __name__=='__main__':unittest.main()
