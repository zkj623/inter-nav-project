import unittest
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState


class PassageCertificateTests(unittest.TestCase):
    def planner(self):
        p=SidePushPlanner(np.ones((201,201),bool),[[-2.5,2.5],[-2.5,2.5]],
            passages={'B':{'center':[0.,0.],'axis':[0.,1.]}})
        x=-2.5+(np.arange(201)+.5)*p.resolution;xx,yy=np.meshgrid(x,x)
        p.free[(abs(yy)<.10)&(abs(xx)>.50)]=False
        p.active_passages=p.passages.copy()
        p.planning_scope='global';p.navigation_route_radius=.30
        return p

    def test_shifted_corridor_can_be_open_before_nominal_center_is_clear(self):
        p=self.planner();box=BoxState('B',(.58,-.55),(.60,.50,.70))
        start=(-.12,-1.5);goal=(-.12,1.2)
        path=p.path(start,goal,[box])
        self.assertTrue(path)
        self.assertFalse(p.clearance(box)['side_parked'])
        certificate=p.certify_cleared_passage(box,path,start,[box])
        self.assertIsNotNone(certificate)
        self.assertTrue(certificate['side_parked'])
        self.assertLess(certificate['center'][0],-.02)
        self.assertEqual(certificate['body_margin_m'],.04)

    def test_another_box_still_blocks_the_full_body_corridor(self):
        p=self.planner();box=BoxState('B',(.58,-.55),(.60,.50,.70))
        blocker=BoxState('C',(-.12,.4),(.35,.25,.50))
        path=[(-.12,-.8),(-.12,1.2)]
        self.assertIsNone(p.certify_cleared_passage(box,path,(-.12,-1.5),[box,blocker]))

    def test_route_must_cross_the_actual_door_plane(self):
        p=self.planner();box=BoxState('B',(.58,-.55),(.60,.50,.70))
        self.assertIsNone(p.certify_cleared_passage(box,[(-.12,-.2)],(-.12,-1.5),[box]))


if __name__=='__main__':unittest.main()
