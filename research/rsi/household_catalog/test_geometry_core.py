import unittest,itertools
import numpy as np
from scipy.spatial.transform import Rotation
from cooked_geometry import ConvexPart,CookedGeometry,transform_points,metric_bounds_diagnostics,grasp_target_errors

def box(center,size):
    return np.array(list(itertools.product(*[(-s/2,s/2) for s in size])))+center

class GeometryTests(unittest.TestCase):
    def test_hollow_cavity_remains_empty(self):
        walls=[ConvexPart(box(np.array(c),s),str(i)) for i,(c,s) in enumerate([
            ((.1,0,0),(.02,.22,.2)),((-.1,0,0),(.02,.22,.2)),
            ((0,.1,0),(.18,.02,.2)),((0,-.1,0),(.18,.02,.2))])]
        g=CookedGeometry(walls);finger=ConvexPart(box(np.zeros(3),(.01,.01,.01)),'finger')
        self.assertGreater(g.gap(finger.shape,finger.points,np.eye(4)),.08)
    def test_true_penetration_and_threshold(self):
        g=CookedGeometry([ConvexPart(box(np.zeros(3),(.2,.2,.2)),'solid')])
        finger=ConvexPart(box(np.zeros(3),(.02,.02,.02)),'finger')
        for depth in (.001,.003):
            pose=np.eye(4);pose[1,3]=.11-depth
            gap=g.gap(finger.shape,finger.points,pose)
            self.assertAlmostEqual(gap,-depth,places=6)
            self.assertEqual(gap<-.002,depth>.002)
    def test_rotation_and_translation(self):
        part=ConvexPart(box(np.zeros(3),(.2,.2,.2)),'solid');g=CookedGeometry([part])
        pose=np.eye(4);pose[1,3]=.15
        world=np.eye(4);world[:3,:3]=Rotation.from_euler('xyz',[.4,-.2,.7]).as_matrix();world[:3,3]=[3,-4,2]
        finger=ConvexPart(box(np.zeros(3),(.02,.02,.02)),'finger')
        self.assertAlmostEqual(g.gap(finger.shape,finger.points,np.linalg.inv(world)@(world@pose)),.04,places=7)
    def test_metric_transform(self):
        transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('x',90,degrees=True).as_matrix()*.01
        transformed=transform_points([[0,0,0],[100,100,100]],transform)
        np.testing.assert_allclose(transformed[1],[1,-1,1],atol=1e-12)
    def test_handle_does_not_define_body_grip(self):
        body=ConvexPart(box(np.array([0,0,.15]),(.3,.2,.3)),'body')
        handle=ConvexPart(box(np.array([0,.15,.29]),(.05,.1,.02)),'handle')
        g=CookedGeometry([body,handle]);p=g.grasp_profile(.15)
        self.assertAlmostEqual(p['half_width_m'],.1)
        self.assertAlmostEqual(p['center_y_m'],0.)
        self.assertGreater(g.bounds[1,1],p['left_surface_m'])
    def test_missing_geometry_is_rejected(self):
        with self.assertRaises(RuntimeError):CookedGeometry([])
    def test_cooking_expansion_is_not_a_unit_conversion_error(self):
        bounds=[[-.103,-.103,-.003],[.103,.103,.203]]
        d=metric_bounds_diagnostics(bounds,[.2,.2,.2])
        self.assertTrue(d['valid_metric_transform'])
        self.assertTrue(d['cooking_expansion_above_3mm'])
        self.assertFalse(metric_bounds_diagnostics(np.asarray(bounds)*100,[.2,.2,.2])['valid_metric_transform'])
    def test_empty_patch_is_rejected(self):
        g=CookedGeometry([ConvexPart(box(np.array([0,0,1]),(.1,.1,.1)),'tiny')])
        with self.assertRaises(RuntimeError):g.grasp_profile(.2)
    def test_normal_grasp_reserve_is_not_weakened(self):
        desired=np.zeros((2,3));normal=[0,1,0]
        self.assertTrue(grasp_target_errors([[.0005,0,0],[0,0,.0005]],desired,normal,1e-3/3)['passed'])
        self.assertFalse(grasp_target_errors([[0,.0005,0],[0,0,0]],desired,normal,1e-3/3)['passed'])
        self.assertFalse(grasp_target_errors([[.002,0,0],[0,0,0]],desired,normal,1e-3/3)['passed'])

if __name__=='__main__':unittest.main(verbosity=2)
