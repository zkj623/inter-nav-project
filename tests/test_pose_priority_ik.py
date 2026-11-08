from types import SimpleNamespace
import unittest
import numpy as np
from grutopia_extension.manipulation.bimanual_floor_lift import BimanualUrdfKinematics


class PosePriorityTests(unittest.TestCase):
    def solve(self, **kwargs):
        model=BimanualUrdfKinematics.__new__(BimanualUrdfKinematics)
        model.lower=-np.ones(15);model.upper=np.ones(15);model.names=[f'q{i}' for i in range(15)]
        def forward(q):
            tool=np.eye(4);tool[0,3]=q[0]
            return tool,tool.copy()
        model.forward=forward
        collision=SimpleNamespace(forward=lambda q:q,
            box_clearances=lambda q,*a:{'body':.045-q['q0']})
        return model.solve((.004,0.,0.),(.004,0.,0.),np.eye(3),np.eye(3),np.zeros(15),
            collision,np.eye(4),(.3,.3,.3),clearance=.05,**kwargs)

    def test_explicit_default_keeps_original_solution(self):
        np.testing.assert_array_equal(self.solve(),self.solve(pose_weights=(6.,.6)))

    def test_precise_pose_resolves_soft_margin_conflict_more_accurately(self):
        ordinary=self.solve();precise=self.solve(pose_weights=(30.,3.))
        self.assertGreater(abs(ordinary[0]-.004),.005)
        self.assertLess(abs(precise[0]-.004),.001)
        self.assertTrue(np.all(np.abs(precise)<1.))

    def test_invalid_weights_are_rejected(self):
        for weights in ((0.,1.),(-1.,1.),(1.,float('nan')),(1.,float('inf')),(1.,),1.):
            with self.subTest(weights=weights),self.assertRaisesRegex(ValueError,'Pose residual weights'):
                self.solve(pose_weights=weights)


if __name__=='__main__':unittest.main()
