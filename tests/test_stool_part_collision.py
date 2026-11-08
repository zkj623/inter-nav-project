from pathlib import Path
import tempfile
import unittest
import shutil

import fcl
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from grutopia_extension.manipulation.bimanual_collision import RobotCollisionModel


class StoolPartCollisionTests(unittest.TestCase):
    size=(.34,.42,.28)

    def model(self):
        model=RobotCollisionModel.__new__(RobotCollisionModel)
        model.fcl=fcl
        model.names=[f'{side}_gripper_finger_link{i}' for side in ('left','right') for i in (1,2)]+['torso_link']
        model.noncontact_names=['torso_link']
        model.envelopes={name:fcl.Box(.01,.01,.01) for name in model.names}
        return model

    def poses(self,model,position):
        poses={name:np.eye(4) for name in model.names}
        for pose in poses.values():pose[:3,3]=position
        box=np.eye(4);box[2,3]=self.size[2]/2
        return poses,box

    def stage(self,folder):
        # A byte-preserved actual SDK-authored asset; core USD can read its
        # geometry without requiring the SDK-only PhysxSchema authoring plugin.
        asset=Path(folder)/'stool.usd'
        shutil.copy2(Path(__file__).parent/'fixtures/native_stool_small.usd',asset)
        stage=Usd.Stage.CreateInMemory()
        stage.DefinePrim('/World/LiftObject','Xform').GetReferences().AddReference(str(asset))
        return stage

    def test_empty_underseat_space_is_not_a_finger_collision(self):
        model=self.model();poses,box=self.poses(model,(0.,.195,.20))
        old=model.box_clearances(poses,box,self.size,include_fingers=True)
        self.assertLess(min(old.values()),-.002)
        self.assertGreater(min(model.stool_finger_clearances(poses,box,self.size).values()),.014)

    def test_seat_penetration_still_exceeds_same_two_mm_limit(self):
        model=self.model();poses,box=self.poses(model,(0.,.195,.224))
        self.assertLess(min(model.stool_finger_clearances(poses,box,self.size).values()),-.002)

    def test_leg_penetration_is_not_hidden_by_seat_only_geometry(self):
        model=self.model();poses,box=self.poses(model,(.115,.155,.10))
        self.assertLess(min(model.stool_finger_clearances(poses,box,self.size).values()),-.002)

    def test_rotated_and_translated_stool_uses_its_actual_pose(self):
        model=self.model();poses,box=self.poses(model,(0.,.195,.20))
        expected=model.stool_finger_clearances(poses,box,self.size)
        world=np.eye(4);world[:3,:3]=Rotation.from_euler('xyz',(.2,-.1,1.3)).as_matrix();world[:3,3]=(.8,-.6,.22)
        actual=model.stool_finger_clearances({n:world@p for n,p in poses.items()},world@box,self.size)
        np.testing.assert_allclose(list(actual.values()),list(expected.values()),atol=1e-6)

    def test_missing_finger_is_rejected(self):
        model=self.model();poses,box=self.poses(model,(0.,.195,.20));model.names=model.names[1:]
        with self.assertRaisesRegex(ValueError,'all four fingers'):
            model.stool_finger_clearances(poses,box,self.size)

    def test_body_keeps_full_envelope_even_in_underseat_void(self):
        model=self.model();poses,box=self.poses(model,(0.,.195,.20))
        before=model.box_clearances(poses,box,self.size)
        model.stool_finger_clearances(poses,box,self.size)
        self.assertEqual(before,model.box_clearances(poses,box,self.size))
        self.assertLess(before['torso_link'],-.002)

    def test_native_export_matches_nominal_five_parts(self):
        model=self.model();poses,box=self.poses(model,(0.,.195,.20))
        expected=model.stool_finger_clearances(poses,box,self.size)
        with tempfile.TemporaryDirectory() as folder:
            proof=model.verify_stool_stage(self.stage(folder),self.size)
        self.assertEqual(proof['parts'],['Leg0','Leg1','Leg2','Leg3','Seat'])
        np.testing.assert_allclose(list(model.stool_finger_clearances(poses,box,self.size).values()),list(expected.values()),atol=1e-6)

    def test_disabled_collider_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            stage=self.stage(folder)
            UsdPhysics.CollisionAPI(stage.GetPrimAtPath('/World/LiftObject/Leg0')).GetCollisionEnabledAttr().Set(False)
            with self.assertRaisesRegex(ValueError,'disabled'):
                self.model().verify_stool_stage(stage,self.size)

    def test_tampered_native_vertices_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            stage=self.stage(folder);mesh=UsdGeom.Mesh(stage.GetPrimAtPath('/World/LiftObject/Seat'))
            points=list(mesh.GetPointsAttr().Get());points[0]=Gf.Vec3f(points[0][0]+.002,points[0][1],points[0][2])
            mesh.GetPointsAttr().Set(points)
            with self.assertRaisesRegex(ValueError,'differs'):
                self.model().verify_stool_stage(stage,self.size)

    def test_missing_leg_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            stage=self.stage(folder);stage.GetPrimAtPath('/World/LiftObject/Leg0').SetActive(False)
            with self.assertRaisesRegex(ValueError,'coverage'):
                self.model().verify_stool_stage(stage,self.size)


if __name__=='__main__':unittest.main()
