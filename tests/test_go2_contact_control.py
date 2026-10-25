import unittest
import numpy as np
from grutopia_extension.interactive_navigation.go2_contact_control import (
    contact_centering_velocity,contact_braking_required,contact_pose_ready)


class ContactControlTests(unittest.TestCase):
    def test_small_side_component_is_corrected_before_forward_contact(self):
        # The previous normalized .25 request gave only .024 m/s sideways.
        velocity=contact_centering_velocity((-.40,-.04),(-1,0),.25)
        np.testing.assert_allclose(velocity,(0,-.35))

    def test_centered_contact_advances_without_diagonal_motion(self):
        velocity=contact_centering_velocity((-.20,-.02),(-1,0),.25)
        np.testing.assert_allclose(velocity,(-.25,0))

    def test_reverse_and_rotated_contacts_keep_the_correct_world_axis(self):
        np.testing.assert_allclose(contact_centering_velocity((.20,0),(-1,0),.25),(.35,0))
        np.testing.assert_allclose(contact_centering_velocity((-.05,.20),(0,1),.25),(-.35,0))

    def test_braking_starts_before_the_old_half_second_threshold(self):
        self.assertTrue(contact_braking_required((-.18,0),(-.20,0)))
        self.assertFalse(contact_braking_required((-.18,0),(.20,0)))
        self.assertFalse(contact_braking_required((-.18,0),(0,0)))

    def test_safe_interval_accepts_the_original_standoff_after_braking(self):
        self.assertTrue(contact_pose_ready((4.316,0),np.pi,(3.3,0),(.85,.65,.7),(-1,0),.40,.08))
        self.assertFalse(contact_pose_ready((4.12,0),np.pi,(3.3,0),(.85,.65,.7),(-1,0),.40,.08))
        self.assertFalse(contact_pose_ready((4.316,.10),np.pi,(3.3,0),(.85,.65,.7),(-1,0),.40,.08))


if __name__=='__main__':unittest.main()
