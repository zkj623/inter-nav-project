import unittest
import numpy as np
from grutopia_extension.interactive_navigation.go2_contact_control import (
    contact_axis_delta,contact_braking_required,contact_centering_velocity,ContactAlignmentGate)


class ContactAxisBrakingTests(unittest.TestCase):
    def test_forward_standoff_error_cannot_hide_lateral_stopping_distance(self):
        delta=np.array([.20,.06]);velocity=np.array([0.,.20])
        self.assertFalse(contact_braking_required(delta,velocity,horizon=.4))
        self.assertTrue(contact_braking_required(contact_axis_delta(delta,[1,0],'lateral'),velocity,horizon=.4))
    def test_command_respects_the_alignment_gates_selected_axis(self):
        command=contact_centering_velocity([.20,.05],[1,0],.25,axis='longitudinal')
        np.testing.assert_allclose(command,[.25,0.])
        np.testing.assert_allclose(contact_centering_velocity([.20,.05],[1,0],.25,axis='lateral'),[0.,.35])
    def test_settle_waits_for_motion_but_does_not_interrupt_an_active_correction(self):
        gate=ContactAlignmentGate()
        self.assertTrue(gate.settling_required('lateral',0,measured_speed=.3))
        self.assertTrue(gate.settling_required('lateral',101,measured_speed=.15))
        self.assertFalse(gate.settling_required('lateral',120,measured_speed=.02))
        self.assertFalse(gate.settling_required('lateral',121,measured_speed=.3))
        self.assertTrue(gate.settling_required('yaw',122,measured_speed=.3))

if __name__=='__main__':unittest.main()
