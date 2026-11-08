import unittest
from unittest.mock import Mock,patch
import numpy as np
from scipy.spatial.transform import Rotation
import test_individual_finger_calibration as fixtures
from grutopia_extension.interaction_clearance.primitives import SeatEdgeArmExecutePolicy,SeatEdgeArmPlanPolicy


class SupportCenteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.IndividualFingerCalibrationTests.setUpClass()

    def environment(self):return fixtures.IndividualFingerCalibrationTests().environment()

    def test_measured_support_behind_com_is_corrected_by_hand_translation(self):
        env,poses,measured,box=self.environment();policy=SeatEdgeArmExecutePolicy(env,'grasp')
        alignment=policy.support_alignment(poses,measured,box);error=alignment['support_x_error_m']
        self.assertLess(error,-.002);self.assertGreater(error,-.008)
        moved={n:p.copy() for n,p in poses.items()}
        for name in moved:
            if 'gripper_finger' in name:moved[name][:3,3]-=box[:3,0]*error
        corrected=policy.support_alignment(moved,measured,box)
        self.assertLess(abs(corrected['support_x_error_m']),.0001)

    def test_support_alignment_is_invariant_to_world_pose(self):
        env,poses,measured,box=self.environment();policy=SeatEdgeArmExecutePolicy(env,'grasp')
        expected=policy.support_alignment(poses,measured,box)['support_x_error_m']
        world=np.eye(4);world[:3,:3]=Rotation.from_euler('xyz',(.2,-.1,.7)).as_matrix();world[:3,3]=(.8,-.3,.2)
        actual=policy.support_alignment({n:world@p for n,p in poses.items()},measured,world@box)['support_x_error_m']
        self.assertAlmostEqual(actual,expected,places=5)

    def test_upper_finger_position_does_not_replace_lower_support_line(self):
        env,poses,measured,box=self.environment();policy=SeatEdgeArmExecutePolicy(env,'grasp')
        expected=policy.support_alignment(poses,measured,box)['support_x_error_m']
        for name in ('left_gripper_finger_link2','right_gripper_finger_link1'):poses[name][:3,3]+=box[:3,0]*.02
        self.assertAlmostEqual(policy.support_alignment(poses,measured,box)['support_x_error_m'],expected,places=6)

    def test_hand_offset_does_not_move_planned_collision_object(self):
        env,_,measured,_=self.environment();planner=SeatEdgeArmPlanPolicy(env);origin=env.object.get_world_pose()[0];rotation=env.rotation(env.object.get_world_pose()[1])
        env.recorder=None
        with patch.object(env.bimanual,'solve',return_value=np.zeros(15)) as solve:
            planner.solve_edge(origin,.20,measured[:15],rotation);first=solve.call_args
            env.edge_grasp_x_offset=.006;planner.solve_edge(origin,.20,measured[:15],rotation);second=solve.call_args
        np.testing.assert_array_equal(first.args[6],second.args[6])
        p,q=env.base.get_world_pose();expected=env.rotation(q).T@rotation@np.array((.006,0.,0.))
        for i in (0,1):np.testing.assert_allclose(second.args[i]-first.args[i],expected,atol=1e-10)

    def test_open_alignment_certifies_motion_and_updates_offset(self):
        env,poses,measured,box=self.environment();policy=SeatEdgeArmExecutePolicy(env,'grasp')
        policy.measured_calibration_state=Mock(return_value=(poses,measured,box))
        policy.support_alignment=Mock(side_effect=[{'support_x_error_m':-.004},{'support_x_error_m':0.}]);policy.interpolate=Mock()
        with patch.object(policy,'local_support_target',return_value=env.command.copy()), \
             patch.object(policy,'certify_support_translation',return_value=[{'minimum_noncontact_clearance_m':.04,'minimum_wall_clearance_m':.18}]) as certify:
            policy.center_support_open()
        self.assertAlmostEqual(env.edge_grasp_x_offset,.004);certify.assert_called_once();policy.interpolate.assert_called_once()
        self.assertEqual(policy.interpolate.call_args.args[1:],(.8,'approach'))

    def test_alignment_rejects_excessive_offset_before_motion(self):
        env,poses,measured,box=self.environment();env.edge_grasp_x_offset=.019;policy=SeatEdgeArmExecutePolicy(env,'grasp')
        policy.measured_calibration_state=Mock(return_value=(poses,measured,box));policy.support_alignment=Mock(return_value={'support_x_error_m':-.004});policy.interpolate=Mock()
        with self.assertRaisesRegex(RuntimeError,'BOUNDED_REACH'):policy.center_support_open()
        policy.interpolate.assert_not_called()

    def test_loaded_support_must_be_centered_after_bounded_recalibration(self):
        env,poses,measured,box=self.environment();policy=SeatEdgeArmExecutePolicy(env,'grasp')
        policy.support_target_x=Mock(return_value=0.)  # Isolate the zero-bias convergence protocol.
        policy.center_support_open=Mock();policy.calibrate_contact=Mock();policy.reopen_for_alignment=Mock()
        policy.measured_calibration_state=Mock(return_value=(poses,measured,box));policy.support_alignment=Mock(return_value={'support_x_error_m':.002})
        with self.assertRaisesRegex(RuntimeError,'GRASP_SUPPORT_NOT_CENTERED'):policy.prepare_centered_grasp()
        self.assertEqual(policy.calibrate_contact.call_count,3);self.assertEqual(policy.reopen_for_alignment.call_count,2)

    def test_loaded_error_updates_open_target_without_changing_preload(self):
        env,poses,measured,box=self.environment();policy=SeatEdgeArmExecutePolicy(env,'grasp')
        policy.support_target_x=Mock(return_value=0.)
        policy.center_support_open=Mock();policy.calibrate_contact=Mock();policy.reopen_for_alignment=Mock()
        policy.measured_calibration_state=Mock(return_value=(poses,measured,box));policy.support_alignment=Mock(side_effect=[{'support_x_error_m':-.0006},{'support_x_error_m':.00002}])
        policy.prepare_centered_grasp()
        self.assertAlmostEqual(env.edge_support_open_target_x,.0006)
        self.assertEqual(policy.calibration_preload_m,.0009)
        self.assertEqual(policy.calibrate_contact.call_count,2);policy.reopen_for_alignment.assert_called_once()

    def test_open_alignment_tracks_learned_closure_bias(self):
        env,poses,measured,box=self.environment();env.edge_support_open_target_x=.0006;policy=SeatEdgeArmExecutePolicy(env,'grasp')
        policy.measured_calibration_state=Mock(return_value=(poses,measured,box));policy.support_alignment=Mock(side_effect=[{'support_x_error_m':0.},{'support_x_error_m':.0006}]);policy.interpolate=Mock()
        with patch.object(policy,'local_support_target',return_value=env.command.copy()), \
             patch.object(policy,'certify_support_translation',return_value=[{'minimum_noncontact_clearance_m':.04,'minimum_wall_clearance_m':.18}]):
            policy.center_support_open()
        self.assertAlmostEqual(env.edge_grasp_x_offset,.0006);policy.interpolate.assert_called_once()

    def test_excessive_closure_bias_is_rejected_before_reopening(self):
        env,poses,measured,box=self.environment();policy=SeatEdgeArmExecutePolicy(env,'grasp')
        policy.support_target_x=Mock(return_value=0.)
        policy.center_support_open=Mock();policy.calibrate_contact=Mock();policy.reopen_for_alignment=Mock()
        policy.measured_calibration_state=Mock(return_value=(poses,measured,box));policy.support_alignment=Mock(return_value={'support_x_error_m':-.006})
        with self.assertRaisesRegex(RuntimeError,'SUPPORT_CLOSURE_BIAS_LIMIT'):policy.prepare_centered_grasp()
        policy.reopen_for_alignment.assert_not_called()

    def test_short_seat_bias_is_bounded_and_large_seat_remains_unbiased(self):
        env,_,_,_=self.environment();policy=SeatEdgeArmExecutePolicy(env,'grasp')
        self.assertLess(policy.support_target_x(),0.)
        self.assertGreaterEqual(policy.support_target_x(),-.001)
        from types import SimpleNamespace
        env.task=SimpleNamespace(dimensions=(.44,.52,.36))
        self.assertEqual(policy.support_target_x(),0.)
        env.task=SimpleNamespace(dimensions=(.6,.7,.5))
        self.assertEqual(policy.support_target_x(),0.)
        self.assertEqual(policy.calibration_preload_m,.0009)

    def test_short_seat_closure_converges_to_biased_target_with_original_tolerance(self):
        env,poses,measured,box=self.environment();policy=SeatEdgeArmExecutePolicy(env,'grasp')
        policy.center_support_open=Mock();policy.calibrate_contact=Mock();policy.reopen_for_alignment=Mock()
        policy.measured_calibration_state=Mock(return_value=(poses,measured,box))
        policy.support_alignment=Mock(side_effect=[{'support_x_error_m':-.0016},{'support_x_error_m':-.00098}])
        policy.prepare_centered_grasp()
        self.assertAlmostEqual(env.edge_support_open_target_x,-.0004)
        self.assertEqual(policy.calibrate_contact.call_count,2)
        policy.reopen_for_alignment.assert_called_once()
        event=env.recorder.event.call_args.args[1]
        self.assertLess(abs(event['target_error_m']),.0001)
        self.assertEqual(policy.calibration_preload_m,.0009)


if __name__=='__main__':unittest.main()
