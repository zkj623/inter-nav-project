from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from grutopia_extension.interaction_clearance.primitives import (
    ArmPlanPolicy, SeatEdgeArmPlanPolicy, SeatEdgeArmExecutePolicy,
    grasp_contact_established, primitive_registry,
)
from grutopia_extension.interaction_clearance.task import ClearanceTask


class SeatEdgeGripTests(unittest.TestCase):
    def environment(self, kind='stool'):
        rotation = Rotation.from_euler('xyz', [.08,-.04,.8]).as_matrix()
        q = Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
        origin = np.array((.72,0.,.0001))
        task = ClearanceTask(object_kind=kind, dimensions=(.34,.42,.28), mass_kg=1.1)
        return SimpleNamespace(task=task, grasp_mode='stool_seat_edge',
            finger_force_vectors=np.array(((0.,0.,-5.),(0.,0.,4.),(0.,0.,4.),(0.,0.,-5.))) @ rotation.T,
            object=SimpleNamespace(get_world_pose=lambda:(origin,q)),
            base=SimpleNamespace(get_world_pose=lambda:(np.zeros(3),np.array((1.,0.,0.,0.)))),
            rotation=lambda q: Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix(),
            box=SimpleNamespace(dimensions=task.dimensions,grasp_height=.265,squeeze=.003),
            initial_control=np.zeros(19),recorder=None)

    def test_four_correct_loads_follow_object_frame(self):
        self.assertTrue(grasp_contact_established(self.environment()))

    def test_one_unloaded_finger_cannot_be_hidden_in_hand_sum(self):
        for i in range(4):
            env=self.environment(); env.finger_force_vectors[i]=0.
            self.assertFalse(grasp_contact_established(env))

    def test_large_wrong_direction_forces_do_not_establish_grasp(self):
        env=self.environment(); env.finger_force_vectors *= -10
        self.assertFalse(grasp_contact_established(env))

    def test_side_grip_forces_do_not_establish_edge_grasp(self):
        env=self.environment()
        env.finger_force_vectors=np.array(((0.,20.,0.),(0.,20.,0.),(0.,-20.,0.),(0.,-20.,0.)))
        self.assertFalse(grasp_contact_established(env))

    def test_bad_force_values_fail_closed(self):
        for value in (float('nan'),float('inf')):
            env=self.environment();env.finger_force_vectors[0,0]=value
            self.assertFalse(grasp_contact_established(env))

    def test_production_registry_selects_edge_only_for_stools(self):
        env=self.environment()
        registry=primitive_registry(env)
        self.assertIsInstance(registry['arm_plan'],SeatEdgeArmPlanPolicy)
        self.assertIsInstance(registry['grasp_lift'],SeatEdgeArmExecutePolicy)
        self.assertIsInstance(registry['place_release'],SeatEdgeArmExecutePolicy)
        for kind in ('box','bin'):
            env=self.environment(kind);del env.grasp_mode
            self.assertIs(type(primitive_registry(env)['arm_plan']),ArmPlanPolicy)

    def test_insert_with_open_jaws_before_separate_closure(self):
        env=self.environment();plan=SeatEdgeArmPlanPolicy(env)
        with patch.object(plan,'solve_edge',return_value=np.zeros(15)),patch.object(plan,'check_lift_grip'):
            targets,phases,durations=plan.targets('grasp',np.array((.72,0.,.0001)),np.zeros(19),.05)
        self.assertEqual(phases[:3],['approach','approach','squeeze'])
        self.assertTrue(np.all(np.abs(targets[1][15:])>np.abs(targets[2][15:])))
        np.testing.assert_array_equal(targets[1][:15],targets[2][:15])
        self.assertTrue(all(np.array_equal(t[15:],targets[2][15:]) for t in targets[3:]))
        self.assertEqual(phases[3:],['lift']*12)
        self.assertEqual(env.box.squeeze,.003)

    def test_release_opens_before_sideways_withdrawal(self):
        env=self.environment();plan=SeatEdgeArmPlanPolicy(env)
        with patch.object(plan,'solve_edge',return_value=np.ones(15)):
            targets,phases,_=plan.targets('place',np.array((.72,0.,.2201)),np.zeros(19),.05)
        first_release=phases.index('release')
        self.assertTrue(all(p=='lower' for p in phases[:first_release]))
        np.testing.assert_array_equal(targets[first_release][:15],targets[first_release-1][:15])
        self.assertTrue(np.all(np.abs(targets[first_release][15:])>np.abs(targets[first_release-1][15:])))
        self.assertEqual(phases[first_release:],['release','release','retract'])

    def test_missing_edge_contact_stops_before_any_lift(self):
        env=self.environment();env.finger_force_vectors[0]=0.
        policy=SeatEdgeArmExecutePolicy(env,'grasp')
        policy.interpolate=Mock();policy.lift_segment=Mock()
        policy.prepare_centered_grasp=Mock()
        plan={'phases':['approach','approach','squeeze']+['lift']*6,
              'targets':[np.zeros(19)]*9,'durations':[1.]*9}
        with self.assertRaisesRegex(RuntimeError,'FOUR_EDGE_CONTACTS_MISSING'):
            policy.grasp_motion(plan)
        policy.lift_segment.assert_not_called()

    def test_refinement_preserves_full_lift_and_limits_segment_height(self):
        env=self.environment(); env.recorder=Mock()
        plan=SeatEdgeArmPlanPolicy(env)
        origin=np.array((.72,0.,.0001))
        with patch.object(plan,'solve_edge',return_value=np.zeros(15)) as solve,patch.object(plan,'check_lift_grip'):
            targets,phases,durations=plan.targets('grasp',origin,np.zeros(19),.05)
        planned=np.array([call.args[0][2] for call in solve.call_args_list[2:]])
        self.assertAlmostEqual(planned[-1]-origin[2],.22)
        self.assertLessEqual(max(np.diff(np.r_[origin[2],planned])),.02)
        policy=SeatEdgeArmExecutePolicy(env,'grasp')
        policy.interpolate=Mock();policy.lift_segment=Mock()
        policy.prepare_centered_grasp=Mock()
        policy.grasp_motion({'targets':targets,'phases':phases,'durations':durations})
        executed=np.array([call.args[0] for call in policy.lift_segment.call_args_list])
        np.testing.assert_allclose(executed,planned)
        self.assertTrue(all(duration>=.8 for duration in durations[3:]))

    def test_lost_contact_while_lowering_stops_before_release(self):
        env=self.environment();env.command=np.zeros(19);env.control_indices=np.arange(19)
        env.control_dt=.02;env.advance=Mock();env.object_bottom=lambda:.10
        env.finger_force_vectors[0]=0.
        policy=SeatEdgeArmExecutePolicy(env,'place');policy.phase_end=Mock()
        with self.assertRaisesRegex(RuntimeError,'FOUR_EDGE_CONTACTS_LOST_DURING_LOWER'):
            policy.interpolate(np.zeros(19),1.,'lower')
        self.assertEqual(env.advance.call_count,1)
        policy.phase_end.assert_not_called()


if __name__=='__main__':
    unittest.main()
