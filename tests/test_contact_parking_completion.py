import unittest
from grutopia_extension.interactive_navigation.go2_contact_control import push_completion_state


class ParkingCompletionTests(unittest.TestCase):
    def test_recorded_side_parking_stops_force_before_observation_is_complete(self):
        clearance={'side_parked':True,'passage_clearance_margin_m':.12301133103305084}
        self.assertEqual(push_completion_state(None,clearance,.926,True),'side_parked_pending_observation')
    def test_insufficient_clearance_or_no_push_cannot_finish_parking(self):
        self.assertIsNone(push_completion_state(None,{'side_parked':True,'passage_clearance_margin_m':.01},.8,True))
        self.assertIsNone(push_completion_state(None,{'side_parked':True,'passage_clearance_margin_m':.1},.1,True))
        self.assertIsNone(push_completion_state(None,{'side_parked':False,'passage_clearance_margin_m':-.1},.8,True))
    def test_legacy_prior_mode_keeps_its_certificate_requirement(self):
        self.assertIsNone(push_completion_state(None,{'side_parked':True,'passage_clearance_margin_m':.1},.8,False))
    def test_certified_offset_corridor_can_finish_before_nominal_center_clearance(self):
        proof={'side_parked':True,'completion_rule':'current_global_route_and_full_body_local_corridor'}
        self.assertEqual(push_completion_state(proof,{'side_parked':False,'passage_clearance_margin_m':-.03},.74,True),'observed_corridor_clear')

if __name__=='__main__':unittest.main()
