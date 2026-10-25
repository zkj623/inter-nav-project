"""Clock-reference checks only; never sets a clock or imports ROS."""
import unittest
from deploy.sync_time import source_offset, ages_valid


class ClockReferenceTests(unittest.TestCase):
    def samples(self):
        return [(100+i*.01, 1800000000+i*.01, 1000+i*.01) for i in range(30)]

    def test_wrong_wall_epoch_requires_alignment(self):
        samples = self.samples()
        offset = source_offset(samples)
        self.assertFalse(ages_valid(samples, .4))
        self.assertTrue(ages_valid(samples, .4, offset))

    def test_other_source_epoch_or_stale_cloud_is_not_repaired(self):
        samples = self.samples()
        offset = source_offset(samples)
        for delta in (-2., 2.):
            cloud = [(m, s+delta, w) for m, s, w in samples]
            self.assertFalse(ages_valid(cloud, .4, offset))

    def test_invalid_sources_rejected(self):
        samples = self.samples()
        cases = [samples[:5], list(reversed(samples)), [(m, 0, w) for m, s, w in samples],
                 [(m, float('nan'), w) for m, s, w in samples],
                 samples[:-1]+[(samples[-1][0]+.1, samples[-1][1], samples[-1][2])]]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                source_offset(case)

    def test_existing_age_limits_are_preserved(self):
        stamp = 1800000000.
        for age, expected in [(-.06, False), (.1, True), (.41, False)]:
            self.assertEqual(ages_valid([(100., stamp, stamp+age)], .4), expected)
        self.assertFalse(ages_valid([], .4))
