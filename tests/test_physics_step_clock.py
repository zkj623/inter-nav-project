import unittest

from grutopia_extension.manipulation.physics_step import advance_control, validate_recorded_clock


class FakeWorld:
    def __init__(self, bad_render=False):
        self.current_time_step_index = 0
        self.current_time = 0.0
        self.render_calls = 0
        self.bad_render = bad_render

    def step(self, render=True):
        # Reproduces the installed SDK: app.update advances four native ticks.
        ticks = 4 if render else 1
        self.current_time_step_index += ticks
        self.current_time += ticks * 0.005

    def render(self):
        self.render_calls += 1
        if self.bad_render:
            self.step(render=True)


class PhysicsStepClockTests(unittest.TestCase):
    def test_old_loop_demonstrates_the_sdk_seven_tick_bug(self):
        world = FakeWorld()
        for index in range(4):
            world.step(render=index == 3)
        self.assertEqual(world.current_time_step_index, 7)

    def test_correct_control_samples_all_four_ticks_and_render_adds_none(self):
        world = FakeWorld()
        samples = []
        advance_control(world, 0.005, lambda: samples.append(world.current_time_step_index))
        self.assertEqual(samples, [1, 2, 3, 4])
        self.assertAlmostEqual(world.current_time, 0.02)
        self.assertEqual(world.render_calls, 1)

    def test_guard_rejects_render_that_advances_physics(self):
        with self.assertRaisesRegex(RuntimeError, 'render-only'):
            advance_control(FakeWorld(bad_render=True), 0.005, lambda: None)

    def test_safety_exception_still_renders_without_advancing(self):
        world = FakeWorld()

        def reject():
            raise RuntimeError('collision')

        with self.assertRaisesRegex(RuntimeError, 'collision'):
            advance_control(world, 0.005, reject)
        self.assertEqual(world.current_time_step_index, 1)
        self.assertEqual(world.render_calls, 1)

    def recording(self, extra=0):
        samples = [{'physics_step_index': index, 'physics_time_seconds': index * 0.005}
                   for index in range(101, 149 + extra)]
        clock = {'physics_dt': 0.005, 'control_dt': 0.02, 'control_start_index': 140,
                 'control_start_time': 0.7, 'end_index': 148 + extra, 'end_time': (148 + extra) * 0.005}
        result = {'frames': 2, 'success': extra == 0,
                  'reason': 'ROAD_CLEARED' if extra == 0 else 'SAFETY_REJECTION',
                  'actual_physics_seconds': 0.04 + extra * 0.005}
        return {'clock_schema_version': 1, 'physics_clock': clock, 'samples': samples}, result

    def test_contiguous_physics_and_terminal_safety_samples_are_accounted_for(self):
        for extra in range(5):
            with self.subTest(extra=extra):
                audit = validate_recorded_clock(*self.recording(extra))
                self.assertEqual(audit['aborted_control_substeps'], extra)

    def test_missing_tick_is_rejected_even_if_legacy_frame_count_matches(self):
        recording, result = self.recording()
        recording['samples'][44]['physics_step_index'] += 3
        with self.assertRaises(ValueError):
            validate_recorded_clock(recording, result)

    def test_unrecorded_final_render_and_legacy_records_are_rejected(self):
        recording, result = self.recording()
        recording['physics_clock']['end_index'] += 3
        with self.assertRaises(ValueError):
            validate_recorded_clock(recording, result)
        recording, result = self.recording()
        del recording['clock_schema_version']
        with self.assertRaises(ValueError):
            validate_recorded_clock(recording, result)


if __name__ == '__main__':
    unittest.main()
