"""Explicit physics stepping and independently checkable native-clock coverage."""
import math


def advance_control(world, physics_dt, after_step, substeps=4):
    start_index = int(world.current_time_step_index)
    start_time = float(world.current_time)
    try:
        for _ in range(substeps):
            before = int(world.current_time_step_index)
            world.step(render=False)
            if int(world.current_time_step_index) != before + 1:
                raise RuntimeError('PHYSICS_CLOCK: explicit step did not advance exactly one native tick')
            after_step()
    finally:
        before = int(world.current_time_step_index)
        world.render()
        if int(world.current_time_step_index) != before:
            raise RuntimeError('PHYSICS_CLOCK: render-only call advanced physics')
    elapsed = float(world.current_time) - start_time
    if int(world.current_time_step_index) - start_index != substeps or not math.isclose(
            elapsed, substeps * physics_dt, rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError('PHYSICS_CLOCK: actual control interval differs from configured physics time')


def validate_recorded_clock(recording, result):
    if recording.get('clock_schema_version') != 1:
        raise ValueError('Missing native physics clock; full step coverage cannot be established')
    clock = recording['physics_clock']
    samples = recording['samples']
    dt = clock['physics_dt']
    if len(samples) < 40 or not math.isclose(dt, 0.005) or not math.isclose(clock['control_dt'], 4 * dt):
        raise ValueError('Invalid recorded physics/control timestep')
    previous_index, previous_time = None, None
    for sample in samples:
        index, stamp = sample['physics_step_index'], sample['physics_time_seconds']
        if type(index) is not int or not math.isfinite(stamp):
            raise ValueError('Invalid native physics clock sample')
        if previous_index is not None and (index != previous_index + 1 or not math.isclose(
                stamp - previous_time, dt, rel_tol=1e-6, abs_tol=1e-7)):
            raise ValueError('Missing, duplicated, or mistimed native physics tick')
        previous_index, previous_time = index, stamp
    if samples[-1]['physics_step_index'] != clock['end_index'] or not math.isclose(
            samples[-1]['physics_time_seconds'], clock['end_time'], abs_tol=1e-7):
        raise ValueError('Physics advanced after the final audited sample')
    origin = clock['control_start_index']
    initialization_samples = origin - samples[0]['physics_step_index'] + 1
    if initialization_samples != 40 or samples[39]['physics_step_index'] != origin:
        raise ValueError('Unaccounted initialization physics ticks')
    if not math.isclose(samples[39]['physics_time_seconds'], clock['control_start_time'], abs_tol=1e-7):
        raise ValueError('Control clock origin does not match the final initialization sample')
    control_ticks = clock['end_index'] - origin
    extra = control_ticks - result['frames'] * 4
    if extra != 0 and (result['success'] or result['reason'] != 'SAFETY_REJECTION' or not 1 <= extra <= 4):
        raise ValueError('Unaccounted physics ticks outside completed control intervals')
    elapsed = clock['end_time'] - clock['control_start_time']
    if not math.isclose(elapsed, control_ticks * dt, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError('Native physics elapsed time disagrees with tick count')
    if not math.isclose(result['actual_physics_seconds'], elapsed, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError('Reported physical elapsed time disagrees with native clock')
    return {'passed': True, 'physics_dt': dt, 'control_dt': clock['control_dt'],
            'initialization_samples': initialization_samples, 'control_physics_steps': control_ticks,
            'aborted_control_substeps': extra, 'actual_physics_seconds': elapsed}
