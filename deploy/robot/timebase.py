"""Fixed source-clock to monotonic-clock mappings with explicit bootstrap basis.

An aligned-clock anchor preserves independently known data age. The startup
sample-window alternative only establishes relative age: an unknown common
baseline delay cannot be recovered from one-way messages. Neither mode follows
arrivals after bootstrap or changes the operating system clock.
"""
import math
import time


class SourceTimeBridge:
    def __init__(self, source_ns, monotonic_ns, *, uncertainty_ns=0,
                 bootstrap='local_ros_clock_must_already_match_sensor_clock',
                 receipt_offset_estimation=False):
        for name, value in (('source_ns', source_ns), ('monotonic_ns', monotonic_ns),
                            ('uncertainty_ns', uncertainty_ns)):
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be nonnegative integer nanoseconds')
        self.source_ns = source_ns
        self.monotonic_ns = monotonic_ns
        self.uncertainty_ns = uncertainty_ns
        self.bootstrap = bootstrap
        self.receipt_offset_estimation = receipt_offset_estimation
        self._sample_metadata = {}
        self._delay_reserve_ns = 0

    @classmethod
    def from_clock(cls, source_now_ns, *, monotonic_now_ns=None, max_bracket_s=.01):
        """Bracket an already-aligned clock read; use its earliest time bound."""
        if not math.isfinite(max_bracket_s) or max_bracket_s <= 0:
            raise ValueError('Clock sampling limit must be finite and positive')
        monotonic_now_ns = monotonic_now_ns or time.monotonic_ns
        before = monotonic_now_ns()
        source = source_now_ns()
        after = monotonic_now_ns()
        span = after - before
        if span < 0 or span > max_bracket_s * 1e9:
            raise ValueError('Clock anchor sampling was interrupted; restart after checking clocks')
        # Source read happened somewhere within [before, after]. The lower
        # bound conservatively retains any sampling uncertainty as data age.
        return cls(source, after, uncertainty_ns=span)

    @classmethod
    def from_samples(cls, samples, *, max_age_s=.4, future_tolerance_s=.05,
                     reserve_s=.05, min_duration_s=3., max_offset_spread_s=.05):
        """Freeze one relative offset from {stream: [(receipt_ns, source_ns)]}.

        A fixed stream-wide delay is indistinguishable from a clock offset.
        ``reserve_s`` is extra age withheld from the command budget, not a
        measured upper bound on that unknown delay. Repeated calls during a
        running session must not be used to absorb a backlog or clock jump.
        """
        for name, value in (('max_age_s', max_age_s), ('min_duration_s', min_duration_s),
                            ('max_offset_spread_s', max_offset_spread_s)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(reserve_s) or not 0 <= reserve_s < max_age_s:
            raise ValueError('Clock delay reserve must be nonnegative and below the age limit')
        if not math.isfinite(future_tolerance_s) or future_tolerance_s < 0:
            raise ValueError('Future tolerance must be finite and nonnegative')
        if not isinstance(samples, dict) or not {'odom', 'lidar'} <= set(samples):
            raise ValueError('Clock bootstrap requires both odom and lidar samples')
        for kind, values in samples.items():
            if not isinstance(values, (list, tuple)):
                raise ValueError(f'{kind}: invalid clock bootstrap samples')
            if len(values) < (20 if kind == 'odom' else 2):
                raise ValueError(f'{kind}: insufficient clock bootstrap samples')
            if any(not isinstance(sample, (list, tuple)) or len(sample) != 2
                   or any(type(v) is not int for v in sample)
                   or sample[0] < 0 or sample[1] <= 0 for sample in values):
                raise ValueError(f'{kind}: invalid clock bootstrap sample')
            for left, right in zip(values, values[1:]):
                if right[0] <= left[0] or right[1] <= left[1]:
                    raise ValueError(f'{kind}: non-increasing bootstrap timestamps')
                if right[0]-left[0] > max_age_s*1e9:
                    raise ValueError(f'{kind}: bootstrap receive gap exceeds age limit')
        odom = samples['odom']
        duration_ns = odom[-1][0] - odom[0][0]
        if duration_ns < min_duration_s*1e9:
            raise ValueError('Clock bootstrap observation window is too short')
        offsets = [received-source for received, source in odom]
        minimum = min(offsets)
        spread_ns = max(offsets)-minimum
        if spread_ns > max_offset_spread_s*1e9:
            raise ValueError('Odometry clock bootstrap offset spread exceeds limit')
        end_ns = max(values[-1][0] for values in samples.values())
        reserve_ns = round(reserve_s*1e9)
        source_anchor = odom[-1][1]
        bridge = cls(source_anchor, source_anchor+minimum,
                     bootstrap='startup_source_window_relative_age', receipt_offset_estimation=True)
        bridge._delay_reserve_ns = reserve_ns
        for kind, values in samples.items():
            if end_ns-values[-1][0] > max_age_s*1e9:
                raise ValueError(f'{kind}: stream stopped during clock bootstrap')
            for received, source in values:
                # One mapping for all streams; never independently align lidar.
                bridge.captured_at(source, received, max_age_s, future_tolerance_s)
        bridge._sample_metadata = {
            'unknown_baseline_delay': True, 'absolute_age_verified': False,
            'delay_reserve_s': reserve_s, 'delay_reserve_is_measured_bound': False,
            'bootstrap_duration_s': duration_ns*1e-9,
            'bootstrap_offset_spread_s': spread_ns*1e-9,
            'bootstrap_sample_counts': {kind: len(values) for kind, values in samples.items()},
            'bootstrap_first_receipt_ns': odom[0][0], 'bootstrap_last_receipt_ns': end_ns,
        }
        return bridge

    @property
    def metadata(self):
        return {'reference_kind': 'fixed_source_to_monotonic', 'source_anchor_ns': self.source_ns,
                'monotonic_anchor_ns': self.monotonic_ns,
                'anchor_uncertainty_ns': self.uncertainty_ns,
                'bootstrap': self.bootstrap,
                'receipt_offset_estimation': self.receipt_offset_estimation,
                **self._sample_metadata}

    def captured_at(self, stamp_ns, received_ns, max_age_s, future_tolerance_s=.05):
        if type(stamp_ns) is not int or stamp_ns <= 0:
            raise ValueError('ROS timestamp invalid')
        if type(received_ns) is not int or received_ns < 0:
            raise ValueError('Monotonic receipt timestamp invalid')
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('Maximum data age must be finite and positive')
        if not math.isfinite(future_tolerance_s) or future_tolerance_s < 0:
            raise ValueError('Future tolerance must be finite and nonnegative')
        # Subtract source epochs as integers before converting to seconds.
        captured_ns = self.monotonic_ns + (stamp_ns - self.source_ns) - self.uncertainty_ns
        unreserved_age = (received_ns - captured_ns) * 1e-9
        if unreserved_age < -future_tolerance_s:
            raise ValueError(f'ROS clock mismatch: message ahead of anchored monotonic clock by '
                             f'{-unreserved_age*1000:.1f} ms; check source clock and initial alignment')
        # The relative bootstrap reserve only ages data. It must not make a
        # source-clock advance look smaller and widen the future tolerance.
        captured_ns -= self._delay_reserve_ns
        age = (received_ns - captured_ns) * 1e-9
        if age > max_age_s:
            raise ValueError(f'ROS timestamp stale: age={age*1000:.1f} ms, '
                             f'limit={max_age_s*1000:.0f} ms; check source clock and receive delay')
        # Preserve the previous adapter's conservative treatment of tolerated
        # future skew: never extend a command's sensor expiry past receipt.
        return min(captured_ns, received_ns) * 1e-9
