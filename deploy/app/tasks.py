"""Wall-clock task lifecycle around the simulation release's online engine."""
from contextlib import contextmanager
from time import perf_counter
import math
import numpy as np
from .navigation import NavigationEngine


class Tasks:
    def __init__(self, config, perception=None, interaction=None):
        self.config, self.perception = config, perception
        self.engine = NavigationEngine(config, interaction=interaction)
        self.mode = self.state = 'observe'
        self.elapsed = 0.
        self.last_time = None
        self.last_step = -1
        self.last_lidar = self.last_camera = None
        self.result = None
        self.map_ready = False
        self.last_mapping_at = None
        self.last_timings = {}
        self._prepared = False
        self._perception_sample = None

    def start(self, mode, value, now):
        if mode != 'navigate':
            raise ValueError('Only coordinate navigation is supported; semantic search was retired')
        if len(value) != 3 or not np.isfinite(value).all():
            raise ValueError('Goal must contain three finite map coordinates')
        if self.perception is not None:
            self.perception.reset()
        self._perception_sample = None
        self.engine.set_goal(tuple(value), now)
        self.mode, self.state = 'navigate', 'active'
        self.elapsed, self.last_time, self.last_step = 0., now, -1
        self.result = None
        self._prepared = False

    def pause(self):
        if self.state == 'active':
            self.engine.pause()
            self.state = 'paused'
        self._prepared = False

    def resume(self, now):
        if self.state != 'paused':
            raise ValueError('No paused task')
        if self.perception is not None:
            self.perception.reset()
        self._perception_sample = None
        self.engine.set_goal(self.engine.goal, now)
        self.last_time, self.state = now, 'active'
        self._prepared = False

    def prepare_for_motion(self, snapshot, now):
        """Build the first route while the SDK writer is still disarmed.

        The resulting velocity is deliberately discarded. Activation preserves
        the route, but the next live step must check a fresh observation before
        requesting any motion.
        """
        if self.state != 'active':
            raise ValueError('No navigation task to prepare')
        self.update(snapshot, now)
        if self.state != 'active':
            return False
        self.state = 'paused'
        # A successful computation may still say "wait, no observed route".
        # Keep that goal/map, but do not arm a writer with nothing to execute.
        self._prepared = (self.engine.decision or {}).get('kind') != 'wait'
        return self._prepared

    def activate_prepared(self, now):
        if self.state != 'paused' or not self._prepared:
            raise ValueError('No prepared navigation task')
        self._prepared = False
        self.last_time, self.state = now, 'active'
        # Planning is not execution: never seed the slew limiter with an
        # unsent recovery/turn request generated during preparation.
        self.engine.limiter.accept(np.zeros(3), now)

    def cancel(self):
        if self.perception is not None:
            self.perception.reset()
        self._perception_sample = None
        self.engine.cancel()
        self.state = self.mode = 'observe'
        self.result = None
        self.elapsed, self.last_time = 0., None
        self._prepared = False

    def fail(self, reason):
        self.engine.pause()
        self.state = 'fault'
        self.result = {'success': False, 'reason': reason}

    def check_navigation_goal(self, goal):
        if len(goal) < 2 or not all(math.isfinite(v) for v in goal[:2]):
            raise ValueError('目标坐标必须为有限数值')
        if not self.map_ready:
            raise ValueError('导航地图尚未就绪，请先观察建图')
        planner, layers = self.engine.planner, self.engine.layers
        cell = planner.cell(goal[:2])
        limits = self.config['mapping']
        inside = limits['x_limits'][0] <= goal[0] < limits['x_limits'][1] and limits['y_limits'][0] <= goal[1] < limits['y_limits'][1]
        if not inside or not planner.valid_cell(cell):
            raise ValueError('目标超出地图范围')
        # An unseen goal now drives observed-frontier exploration. Only reject
        # a directly measured occupied target; do not mark unknown cells free.
        if layers.observed[cell] and layers.local_occupied[cell]:
            raise ValueError('目标位于地图障碍内')

    @contextmanager
    def _measure(self, stage):
        start = perf_counter()
        try:
            yield
        finally:
            self.last_timings[stage] = perf_counter()-start

    def update(self, snapshot, now):
        self.last_timings = {}
        with self._measure('total_s'):
            return self._update(snapshot, now)

    def observe_only(self, snapshot, now):
        """Maintain live geometry/maps during a writer hold without advancing skills.

        This does not reset task identity, extend execution deadlines, or submit
        motion. A same-cycle preparation reuses the perception sample below.
        """
        self.last_timings = {}
        with self._measure('total_s'):
            return self._update(snapshot, now, advance=False)

    def _perceive(self, snapshot, now):
        cached = self._perception_sample
        if cached is not None and cached[0] is snapshot and cached[1] == now:
            return cached[2]
        result = ((), None) if self.perception is None else self.perception.update(snapshot, now)
        self._perception_sample = (snapshot, now, result)
        return result

    def _update(self, snapshot, now, *, advance=True):
        if not snapshot.geometry_calibrated:
            if self.state == 'active':
                raise ValueError('Navigation requires calibrated ground and LiDAR origin')
            return None
        observation = {**snapshot.observation, 'velocity': tuple(snapshot.velocity),
                       'lidar_captured_at': snapshot.lidar_at}
        with self._measure('perception_s'):
            boxes, recognition = self._perceive(snapshot, now)
        if advance and self.state == 'active':
            self.elapsed += max(0., now-self.last_time)
            self.last_time = now
            self.last_step = max(self.last_step+1, int(self.elapsed*self.config['tick_hz']))
            if self.elapsed >= self.config['task_timeout_s']:
                self.engine.pause()
                self.state = 'completed'
                self.result = {'success': False, 'reason': 'task_timeout'}
                return None
            with self._measure('navigation_s'):
                velocity = self.engine.update(observation, now, boxes=boxes, recognition=recognition)
        else:
            with self._measure('mapping_s'):
                self.engine.observe(observation, now, boxes=boxes)
            velocity = None
        if self.engine.layers.scan_count:
            self.map_ready = True
            self.last_lidar = self.engine.layers.last_scan
            if self.last_lidar == observation['sensors']['lidar'].get('physics_step'):
                self.last_mapping_at = snapshot.lidar_at
        if advance and self.state == 'active':
            with self._measure('goal_check_s'):
                self.check_navigation_goal(self.engine.goal)
        if advance and self.state == 'active' and self.engine.status in ('completed', 'failed'):
            self.state = 'completed'
            self.result = self.engine.result
            return None
        return velocity
