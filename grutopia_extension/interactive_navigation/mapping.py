"""Isaac-independent RGB/LiDAR fusion maps and A* navigation."""

from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import hypot, sqrt
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

Vector3 = Tuple[float, float, float]
GridCell = Tuple[int, int]


class PlanningError(RuntimeError):
    """Raised when no traversable map path can be found."""


@dataclass(frozen=True)
class MappingConfig:
    x_limits: Tuple[float, float] = (-2.0, 9.0)
    y_limits: Tuple[float, float] = (-3.0, 3.0)
    grid_resolution: float = 0.10
    voxel_size: float = 0.10
    obstacle_height: Tuple[float, float] = (0.12, 1.80)
    point_height: Tuple[float, float] = (-0.25, 2.75)
    robot_radius: float = 0.18
    obstacle_inflation_radius: Optional[float] = None
    static_obstacle_inflation_radius: Optional[float] = None
    object_position_window: int = 0
    occupied_threshold: float = 0.55
    free_update: float = 0.30
    occupied_update: float = 0.85
    # Cells whose occupancy evidence reached this level are immune to free
    # rays (scale 0): in static scenes, confirmed obstacles may only be
    # cleared by the robot physically traversing the cell. Raise the scale
    # above zero to let vacated space fade in dynamic environments.
    sticky_occupied_threshold: float = 1.5
    sticky_free_scale: float = 0.0
    unknown_cost: float = 1.75
    max_lidar_rays: int = 900
    allow_lidar_free_only_rays: bool = True
    lidar_free_height_layers: int = 1
    # Azimuth resolution of the flattened top-down lidar scan (0.5 degrees).
    lidar_scan_bins: int = 720
    max_voxel_points_per_update: int = 5000
    # Opt-in removal of old LiDAR-only geometry after distinct scans confirm
    # free space along measured 3D rays. Zero preserves the static-map policy.
    voxel_clear_observations: int = 0
    graph_stride: int = 5
    safe_recovery_y_limits: Tuple[float, float] = (-1.25, 1.25)

    def __post_init__(self):
        if self.x_limits[0] >= self.x_limits[1] or self.y_limits[0] >= self.y_limits[1]:
            raise ValueError('map limits must be increasing')
        if self.safe_recovery_y_limits[0] >= self.safe_recovery_y_limits[1]:
            raise ValueError('safe_recovery_y_limits must be increasing')
        for name in (
            'grid_resolution',
            'voxel_size',
            'robot_radius',
            'occupied_threshold',
            'free_update',
            'occupied_update',
            'unknown_cost',
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.sticky_occupied_threshold <= 0 or not 0.0 <= self.sticky_free_scale <= 1.0:
            raise ValueError('sticky occupancy parameters must be positive (scale within [0, 1])')
        if self.max_lidar_rays <= 0 or self.max_voxel_points_per_update <= 0 or self.graph_stride <= 0:
            raise ValueError('mapping sample limits and graph_stride must be positive')
        if self.lidar_scan_bins <= 0:
            raise ValueError('lidar_scan_bins must be positive')
        if self.lidar_free_height_layers <= 0:
            raise ValueError('lidar_free_height_layers must be positive')
        if type(self.voxel_clear_observations) is not int or self.voxel_clear_observations < 0:
            raise ValueError('voxel_clear_observations must be a nonnegative integer')
        if self.obstacle_height[0] >= self.obstacle_height[1]:
            raise ValueError('obstacle_height must be increasing')
        if self.obstacle_inflation_radius is not None and self.obstacle_inflation_radius <= 0:
            raise ValueError('obstacle_inflation_radius must be positive when provided')
        if self.static_obstacle_inflation_radius is not None and self.static_obstacle_inflation_radius <= 0:
            raise ValueError('static_obstacle_inflation_radius must be positive when provided')


@dataclass(frozen=True)
class SemanticDetection:
    label: str
    position: Vector3
    color: Optional[Tuple[int, int, int]] = None
    confidence: float = 1.0
    embedding: Optional[Tuple[float, ...]] = None
    point_count: int = 1
    step: int = 0
    sources: Tuple[str, ...] = ()
    # Distinct labels observed for this object in one camera frame. Geometry
    # can be fused without discarding a lower-confidence target label.
    label_evidence: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneGraphNode:
    node_id: str
    kind: str
    position: Vector3
    label: str
    color: Optional[Tuple[int, int, int]] = None
    observations: int = 1
    confidence: float = 1.0
    embedding: Optional[Tuple[float, ...]] = None
    point_count: int = 1
    last_seen_step: int = 0
    sources: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneGraphEdge:
    source: str
    target: str
    relation: str
    distance: float


@dataclass(frozen=True)
class SceneGraphSnapshot:
    nodes: Tuple[SceneGraphNode, ...]
    edges: Tuple[SceneGraphEdge, ...]

    @property
    def place_count(self) -> int:
        return sum(node.kind == 'place' for node in self.nodes)

    @property
    def object_count(self) -> int:
        return sum(node.kind != 'place' for node in self.nodes)


@dataclass
class _Voxel:
    position_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    color_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    count: int = 0
    color_count: int = 0
    source_mask: int = 0
    last_step: int = 0
    free_observations: int = 0


class VoxelPointCloudMap:
    """Bounded voxel map retaining LiDAR geometry and RGB color evidence."""

    _SOURCE_BITS = {'lidar': 1, 'rgb': 2}

    def __init__(self, config: MappingConfig):
        self.config = config
        self._voxels: Dict[Tuple[int, int, int], _Voxel] = {}
        self.input_points = 0

    def update(
        self,
        points: np.ndarray,
        step: int,
        source: str,
        colors: Optional[np.ndarray] = None,
    ):
        points = _points_array(points)
        if len(points) == 0:
            return
        if source not in self._SOURCE_BITS:
            raise ValueError(f'unknown point source: {source}')
        valid = np.isfinite(points).all(axis=1)
        valid &= points[:, 2] >= self.config.point_height[0]
        valid &= points[:, 2] <= self.config.point_height[1]
        points = points[valid]
        if colors is not None:
            colors = np.asarray(colors).reshape(-1, 3)[valid]
        if len(points) > self.config.max_voxel_points_per_update:
            indices = np.linspace(
                0,
                len(points) - 1,
                self.config.max_voxel_points_per_update,
                dtype=int,
            )
            points = points[indices]
            if colors is not None:
                colors = colors[indices]

        self.input_points += len(points)
        keys = np.floor(points / self.config.voxel_size).astype(np.int64)
        source_bit = self._SOURCE_BITS[source]
        for index, (point, key_array) in enumerate(zip(points, keys)):
            key = tuple(int(value) for value in key_array)
            voxel = self._voxels.get(key)
            if voxel is None:
                voxel = self._voxels[key] = _Voxel()
            voxel.position_sum += point
            voxel.count += 1
            np.divide(voxel.position_sum, voxel.count, out=voxel.centroid)
            voxel.source_mask |= source_bit
            voxel.last_step = step
            voxel.free_observations = 0
            if colors is not None:
                color = np.asarray(colors[index], dtype=np.float64)
                if np.nanmax(color) <= 1.0:
                    color *= 255.0
                voxel.color_sum += np.clip(color, 0.0, 255.0)
                voxel.color_count += 1

    def clear_lidar_free_space(self, origin: Vector3, points: np.ndarray, max_range: float):
        """Remove contradicted geometry, never geometry merely absent from a scan.

        Only a measured ray passing close to the old centroid and ending
        farther away is evidence. Hits in the current frame protect their
        entire voxel. Occluded, unseen and RGB-fused voxels remain untouched.
        The caller must supply each LiDAR scan only once, as MappingRuntime does.
        """
        threshold = self.config.voxel_clear_observations
        if not threshold or not self._voxels:
            return
        from scipy.spatial import cKDTree

        origin = np.asarray(origin, dtype=np.float64)
        points = _points_array(points)
        points = points[np.isfinite(points).all(axis=1)]
        if origin.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(max_range):
            return
        vectors = points - origin
        ranges = np.linalg.norm(vectors, axis=1)
        valid = (ranges >= .25) & (ranges < max_range * .99)
        points, vectors, ranges = points[valid], vectors[valid], ranges[valid]
        if not len(points):
            return
        hits = {tuple(key) for key in np.floor(points / self.config.voxel_size).astype(np.int64)}
        candidates = [(key, voxel) for key, voxel in self._voxels.items()
                      if key not in hits and voxel.source_mask == self._SOURCE_BITS['lidar']]
        if not candidates:
            return
        # Retain all endpoints: sampling away a nearer return could make a
        # farther ray incorrectly clear an occluded voxel.
        directions = vectors / ranges[:, None]
        deltas = np.asarray([voxel.centroid for _, voxel in candidates]) - origin
        distances = np.linalg.norm(deltas, axis=1)
        eligible = (distances >= .25) & (distances < max_range * .99)
        indices = np.flatnonzero(eligible)
        if not len(indices):
            return
        tree = cKDTree(directions)
        unit = deltas[indices] / distances[indices, None]
        _, ray = tree.query(unit)
        along = np.einsum('ij,ij->i', deltas[indices], directions[ray])
        perpendicular_squared = np.maximum(0., distances[indices]**2 - along**2)
        size = self.config.voxel_size
        clear = ((along >= .25) & (along < ranges[ray] - 2*size)
                 & (perpendicular_squared <= (.25*size)**2))
        for local_index in np.flatnonzero(clear):
            index = indices[local_index]
            # Also respect nearer returns in a small angular neighborhood,
            # including duplicate beam directions and multi-return scans.
            neighbors = tree.query_ball_point(unit[local_index], .5*size/distances[index])
            if neighbors and distances[index] >= np.min(ranges[neighbors]) - 2*size:
                continue
            key, voxel = candidates[index]
            voxel.free_observations += 1
            if voxel.free_observations >= threshold:
                del self._voxels[key]

    def points(self) -> np.ndarray:
        if not self._voxels:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(
            [voxel.centroid for voxel in self._voxels.values()],
            dtype=np.float32,
        )

    def colors(self) -> np.ndarray:
        if not self._voxels:
            return np.empty((0, 3), dtype=np.uint8)
        return np.asarray(
            [
                voxel.color_sum / voxel.color_count if voxel.color_count else np.array([127.0, 127.0, 127.0])
                for voxel in self._voxels.values()
            ],
            dtype=np.uint8,
        )

    def dominant_color_near(self, position: Tuple[float, float], radius: float = 0.45):
        samples = []
        for voxel in self._voxels.values():
            if voxel.color_count == 0:
                continue
            centroid = voxel.position_sum / voxel.count
            if np.linalg.norm(centroid[:2] - np.asarray(position)) <= radius:
                samples.append(voxel.color_sum / voxel.color_count)
        if not samples:
            return None
        color = np.median(np.asarray(samples), axis=0)
        return tuple(int(value) for value in np.clip(color, 0, 255))

    @property
    def voxel_count(self) -> int:
        return len(self._voxels)

    @property
    def lidar_voxel_count(self) -> int:
        return sum(bool(voxel.source_mask & self._SOURCE_BITS['lidar']) for voxel in self._voxels.values())

    @property
    def colored_voxel_count(self) -> int:
        return sum(voxel.color_count > 0 for voxel in self._voxels.values())


class OccupancyGridMap:
    """Log-odds 2D occupancy map updated from LiDAR ray endpoints."""

    def __init__(self, config: MappingConfig):
        self.config = config
        width = int(np.ceil((config.x_limits[1] - config.x_limits[0]) / config.grid_resolution))
        height = int(np.ceil((config.y_limits[1] - config.y_limits[0]) / config.grid_resolution))
        self.log_odds = np.zeros((height, width), dtype=np.float32)
        self.observed = np.zeros((height, width), dtype=bool)
        self.static_occupied = np.zeros((height, width), dtype=bool)
        self._lidar_free_height_seen = (
            np.zeros((config.lidar_free_height_layers, height, width), dtype=bool)
            if config.lidar_free_height_layers > 1 else None
        )
        self.revision = 0

    def update_lidar(self, origin: Vector3, points: np.ndarray, max_range: float):
        """Update occupancy from a lidar frame via a top-down virtual scan.

        The 3D frame is projected into a 2D scan per azimuth bin: the nearest
        return inside the obstacle height band becomes that direction's
        obstacle endpoint, cells before it become free, and cells behind it
        stay unknown. Optional multi-height mode keeps vertical channels
        separate and requires free evidence from every configured layer.
        """

        points = _points_array(points)
        if len(points) == 0:
            return
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if len(points) == 0:
            return

        origin_array = np.asarray(origin, dtype=np.float64)
        origin_cell = self.world_to_cell(origin_array[:2])
        if origin_cell is None:
            return
        min_height, max_height = self.config.obstacle_height

        deltas = points - origin_array
        horizontal = np.hypot(deltas[:, 0], deltas[:, 1])
        distances = np.linalg.norm(deltas, axis=1)
        # Returns near or beyond the sensor limit cannot become obstacles or
        # provide reliable free-space evidence.
        valid = (distances >= 0.25) & (distances < max_range * 0.99) & (horizontal > 1e-6)
        in_band = (
            valid
            & (points[:, 2] >= min_height)
            & (points[:, 2] <= max_height)
        )
        # A single free-height layer selects the simpler projected-2D-scan
        # behavior: one nearest in-band return per azimuth. With multiple
        # layers, keep vertical LiDAR channels separate so each layer has its
        # own visibility evidence. The Go2 LiDAR uses 5 degree vertical
        # sampling, so quantize measured elevation to retain those channels.
        scan_bins = min(self.config.lidar_scan_bins, self.config.max_lidar_rays)
        bins = (
            (np.arctan2(deltas[:, 1], deltas[:, 0]) + np.pi)
            / (2.0 * np.pi)
            * scan_bins
        ).astype(int) % scan_bins
        if self.config.lidar_free_height_layers > 1:
            elevations = np.rad2deg(np.arctan2(deltas[:, 2], horizontal))
            elevation_bins = np.rint(elevations / 5.0).astype(int)
        else:
            elevation_bins = np.zeros(len(points), dtype=int)
        ray_groups: Dict[Tuple[int, int], List[int]] = {}
        for index in np.flatnonzero(valid):
            ray_groups.setdefault(
                (int(bins[index]), int(elevation_bins[index])), []
            ).append(int(index))

        # Bound work while preserving samples from every vertical channel.
        # The previous budget applied to azimuth alone and silently discarded
        # all but one elevation at each angle.
        if len(ray_groups) > self.config.max_lidar_rays:
            by_elevation: Dict[int, List[Tuple[int, int]]] = {}
            for key in sorted(ray_groups):
                by_elevation.setdefault(key[1], []).append(key)
            per_channel = max(1, self.config.max_lidar_rays // len(by_elevation))
            retained = []
            for keys in by_elevation.values():
                if len(keys) <= per_channel:
                    retained.extend(keys)
                else:
                    indices = np.linspace(0, len(keys) - 1, per_channel, dtype=int)
                    retained.extend(keys[i] for i in indices)
            ray_groups = {key: ray_groups[key] for key in retained[:self.config.max_lidar_rays]}

        # Each vertical channel has its own nearest obstacle. A return in one
        # channel must not hide valid free-space rays in another channel.
        nearest_obstacle: Dict[Tuple[int, int], int] = {}
        farthest_seen: Dict[Tuple[int, int], int] = {}
        for key, indices in ray_groups.items():
            nearest = [index for index in indices if in_band[index]]
            if nearest:
                nearest_obstacle[key] = min(nearest, key=lambda index: horizontal[index])
            farthest_seen[key] = max(indices, key=lambda index: horizontal[index])

        frame_free = set()
        frame_occupied = set()
        frame_free_layers = (
            [set() for _ in range(self.config.lidar_free_height_layers)]
            if self._lidar_free_height_seen is not None else None
        )
        if self.config.allow_lidar_free_only_rays:
            selected = (
                (nearest_obstacle.get(key), seen_index)
                for key, seen_index in farthest_seen.items()
            )
        else:
            selected = ((index, index) for index in nearest_obstacle.values())

        for obstacle_index, seen_index in selected:
            endpoint = points[obstacle_index if obstacle_index is not None else seen_index]
            endpoint_cell = self.world_to_cell(endpoint[:2])
            if endpoint_cell is None:
                continue
            ray = _bresenham(origin_cell, endpoint_cell)
            if not ray:
                continue
            if obstacle_index is None:
                free_cells = ray
            else:
                free_cells = ray[:-1]
                frame_occupied.add(ray[-1])
            if frame_free_layers is None:
                frame_free.update(free_cells)
            else:
                for ray_index, cell in enumerate(free_cells):
                    fraction = ray_index / max(1, len(ray) - 1)
                    ray_height = origin_array[2] + fraction * (endpoint[2] - origin_array[2])
                    if ray_height < min_height or ray_height > max_height:
                        continue
                    layer = min(
                        int((ray_height - min_height) / (max_height - min_height)
                            * self.config.lidar_free_height_layers),
                        self.config.lidar_free_height_layers - 1,
                    )
                    frame_free_layers[layer].add(cell)

        if frame_free_layers is not None:
            for layer, cells in enumerate(frame_free_layers):
                for cell in cells:
                    if self.in_bounds(cell):
                        self._lidar_free_height_seen[layer, cell[0], cell[1]] = True
            candidates = set().union(*frame_free_layers)
            frame_free = {
                cell for cell in candidates
                if self.in_bounds(cell)
                and self._lidar_free_height_seen[:, cell[0], cell[1]].all()
            }

        # Rays in one frame are correlated observations, not independent
        # evidence. Count each cell once; an endpoint hit takes precedence
        # over other rays crossing the same discretized cell.
        for cell in frame_free - frame_occupied:
            self._add(cell, -self.config.free_update)
        for cell in frame_occupied:
            self._add(cell, self.config.occupied_update)

        # The sensor origin is not the base/footprint center. The runtime
        # clears the actual robot footprint separately; clearing a robot-size
        # disk here would erase nearby furniture in front of an offset lidar.
        if frame_free or frame_occupied:
            self.revision += 1


    def prune_isolated_occupancy(self):
        """Reset isolated, unconfirmed occupied cells back to unknown.

        Floor-return noise leaking into the obstacle band shows up as lone
        weak cells, unlike furniture which is contiguous and repeatedly hit.
        Such cells would otherwise serve as free-ray endpoints that wash out
        real obstacles behind them.
        """

        occupied = (self.log_odds >= self.config.occupied_threshold) & ~self.static_occupied
        if not occupied.any():
            return
        neighbors = np.zeros(occupied.shape, dtype=np.int16)
        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                if row_offset == 0 and col_offset == 0:
                    continue
                source_rows, target_rows = _shift_slices(occupied.shape[0], row_offset)
                source_cols, target_cols = _shift_slices(occupied.shape[1], col_offset)
                neighbors[target_rows, target_cols] += occupied[source_rows, source_cols]
        isolated = (
            occupied
            & (neighbors == 0)
            & (self.log_odds < self.config.sticky_occupied_threshold)
        )
        if isolated.any():
            self.log_odds[isolated] = 0.0
            self.revision += 1

    def update_obstacle_points(self, points: np.ndarray):
        points = _points_array(points)
        if len(points) == 0:
            return
        valid = np.isfinite(points).all(axis=1)
        valid &= points[:, 2] >= self.config.obstacle_height[0]
        valid &= points[:, 2] <= self.config.obstacle_height[1]
        points = points[valid]
        if len(points) > self.config.max_voxel_points_per_update:
            indices = np.linspace(0, len(points) - 1, self.config.max_voxel_points_per_update, dtype=int)
            points = points[indices]
        updated = False
        for point in points:
            cell = self.world_to_cell(point[:2])
            if cell is not None:
                self._add(cell, self.config.occupied_update)
                updated = True
        if updated:
            self.revision += 1

    def mark_free(self, xy: Iterable[float], radius: float):
        center = self.world_to_cell(tuple(xy))
        if center is None:
            return
        cells = int(np.ceil(radius / self.config.grid_resolution))
        changed = False
        for row_offset in range(-cells, cells + 1):
            for col_offset in range(-cells, cells + 1):
                if hypot(row_offset, col_offset) > cells:
                    continue
                cell = (center[0] + row_offset, center[1] + col_offset)
                if self.in_bounds(cell):
                    if self.static_occupied[cell]:
                        continue
                    free_log_odds = min(self.log_odds[cell], -self.config.free_update)
                    if free_log_odds != self.log_odds[cell] or not self.observed[cell]:
                        changed = True
                        self.log_odds[cell] = free_log_odds
                        self.observed[cell] = True
        if changed:
            self.revision += 1

    def mark_occupied(self, xy: Iterable[float], radius: float, evidence: float = 0.85):
        center = self.world_to_cell(tuple(xy))
        if center is None:
            return
        cells = max(1, int(np.ceil(radius / self.config.grid_resolution)))
        for row_offset in range(-cells, cells + 1):
            for col_offset in range(-cells, cells + 1):
                if hypot(row_offset, col_offset) > cells:
                    continue
                cell = (center[0] + row_offset, center[1] + col_offset)
                if self.in_bounds(cell):
                    self.log_odds[cell] = np.clip(self.log_odds[cell] + evidence, -4.0, 4.0)
                    self.observed[cell] = True
        self.revision += 1

    def mark_box_occupied(
        self,
        minimum_xy: Iterable[float],
        maximum_xy: Iterable[float],
        evidence: float = 4.0,
    ):
        minimum = np.asarray(tuple(minimum_xy), dtype=np.float64)
        maximum = np.asarray(tuple(maximum_xy), dtype=np.float64)
        if minimum.shape != (2,) or maximum.shape != (2,) or np.any(minimum > maximum):
            raise ValueError('occupied box bounds must be increasing 2D coordinates')
        if (
            maximum[0] < self.config.x_limits[0]
            or minimum[0] >= self.config.x_limits[1]
            or maximum[1] < self.config.y_limits[0]
            or minimum[1] >= self.config.y_limits[1]
        ):
            return
        row_min = max(0, int(np.floor((minimum[1] - self.config.y_limits[0]) / self.config.grid_resolution)))
        row_max = min(
            self.log_odds.shape[0] - 1,
            int(np.floor((maximum[1] - self.config.y_limits[0]) / self.config.grid_resolution)),
        )
        col_min = max(0, int(np.floor((minimum[0] - self.config.x_limits[0]) / self.config.grid_resolution)))
        col_max = min(
            self.log_odds.shape[1] - 1,
            int(np.floor((maximum[0] - self.config.x_limits[0]) / self.config.grid_resolution)),
        )
        if row_min > row_max or col_min > col_max:
            return
        self.log_odds[row_min : row_max + 1, col_min : col_max + 1] = np.maximum(
            self.log_odds[row_min : row_max + 1, col_min : col_max + 1],
            evidence,
        )
        self.static_occupied[row_min : row_max + 1, col_min : col_max + 1] = True
        self.observed[row_min : row_max + 1, col_min : col_max + 1] = True
        self.revision += 1

    def occupied_mask(self) -> np.ndarray:
        return self.static_occupied | (self.log_odds >= self.config.occupied_threshold)

    def inflated_mask(self) -> np.ndarray:
        dynamic_occupied = (self.log_odds >= self.config.occupied_threshold) & ~self.static_occupied
        inflation_radius = self.config.obstacle_inflation_radius or self.config.robot_radius
        static_radius = self.config.static_obstacle_inflation_radius or inflation_radius
        return self._inflate(dynamic_occupied, inflation_radius) | self._inflate(
            self.static_occupied,
            static_radius,
        )

    def _inflate(self, occupied: np.ndarray, radius: float) -> np.ndarray:
        if not occupied.any():
            return occupied.copy()
        radius_cells = int(np.ceil(radius / self.config.grid_resolution))
        inflated = occupied.copy()
        for row_offset in range(-radius_cells, radius_cells + 1):
            for col_offset in range(-radius_cells, radius_cells + 1):
                if hypot(row_offset, col_offset) * self.config.grid_resolution > radius:
                    continue
                source_rows, target_rows = _shift_slices(occupied.shape[0], row_offset)
                source_cols, target_cols = _shift_slices(occupied.shape[1], col_offset)
                inflated[target_rows, target_cols] |= occupied[source_rows, source_cols]
        return inflated

    def world_to_cell(self, xy: Iterable[float]) -> Optional[GridCell]:
        x, y = (float(value) for value in xy)
        col = int(np.floor((x - self.config.x_limits[0]) / self.config.grid_resolution))
        row = int(np.floor((y - self.config.y_limits[0]) / self.config.grid_resolution))
        cell = (row, col)
        return cell if self.in_bounds(cell) else None

    def cell_to_world(self, cell: GridCell) -> Tuple[float, float]:
        row, col = cell
        x = self.config.x_limits[0] + (col + 0.5) * self.config.grid_resolution
        y = self.config.y_limits[0] + (row + 0.5) * self.config.grid_resolution
        return x, y

    def in_bounds(self, cell: GridCell) -> bool:
        return 0 <= cell[0] < self.log_odds.shape[0] and 0 <= cell[1] < self.log_odds.shape[1]

    def _add(self, cell: GridCell, value: float):
        if not self.in_bounds(cell):
            return
        if value < 0 and self.static_occupied[cell]:
            return
        if value < 0 and self.log_odds[cell] >= self.config.sticky_occupied_threshold:
            value *= self.config.sticky_free_scale
        # This is one scalar per cell, often thousands of calls per scan.
        # Avoid NumPy's array-dispatch overhead without changing the evidence.
        updated = self.log_odds[cell] + value
        self.log_odds[cell] = -4.0 if updated < -4.0 else 4.0 if updated > 4.0 else updated
        self.observed[cell] = True


class SceneGraphMap:
    """Discrete free-space graph plus persistent semantic/geometry landmarks."""

    def __init__(self, config: MappingConfig):
        self.config = config
        self._objects: Dict[str, SceneGraphNode] = {}
        self._object_position_sums: Dict[str, np.ndarray] = {}
        self._object_embedding_sums: Dict[str, np.ndarray] = {}
        self._object_label_counts: Dict[str, Dict[str, int]] = {}

    def label_counts(self, node_id: str) -> Dict[str, int]:
        return dict(self._object_label_counts.get(node_id, {}))

    def update_detections(self, detections: Iterable[SemanticDetection]):
        # One representative per node and capture step. Process the strongest
        # same-frame proposal first; cached results must not add evidence again.
        for detection in sorted(detections, key=lambda item: (item.step, -item.confidence)):
            label = detection.label.strip() or 'unknown'
            labels = tuple(dict.fromkeys(
                observed.strip()
                for observed in (*detection.label_evidence, label)
                if observed.strip()
            ))
            candidates = [
                node
                for node in self._objects.values()
                # Image similarity alone cannot establish object identity across
                # classes: keep their positions, counts and embeddings separate.
                if (canonical_semantic_label(node.label) == canonical_semantic_label(label)
                    or ('qwen_vl' not in detection.sources and 'qwen_vl' not in node.sources
                        and _embedding_similarity(node.embedding, detection.embedding) >= 0.86))
                and np.linalg.norm(
                    np.asarray(node.position[:2]) - np.asarray(detection.position[:2])
                )
                < 0.75
            ]
            if candidates:
                node = min(
                    candidates,
                    key=lambda candidate: np.linalg.norm(
                        np.asarray(candidate.position[:2]) - np.asarray(detection.position[:2])
                    ),
                )
                if 'qwen_vl' in detection.sources and int(detection.step) <= node.last_seen_step:
                    continue
                count = node.observations + 1
                label_counts = self._object_label_counts.setdefault(
                    node.node_id, {node.label: node.observations}
                )
                for observed in labels:
                    label_counts[observed] = label_counts.get(observed, 0) + 1
                position_sum = self._object_position_sums[node.node_id] + np.asarray(detection.position)
                position = tuple(float(value) for value in position_sum / count)
                if self.config.object_position_window > 0:
                    weight = min(count, self.config.object_position_window)
                    position = tuple(
                        float(value)
                        for value in (
                            np.asarray(node.position) + (np.asarray(detection.position) - node.position) / weight
                        )
                    )
                color = detection.color if detection.color is not None else node.color
                confidence = (
                    node.confidence * node.observations + float(detection.confidence)
                ) / count
                embedding = node.embedding
                if detection.embedding is not None:
                    incoming = np.asarray(detection.embedding, dtype=np.float64)
                    accumulated = self._object_embedding_sums.get(
                        node.node_id,
                        np.zeros_like(incoming),
                    )
                    if accumulated.shape == incoming.shape:
                        accumulated = accumulated + incoming
                        norm = float(np.linalg.norm(accumulated))
                        if norm > 1e-12:
                            embedding = tuple(float(value) for value in accumulated / norm)
                        self._object_embedding_sums[node.node_id] = accumulated
                self._objects[node.node_id] = SceneGraphNode(
                    node_id=node.node_id,
                    kind='object',
                    position=position,
                    label=(
                        label
                        if float(detection.confidence) > node.confidence + 0.10
                        else node.label
                    ),
                    color=color,
                    observations=count,
                    confidence=confidence,
                    embedding=embedding,
                    point_count=node.point_count + max(1, int(detection.point_count)),
                    last_seen_step=max(node.last_seen_step, int(detection.step)),
                    sources=tuple(dict.fromkeys((*node.sources, *detection.sources))),
                )
                self._object_position_sums[node.node_id] = position_sum
            else:
                node_id = f'object:{label}:{len(self._objects)}'
                self._objects[node_id] = SceneGraphNode(
                    node_id=node_id,
                    kind='object',
                    position=detection.position,
                    label=label,
                    color=detection.color,
                    confidence=float(detection.confidence),
                    embedding=detection.embedding,
                    point_count=max(1, int(detection.point_count)),
                    last_seen_step=int(detection.step),
                    sources=tuple(dict.fromkeys(detection.sources)),
                )
                self._object_position_sums[node_id] = np.asarray(detection.position, dtype=np.float64)
                self._object_label_counts[node_id] = {observed: 1 for observed in labels}
                if detection.embedding is not None:
                    self._object_embedding_sums[node_id] = np.asarray(
                        detection.embedding,
                        dtype=np.float64,
                    )

    def object_nodes(self) -> Tuple[SceneGraphNode, ...]:
        return tuple(self._objects.values())

    def snapshot(self, occupancy: OccupancyGridMap, voxels: VoxelPointCloudMap) -> SceneGraphSnapshot:
        inflated = occupancy.inflated_mask()
        nodes: List[SceneGraphNode] = []
        edges: List[SceneGraphEdge] = []
        place_by_cell: Dict[GridCell, SceneGraphNode] = {}
        stride = self.config.graph_stride

        for row in range(0, inflated.shape[0], stride):
            for col in range(0, inflated.shape[1], stride):
                if inflated[row, col] or not occupancy.observed[row, col]:
                    continue
                xy = occupancy.cell_to_world((row, col))
                node = SceneGraphNode(
                    node_id=f'place:{row}:{col}',
                    kind='place',
                    position=(xy[0], xy[1], 0.0),
                    label='free_space',
                )
                nodes.append(node)
                place_by_cell[(row, col)] = node

        neighbor_offsets = (
            (-stride, -stride),
            (-stride, 0),
            (-stride, stride),
            (0, stride),
            (stride, stride),
            (stride, 0),
            (stride, -stride),
        )
        for cell, node in place_by_cell.items():
            for row_offset, col_offset in neighbor_offsets:
                neighbor = place_by_cell.get((cell[0] + row_offset, cell[1] + col_offset))
                if neighbor is None:
                    continue
                distance = hypot(row_offset, col_offset) * self.config.grid_resolution
                edges.append(SceneGraphEdge(node.node_id, neighbor.node_id, 'navigable', distance))

        object_nodes = list(self._objects.values())
        occupied_components = _connected_components(occupancy.occupied_mask())
        for component_index, component in enumerate(occupied_components):
            if len(component) < 2:
                continue
            positions = np.asarray([occupancy.cell_to_world(cell) for cell in component])
            centroid = positions.mean(axis=0)
            color = voxels.dominant_color_near(tuple(centroid))
            node = SceneGraphNode(
                node_id=f'geometry:{component_index}',
                kind='geometry',
                position=(float(centroid[0]), float(centroid[1]), 0.5),
                label=_color_label(color),
                color=color,
                observations=len(component),
            )
            object_nodes.append(node)

        nodes.extend(object_nodes)
        place_nodes = tuple(place_by_cell.values())
        for object_node in object_nodes:
            if not place_nodes:
                continue
            nearest = min(
                place_nodes,
                key=lambda place: _distance_xy(place.position, object_node.position),
            )
            distance = _distance_xy(nearest.position, object_node.position)
            edges.append(SceneGraphEdge(object_node.node_id, nearest.node_id, 'near', distance))

        for index, left in enumerate(object_nodes):
            for right in object_nodes[index + 1 :]:
                distance = _distance_xy(left.position, right.position)
                if distance <= 1.5:
                    edges.append(SceneGraphEdge(left.node_id, right.node_id, 'near', distance))
        return SceneGraphSnapshot(nodes=tuple(nodes), edges=tuple(edges))


class AStarMapPlanner:
    """A* over the fused occupancy layer, with unknown-space penalty."""

    _NEIGHBORS = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, sqrt(2.0)),
        (-1, 1, sqrt(2.0)),
        (1, -1, sqrt(2.0)),
        (1, 1, sqrt(2.0)),
    )

    def __init__(self, occupancy: OccupancyGridMap, observed_only: bool = False):
        self.occupancy = occupancy
        self.observed_only = observed_only

    def plan(self, start: Vector3, goal: Vector3) -> Tuple[Vector3, ...]:
        start_cell = self.occupancy.world_to_cell(start[:2])
        goal_cell = self.occupancy.world_to_cell(goal[:2])
        if start_cell is None or goal_cell is None:
            raise PlanningError('start or goal is outside map bounds')
        blocked = self.occupancy.inflated_mask()
        if self.observed_only:
            blocked = blocked | ~self.occupancy.observed
            if blocked[start_cell]:
                raise PlanningError('start is not observed traversable space')
            if blocked[goal_cell]:
                raise PlanningError('goal is not observed traversable space')
        start_cell = self._nearest_open(start_cell, blocked)
        goal_cell = self._nearest_open(goal_cell, blocked)

        clearance_cost = None
        if self.observed_only:
            from scipy.ndimage import distance_transform_edt

            clearance = distance_transform_edt(~blocked) * self.occupancy.config.grid_resolution
            # Prefer room/doorway centers instead of skimming the boundary
            # of the inflated mask. Inflation is the hard limit; this soft
            # cost reserves additional space for locomotion tracking error.
            clearance_cost = 1.0 + 0.5 / np.maximum(clearance, self.occupancy.config.grid_resolution)

        frontier = [(0.0, start_cell)]
        cost_so_far = {start_cell: 0.0}
        parent: Dict[GridCell, GridCell] = {}
        while frontier:
            _, current = heappop(frontier)
            if current == goal_cell:
                break
            for row_offset, col_offset, move_cost in self._NEIGHBORS:
                neighbor = (current[0] + row_offset, current[1] + col_offset)
                if not self.occupancy.in_bounds(neighbor) or blocked[neighbor]:
                    continue
                if (
                    self.observed_only
                    and row_offset
                    and col_offset
                    and (blocked[current[0] + row_offset, current[1]] or blocked[current[0], current[1] + col_offset])
                ):
                    continue
                observed_cost = 1.0 if self.occupancy.observed[neighbor] else self.occupancy.config.unknown_cost
                if clearance_cost is not None:
                    observed_cost = float(clearance_cost[neighbor])
                new_cost = cost_so_far[current] + move_cost * observed_cost
                if new_cost >= cost_so_far.get(neighbor, float('inf')):
                    continue
                cost_so_far[neighbor] = new_cost
                parent[neighbor] = current
                heuristic = hypot(goal_cell[0] - neighbor[0], goal_cell[1] - neighbor[1])
                heappush(frontier, (new_cost + heuristic, neighbor))
        else:
            raise PlanningError('no map path to goal')

        cells = [goal_cell]
        while cells[-1] != start_cell:
            if cells[-1] not in parent:
                raise PlanningError('no map path to goal')
            cells.append(parent[cells[-1]])
        cells.reverse()
        cells = self._simplify(cells, blocked)
        waypoints = [
            (float(x), float(y), float(goal[2]))
            for x, y in (self.occupancy.cell_to_world(cell) for cell in cells[1:])
        ]
        if not waypoints or _distance_xy(waypoints[-1], goal) > self.occupancy.config.grid_resolution:
            waypoints.append(tuple(float(value) for value in goal))
        else:
            waypoints[-1] = tuple(float(value) for value in goal)
        return tuple(waypoints)


    def _nearest_open(self, cell: GridCell, blocked: np.ndarray) -> GridCell:
        if not blocked[cell]:
            return cell
        for radius in range(1, 10):
            candidates = []
            for row in range(cell[0] - radius, cell[0] + radius + 1):
                for col in range(cell[1] - radius, cell[1] + radius + 1):
                    candidate = (row, col)
                    if self.occupancy.in_bounds(candidate) and not blocked[candidate]:
                        candidates.append(candidate)
            if candidates:
                return min(candidates, key=lambda candidate: hypot(candidate[0] - cell[0], candidate[1] - cell[1]))
        raise PlanningError('no open grid cell near requested endpoint')

    def _simplify(self, cells: List[GridCell], blocked: np.ndarray) -> List[GridCell]:
        if len(cells) <= 2:
            return cells
        if self.observed_only:
            # Keep route corners; continuous lookahead is validated at execution.
            return [cells[0]] + [cells[i] for i in range(1, len(cells) - 1)
                if (cells[i][0] - cells[i - 1][0], cells[i][1] - cells[i - 1][1])
                != (cells[i + 1][0] - cells[i][0], cells[i + 1][1] - cells[i][1])] + [cells[-1]]
        simplified = [cells[0]]
        index = 0
        while index < len(cells) - 1:
            next_index = len(cells) - 1
            while next_index > index + 1:
                if all(not blocked[cell] for cell in _bresenham(cells[index], cells[next_index])):
                    break
                next_index -= 1
            simplified.append(cells[next_index])
            index = next_index
        return simplified

    def segment_clear_with_margin(self, start, end, blocked: np.ndarray, radius: float) -> bool:
        """Conservative continuous-world capsule check, including both endpoints.

        Blocked cells are squares, not point obstacles. Their circumscribed
        circles add half a cell diagonal, so sub-cell endpoints and diagonal
        grazing cannot slip through a centre-line rasterization. Unknown space
        outside the finite map is also forbidden. Only the capsule's bounding
        window is inspected; no clearance cache can outlive a map update.
        """
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if (start.shape != (2,) or end.shape != (2,) or not np.isfinite(start).all()
                or not np.isfinite(end).all() or not np.isfinite(radius) or radius < 0):
            return False
        config = self.occupancy.config
        lower = np.minimum(start, end) - radius
        upper = np.maximum(start, end) + radius
        origin = np.array([config.x_limits[0], config.y_limits[0]])
        limit = np.array([config.x_limits[1], config.y_limits[1]])
        if np.any(lower <= origin) or np.any(upper >= limit):
            return False
        resolution = config.grid_resolution
        col0, row0 = np.maximum(0, np.floor((lower-origin)/resolution).astype(int))
        col1, row1 = np.floor((upper-origin)/resolution).astype(int) + 1
        rows, cols = np.nonzero(blocked[row0:row1, col0:col1])
        if not len(rows):
            return True
        centers = np.column_stack((cols+col0+.5, rows+row0+.5))*resolution + origin
        delta = end-start
        length_squared = float(np.dot(delta, delta))
        ratios = (np.clip((centers-start) @ delta / length_squared, 0., 1.)
                  if length_squared > 0 else np.zeros(len(centers)))
        closest = start + ratios[:, None]*delta
        required = radius + resolution*np.sqrt(2.)*.5
        return bool(np.all(np.sum((centers-closest)**2, axis=1) > required**2))

    @staticmethod
    def segment_clear(start: GridCell, end: GridCell, blocked: np.ndarray) -> bool:
        """Check the whole segment, including cells adjacent to diagonal steps."""
        cells = _bresenham(start, end)
        if any(not (0 <= r < blocked.shape[0] and 0 <= c < blocked.shape[1])
               or blocked[r, c] for r, c in cells):
            return False
        for left, right in zip(cells, cells[1:]):
            if left[0] != right[0] and left[1] != right[1]:
                if blocked[left[0], right[1]] or blocked[right[0], left[1]]:
                    return False
        return True


class FusedMap:
    """Own all map layers and expose planning and scene-graph snapshots."""

    _SEMANTIC_OBSTACLE_RADII = {
        'obstacle': 0.42,
        'pedestal': 0.42,
        'door': 0.22,
        'door_frame_hinge': 0.18,
        'door_frame_latch': 0.18,
    }

    def __init__(self, config: MappingConfig = MappingConfig()):
        self.config = config
        self.voxels = VoxelPointCloudMap(config)
        self.occupancy = OccupancyGridMap(config)
        self.scene_graph = SceneGraphMap(config)
        self.planner = AStarMapPlanner(self.occupancy)
        self.lidar_frames = 0
        self.rgb_frames = 0
        self.lidar_points = 0
        self.rgb_points = 0
        self.plan_count = 0
        self.traversable_semantic_labels = set()

    def update_lidar(self, origin: Vector3, points: np.ndarray, step: int, max_range: float):
        points = _points_array(points)
        if len(points) == 0:
            return
        self.voxels.clear_lidar_free_space(origin, points, max_range)
        self.voxels.update(points, step=step, source='lidar')
        self.occupancy.update_lidar(origin, points, max_range=max_range)
        self.lidar_frames += 1
        self.lidar_points += len(points)

    def update_rgb(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        detections: Iterable[SemanticDetection],
        step: int,
        origin: Optional[Vector3] = None,
        max_range: float = 8.0,
        update_occupancy: bool = True,
        update_semantic_occupancy: bool = True,
        clear_free_space: bool = True,
    ):
        points = _points_array(points)
        if len(points) == 0:
            return
        self.voxels.update(points, colors=colors, step=step, source='rgb')
        if update_occupancy:
            if origin is not None and clear_free_space:
                self.occupancy.update_lidar(origin, points, max_range=max_range)
            else:
                self.occupancy.update_obstacle_points(points)
        detections = tuple(detections)
        if update_semantic_occupancy:
            for detection in detections:
                radius = self._SEMANTIC_OBSTACLE_RADII.get(detection.label)
                if radius is not None:
                    self.occupancy.mark_occupied(detection.position[:2], radius)
        self.scene_graph.update_detections(detections)
        self.rgb_frames += 1
        self.rgb_points += len(points)

    def plan(
        self,
        start: Vector3,
        goal: Vector3,
        reinforce_semantics: bool = True,
    ) -> Tuple[Vector3, ...]:
        if reinforce_semantics:
            self._reinforce_semantic_occupancy()
        path = self.planner.plan(start, goal)
        self.plan_count += 1
        return path

    def _reinforce_semantic_occupancy(self):
        for node in self.scene_graph.object_nodes():
            if node.label in self.traversable_semantic_labels:
                continue
            radius = self._SEMANTIC_OBSTACLE_RADII.get(node.label)
            if radius is not None and node.observations >= 2:
                self.occupancy.mark_occupied(node.position[:2], radius, evidence=2.0)

    def set_semantic_labels_traversable(self, labels: Iterable[str]):
        self.traversable_semantic_labels.update(labels)
        for node in self.scene_graph.object_nodes():
            if node.label not in self.traversable_semantic_labels:
                continue
            radius = self._SEMANTIC_OBSTACLE_RADII.get(node.label)
            if radius is not None:
                self.occupancy.mark_free(node.position[:2], radius + self.config.robot_radius)

    def snapshot(self) -> SceneGraphSnapshot:
        return self.scene_graph.snapshot(self.occupancy, self.voxels)

    def statistics(self) -> dict:
        graph = self.snapshot()
        return {
            'lidar_frames': self.lidar_frames,
            'rgb_frames': self.rgb_frames,
            'lidar_points': self.lidar_points,
            'rgb_points': self.rgb_points,
            'voxel_count': self.voxels.voxel_count,
            'lidar_voxel_count': self.voxels.lidar_voxel_count,
            'colored_voxel_count': self.voxels.colored_voxel_count,
            'occupied_cells': int(self.occupancy.occupied_mask().sum()),
            'observed_cells': int(self.occupancy.observed.sum()),
            'scene_graph_nodes': len(graph.nodes),
            'scene_graph_edges': len(graph.edges),
            'scene_graph_places': graph.place_count,
            'scene_graph_objects': graph.object_count,
            'plans': self.plan_count,
            'traversable_semantic_labels': sorted(self.traversable_semantic_labels),
        }


def _points_array(points) -> np.ndarray:
    if points is None:
        return np.empty((0, 3), dtype=np.float32)
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return array.reshape(-1, 3)


def _shift_slices(length: int, offset: int):
    if offset >= 0:
        return slice(0, length - offset), slice(offset, length)
    return slice(-offset, length), slice(0, length + offset)


def _bresenham(start: GridCell, end: GridCell) -> List[GridCell]:
    row0, col0 = start
    row1, col1 = end
    delta_col = abs(col1 - col0)
    delta_row = -abs(row1 - row0)
    step_col = 1 if col0 < col1 else -1
    step_row = 1 if row0 < row1 else -1
    error = delta_col + delta_row
    cells = []
    while True:
        cells.append((row0, col0))
        if row0 == row1 and col0 == col1:
            break
        doubled = 2 * error
        if doubled >= delta_row:
            error += delta_row
            col0 += step_col
        if doubled <= delta_col:
            error += delta_col
            row0 += step_row
    return cells


def _connected_components(mask: np.ndarray) -> List[List[GridCell]]:
    remaining = {tuple(int(value) for value in cell) for cell in np.argwhere(mask)}
    components = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _color_label(color: Optional[Tuple[int, int, int]]) -> str:
    if color is None:
        return 'geometry'
    channels = ('red', 'green', 'blue')
    dominant = int(np.argmax(color))
    if max(color) - min(color) < 30:
        return 'neutral_geometry'
    return f'{channels[dominant]}_geometry'


def _distance_xy(left: Vector3, right: Vector3) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])


def canonical_semantic_label(value: str) -> str:
    """Normalize exact category names/aliases without guessing from substrings."""
    label = ' '.join(str(value).casefold().replace('_', ' ').replace('-', ' ').split())
    return {
        'couch': 'sofa',
        'fridge': 'refrigerator',
        'potted plant': 'plant',
        'tv': 'television',
        'tv screen': 'television',
        'television screen': 'television',
    }.get(label, label)


def _embedding_similarity(left, right) -> float:
    if left is None or right is None:
        return -1.0
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.size == 0:
        return -1.0
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator <= 1e-12:
        return -1.0
    return float(np.dot(left_array, right_array) / denominator)
