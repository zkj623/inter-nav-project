"""Isaac-independent semantic Voronoi graph extraction for occupancy grids.

The graph is rebuilt from the occupancy map either fully or incrementally.
Incremental updates re-thin only a window around changed cells and splice the
result into the cached skeleton; a seam-band consistency guard falls back to a
full rebuild whenever splicing would not be seamless. On top of the flat graph
a hierarchical region layer groups Voronoi nodes into room-like regions that
are separated by narrow doorway edges.
"""

from collections import deque
from dataclasses import asdict, dataclass, field
from heapq import heappop, heappush
from math import ceil, hypot
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from grutopia_extension.interactive_navigation.mapping import (
    AStarMapPlanner,
    OccupancyGridMap,
    PlanningError,
    SceneGraphNode,
)

GridCell = Tuple[int, int]
GridWindow = Tuple[int, int, int, int]

try:
    from scipy import ndimage as _ndimage
except ImportError:  # pragma: no cover - scipy is available in the runtime env
    _ndimage = None

_EIGHT_CONNECTIVITY = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class SemanticVoronoiConfig:
    """Parameters controlling skeleton extraction and graph compression."""

    min_clearance: float = 0.0
    spur_length: float = 0.30
    frontier_min_size: int = 1
    incremental: bool = True
    verify_incremental: bool = False  # expensive full-rebuild oracle for smoke tests
    incremental_margin_cells: int = 4
    incremental_guard_cells: int = 3
    incremental_max_window_fraction: float = 0.75
    region_segmentation: bool = True
    doorway_max_width: float = 1.6
    doorway_clearance_ratio: float = 1.0

    def __post_init__(self):
        if self.min_clearance < 0.0 or self.spur_length < 0.0:
            raise ValueError('clearance and spur length must be non-negative')
        if self.frontier_min_size < 1:
            raise ValueError('frontier_min_size must be positive')
        if self.incremental_margin_cells < 0 or self.incremental_guard_cells < 1:
            raise ValueError('incremental margin must be non-negative and guard positive')
        if not 0.0 < self.incremental_max_window_fraction <= 1.0:
            raise ValueError('incremental_max_window_fraction must be in (0, 1]')
        if self.doorway_max_width <= 0.0 or not 0.0 < self.doorway_clearance_ratio <= 1.0:
            raise ValueError('doorway thresholds must be positive (ratio at most 1)')


@dataclass(frozen=True)
class VoronoiNode:
    node_id: str
    kind: str
    cell: GridCell
    position: Tuple[float, float]
    clearance: float
    cells: Tuple[GridCell, ...] = ()


@dataclass(frozen=True)
class VoronoiEdge:
    edge_id: str
    source: str
    target: str
    cells: Tuple[GridCell, ...]
    length: float
    min_clearance: float


@dataclass(frozen=True)
class Frontier:
    frontier_id: str
    cells: Tuple[GridCell, ...]
    centroid: Tuple[float, float]
    node_id: Optional[str]
    distance: Optional[float]
    region_id: Optional[str] = None


@dataclass(frozen=True)
class SemanticAttachment:
    object_id: str
    node_id: Optional[str]
    distance: Optional[float]
    label: str


@dataclass(frozen=True)
class Doorway:
    """A narrow passage edge connecting two regions."""

    doorway_id: str
    edge_id: str
    position: Tuple[float, float]
    width: float
    regions: Tuple[str, str]


@dataclass(frozen=True)
class Region:
    """A room-like group of Voronoi nodes bounded by doorways."""

    region_id: str
    node_ids: Tuple[str, ...]
    centroid: Tuple[float, float]
    area: float
    labels: Tuple[Tuple[str, int], ...]
    adjacent: Tuple[str, ...]
    frontier_count: int = 0


@dataclass(frozen=True)
class SemanticVoronoiSnapshot:
    revision: int
    nodes: Tuple[VoronoiNode, ...]
    edges: Tuple[VoronoiEdge, ...]
    frontiers: Tuple[Frontier, ...]
    semantics: Tuple[SemanticAttachment, ...]
    skeleton_cells: Tuple[GridCell, ...]
    regions: Tuple[Region, ...] = ()
    doorways: Tuple[Doorway, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _MutableStatistics:
    builds: int = 0
    cache_hits: int = 0
    unchanged_updates: int = 0
    incremental_updates: int = 0
    incremental_fallbacks: int = 0
    incremental_fallback_reasons: Dict[str, int] = field(default_factory=dict)
    last_changed_cells: int = 0
    last_traversable_changed_cells: int = 0
    last_observed_changed_cells: int = 0
    last_changed_components: int = 0
    last_window_count: int = 0
    last_window_cells: int = 0
    last_window_work_cells: int = 0
    last_seam_band_cells: int = 0
    last_window_fraction: float = 0.0
    last_window_work_fraction: float = 0.0
    last_window_to_changed_ratio: float = 0.0
    last_incremental_fallback_reason: Optional[str] = None
    incremental_window_evaluations: int = 0
    incremental_changed_cells_total: int = 0
    incremental_window_cells_total: int = 0
    incremental_window_work_cells_total: int = 0
    incremental_seconds: float = 0.0
    full_build_seconds: float = 0.0
    verification_seconds: float = 0.0
    incremental_max_window_count: int = 0
    incremental_max_window_fraction: float = 0.0
    incremental_seam_checks: int = 0
    incremental_seam_mismatches: int = 0
    incremental_verification_checks: int = 0
    incremental_verification_mismatches: int = 0
    incremental_verification_mismatched_cells: int = 0
    last_verification_mismatch_revision: Optional[int] = None


class SemanticVoronoiGraph:
    """Build and cache a compact safe medial-axis graph for an occupancy map.

    ``update`` diffs the traversable mask against the previous call and, when
    the change is local, re-thins only a padded window around it. The spliced
    skeleton must match the cached one on a seam band placed beyond both the
    change propagation distance and the window truncation artifact distance;
    otherwise the update falls back to a deterministic full rebuild.
    """

    def __init__(
        self,
        occupancy: OccupancyGridMap,
        config: Optional[SemanticVoronoiConfig] = None,
    ):
        self.occupancy = occupancy
        self.config = config or SemanticVoronoiConfig()
        self._snapshot: Optional[SemanticVoronoiSnapshot] = None
        self._semantics: Dict[str, SceneGraphNode] = {}
        self._stats = _MutableStatistics()
        self._last_traversable: Optional[np.ndarray] = None
        self._last_observed: Optional[np.ndarray] = None
        self._last_clearance: Optional[np.ndarray] = None
        self._last_skeleton: Optional[np.ndarray] = None
        self._last_region_assignment: Optional[Tuple[np.ndarray, Tuple[str, ...]]] = None

    @property
    def nodes(self) -> Tuple[VoronoiNode, ...]:
        return self.snapshot().nodes

    @property
    def edges(self) -> Tuple[VoronoiEdge, ...]:
        return self.snapshot().edges

    @property
    def frontiers(self) -> Tuple[Frontier, ...]:
        return self.snapshot().frontiers

    @property
    def regions(self) -> Tuple[Region, ...]:
        return self.snapshot().regions

    def update(
        self,
        changed_cells: Optional[Iterable[GridCell]] = None,
        force: bool = False,
    ) -> SemanticVoronoiSnapshot:
        changed_hint = tuple(changed_cells or ())
        if self._snapshot is not None and self._snapshot.revision == self.occupancy.revision and not force:
            self._stats.cache_hits += 1
            return self._snapshot

        inflated = np.asarray(self.occupancy.inflated_mask(), dtype=bool)
        observed = np.asarray(self.occupancy.observed, dtype=bool).copy()
        traversable = observed & ~inflated
        resolution = float(self.occupancy.config.grid_resolution)

        traversable_unchanged = (
            self._snapshot is not None
            and self._last_traversable is not None
            and np.array_equal(traversable, self._last_traversable)
        )
        observed_unchanged = (
            self._last_observed is not None
            and np.array_equal(observed, self._last_observed)
        )
        if traversable_unchanged and observed_unchanged and not force:
            self._stats.last_changed_cells = 0
            self._stats.last_traversable_changed_cells = 0
            self._stats.last_observed_changed_cells = 0
            self._reset_incremental_diagnostics()
            self._stats.unchanged_updates += 1
            self._snapshot = SemanticVoronoiSnapshot(
                revision=int(self.occupancy.revision),
                nodes=self._snapshot.nodes,
                edges=self._snapshot.edges,
                frontiers=self._snapshot.frontiers,
                semantics=self._snapshot.semantics,
                skeleton_cells=self._snapshot.skeleton_cells,
                regions=self._snapshot.regions,
                doorways=self._snapshot.doorways,
            )
            return self._snapshot

        clearance = (
            self._last_clearance
            if traversable_unchanged and self._last_clearance is not None
            else _distance_to_false(traversable) * resolution
        )

        skeleton: Optional[np.ndarray] = None
        can_increment = (
            not force
            and self.config.incremental
            and self._last_skeleton is not None
            and self._last_traversable is not None
        )
        if can_increment:
            # The authoritative change set is the traversable-mask diff; the
            # caller-provided cells are only a hint and never force a rebuild.
            changed = traversable != self._last_traversable
            observed_changed = (
                observed != self._last_observed if self._last_observed is not None else np.ones_like(observed)
            )
            self._stats.last_changed_cells = int((changed | observed_changed).sum())
            self._stats.last_traversable_changed_cells = int(changed.sum())
            self._stats.last_observed_changed_cells = int(observed_changed.sum())
            self._reset_incremental_diagnostics()
            if not changed.any():
                skeleton = self._last_skeleton
                self._stats.unchanged_updates += 1
            else:
                started = perf_counter()
                skeleton = self._incremental_skeleton(traversable, clearance, changed, resolution)
                self._stats.incremental_seconds += perf_counter() - started
                if skeleton is not None:
                    self._stats.incremental_updates += 1
                else:
                    self._stats.incremental_fallbacks += 1
                    reason = self._stats.last_incremental_fallback_reason or 'unknown'
                    self._stats.incremental_fallback_reasons[reason] = (
                        self._stats.incremental_fallback_reasons.get(reason, 0) + 1
                    )
        else:
            self._stats.last_changed_cells = len(changed_hint)
            self._stats.last_traversable_changed_cells = 0
            self._stats.last_observed_changed_cells = 0
            self._reset_incremental_diagnostics()

        if skeleton is None:
            started = perf_counter()
            skeleton = self._build_skeleton(traversable, clearance, resolution)
            self._stats.full_build_seconds += perf_counter() - started
            self._stats.builds += 1

        # When the traversable mask is unchanged the clearance field and the
        # skeleton are unchanged too, so the compressed graph can be reused.
        reuse_graph = traversable_unchanged
        if reuse_graph:
            nodes: List[VoronoiNode] = list(self._snapshot.nodes)
            edges: List[VoronoiEdge] = list(self._snapshot.edges)
        else:
            nodes, edges = self._compress(skeleton, clearance)
        frontiers = self._extract_frontiers(traversable, nodes)
        semantics = self._semantic_attachments(nodes, traversable)
        regions: List[Region] = []
        doorways: List[Doorway] = []
        if self.config.region_segmentation:
            regions, doorways, frontiers = self._segment_regions(
                nodes,
                edges,
                frontiers,
                semantics,
                traversable,
                skeleton,
                clearance,
            )
        self._snapshot = SemanticVoronoiSnapshot(
            revision=int(self.occupancy.revision),
            nodes=tuple(nodes),
            edges=tuple(edges),
            frontiers=tuple(frontiers),
            semantics=tuple(semantics),
            skeleton_cells=tuple(_mask_cells(skeleton)),
            regions=tuple(regions),
            doorways=tuple(doorways),
        )
        self._last_traversable = traversable
        self._last_observed = observed
        self._last_clearance = clearance
        self._last_skeleton = skeleton
        return self._snapshot

    def snapshot(self) -> SemanticVoronoiSnapshot:
        return self.update()

    def cached_snapshot(self) -> SemanticVoronoiSnapshot:
        """Return the latest graph without rebuilding for a newer occupancy revision."""

        return self.update() if self._snapshot is None else self._snapshot

    def region_assignment(self) -> Optional[Tuple[np.ndarray, Tuple[str, ...]]]:
        """Return the cached per-cell region index grid and sorted region ids.

        The grid matches the occupancy shape with ``-1`` for unassigned cells.
        Consistent with :meth:`cached_snapshot`, no rebuild is triggered.
        """

        return self._last_region_assignment

    def attach_semantics(self, objects: Iterable[SceneGraphNode]) -> Tuple[SemanticAttachment, ...]:
        """Attach semantic objects to their nearest reachable node and refresh the snapshot."""

        self._semantics = {obj.node_id: obj for obj in objects}
        snapshot = self.update()
        traversable = np.asarray(self.occupancy.observed, dtype=bool) & ~self.occupancy.inflated_mask()
        semantics = tuple(self._semantic_attachments(snapshot.nodes, traversable))
        regions = snapshot.regions
        frontiers = snapshot.frontiers
        if self.config.region_segmentation and self._last_skeleton is not None and self._last_clearance is not None:
            region_list, doorway_list, frontier_list = self._segment_regions(
                list(snapshot.nodes),
                list(snapshot.edges),
                list(snapshot.frontiers),
                list(semantics),
                traversable,
                self._last_skeleton,
                self._last_clearance,
            )
            regions = tuple(region_list)
            frontiers = tuple(frontier_list)
            doorways = tuple(doorway_list)
        else:
            doorways = snapshot.doorways
        self._snapshot = SemanticVoronoiSnapshot(
            revision=snapshot.revision,
            nodes=snapshot.nodes,
            edges=snapshot.edges,
            frontiers=frontiers,
            semantics=semantics,
            skeleton_cells=snapshot.skeleton_cells,
            regions=regions,
            doorways=doorways,
        )
        return semantics

    def statistics(self) -> dict:
        snapshot = self.cached_snapshot()
        result = asdict(self._stats)
        result.update(
            {
                'revision': snapshot.revision,
                'node_count': len(snapshot.nodes),
                'edge_count': len(snapshot.edges),
                'frontier_count': len(snapshot.frontiers),
                'semantic_count': len(snapshot.semantics),
                'skeleton_cell_count': len(snapshot.skeleton_cells),
                'region_count': len(snapshot.regions),
                'doorway_count': len(snapshot.doorways),
            }
        )
        return result

    def _reset_incremental_diagnostics(self):
        self._stats.last_changed_components = 0
        self._stats.last_window_count = 0
        self._stats.last_window_cells = 0
        self._stats.last_window_work_cells = 0
        self._stats.last_seam_band_cells = 0
        self._stats.last_window_fraction = 0.0
        self._stats.last_window_work_fraction = 0.0
        self._stats.last_window_to_changed_ratio = 0.0
        self._stats.last_incremental_fallback_reason = None

    def _record_incremental_window_evaluation(self, changed_cells: int):
        self._stats.incremental_window_evaluations += 1
        self._stats.incremental_changed_cells_total += int(changed_cells)
        self._stats.incremental_window_cells_total += self._stats.last_window_cells
        self._stats.incremental_window_work_cells_total += self._stats.last_window_work_cells
        self._stats.incremental_max_window_count = max(
            self._stats.incremental_max_window_count,
            self._stats.last_window_count,
        )
        self._stats.incremental_max_window_fraction = max(
            self._stats.incremental_max_window_fraction,
            self._stats.last_window_fraction,
        )

    def to_dict(self) -> dict:
        return self.cached_snapshot().to_dict()

    def _build_skeleton(
        self,
        traversable: np.ndarray,
        clearance: np.ndarray,
        resolution: float,
    ) -> np.ndarray:
        spur_cells = int(np.ceil(self.config.spur_length / resolution))
        skeleton = _thin(traversable)
        skeleton = _prune_spurs(skeleton, spur_cells)
        if self.config.min_clearance > 0.0:
            skeleton &= clearance + 1e-12 >= self.config.min_clearance
            skeleton = _prune_spurs(skeleton, spur_cells)
        skeleton &= traversable
        return skeleton

    def _incremental_skeleton(
        self,
        traversable: np.ndarray,
        clearance: np.ndarray,
        changed: np.ndarray,
        resolution: float,
    ) -> Optional[np.ndarray]:
        """Re-thin a window around changed cells and splice it into the cache.

        Returns ``None`` when the update cannot be proven seamless, in which
        case the caller performs a full rebuild.
        """

        if not changed.any():
            return self._last_skeleton

        shape = traversable.shape
        spur_cells = int(np.ceil(self.config.spur_length / resolution))
        guard = int(self.config.incremental_guard_cells)
        base_clusters = [set(component) for component in _components(set(_mask_cells(changed)))]
        self._stats.last_changed_components = len(base_clusters)

        # A single bounding box around every changed cell is pathological for
        # sparse updates: two small, distant changes make the untouched space
        # between them part of the update window. Build one independently
        # guarded window per local cluster and merge only overlapping seam/write
        # zones. Start from a deliberately local reach; when the seam proves it
        # insufficient, grow it geometrically and retry before falling back.
        reach_boost = 0
        while True:
            clusters = [set(cluster) for cluster in base_clusters]
            while True:
                specifications = [
                    self._incremental_window(
                        cluster,
                        clearance,
                        resolution,
                        shape,
                        spur_cells,
                        guard,
                        reach_boost,
                    )
                    for cluster in clusters
                ]
                parent = list(range(len(clusters)))

                def find(index: int) -> int:
                    while parent[index] != index:
                        parent[index] = parent[parent[index]]
                        index = parent[index]
                    return index

                def union(left: int, right: int):
                    left_root, right_root = find(left), find(right)
                    if left_root != right_root:
                        parent[right_root] = left_root

                for left in range(len(specifications)):
                    for right in range(left + 1, len(specifications)):
                        if _windows_overlap(specifications[left][2], specifications[right][2]):
                            union(left, right)

                merged: Dict[int, Set[GridCell]] = {}
                for index, cluster in enumerate(clusters):
                    merged.setdefault(find(index), set()).update(cluster)
                next_clusters = sorted(merged.values(), key=min)
                if len(next_clusters) == len(clusters):
                    break
                clusters = next_clusters

            windows = [specification[0] for specification in specifications]
            window_cells = _window_union_area(windows, shape)
            window_work_cells = sum(_window_area(window) for window in windows)
            seam_band_cells = sum(
                _window_area(specification[2]) - _window_area(specification[1])
                for specification in specifications
            )
            changed_cells = int(changed.sum())
            self._stats.last_window_count = len(specifications)
            self._stats.last_window_cells = int(window_cells)
            self._stats.last_window_work_cells = int(window_work_cells)
            self._stats.last_seam_band_cells = int(seam_band_cells)
            self._stats.last_window_fraction = float(window_cells / traversable.size)
            self._stats.last_window_work_fraction = float(window_work_cells / traversable.size)
            self._stats.last_window_to_changed_ratio = float(window_cells / max(changed_cells, 1))
            if window_cells > self.config.incremental_max_window_fraction * traversable.size:
                self._stats.last_incremental_fallback_reason = 'window_fraction'
                self._record_incremental_window_evaluation(changed_cells)
                return None

            spliced = self._last_skeleton.copy()
            seam_mismatch = False
            for window, inner, band in specifications:
                sub_traversable = traversable[window[0] : window[1], window[2] : window[3]]
                sub_clearance = clearance[window[0] : window[1], window[2] : window[3]]
                sub_skeleton = _thin(sub_traversable)
                sub_skeleton = _prune_spurs(sub_skeleton, spur_cells)
                if self.config.min_clearance > 0.0:
                    sub_skeleton &= sub_clearance + 1e-12 >= self.config.min_clearance
                    sub_skeleton = _prune_spurs(sub_skeleton, spur_cells)
                sub_skeleton &= sub_traversable

                old_band = self._last_skeleton[band[0] : band[1], band[2] : band[3]]
                new_band = sub_skeleton[
                    band[0] - window[0] : band[1] - window[0],
                    band[2] - window[2] : band[3] - window[2],
                ]
                ring = np.ones(old_band.shape, dtype=bool)
                interior = (
                    inner[0] - band[0],
                    old_band.shape[0] - (band[1] - inner[1]),
                    inner[2] - band[2],
                    old_band.shape[1] - (band[3] - inner[3]),
                )
                ring[interior[0] : interior[1], interior[2] : interior[3]] = False
                self._stats.incremental_seam_checks += 1
                if np.any(old_band[ring] != new_band[ring]):
                    self._stats.incremental_seam_mismatches += 1
                    seam_mismatch = True
                    break

                spliced[inner[0] : inner[1], inner[2] : inner[3]] = sub_skeleton[
                    inner[0] - window[0] : inner[1] - window[0],
                    inner[2] - window[2] : inner[3] - window[2],
                ]

            if not seam_mismatch:
                spliced &= traversable
                self._record_incremental_window_evaluation(changed_cells)
                if self.config.verify_incremental:
                    # The seam is a cheap guard, not a proof of global equality.
                    # Smoke runs compare the complete result and fall back safely.
                    started = perf_counter()
                    reference = self._build_skeleton(traversable, clearance, resolution)
                    self._stats.verification_seconds += perf_counter() - started
                    mismatched_cells = int(np.count_nonzero(spliced != reference))
                    self._stats.incremental_verification_checks += 1
                    if mismatched_cells:
                        self._stats.incremental_verification_mismatches += 1
                        self._stats.incremental_verification_mismatched_cells += mismatched_cells
                        self._stats.last_verification_mismatch_revision = int(self.occupancy.revision)
                        self._stats.last_incremental_fallback_reason = 'verification_mismatch'
                        return None
                return spliced

            next_boost = (
                max(guard, int(self.config.incremental_margin_cells))
                if reach_boost == 0
                else 2 * reach_boost
            )
            if next_boost > max(shape):
                self._stats.last_incremental_fallback_reason = 'seam_mismatch'
                self._record_incremental_window_evaluation(changed_cells)
                return None
            reach_boost = next_boost

    def _incremental_window(
        self,
        changed_cells: Set[GridCell],
        clearance: np.ndarray,
        resolution: float,
        shape: Tuple[int, int],
        spur_cells: int,
        guard: int,
        reach_boost: int = 0,
    ) -> Optional[Tuple[GridWindow, GridWindow, GridWindow]]:
        rows = [cell[0] for cell in changed_cells]
        cols = [cell[1] for cell in changed_cells]
        row_min, row_max = min(rows), max(rows)
        col_min, col_max = min(cols), max(cols)

        # Estimate propagation from the cells that actually changed. Expanding
        # the probe recursively pulled unrelated high-clearance room centers
        # into the estimate and made most windows nearly map-sized. Both the
        # previous and current clearance are sampled so inserting an obstacle
        # in open space still receives the old free-space radius. If this local
        # estimate is optimistic, the seam comparison below rejects the splice
        # and safely falls back to a full rebuild.
        local = clearance[rows, cols]
        previous = self._last_clearance[rows, cols] if self._last_clearance is not None else local
        max_clearance = max(
            float(local.max()) if local.size else 0.0,
            float(previous.max()) if previous.size else 0.0,
        )
        reach = (
            int(ceil(max_clearance / resolution))
            + spur_cells
            + reach_boost
        )

        window = (
            max(0, row_min - 2 * reach - guard),
            min(shape[0], row_max + 2 * reach + guard + 1),
            max(0, col_min - 2 * reach - guard),
            min(shape[1], col_max + 2 * reach + guard + 1),
        )
        inner = (
            max(window[0], row_min - reach),
            min(window[1], row_max + reach + 1),
            max(window[2], col_min - reach),
            min(window[3], col_max + reach + 1),
        )
        band = (
            max(window[0], inner[0] - guard),
            min(window[1], inner[1] + guard),
            max(window[2], inner[2] - guard),
            min(window[3], inner[3] + guard),
        )
        return window, inner, band

    def _compress(
        self,
        skeleton: np.ndarray,
        clearance: np.ndarray,
    ) -> Tuple[List[VoronoiNode], List[VoronoiEdge]]:
        cells = set(_mask_cells(skeleton))
        if not cells:
            return [], []
        adjacency = {cell: tuple(_connected_neighbors(cell, cells)) for cell in cells}
        junction_cells = {cell for cell, neighbors in adjacency.items() if len(neighbors) >= 3}
        junction_groups = _components(junction_cells)
        node_groups: List[Tuple[str, Set[GridCell]]] = [('junction', group) for group in junction_groups]
        node_groups.extend(('endpoint', {cell}) for cell in cells if len(adjacency[cell]) <= 1)

        covered = set().union(*(group for _, group in node_groups)) if node_groups else set()
        components = _components(cells)
        for component in components:
            if not component & covered:
                anchor = min(component)
                node_groups.append(('corridor', {anchor}))
                covered.add(anchor)

        cell_to_node: Dict[GridCell, str] = {}
        nodes = []
        for kind, group in sorted(node_groups, key=lambda item: (min(item[1]), item[0])):
            representative = max(sorted(group), key=lambda cell: (clearance[cell], -cell[0], -cell[1]))
            node_id = f'voronoi:{kind}:{min(group)[0]}:{min(group)[1]}'
            node = VoronoiNode(
                node_id=node_id,
                kind=kind,
                cell=representative,
                position=self.occupancy.cell_to_world(representative),
                clearance=float(clearance[representative]),
                cells=tuple(sorted(group)),
            )
            nodes.append(node)
            for cell in group:
                cell_to_node[cell] = node_id

        node_by_id = {node.node_id: node for node in nodes}
        edges: List[VoronoiEdge] = []
        visited_links: Set[Tuple[GridCell, GridCell]] = set()
        for start_cell in sorted(cell_to_node):
            source = cell_to_node[start_cell]
            for neighbor in adjacency[start_cell]:
                link = _ordered_link(start_cell, neighbor)
                if link in visited_links or cell_to_node.get(neighbor) == source:
                    visited_links.add(link)
                    continue
                path = [start_cell, neighbor]
                visited_links.add(link)
                previous, current = start_cell, neighbor
                while current not in cell_to_node:
                    choices = [cell for cell in adjacency[current] if cell != previous]
                    if not choices:
                        break
                    next_cell = min(choices)
                    visited_links.add(_ordered_link(current, next_cell))
                    path.append(next_cell)
                    previous, current = current, next_cell
                target = cell_to_node.get(current)
                if target is None:
                    continue
                edge_cells = tuple(path)
                length = sum(_cell_distance(a, b) for a, b in zip(edge_cells, edge_cells[1:]))
                length *= self.occupancy.config.grid_resolution
                edge_key = min(source, target), max(source, target), edge_cells
                edge_id = f'edge:{edge_key[0]}:{edge_key[1]}:{len(edges)}'
                edges.append(
                    VoronoiEdge(
                        edge_id=edge_id,
                        source=source,
                        target=target,
                        cells=edge_cells,
                        length=float(length),
                        min_clearance=float(min(clearance[cell] for cell in edge_cells)),
                    )
                )
        edges.sort(key=lambda edge: (edge.source, edge.target, edge.cells))
        # IDs should not depend on traversal order.
        edges = [
            VoronoiEdge(
                edge_id=f'edge:{edge.source}:{edge.target}:{edge.cells[0][0]}:{edge.cells[0][1]}',
                source=edge.source,
                target=edge.target,
                cells=edge.cells,
                length=edge.length,
                min_clearance=edge.min_clearance,
            )
            for edge in edges
        ]
        return sorted(node_by_id.values(), key=lambda node: node.node_id), edges

    def _extract_frontiers(
        self,
        traversable: np.ndarray,
        nodes: Sequence[VoronoiNode],
    ) -> List[Frontier]:
        unknown = ~np.asarray(self.occupancy.observed, dtype=bool)
        adjacent_unknown = np.zeros_like(unknown)
        adjacent_unknown[1:] |= unknown[:-1]
        adjacent_unknown[:-1] |= unknown[1:]
        adjacent_unknown[:, 1:] |= unknown[:, :-1]
        adjacent_unknown[:, :-1] |= unknown[:, 1:]
        frontier_mask = traversable & adjacent_unknown
        groups = [
            group
            for group in _components_from_mask(frontier_mask)
            if len(group) >= self.config.frontier_min_size
        ]
        component_labels = _component_labels(traversable)
        result = []
        for group in sorted(groups, key=min):
            cells = tuple(sorted(group))
            world = np.asarray([self.occupancy.cell_to_world(cell) for cell in cells])
            centroid = tuple(float(value) for value in world.mean(axis=0))
            labels = [component_labels[cell] for cell in cells if component_labels[cell] >= 0]
            reachable_nodes = []
            if labels:
                label = min(labels)
                reachable_nodes = [node for node in nodes if component_labels[node.cell] == label]
            nearest = _nearest_node(centroid, reachable_nodes)
            result.append(
                Frontier(
                    frontier_id=f'frontier:{cells[0][0]}:{cells[0][1]}',
                    cells=cells,
                    centroid=centroid,
                    node_id=nearest[0].node_id if nearest else None,
                    distance=nearest[1] if nearest else None,
                )
            )
        return result

    def _semantic_attachments(
        self,
        nodes: Sequence[VoronoiNode],
        traversable: np.ndarray,
    ) -> List[SemanticAttachment]:
        labels = _component_labels(traversable)
        attachments = []
        for obj in sorted(self._semantics.values(), key=lambda item: item.node_id):
            xy = (float(obj.position[0]), float(obj.position[1]))
            cell = self.occupancy.world_to_cell(xy)
            candidates: List[VoronoiNode] = []
            if cell is not None:
                anchor = cell if labels[cell] >= 0 else _nearest_true_cell(cell, traversable)
                if anchor is not None and labels[anchor] >= 0:
                    candidates = [node for node in nodes if labels[node.cell] == labels[anchor]]
            nearest = _nearest_node(xy, candidates)
            attachments.append(
                SemanticAttachment(
                    object_id=obj.node_id,
                    node_id=nearest[0].node_id if nearest else None,
                    distance=nearest[1] if nearest else None,
                    label=obj.label,
                )
            )
        return attachments

    def _segment_regions(
        self,
        nodes: Sequence[VoronoiNode],
        edges: Sequence[VoronoiEdge],
        frontiers: Sequence[Frontier],
        semantics: Sequence[SemanticAttachment],
        traversable: np.ndarray,
        skeleton: np.ndarray,
        clearance: np.ndarray,
    ) -> Tuple[List[Region], List[Doorway], List[Frontier]]:
        """Split the node graph at doorway edges into room-like regions."""

        if not nodes:
            return [], [], list(frontiers)
        node_by_id = {node.node_id: node for node in nodes}

        # Multigraph adjacency keyed by edge index so parallel edges (loops
        # around furniture) are distinguished from genuine separations.
        neighbor_edges: Dict[str, List[Tuple[str, int]]] = {node.node_id: [] for node in nodes}
        for index, edge in enumerate(edges):
            if edge.source in neighbor_edges and edge.target in neighbor_edges:
                neighbor_edges[edge.source].append((edge.target, index))
                neighbor_edges[edge.target].append((edge.source, index))

        def is_bridge(excluded_index: int, edge: VoronoiEdge) -> bool:
            frontier_nodes = [edge.source]
            seen = {edge.source}
            while frontier_nodes:
                current = frontier_nodes.pop()
                for neighbor, index in neighbor_edges[current]:
                    if index == excluded_index or neighbor in seen:
                        continue
                    if neighbor == edge.target:
                        return False
                    seen.add(neighbor)
                    frontier_nodes.append(neighbor)
            return True

        def is_doorway(index: int, edge: VoronoiEdge) -> bool:
            source = node_by_id.get(edge.source)
            target = node_by_id.get(edge.target)
            if source is None or target is None or edge.source == edge.target:
                return False
            # A dead-end branch cannot separate two rooms.
            if source.kind == 'endpoint' or target.kind == 'endpoint':
                return False
            if 2.0 * edge.min_clearance > self.config.doorway_max_width:
                return False
            wider_side = min(source.clearance, target.clearance)
            if edge.min_clearance >= self.config.doorway_clearance_ratio * wider_side:
                return False
            # Real doorways are cut edges; narrow gaps beside furniture have a
            # detour around the obstacle and must not split the room.
            return is_bridge(index, edge)

        doorway_edges = [edge for index, edge in enumerate(edges) if is_doorway(index, edge)]
        doorway_ids = {edge.edge_id for edge in doorway_edges}

        # Region grouping: connected components of the node graph without
        # doorway edges. Nodes disconnected in grid space stay separate too.
        parent = {node.node_id: node.node_id for node in nodes}

        def find(node_id: str) -> str:
            while parent[node_id] != node_id:
                parent[node_id] = parent[parent[node_id]]
                node_id = parent[node_id]
            return node_id

        for edge in edges:
            if edge.edge_id in doorway_ids:
                continue
            left, right = find(edge.source), find(edge.target)
            if left != right:
                parent[max(left, right)] = min(left, right)

        groups: Dict[str, List[VoronoiNode]] = {}
        for node in nodes:
            groups.setdefault(find(node.node_id), []).append(node)
        region_id_by_node: Dict[str, str] = {}
        region_members: Dict[str, List[VoronoiNode]] = {}
        for members in groups.values():
            anchor = min(min(node.cells) if node.cells else node.cell for node in members)
            region_id = f'region:{anchor[0]}:{anchor[1]}'
            region_members[region_id] = members
            for node in members:
                region_id_by_node[node.node_id] = region_id

        doorways: List[Doorway] = []
        adjacency: Dict[str, Set[str]] = {region_id: set() for region_id in region_members}
        for edge in sorted(doorway_edges, key=lambda item: item.edge_id):
            left = region_id_by_node[edge.source]
            right = region_id_by_node[edge.target]
            if left == right:
                continue
            narrowest = min(
                edge.cells,
                key=lambda cell: (float(clearance[cell]), cell),
            )
            doorways.append(
                Doorway(
                    doorway_id=f'doorway:{narrowest[0]}:{narrowest[1]}',
                    edge_id=edge.edge_id,
                    position=self.occupancy.cell_to_world(narrowest),
                    width=float(2.0 * edge.min_clearance),
                    regions=(min(left, right), max(left, right)),
                )
            )
            adjacency[left].add(right)
            adjacency[right].add(left)

        areas = self._region_areas(region_members, region_id_by_node, edges, doorway_ids, traversable, skeleton)
        label_counts: Dict[str, Dict[str, int]] = {region_id: {} for region_id in region_members}
        for attachment in semantics:
            if attachment.node_id is None:
                continue
            region_id = region_id_by_node.get(attachment.node_id)
            if region_id is None:
                continue
            counts = label_counts[region_id]
            counts[attachment.label] = counts.get(attachment.label, 0) + 1

        located_frontiers: List[Frontier] = []
        frontier_counts: Dict[str, int] = {region_id: 0 for region_id in region_members}
        for frontier in frontiers:
            region_id = None if frontier.node_id is None else region_id_by_node.get(frontier.node_id)
            if region_id is not None:
                frontier_counts[region_id] += 1
            located_frontiers.append(
                Frontier(
                    frontier_id=frontier.frontier_id,
                    cells=frontier.cells,
                    centroid=frontier.centroid,
                    node_id=frontier.node_id,
                    distance=frontier.distance,
                    region_id=region_id,
                )
            )

        regions: List[Region] = []
        for region_id in sorted(region_members):
            members = region_members[region_id]
            positions = np.asarray([node.position for node in members], dtype=np.float64)
            counts = sorted(label_counts[region_id].items(), key=lambda item: (-item[1], item[0]))
            regions.append(
                Region(
                    region_id=region_id,
                    node_ids=tuple(sorted(node.node_id for node in members)),
                    centroid=tuple(float(value) for value in positions.mean(axis=0)),
                    area=float(areas.get(region_id, 0.0)),
                    labels=tuple((label, int(count)) for label, count in counts),
                    adjacent=tuple(sorted(adjacency[region_id])),
                    frontier_count=frontier_counts[region_id],
                )
            )
        return regions, doorways, located_frontiers

    def _region_areas(
        self,
        region_members: Mapping[str, Sequence[VoronoiNode]],
        region_id_by_node: Mapping[str, str],
        edges: Sequence[VoronoiEdge],
        doorway_ids: Set[str],
        traversable: np.ndarray,
        skeleton: np.ndarray,
    ) -> Dict[str, float]:
        """Assign traversable cells to the region of their nearest skeleton cell."""

        resolution = float(self.occupancy.config.grid_resolution)
        cell_area = resolution * resolution
        region_index = {region_id: index for index, region_id in enumerate(sorted(region_members))}
        skeleton_region = np.full(traversable.shape, -1, dtype=np.int32)
        for region_id, members in region_members.items():
            index = region_index[region_id]
            for node in members:
                for cell in node.cells:
                    skeleton_region[cell] = index
        for edge in edges:
            source_region = region_id_by_node[edge.source]
            target_region = region_id_by_node[edge.target]
            cells = edge.cells
            if edge.edge_id in doorway_ids and source_region != target_region:
                half = len(cells) // 2
                for cell in cells[:half]:
                    skeleton_region[cell] = region_index[source_region]
                for cell in cells[half:]:
                    skeleton_region[cell] = region_index[target_region]
            else:
                for cell in cells:
                    skeleton_region[cell] = region_index[source_region]

        region_ids = tuple(sorted(region_index, key=lambda region_id: region_index[region_id]))
        if _ndimage is None or not skeleton.any():
            assigned = _propagate_region_labels(skeleton_region, traversable)
            self._last_region_assignment = (assigned.astype(np.int32), region_ids)
            counts = np.bincount(
                assigned[assigned >= 0].reshape(-1),
                minlength=len(region_index),
            )
            return {region_id: float(counts[index] * cell_area) for region_id, index in region_index.items()}

        _, (nearest_rows, nearest_cols) = _ndimage.distance_transform_edt(
            ~(skeleton_region >= 0),
            return_indices=True,
        )
        assigned = skeleton_region[nearest_rows, nearest_cols]
        assigned = np.where(traversable, assigned, -1)
        self._last_region_assignment = (assigned.astype(np.int32), region_ids)
        counts = np.bincount(assigned[assigned >= 0].reshape(-1), minlength=len(region_index))
        return {region_id: float(counts[index] * cell_area) for region_id, index in region_index.items()}


# Short alias for users who prefer the map-like name.
SemanticVoronoi = SemanticVoronoiGraph


class VoronoiPathPlanner:
    """Plan waypoints that follow the safe Voronoi skeleton.

    A path consists of a short free-space attach segment from the start to
    the nearest reachable skeleton cell, an A* route along the skeleton, and
    a detach segment towards the goal. Waypoints are sampled directly from
    the planned cells at ``waypoint_spacing`` arc length, so the executed
    route stays glued to the max-clearance medial axis instead of cutting
    long line-of-sight chords across the corridor. ``PlanningError`` is
    raised whenever a skeleton route does not exist so callers can fall back
    to grid A*.
    """

    _NEIGHBORS = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.4142135623730951),
        (-1, 1, 1.4142135623730951),
        (1, -1, 1.4142135623730951),
        (1, 1, 1.4142135623730951),
    )

    def __init__(
        self,
        graph: SemanticVoronoiGraph,
        attach_radius: float = 6.0,
        waypoint_spacing: float = 0.30,
        observed_only: bool = False,
    ):
        if attach_radius <= 0 or waypoint_spacing <= 0:
            raise ValueError('attach_radius and waypoint_spacing must be positive')
        self.graph = graph
        self.attach_radius = float(attach_radius)
        self.waypoint_spacing = float(waypoint_spacing)
        self.observed_only = observed_only

    def plan(self, start: Sequence[float], goal: Sequence[float]) -> Tuple[Tuple[float, float, float], ...]:
        occupancy = self.graph.occupancy
        snapshot = self.graph.cached_snapshot()
        blocked = occupancy.inflated_mask()
        if self.observed_only:
            blocked = blocked | ~occupancy.observed
        # The cached topology can lag behind the occupancy layer, so drop
        # skeleton cells that have become unsafe in the meantime.
        skeleton = {cell for cell in snapshot.skeleton_cells if not blocked[cell]}
        if not skeleton:
            raise PlanningError('voronoi skeleton is empty')
        start_cell = occupancy.world_to_cell(start[:2])
        goal_cell = occupancy.world_to_cell(goal[:2])
        if start_cell is None or goal_cell is None:
            raise PlanningError('start or goal is outside map bounds')
        if self.observed_only and (blocked[start_cell] or blocked[goal_cell]):
            raise PlanningError('start or goal is not observed traversable space')
        resolution = float(occupancy.config.grid_resolution)
        attach_limit = max(1, int(ceil(self.attach_radius / resolution)))
        start_attach = self._attach_path(start_cell, skeleton, blocked, occupancy, attach_limit)
        goal_attach = self._attach_path(goal_cell, skeleton, blocked, occupancy, attach_limit)
        spine = self._skeleton_path(start_attach[-1], goal_attach[-1], skeleton, blocked)
        detach = list(reversed(goal_attach))[1:]
        cells = start_attach[:-1] + spine + detach
        cells = self._sample_along_path(cells, blocked, resolution)
        waypoints = [
            (float(x), float(y), float(goal[2]))
            for x, y in (occupancy.cell_to_world(cell) for cell in cells[1:])
        ]
        goal_point = (float(goal[0]), float(goal[1]), float(goal[2]))
        if not waypoints or hypot(waypoints[-1][0] - goal_point[0], waypoints[-1][1] - goal_point[1]) > resolution:
            waypoints.append(goal_point)
        else:
            waypoints[-1] = goal_point
        if self.observed_only:
            # Check exported chords, including the exact goal, using the
            # same unknown-space and diagonal-corner rules as grid A*.
            checker = AStarMapPlanner(occupancy, observed_only=True)
            route = [start_cell] + [occupancy.world_to_cell(point[:2]) for point in waypoints]
            if any(not checker.segment_clear(a, b, blocked) for a, b in zip(route, route[1:])):
                raise PlanningError('voronoi route crosses an unsafe corner')
        return tuple(waypoints)

    def _attach_path(
        self,
        cell: GridCell,
        skeleton: Set[GridCell],
        blocked: np.ndarray,
        occupancy: OccupancyGridMap,
        max_radius_cells: int,
    ) -> List[GridCell]:
        cell = self._nearest_open(cell, blocked, occupancy)
        if cell in skeleton:
            return [cell]
        parents: Dict[GridCell, Optional[GridCell]] = {cell: None}
        queue = deque([cell])
        while queue:
            current = queue.popleft()
            for row_offset, col_offset, _cost in self._NEIGHBORS:
                neighbor = (current[0] + row_offset, current[1] + col_offset)
                if neighbor in parents or not occupancy.in_bounds(neighbor) or blocked[neighbor]:
                    continue
                if row_offset and col_offset and (
                    blocked[current[0], neighbor[1]] or blocked[neighbor[0], current[1]]
                ):
                    continue
                if max(abs(neighbor[0] - cell[0]), abs(neighbor[1] - cell[1])) > max_radius_cells:
                    continue
                parents[neighbor] = current
                if neighbor in skeleton:
                    path = [neighbor]
                    while path[-1] is not None and parents[path[-1]] is not None:
                        path.append(parents[path[-1]])
                    path.reverse()
                    return path
                queue.append(neighbor)
        raise PlanningError('no reachable voronoi skeleton near endpoint')

    @staticmethod
    def _nearest_open(cell: GridCell, blocked: np.ndarray, occupancy: OccupancyGridMap) -> GridCell:
        if occupancy.in_bounds(cell) and not blocked[cell]:
            return cell
        for radius in range(1, 12):
            candidates = [
                (row, col)
                for row in range(cell[0] - radius, cell[0] + radius + 1)
                for col in range(cell[1] - radius, cell[1] + radius + 1)
                if occupancy.in_bounds((row, col)) and not blocked[row, col]
            ]
            if candidates:
                return min(candidates, key=lambda item: hypot(item[0] - cell[0], item[1] - cell[1]))
        raise PlanningError('no open grid cell near requested endpoint')

    def _skeleton_path(
        self,
        start: GridCell,
        goal: GridCell,
        skeleton: Set[GridCell],
        blocked: np.ndarray,
    ) -> List[GridCell]:
        if start == goal:
            return [start]
        frontier = [(0.0, start)]
        cost_so_far = {start: 0.0}
        parents: Dict[GridCell, GridCell] = {}
        while frontier:
            _, current = heappop(frontier)
            if current == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(parents[path[-1]])
                path.reverse()
                return path
            for row_offset, col_offset, move_cost in self._NEIGHBORS:
                neighbor = (current[0] + row_offset, current[1] + col_offset)
                if neighbor not in skeleton:
                    continue
                if row_offset and col_offset and (
                    blocked[current[0], neighbor[1]] or blocked[neighbor[0], current[1]]
                ):
                    continue
                new_cost = cost_so_far[current] + move_cost
                if new_cost >= cost_so_far.get(neighbor, float('inf')):
                    continue
                cost_so_far[neighbor] = new_cost
                parents[neighbor] = current
                heuristic = hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                heappush(frontier, (new_cost + heuristic, neighbor))
        raise PlanningError('start and goal connect to different skeleton components')

    def _sample_along_path(
        self,
        cells: List[GridCell],
        blocked: np.ndarray,
        resolution: float,
    ) -> List[GridCell]:
        """Sample waypoints from the planned cells at a short arc-length spacing.

        Every waypoint lies on the attach/skeleton/detach path itself, so the
        chords driven between consecutive waypoints stay within one spacing of
        the max-clearance medial axis. Long line-of-sight shortcuts are
        deliberately avoided: they only check the inflated mask, which
        under-protects thin obstacles such as standing people.
        """

        if len(cells) <= 2:
            return cells
        spacing = max(resolution, self.waypoint_spacing)
        sampled = [cells[0]]
        checker = AStarMapPlanner(self.graph.occupancy, observed_only=True)
        index = 0
        while index < len(cells) - 1:
            next_index = index
            travelled = 0.0
            while next_index < len(cells) - 1 and travelled < spacing:
                travelled += _cell_distance(cells[next_index], cells[next_index + 1]) * resolution
                next_index += 1
            # Guard the short chord against the inflated mask; adjacent path
            # cells are always safe, so the fallback terminates.
            while next_index > index + 1 and not checker.segment_clear(cells[index], cells[next_index], blocked):
                next_index -= 1
            sampled.append(cells[next_index])
            index = next_index
        return sampled


def _distance_to_false(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    if _ndimage is not None:
        return np.asarray(_ndimage.distance_transform_edt(padded)[1:-1, 1:-1], dtype=np.float64)
    false_cells = np.argwhere(~padded)
    result = np.zeros(mask.shape, dtype=np.float64)
    for row, col in np.argwhere(mask):
        delta = false_cells - np.asarray((row + 1, col + 1))
        result[row, col] = float(np.sqrt(np.min(np.sum(delta * delta, axis=1))))
    return result


def _thin(mask: np.ndarray) -> np.ndarray:
    image = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    changed = True
    while changed:
        changed = False
        for first_pass in (True, False):
            center = image[1:-1, 1:-1]
            p2, p3, p4 = image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:]
            p5, p6, p7 = image[2:, 2:], image[2:, 1:-1], image[2:, :-2]
            p8, p9 = image[1:-1, :-2], image[:-2, :-2]
            neighbors = np.sum(np.stack((p2, p3, p4, p5, p6, p7, p8, p9)), axis=0)
            transitions = np.sum(
                np.stack(
                    (
                        ~p2 & p3,
                        ~p3 & p4,
                        ~p4 & p5,
                        ~p5 & p6,
                        ~p6 & p7,
                        ~p7 & p8,
                        ~p8 & p9,
                        ~p9 & p2,
                    )
                ),
                axis=0,
            )
            if first_pass:
                removable = center & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1)
                removable &= ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                removable = center & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1)
                removable &= ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            if np.any(removable):
                center[removable] = False
                changed = True
    return image[1:-1, 1:-1]


def _prune_spurs(skeleton: np.ndarray, maximum_length: int) -> np.ndarray:
    result = skeleton.copy()
    if maximum_length <= 0:
        return result
    cells = set(_mask_cells(result))
    adjacency = {cell: set(_connected_neighbors(cell, cells)) for cell in cells}
    while True:
        remove: Set[GridCell] = set()
        for endpoint in sorted(cell for cell in cells if len(adjacency[cell]) <= 1):
            if not adjacency[endpoint]:
                continue
            path = [endpoint]
            previous, current = endpoint, next(iter(adjacency[endpoint]))
            while len(adjacency[current]) == 2 and len(path) <= maximum_length:
                path.append(current)
                next_cells = [cell for cell in adjacency[current] if cell != previous]
                previous, current = current, next_cells[0]
            if len(adjacency[current]) >= 3 and len(path) <= maximum_length:
                remove.update(path)
        if not remove:
            return result
        for cell in remove:
            result[cell] = False
            cells.discard(cell)
            for neighbor in adjacency.pop(cell, ()):
                adjacency.get(neighbor, set()).discard(cell)
        # Removing spur cells changes diagonal-bridge visibility for their
        # neighbors, so recompute adjacency around the removed area.
        affected = {
            neighbor
            for cell in remove
            for neighbor in _all_neighbors(cell)
            if neighbor in cells
        }
        for cell in affected:
            fresh = set(_connected_neighbors(cell, cells))
            for stale in adjacency[cell] - fresh:
                adjacency[stale].discard(cell)
            for added in fresh - adjacency[cell]:
                adjacency[added].add(cell)
            adjacency[cell] = fresh


def _connected_neighbors(cell: GridCell, cells: Set[GridCell]) -> List[GridCell]:
    row, col = cell
    result = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            neighbor = row + dr, col + dc
            if neighbor not in cells:
                continue
            if dr and dc and ((row + dr, col) in cells or (row, col + dc) in cells):
                continue
            result.append(neighbor)
    return sorted(result)


def _window_area(window: GridWindow) -> int:
    return (window[1] - window[0]) * (window[3] - window[2])


def _window_union_area(windows: Sequence[GridWindow], shape: Tuple[int, int]) -> int:
    coverage = np.zeros(shape, dtype=bool)
    for row_start, row_stop, column_start, column_stop in windows:
        coverage[row_start:row_stop, column_start:column_stop] = True
    return int(coverage.sum())


def _windows_overlap(
    left: GridWindow,
    right: GridWindow,
) -> bool:
    return not (
        left[1] <= right[0]
        or right[1] <= left[0]
        or left[3] <= right[2]
        or right[3] <= left[2]
    )


def _components(cells: Set[GridCell]) -> List[Set[GridCell]]:
    if not cells:
        return []
    if _ndimage is not None:
        rows = [cell[0] for cell in cells]
        cols = [cell[1] for cell in cells]
        row_min, col_min = min(rows), min(cols)
        mask = np.zeros((max(rows) - row_min + 1, max(cols) - col_min + 1), dtype=bool)
        for row, col in cells:
            mask[row - row_min, col - col_min] = True
        labeled, count = _ndimage.label(mask, structure=_EIGHT_CONNECTIVITY)
        groups: List[Set[GridCell]] = [set() for _ in range(count)]
        for row, col in zip(*np.nonzero(mask)):
            groups[labeled[row, col] - 1].add((int(row) + row_min, int(col) + col_min))
        return sorted(groups, key=min)
    remaining = set(cells)
    groups = []
    while remaining:
        seed = min(remaining)
        group = {seed}
        stack = [seed]
        remaining.remove(seed)
        while stack:
            cell = stack.pop()
            for neighbor in _all_neighbors(cell):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    group.add(neighbor)
                    stack.append(neighbor)
        groups.append(group)
    return sorted(groups, key=min)


def _components_from_mask(mask: np.ndarray) -> List[Set[GridCell]]:
    if not mask.any():
        return []
    if _ndimage is not None:
        labeled, count = _ndimage.label(mask, structure=_EIGHT_CONNECTIVITY)
        groups: List[Set[GridCell]] = [set() for _ in range(count)]
        for row, col in zip(*np.nonzero(mask)):
            groups[labeled[row, col] - 1].add((int(row), int(col)))
        return sorted(groups, key=min)
    return _components(set(_mask_cells(mask)))


def _component_labels(mask: np.ndarray) -> np.ndarray:
    if _ndimage is not None:
        labeled, _ = _ndimage.label(np.asarray(mask, dtype=bool), structure=_EIGHT_CONNECTIVITY)
        return np.asarray(labeled, dtype=np.int32) - 1
    labels = np.full(mask.shape, -1, dtype=np.int32)
    for label, group in enumerate(_components(set(_mask_cells(mask)))):
        for cell in group:
            labels[cell] = label
    return labels


def _nearest_true_cell(origin: GridCell, mask: np.ndarray) -> Optional[GridCell]:
    """Return the nearest true cell without silently crossing map bounds."""

    if mask[origin]:
        return origin
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    distances = (rows - origin[0]) ** 2 + (cols - origin[1]) ** 2
    index = int(np.argmin(distances))
    return int(rows[index]), int(cols[index])


def _propagate_region_labels(seeds: np.ndarray, traversable: np.ndarray) -> np.ndarray:
    """Assign every reachable free cell to a seeded region without SciPy."""

    assigned = np.where(traversable, seeds, -1).astype(np.int32)
    queue = deque(tuple(int(value) for value in cell) for cell in np.argwhere(assigned >= 0))
    while queue:
        row, col = queue.popleft()
        for neighbor in _all_neighbors((row, col)):
            nr, nc = neighbor
            if not (0 <= nr < assigned.shape[0] and 0 <= nc < assigned.shape[1]):
                continue
            if not traversable[neighbor] or assigned[neighbor] >= 0:
                continue
            assigned[neighbor] = assigned[row, col]
            queue.append(neighbor)
    return assigned


def _all_neighbors(cell: GridCell) -> Iterable[GridCell]:
    row, col = cell
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr != 0 or dc != 0:
                yield row + dr, col + dc


def _mask_cells(mask: np.ndarray) -> List[GridCell]:
    return [tuple(int(value) for value in cell) for cell in np.argwhere(mask)]


def _ordered_link(first: GridCell, second: GridCell) -> Tuple[GridCell, GridCell]:
    return (first, second) if first <= second else (second, first)


def _cell_distance(first: GridCell, second: GridCell) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])


def _nearest_node(
    position: Tuple[float, float],
    nodes: Sequence[VoronoiNode],
) -> Optional[Tuple[VoronoiNode, float]]:
    if not nodes:
        return None
    nearest = min(
        nodes,
        key=lambda node: (hypot(node.position[0] - position[0], node.position[1] - position[1]), node.node_id),
    )
    return nearest, float(hypot(nearest.position[0] - position[0], nearest.position[1] - position[1]))


__all__ = [
    'Doorway',
    'Frontier',
    'Region',
    'SemanticAttachment',
    'SemanticVoronoi',
    'SemanticVoronoiConfig',
    'SemanticVoronoiGraph',
    'SemanticVoronoiSnapshot',
    'VoronoiEdge',
    'VoronoiNode',
    'VoronoiPathPlanner',
]
