"""Reachable observation goals on an online map; no simulator dependencies."""

from dataclasses import dataclass
from heapq import heappop, heappush
from math import hypot

import numpy as np

from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner


@dataclass(frozen=True)
class ObservationGoal:
    key: str
    position: tuple
    distance: float
    information_gain: float
    score: float


def reachable_distances(occupancy, position):
    """One Dijkstra traversal for all candidates, with no unknown/corner cuts."""
    blocked = occupancy.inflated_mask() | ~occupancy.observed
    start = occupancy.world_to_cell(position[:2])
    distances = {}
    if start is None or blocked[start]:
        return distances
    distances[start] = 0.0
    queue = [(0.0, start)]
    resolution = occupancy.config.grid_resolution
    while queue:
        distance, cell = heappop(queue)
        if distance != distances[cell]:
            continue
        for dr, dc, cost in AStarMapPlanner._NEIGHBORS:
            neighbor = (cell[0] + dr, cell[1] + dc)
            if not occupancy.in_bounds(neighbor) or blocked[neighbor]:
                continue
            if dr and dc and (blocked[cell[0] + dr, cell[1]] or blocked[cell[0], cell[1] + dc]):
                continue
            candidate = distance + cost * resolution
            if candidate < distances.get(neighbor, float('inf')):
                distances[neighbor] = candidate
                heappush(queue, (candidate, neighbor))
    return distances


def local_unknown(occupancy, position, radius=0.8):
    cell = occupancy.world_to_cell(position)
    if cell is None:
        return 0
    radius = int(np.ceil(radius / occupancy.config.grid_resolution))
    row, col = cell
    patch = occupancy.observed[max(0, row - radius) : row + radius + 1, max(0, col - radius) : col + radius + 1]
    return int((~patch).sum())


def frontier_goals(occupancy, snapshot, position, excluded, redundancy_radius=1.0):
    """Create one reachable goal per connected or nearby frontier cluster.

    Frontier cells are first filtered by map reachability. Nearby fragments
    from the same region are then merged using an RRT-style redundancy radius.
    The target is an actual boundary cell near the cluster center, never the
    arithmetic centroid inside a wall.
    """
    distances = reachable_distances(occupancy, position)
    components = []
    for frontier in snapshot.frontiers:
        cells = tuple(sorted({cell for cell in frontier.cells if cell in distances}))
        if not cells:
            continue
        points = np.asarray([occupancy.cell_to_world(cell) for cell in cells], dtype=np.float64)
        components.append({
            'region': frontier.region_id or 'unsegmented',
            'cells': cells,
            'center': points.mean(axis=0),
        })
    if not components:
        return []

    parents = list(range(len(components)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    radius = max(float(redundancy_radius), occupancy.config.grid_resolution)
    buckets = {}
    for index, component in enumerate(components):
        center = component['center']
        bx, by = int(np.floor(center[0] / radius)), int(np.floor(center[1] / radius))
        region = component['region']
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in buckets.get((region, bx + dx, by + dy), ()):
                    if np.linalg.norm(center - components[other]['center']) <= radius:
                        union(index, other)
        buckets.setdefault((region, bx, by), []).append(index)

    clusters = {}
    for index, component in enumerate(components):
        root = find(index)
        cluster = clusters.setdefault(root, {'region': component['region'], 'cells': set()})
        cluster['cells'].update(component['cells'])

    goals = []
    for cluster in clusters.values():
        cells = tuple(sorted(cluster['cells']))
        points = np.asarray([occupancy.cell_to_world(cell) for cell in cells], dtype=np.float64)
        center = points.mean(axis=0)
        cell = min(
            cells,
            key=lambda item: (
                float(np.sum((np.asarray(occupancy.cell_to_world(item)) - center) ** 2)),
                distances[item],
                item,
            ),
        )
        xy = occupancy.cell_to_world(cell)
        key = f'{cluster["region"]}:{cell[0]}:{cell[1]}'
        if key in excluded:
            continue
        gain = local_unknown(occupancy, xy) * occupancy.config.grid_resolution**2
        distance = distances[cell]
        goals.append(ObservationGoal(key, tuple(xy), distance, gain, gain / (1.0 + distance)))
    return sorted(goals, key=lambda item: (-item.score, item.distance, item.key))



def buffer_exit_goal(occupancy, position, preferred=None):
    """Leave a planning buffer without dropping below the starting clearance.

    The physical footprint remains blocked. This is a short local segment,
    not a relaxed route through walls, unknown cells, or another doorway.
    """
    from scipy.ndimage import distance_transform_edt

    start = occupancy.world_to_cell(position[:2])
    if start is None or not occupancy.observed[start]:
        return None
    resolution = occupancy.config.grid_resolution
    clearance = distance_transform_edt(~occupancy.occupied_mask()) * resolution
    if clearance[start] <= occupancy.config.robot_radius:
        return None
    inflation = occupancy.config.obstacle_inflation_radius or occupancy.config.robot_radius
    blocked = ~occupancy.observed | (clearance < clearance[start] - 1e-9)
    safe = occupancy.observed & ~occupancy.inflated_mask() & (clearance >= inflation + resolution)
    planner = AStarMapPlanner(occupancy)

    def valid(xy):
        cell = occupancy.world_to_cell(xy)
        distance = np.linalg.norm(np.asarray(xy) - position[:2])
        return (
            cell is not None
            and safe[cell]
            and distance <= 0.8
            and planner.segment_clear(start, cell, blocked)
        )

    if preferred is not None and valid(preferred):
        return tuple(preferred)
    reach = int(np.ceil(0.8 / resolution))
    row, col = start
    r0, c0 = max(0, row - reach), max(0, col - reach)
    window = safe[r0 : row + reach + 1, c0 : col + reach + 1]
    candidates = [occupancy.cell_to_world((r + r0, c + c0)) for r, c in np.argwhere(window)]
    candidates.sort(key=lambda xy: np.linalg.norm(np.asarray(xy) - position[:2]))
    return next((xy for xy in candidates if valid(xy)), None)
