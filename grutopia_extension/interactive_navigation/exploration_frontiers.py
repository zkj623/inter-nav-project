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


def frontier_goals(occupancy, snapshot, position, excluded, spacing=0.6):
    """Split long boundaries spatially and choose a reachable *boundary cell*.

    A group's arithmetic centroid can lie inside a wall or far from its
    boundary. Candidate endpoints are actual observed cells instead. Spatial
    keys keep feedback meaningful when topology IDs change as the map grows.
    """
    distances = reachable_distances(occupancy, position)
    buckets = {}
    for frontier in snapshot.frontiers:
        for cell in frontier.cells:
            xy = occupancy.cell_to_world(cell)
            if cell not in distances:
                continue
            key = f'{int(np.floor(xy[0] / spacing))}:{int(np.floor(xy[1] / spacing))}'
            if key in excluded:
                continue
            buckets.setdefault(key, []).append(cell)
    goals = []
    for key, cells in buckets.items():
        center = np.mean(cells, axis=0)
        cell = min(cells, key=lambda item: (hypot(item[0] - center[0], item[1] - center[1]), distances[item]))
        xy = occupancy.cell_to_world(cell)
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
