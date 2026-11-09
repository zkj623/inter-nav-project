"""Offline map-video replay from a persisted semantic exploration run.

Reads ``<prefix>.npz`` and ``<prefix>.json`` written by
``MapNavigationRuntime.save`` and re-renders the top-down mapping video in
seconds, without Isaac. The exploration process is approximated by revealing
the final map along the recorded trajectory with a sensor-radius disk.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REGION_PALETTE = np.asarray(
    [
        (170, 210, 120),
        (120, 170, 220),
        (200, 150, 120),
        (130, 210, 210),
        (200, 130, 190),
        (120, 220, 170),
        (170, 130, 220),
        (210, 200, 120),
        (140, 140, 230),
        (220, 170, 140),
        (150, 220, 130),
        (190, 120, 150),
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map-prefix', required=True)
    parser.add_argument('--output', default='')
    parser.add_argument('--fps', type=float, default=12.0)
    parser.add_argument('--stride', type=int, default=20, help='trajectory points per frame')
    parser.add_argument('--reveal-radius', type=float, default=5.0, help='sensor disk in metres')
    parser.add_argument('--hold-frames', type=int, default=30, help='freeze frames at the end')
    parser.add_argument('--max-labels', type=int, default=24)
    return parser.parse_args()


def region_tint(topology, shape):
    regions = topology.get('regions') or ()
    if not regions:
        return None
    node_region = {}
    for index, region in enumerate(sorted(regions, key=lambda item: item['region_id'])):
        for node_id in region.get('node_ids', ()):
            node_region[node_id] = index
    doorway_edges = {doorway['edge_id'] for doorway in topology.get('doorways', ())}
    skeleton_region = np.full(shape, -1, dtype=np.int32)
    for node in topology['nodes']:
        index = node_region.get(node['node_id'])
        if index is None:
            continue
        for row, col in node.get('cells', ()):
            skeleton_region[row, col] = index
    for edge in topology['edges']:
        source_index = node_region.get(edge['source'])
        target_index = node_region.get(edge['target'])
        if source_index is None or target_index is None:
            continue
        cells = edge.get('cells', ())
        if edge['edge_id'] in doorway_edges and source_index != target_index:
            half = len(cells) // 2
            for row, col in cells[:half]:
                skeleton_region[row, col] = source_index
            for row, col in cells[half:]:
                skeleton_region[row, col] = target_index
        else:
            for row, col in cells:
                skeleton_region[row, col] = source_index
    if not (skeleton_region >= 0).any():
        return None
    try:
        from scipy.ndimage import distance_transform_edt

        _, (rows, cols) = distance_transform_edt(~(skeleton_region >= 0), return_indices=True)
        assigned = skeleton_region[rows, cols]
    except ImportError:
        assigned = skeleton_region
    palette = _REGION_PALETTE[np.arange(len(regions)) % len(_REGION_PALETTE)]
    tint = palette[np.clip(assigned, 0, None)]
    tint[assigned < 0] = 0
    return tint, assigned >= 0


def disk_offsets(radius_cells):
    span = np.arange(-radius_cells, radius_cells + 1)
    rows, cols = np.meshgrid(span, span, indexing='ij')
    keep = rows * rows + cols * cols <= radius_cells * radius_cells
    return rows[keep], cols[keep]


def main():
    args = parse_args()
    prefix = Path(args.map_prefix)
    metadata = json.loads(Path(str(prefix) + '.json').read_text(encoding='utf-8'))
    arrays = np.load(str(prefix) + '.npz')
    topology = metadata['semantic_voronoi']
    exploration = metadata.get('semantic_exploration', {})
    trajectory = exploration.get('trajectory') or []
    if not trajectory:
        print('no trajectory found in metadata; nothing to replay', file=sys.stderr)
        return 2
    decisions = exploration.get('decisions', [])

    observed = np.asarray(arrays['occupancy_observed'], dtype=bool)
    occupied = np.asarray(arrays['occupancy_log_odds']) >= 0.55
    inflated = np.asarray(arrays['occupancy_inflated'], dtype=bool)
    skeleton = np.asarray(arrays['semantic_voronoi_skeleton'], dtype=bool)
    mapping = metadata.get('mapping_config', {})
    resolution = float(mapping.get('grid_resolution', 0.1))
    x_limits = mapping.get('x_limits', (0.0, observed.shape[1] * resolution))
    y_limits = mapping.get('y_limits', (0.0, observed.shape[0] * resolution))

    # Full-knowledge base layers, revealed progressively along the trajectory.
    base = np.zeros((*observed.shape, 3), dtype=np.uint8)
    base[:] = (25, 25, 25)
    base[observed] = (205, 205, 205)
    tinted = region_tint(topology, observed.shape)
    if tinted is not None:
        tint, mask = tinted
        blend_mask = mask & observed
        base[blend_mask] = (0.55 * base[blend_mask] + 0.45 * tint[blend_mask]).astype(np.uint8)
    base[inflated & ~occupied] = (120, 190, 235)
    base[occupied] = (20, 20, 220)
    base[skeleton] = (200, 90, 20)
    junction_cells = [
        tuple(node['cell']) for node in topology['nodes'] if node.get('kind') == 'junction'
    ]
    doorway_cells = []
    edge_by_id = {edge['edge_id']: edge for edge in topology['edges']}
    for doorway in topology.get('doorways', ()):
        edge = edge_by_id.get(doorway['edge_id'])
        if edge and edge.get('cells'):
            doorway_cells.append(tuple(edge['cells'][len(edge['cells']) // 2]))
    dark = np.zeros_like(base)
    dark[:] = (12, 12, 12)

    def world_to_cell(x, y):
        col = int((x - x_limits[0]) / resolution)
        row = int((y - y_limits[0]) / resolution)
        return min(max(row, 0), observed.shape[0] - 1), min(max(col, 0), observed.shape[1] - 1)

    scale = max(4, min(10, 1600 // observed.shape[1]))
    frame_size = (observed.shape[1] * scale, observed.shape[0] * scale)

    def to_pixel(row, col):
        return int((col + 0.5) * scale), int((observed.shape[0] - 1 - row + 0.5) * scale)

    def world_to_pixel(x, y):
        return to_pixel(*world_to_cell(x, y))

    labels = sorted(
        metadata.get('scene_graph', {}).get('nodes', []),
        key=lambda node: node.get('observations', 0),
        reverse=True,
    )
    labels = [node for node in labels if node.get('kind') != 'place'][: args.max_labels]

    output = Path(args.output or str(prefix) + '_replay.mp4')
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*'mp4v'),
        args.fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f'could not open video writer for {output}')

    radius_cells = max(1, int(round(args.reveal_radius / resolution)))
    disk_rows, disk_cols = disk_offsets(radius_cells)
    reveal = np.zeros(observed.shape, dtype=bool)
    frame_indices = list(range(0, len(trajectory), max(1, args.stride)))

    def compose(step_index, reveal_mask, robot_xy, goal_xy):
        grid = np.where(reveal_mask[..., None], base, dark)
        frame = cv2.resize(np.flipud(grid), frame_size, interpolation=cv2.INTER_NEAREST)
        for row, col in junction_cells:
            if reveal_mask[row, col]:
                cv2.circle(frame, to_pixel(row, col), max(2, scale // 2), (30, 140, 230), -1)
        for row, col in doorway_cells:
            if reveal_mask[row, col]:
                cv2.drawMarker(frame, to_pixel(row, col), (255, 255, 255), cv2.MARKER_DIAMOND, 3 * scale, 2)
        points = [
            world_to_pixel(point[0], point[1])
            for point in trajectory[: step_index + 1 : max(1, args.stride // 4)]
        ]
        if len(points) >= 2:
            cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], False, (20, 160, 20), 2)
        if goal_xy is not None:
            cv2.drawMarker(
                frame,
                world_to_pixel(goal_xy[0], goal_xy[1]),
                (0, 255, 255),
                cv2.MARKER_TILTED_CROSS,
                3 * scale,
                2,
            )
        cv2.circle(frame, world_to_pixel(robot_xy[0], robot_xy[1]), max(4, scale), (20, 230, 20), -1)
        used_boxes = []
        for node in labels:
            row, col = world_to_cell(node['position'][0], node['position'][1])
            if not reveal_mask[row, col]:
                continue
            org = to_pixel(row, col)
            text = str(node.get('label', ''))[:14]
            (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            box = (org[0], org[1] - height - baseline, org[0] + width, org[1] + baseline)
            if any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in used_boxes):
                continue
            used_boxes.append(box)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 10, 10), 2, cv2.LINE_AA)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (frame_size[0], 30), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f'semantic exploration replay | step {step_index} / {len(trajectory) - 1}',
            (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return frame

    goal = None
    decision_cursor = 0
    for step_index in frame_indices:
        row, col = world_to_cell(trajectory[step_index][0], trajectory[step_index][1])
        rows = np.clip(disk_rows + row, 0, observed.shape[0] - 1)
        cols = np.clip(disk_cols + col, 0, observed.shape[1] - 1)
        reveal[rows, cols] = True
        while decision_cursor < len(decisions) and decisions[decision_cursor].get('step', 0) <= step_index:
            goal = decisions[decision_cursor].get('goal')
            decision_cursor += 1
        writer.write(compose(step_index, reveal & observed, trajectory[step_index], goal))

    final = compose(len(trajectory) - 1, observed, trajectory[-1], None)
    for _ in range(max(0, args.hold_frames)):
        writer.write(final)
    writer.release()
    seconds = (len(frame_indices) + args.hold_frames) / args.fps
    print(json.dumps({'event': 'replay_written', 'output': str(output), 'video_seconds': round(seconds, 1)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
