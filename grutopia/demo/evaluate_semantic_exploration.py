"""Evaluate and visualize a persisted semantic Voronoi exploration map."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map-prefix', required=True)
    parser.add_argument('--output', default='')
    parser.add_argument('--target-label', default='')
    parser.add_argument('--expected-target', type=float, nargs=3)
    parser.add_argument('--max-target-error', type=float, default=0.35)
    return parser.parse_args()


def _ratio(values):
    return 1.0 if not values else sum(values) / len(values)


def evaluate(prefix: Path, target_label='', expected_target=None, max_target_error=0.35):
    metadata = json.loads(Path(str(prefix) + '.json').read_text(encoding='utf-8'))
    arrays = np.load(str(prefix) + '.npz')
    observed = np.asarray(arrays['occupancy_observed'], dtype=bool)
    inflated = np.asarray(arrays['occupancy_inflated'], dtype=bool)
    skeleton = np.asarray(arrays['semantic_voronoi_skeleton'], dtype=bool)
    topology = metadata['semantic_voronoi']
    node_ids = {node['node_id'] for node in topology['nodes']}
    invalid_edges = [
        edge['edge_id']
        for edge in topology['edges']
        if edge['source'] not in node_ids or edge['target'] not in node_ids
    ]
    frontier_attached = _ratio(
        [frontier.get('node_id') is not None for frontier in topology['frontiers']]
    )
    semantic_attached = _ratio(
        [semantic.get('node_id') is not None for semantic in topology['semantics']]
    )
    target_error = None
    matched_target = None
    if target_label:
        normalized = target_label.casefold().replace(' ', '_')
        graph = metadata.get('scene_graph', {})
        label_counts = graph.get('label_counts', {})
        candidates = [
            node
            for node in graph.get('nodes', [])
            if normalized in node.get('label', '').casefold().replace(' ', '_')
            or any(
                normalized in label.casefold().replace(' ', '_') and count > 0
                for label, count in label_counts.get(node['node_id'], {}).items()
            )
        ]
        if candidates:
            selected_id = metadata.get('semantic_exploration', {}).get('statistics', {}).get('target_node')
            matched_target = next(
                (node for node in candidates if node['node_id'] == selected_id),
                None,
            ) or max(candidates, key=lambda node: node.get('observations', 0))
    if matched_target is not None and expected_target is not None:
        target_error = float(
            np.linalg.norm(
                np.asarray(matched_target['position'], dtype=np.float64)
                - np.asarray(expected_target, dtype=np.float64)
            )
        )

    checks = {
        'has_skeleton': bool(skeleton.any()),
        'skeleton_avoids_inflation': not bool((skeleton & inflated).any()),
        'skeleton_is_observed': not bool((skeleton & ~observed).any()),
        'all_edges_reference_nodes': not invalid_edges,
        'frontier_projection_ratio': frontier_attached >= 0.90,
        'semantic_attachment_ratio': semantic_attached >= 0.90,
        'target_detected': not target_label or matched_target is not None,
        'target_localization': target_error is None or target_error <= max_target_error,
    }
    report = {
        'passed': all(checks.values()),
        'checks': checks,
        'metrics': {
            'observed_cells': int(observed.sum()),
            'skeleton_cells': int(skeleton.sum()),
            'unsafe_skeleton_cells': int((skeleton & inflated).sum()),
            'unknown_skeleton_cells': int((skeleton & ~observed).sum()),
            'voronoi_nodes': len(topology['nodes']),
            'voronoi_edges': len(topology['edges']),
            'frontiers': len(topology['frontiers']),
            'frontier_projection_ratio': frontier_attached,
            'semantics': len(topology['semantics']),
            'semantic_attachment_ratio': semantic_attached,
            'invalid_edges': invalid_edges,
            'target_error_metres': target_error,
        },
        'target': None if matched_target is None else {
            key: matched_target[key]
            for key in ('node_id', 'label', 'position', 'observations')
            if key in matched_target
        } | {
            'label_counts': metadata.get('scene_graph', {}).get('label_counts', {}).get(
                matched_target['node_id'], {}
            ),
        },
    }
    return report, metadata, arrays


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


def _region_tint(topology, observed, inflated, skeleton):
    """Per-cell BGR region tint reconstructed from the persisted hierarchy."""
    regions = topology.get('regions') or ()
    if not regions:
        return None
    node_cells = {node['node_id']: node.get('cells', ()) for node in topology['nodes']}
    edge_by_id = {edge['edge_id']: edge for edge in topology['edges']}
    node_region = {}
    for index, region in enumerate(sorted(regions, key=lambda item: item['region_id'])):
        for node_id in region.get('node_ids', ()):
            node_region[node_id] = index
    doorway_edges = {doorway['edge_id'] for doorway in topology.get('doorways', ())}
    skeleton_region = np.full(observed.shape, -1, dtype=np.int32)
    for node_id, cells in node_cells.items():
        index = node_region.get(node_id)
        if index is None:
            continue
        for row, col in cells:
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
    assigned = np.where(observed & ~inflated, assigned, -1)
    palette = _REGION_PALETTE[np.arange(len(regions)) % len(_REGION_PALETTE)]
    tint = palette[np.clip(assigned, 0, None)]
    tint[assigned < 0] = 0
    return tint, assigned >= 0


def render_diagnostic(metadata, arrays, output: Path):
    observed = np.asarray(arrays['occupancy_observed'], dtype=bool)
    occupied = np.asarray(arrays['occupancy_log_odds']) >= 0.55
    inflated = np.asarray(arrays['occupancy_inflated'], dtype=bool)
    skeleton = np.asarray(arrays['semantic_voronoi_skeleton'], dtype=bool)
    topology = metadata['semantic_voronoi']

    frame = np.zeros((*observed.shape, 3), dtype=np.uint8)
    frame[:] = (25, 25, 25)
    frame[observed] = (205, 205, 205)
    tinted = _region_tint(topology, observed, inflated, skeleton)
    if tinted is not None:
        tint, mask = tinted
        frame[mask] = (0.55 * frame[mask] + 0.45 * tint[mask]).astype(np.uint8)
    frame[inflated & ~occupied] = (120, 190, 235)
    frame[occupied] = (20, 20, 220)
    for frontier in topology['frontiers']:
        for row, col in frontier['cells']:
            if 0 <= row < frame.shape[0] and 0 <= col < frame.shape[1]:
                frame[row, col] = (255, 255, 0)

    scale = max(4, min(10, 1600 // observed.shape[1]))
    frame = cv2.resize(
        np.flipud(frame),
        (observed.shape[1] * scale, observed.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )

    def to_pixel(cell):
        row, col = cell
        return (
            int((col + 0.5) * scale),
            int((observed.shape[0] - 1 - row + 0.5) * scale),
        )

    for edge in topology['edges']:
        cells = edge.get('cells', ())
        if len(cells) < 2:
            continue
        points = np.asarray([to_pixel(cell) for cell in cells], dtype=np.int32)
        cv2.polylines(frame, [points], False, (200, 90, 20), max(1, scale // 4), cv2.LINE_AA)
    for node in topology['nodes']:
        if node.get('kind') != 'junction':
            continue
        cv2.circle(frame, to_pixel(node['cell']), max(2, scale // 2), (30, 140, 230), -1)
    for doorway in topology.get('doorways', ()):
        # Doorway positions are stored in world coordinates; recover the cell
        # from the narrowest edge cell instead for a resolution-free overlay.
        edge = next(
            (item for item in topology['edges'] if item['edge_id'] == doorway['edge_id']),
            None,
        )
        if edge is None or not edge.get('cells'):
            continue
        middle = edge['cells'][len(edge['cells']) // 2]
        cv2.drawMarker(frame, to_pixel(middle), (255, 255, 255), cv2.MARKER_DIAMOND, 3 * scale, 2)

    semantic = metadata.get('semantic_exploration', {})
    target_id = semantic.get('statistics', {}).get('target_node')
    target_node = next(
        (node for node in metadata.get('scene_graph', {}).get('nodes', []) if node.get('node_id') == target_id),
        None,
    ) if target_id else None
    if target_node is not None:
        config = metadata['mapping_config']
        resolution = config['grid_resolution']
        x, y = target_node['position'][:2]
        col = int(np.floor((x - config['x_limits'][0]) / resolution))
        row = int(np.floor((y - config['y_limits'][0]) / resolution))
        if 0 <= row < observed.shape[0] and 0 <= col < observed.shape[1]:
            pixel = to_pixel((row, col))
            cv2.drawMarker(frame, pixel, (0, 255, 255), cv2.MARKER_CROSS, 4 * scale, 2)
            state = 'TARGET' if semantic.get('statistics', {}).get('target_confirmed') else 'CANDIDATE'
            label = f"{state}: {semantic.get('target_query', target_node['label'])}"
            width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
            label_x = min(pixel[0] + 2 * scale, max(5, frame.shape[1] - width - 5))
            label_y = max(52, pixel[1] - scale)
            cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        frame,
        'observed=gray regions=tinted occupied=red safety=light-orange '
        'skeleton=blue junction=orange doorway=white frontier=cyan',
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), frame)


def main():
    args = parse_args()
    prefix = Path(args.map_prefix)
    report, metadata, arrays = evaluate(
        prefix,
        target_label=args.target_label,
        expected_target=args.expected_target,
        max_target_error=args.max_target_error,
    )
    output = Path(args.output or str(prefix) + '_evaluation.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    render_diagnostic(metadata, arrays, output.with_suffix('.png'))
    print(json.dumps(report))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
