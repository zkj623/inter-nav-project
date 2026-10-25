"""Accumulate camera semantic-segmentation pixels in the navigation grid."""

import json
from pathlib import Path

import numpy as np


_STRUCTURAL_LABELS = {'floor', 'ground', 'wall', 'ceiling'}
_UNLABELED = {'', 'unlabelled', 'unlabeled', 'background', 'none', 'other'}


def _class_label(value):
    """Read Isaac/Replicator idToLabels entries without guessing categories."""
    if isinstance(value, dict):
        value = value.get('class', '')
    if value is None:
        return ''
    label = str(value).strip().casefold()
    if label.startswith('class:'):
        label = label[6:].strip()
    # GRScenes appends an instance/model path after the semantic class.
    return label.split('/', 1)[0].strip()


class CameraSemanticGrid:
    """Observed class evidence from aligned segmentation, depth, and world points."""

    def __init__(self, config, shape, *, max_depth_m=5.0, pixel_stride=3):
        self.config = config
        self.shape = tuple(shape)
        self.max_depth_m = float(max_depth_m)
        self.pixel_stride = int(pixel_stride)
        self.votes = {}
        self.frames = 0
        self.last_capture_id = None
        self._cached = None
        self._dirty = True
        if self.max_depth_m <= 0 or self.pixel_stride < 1:
            raise ValueError('invalid semantic grid sampling settings')

    def update(self, camera):
        capture_id = camera.get('capture_id')
        if capture_id is not None and capture_id == self.last_capture_id:
            return False
        payload = camera.get('semantic_segmentation')
        depth = camera.get('depth')
        point_image = camera.get('pointcloud_image')
        if not isinstance(payload, dict) or depth is None or point_image is None:
            return False
        ids = np.asarray(payload.get('data'))
        depth = np.asarray(depth)
        points = np.asarray(point_image)
        if ids.size != depth.size or points.shape != (*depth.shape, 3):
            return False
        ids = ids.reshape(depth.shape)[::self.pixel_stride, ::self.pixel_stride]
        depths = depth[::self.pixel_stride, ::self.pixel_stride]
        points = points[::self.pixel_stride, ::self.pixel_stride]
        valid = (
            np.isfinite(depths) & (depths > 0.01) & (depths <= self.max_depth_m)
            & np.isfinite(points).all(axis=2)
            & (points[:, :, 2] >= self.config.point_height[0])
            & (points[:, :, 2] <= self.config.point_height[1])
        )
        x_min, x_max = self.config.x_limits
        y_min, y_max = self.config.y_limits
        safe_x = np.where(np.isfinite(points[:, :, 0]), points[:, :, 0], x_min - 1.0)
        safe_y = np.where(np.isfinite(points[:, :, 1]), points[:, :, 1], y_min - 1.0)
        rows = ((safe_y - y_min) / self.config.grid_resolution).astype(np.int32)
        cols = ((safe_x - x_min) / self.config.grid_resolution).astype(np.int32)
        valid &= (
            (points[:, :, 0] >= x_min) & (points[:, :, 0] < x_max)
            & (points[:, :, 1] >= y_min) & (points[:, :, 1] < y_max)
            & (rows >= 0) & (rows < self.shape[0])
            & (cols >= 0) & (cols < self.shape[1])
        )
        lookup = payload.get('info', {}).get('idToLabels', {})
        if not isinstance(lookup, dict):
            return False
        observed_labels = []
        for raw_id in np.unique(ids[valid]):
            semantic_id = int(raw_id)
            entry = lookup.get(str(semantic_id), lookup.get(semantic_id))
            label = _class_label(entry)
            if label in _UNLABELED:
                continue
            mask = valid & (ids == raw_id)
            cells = np.unique(rows[mask] * self.shape[1] + cols[mask])
            if cells.size == 0:
                continue
            votes = self.votes.setdefault(label, np.zeros(self.shape, dtype=np.uint16))
            flat = votes.reshape(-1)
            flat[cells] = np.minimum(flat[cells].astype(np.uint32) + 1, 65535).astype(np.uint16)
            observed_labels.append(label)
        if capture_id is not None:
            self.last_capture_id = capture_id
        self.frames += 1
        self._dirty = True
        if self.frames == 1:
            print(json.dumps({
                'event': 'camera_semantic_frame',
                'shape': list(ids.shape),
                'id_count': int(len(lookup)),
                'mapped_labels': sorted(set(observed_labels)),
                'sampled_pixels': int(np.count_nonzero(valid)),
            }), flush=True)
        return bool(observed_labels)

    def class_grid(self):
        if not self._dirty and self._cached is not None:
            return self._cached
        labels = sorted(self.votes)
        result = np.full(self.shape, -1, dtype=np.int16)
        best = np.zeros(self.shape, dtype=np.uint16)
        for index, label in enumerate(labels):
            if label.split(',')[-1].strip() in _STRUCTURAL_LABELS:
                continue
            score = self.votes[label]
            mask = score > best
            result[mask] = index
            best[mask] = score[mask]
        for index, label in enumerate(labels):
            if label.split(',')[-1].strip() not in _STRUCTURAL_LABELS:
                continue
            score = self.votes[label]
            mask = (result < 0) & (score > best)
            result[mask] = index
            best[mask] = score[mask]
        self._cached = (result, labels)
        self._dirty = False
        return self._cached

    def save(self, output_dir):
        output = Path(output_dir)
        class_ids, labels = self.class_grid()
        ordered_votes = np.stack([self.votes[label] for label in labels]) if labels else np.empty((0, *self.shape), np.uint16)
        np.savez_compressed(
            output / 'semantic_grid.npz',
            class_ids=class_ids,
            labels=np.asarray(labels),
            votes=ordered_votes,
        )
        summary = {
            'source': 'Isaac Sim semantic_segmentation ground truth projected with RGB-D world points',
            'frames': self.frames,
            'max_depth_m': self.max_depth_m,
            'pixel_stride': self.pixel_stride,
            'grid_resolution_m': self.config.grid_resolution,
            'mapped_cells': int(np.count_nonzero(class_ids >= 0)),
            'labels': labels,
        }
        (output / 'semantic_grid.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary
