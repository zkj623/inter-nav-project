"""Export the recorded 3D cloud as a static image, without a browser/GPU."""
import cv2
import numpy as np


def render_cloud(points, position, size=720, ground_relative_z=False, *, orientation=None, mode='overview'):
    if mode == 'follow':
        if not ground_relative_z:
            raise ValueError('Simulation height slice requires ground-relative coordinates')
        from .simulation_view import render_follow_cloud
        return render_follow_cloud(points, position, [1., 0., 0., 0.] if orientation is None else orientation, size)
    if mode != 'overview':
        raise ValueError('Cloud view must be follow or overview')
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    frame = np.full((size, size, 3), 249, dtype=np.uint8)
    if not len(points):
        return frame
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > 12000:
        points = points[np.linspace(0, len(points)-1, 12000, dtype=int)]
    position = np.asarray(position, dtype=float)
    bounds = np.vstack((points, position))
    center = (bounds.min(axis=0)+bounds.max(axis=0))/2
    local = bounds-center
    yaw, pitch = -.75, .7
    u = np.cos(yaw)*local[:, 0]-np.sin(yaw)*local[:, 1]
    v = np.sin(yaw)*local[:, 0]+np.cos(yaw)*local[:, 1]
    vertical = v*np.sin(pitch)+local[:, 2]*np.cos(pitch)
    scale = min(size*.42/max(1., np.abs(u).max()),
                size*.42/max(1., np.abs(vertical).max()))
    col = (size/2+u*scale).astype(int)
    row = (size/2-vertical*scale).astype(int)
    robot_pixel = (int(col[-1]), int(row[-1]))
    col, row, v, local = col[:-1], row[:-1], v[:-1], local[:-1]
    depth = v*np.cos(pitch)-local[:, 2]*np.sin(pitch)
    valid = (col >= 1) & (col < size-1) & (row >= 1) & (row < size-1)
    order = np.argsort(depth[valid])[::-1]
    row, col = row[valid][order], col[valid][order]
    heights = np.clip((points[valid, 2][order]-.1)/.7*255, 0, 255).astype(np.uint8)
    if len(heights):
        colors = cv2.applyColorMap(heights[:, None], cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
        if ground_relative_z:
            colors[np.abs(points[valid, 2][order]) <= .08] = (207, 196, 184)  # RGB #b8c4cf
        for dx, dy in ((0, 0), (1, 0), (0, 1)):
            frame[row+dy, col+dx] = colors
    cv2.circle(frame, robot_pixel, 5, (80, 180, 255), -1)
    return frame
