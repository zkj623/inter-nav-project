"""MP4 visualizations for RGB observations and fused navigation maps."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np

from grutopia_extension.interactive_navigation.mapping import canonical_semantic_label
from grutopia_extension.interactive_navigation.exploration_frontiers import frontier_goals
from grutopia_extension.interactive_navigation.semantic_grid import CameraSemanticGrid


def _frontier_display_markers(snapshot, occupancy, robot_position, selected_frontier_id=None):
    """Show one marker per nearby, actually reachable observation goal."""

    goals = frontier_goals(
        occupancy, snapshot, robot_position, excluded=set(), redundancy_radius=1.0,
    )
    return [
        (goal.position, goal.key == selected_frontier_id)
        for goal in goals
    ]


class FixedOverviewCamera:
    """Fixed wide camera covering the complete generated household scene."""

    def __init__(
        self,
        resolution=(640, 360),
        position=(4.0, -7.0, 4.5),
        look_at=(4.0, 0.0, 0.8),
        focal_length=18.0,
    ):
        import omni.replicator.core as rep

        self.camera = rep.create.camera(
            position=position,
            look_at=look_at,
            focal_length=focal_length,
        )
        self.render_product = rep.create.render_product(self.camera, resolution)
        self.annotator = rep.AnnotatorRegistry.get_annotator('LdrColor')
        self.annotator.attach(self.render_product)
        self.bounding_box_annotator = rep.AnnotatorRegistry.get_annotator('bounding_box_2d_tight')
        self.bounding_box_annotator.attach(self.render_product)

    def get_data(self):
        return {
            'rgba': self.annotator.get_data(),
            'bounding_box_2d_tight': self.bounding_box_annotator.get_data(),
        }


def following_camera_world_pose(position, yaw, offset, pitch_degrees):
    """Place a +X-forward camera in the robot body frame and pitch it downward."""

    position = np.asarray(position, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    half_yaw = 0.5 * float(yaw)
    yaw_quaternion = np.array(
        [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
        dtype=np.float64,
    )
    pitch_radians = np.deg2rad(pitch_degrees)
    pitch_quaternion = np.array(
        [np.cos(pitch_radians / 2.0), 0.0, np.sin(pitch_radians / 2.0), 0.0],
        dtype=np.float64,
    )
    camera_position = position + _quaternion_matrix(yaw_quaternion) @ offset
    camera_orientation = _quaternion_multiply(yaw_quaternion, pitch_quaternion)
    return camera_position, camera_orientation


class FollowingRobotCamera:
    """World-frame RGB camera following a robot whose +X axis is forward."""

    def __init__(
        self,
        resolution=(640, 360),
        offset=(0.14, 0.0, 0.48),
        pitch_degrees=3.0,
        horizontal_fov_degrees=96.0,
        clipping_range=(0.01, 80.0),
        enable_depth=False,
        enable_semantics=False,
        enable_bounding_boxes=True,
        prim_path='/World/FollowingRobotCamera',
    ):
        from omni.isaac.sensor import Camera

        self.offset = np.asarray(offset, dtype=np.float64)
        self.pitch_degrees = float(pitch_degrees)
        self.smoothing = 0.20
        self._smoothed_position = None
        self._smoothed_yaw = None
        self.enable_depth = bool(enable_depth)
        self.enable_semantics = bool(enable_semantics)
        self.enable_bounding_boxes = bool(enable_bounding_boxes)
        self._capture_id = 0
        self.camera = Camera(
            prim_path=prim_path,
            resolution=resolution,
        )
        self.camera.initialize()
        # USD cameras default to a 1 m near plane, which hides tabletops and
        # grasped objects as soon as the robot steps inside that distance.
        self.clipping_range = (float(clipping_range[0]), float(clipping_range[1]))
        self._apply_clipping_range()
        horizontal_fov_radians = np.deg2rad(horizontal_fov_degrees)
        focal_length = self.camera.get_horizontal_aperture() / (
            2.0 * np.tan(horizontal_fov_radians / 2.0)
        )
        self.camera.set_focal_length(float(focal_length))
        if self.enable_bounding_boxes:
            self.camera.add_bounding_box_2d_tight_to_frame()
        if self.enable_depth:
            self.camera.add_distance_to_image_plane_to_frame()
        if self.enable_semantics:
            self.camera.add_semantic_segmentation_to_frame()

    def update_pose(self, position, orientation):
        position = np.asarray(position, dtype=np.float64)
        w, x, y, z = np.asarray(orientation, dtype=np.float64)
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        if self._smoothed_position is None:
            self._smoothed_position = position.copy()
            self._smoothed_yaw = yaw
        else:
            self._smoothed_position = (
                (1.0 - self.smoothing) * self._smoothed_position + self.smoothing * position
            )
            yaw_delta = (yaw - self._smoothed_yaw + np.pi) % (2.0 * np.pi) - np.pi
            self._smoothed_yaw += self.smoothing * yaw_delta
        camera_position, camera_orientation = following_camera_world_pose(
            self._smoothed_position,
            self._smoothed_yaw,
            self.offset,
            self.pitch_degrees,
        )
        self.camera.set_world_pose(
            position=camera_position,
            orientation=camera_orientation,
            camera_axes='world',
        )
        self._apply_clipping_range()

    def _apply_clipping_range(self):
        from pxr import Gf, UsdGeom

        near_distance, far_distance = self.clipping_range
        self.camera.set_clipping_range(near_distance=near_distance, far_distance=far_distance)
        usd_camera = UsdGeom.Camera(self.camera.prim)
        usd_camera.GetClippingRangeAttr().Set(Gf.Vec2f(near_distance, far_distance))

    def get_data(self):
        # Read the RGB image and its metadata from the same completed render
        # event. Calling get_rgba() separately can observe the next in-flight
        # render product while the camera is moving, leaving torn bands at the
        # bottom of the recorded frame.
        frame = self.camera.get_current_frame(clone=True)
        self._capture_id += 1
        position, orientation = self.camera.get_world_pose(camera_axes='world')
        data = {
            'rgba': frame.get('rgba'),
            'bounding_box_2d_tight': frame.get('bounding_box_2d_tight'),
            'position': np.asarray(position, dtype=np.float32),
            'orientation': np.asarray(orientation, dtype=np.float32),
            'capture_id': self._capture_id,
        }
        if self.enable_semantics:
            data['semantic_segmentation'] = frame.get('semantic_segmentation')
        if self.enable_depth:
            depth = frame.get('distance_to_image_plane')
            data['depth'] = depth
            if depth is None:
                data['pointcloud'] = None
                if self.enable_semantics:
                    data['pointcloud_image'] = None
            else:
                depth_array = np.asarray(depth)
                world_points = np.asarray(self.camera.get_pointcloud()).reshape((*depth_array.shape, 3))
                if self.enable_semantics:
                    data['pointcloud_image'] = world_points
                valid = np.isfinite(depth_array) & (depth_array > 0.01) & (depth_array < 10000.0)
                data['pointcloud'] = world_points[valid]
        return data


class InteractionVideoRecorder:
    """Record robot RGB, fused top-down map, and a combined review video."""

    def __init__(
        self,
        output_dir: str,
        capture_interval: int = 20,
        fps: float = 12.0,
        show_scene_graph: bool = True,
        show_semantic_visuals: bool = True,
        record_robot_topdown: bool = True,
    ):
        if capture_interval <= 0 or fps <= 0:
            raise ValueError('capture_interval and fps must be positive')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.capture_interval = capture_interval
        self.fps = fps
        self.show_scene_graph = show_scene_graph
        self.show_semantic_visuals = bool(show_semantic_visuals)
        # Only H1 robots carry a tp_camera; other platforms would record an
        # all-black stream, so callers without that sensor disable this track.
        self.record_robot_topdown = bool(record_robot_topdown)
        self.rgb_size = (640, 360)
        # The map panes render at 1280x720. The final combined video resizes
        # each of four views into a 960x540 quadrant (1920x1080 total).
        self.map_size = (1280, 720)
        self.map_plot_width = 1000
        self._map_viewport = (0, 0, self.map_plot_width, self.map_size[1])
        self.combined_panel_size = (960, 540)
        self.combined_size = (self.combined_panel_size[0] * 2,
                              self.combined_panel_size[1] * 2)
        self._capture_paths = {}
        self._writers = {
            'robot_rgb': self._writer('robot_rgb.mp4', self.rgb_size),
            'third_person': self._writer('third_person.mp4', self.rgb_size),
            'map_topdown': self._writer('map_topdown.mp4', self.map_size),
            'pointcloud_map': self._writer(
                'pointcloud_map.mp4', (self.map_size[0] * 2, self.map_size[1])
            ),
            'combined': self._writer('combined.mp4', self.combined_size),
        }
        if self.show_semantic_visuals:
            self._writers['semantic_map'] = self._writer('semantic_map.mp4', self.map_size)
            self._writers['map_overlay'] = self._writer('map_overlay.mp4', self.map_size)
        if self.record_robot_topdown:
            self._writers['robot_topdown'] = self._writer('robot_topdown.mp4', self.rgb_size)
        self.frame_count = 0
        self._trajectory = []
        self._trajectory_breaks = set()
        self._last_frames = {}
        self._perception_events = None
        self._last_perception_step = None
        self._perception_frame_count = 0
        self._semantic_grid = None
        self._video_frames = (self.output_dir / 'video_frames.jsonl').open(
            'w', encoding='utf-8', buffering=1
        )

    def _writer(self, filename: str, size):
        capture_path = self.output_dir / f'{Path(filename).stem}_capture.avi'
        writer = cv2.VideoWriter(
            str(capture_path),
            cv2.VideoWriter_fourcc(*'FFV1'),
            self.fps,
            size,
        )
        if not writer.isOpened():
            raise RuntimeError(f'OpenCV could not open lossless capture writer for {filename}')
        self._capture_paths[filename] = capture_path
        return writer

    def capture(self, step: int, state, robot_observation: dict, interaction_observation, mapping_runtime, target_context=None):
        if step % self.capture_interval != 0:
            return
        position = tuple(float(value) for value in interaction_observation.robot_position)
        if self._trajectory and np.linalg.norm(np.asarray(position[:2]) - np.asarray(self._trajectory[-1][:2])) > 0.75:
            self._trajectory_breaks.add(len(self._trajectory))
        self._trajectory.append(position)
        state_name = state.value if hasattr(state, 'value') else str(state)
        rgb_title = 'Robot RGB' if not self.show_semantic_visuals else None
        sensors = robot_observation.get('sensors', {})
        camera_data = sensors.get('camera', {})
        if self.show_semantic_visuals and self._semantic_grid is None and camera_data.get('semantic_segmentation') is not None:
            self._semantic_grid = CameraSemanticGrid(
                mapping_runtime.map.config, mapping_runtime.map.occupancy.observed.shape,
                max_depth_m=5.0, pixel_stride=3,
            )
        if self._semantic_grid is not None:
            self._semantic_grid.update(camera_data)
        mode = getattr(mapping_runtime, 'semantic_detection_mode', None)
        uses_open_vocabulary = bool(getattr(mode, 'uses_open_vocabulary', False))
        perception = getattr(mapping_runtime, 'open_vocabulary_perception', None)
        debug_frame = getattr(perception, 'last_debug_frame', None)
        if not isinstance(debug_frame, dict) or debug_frame.get('step') != step:
            debug_frame = None
        rgb_sensor_name = 'recording_camera' if 'recording_camera' in sensors else 'camera'
        rgb_frame = self._render_camera(
            step,
            state_name,
            robot_observation,
            rgb_sensor_name,
            rgb_title or (('Robot RGB (DINO + Qwen-VL)' if getattr(getattr(perception, 'config', None), 'semantic_classifier', '') == 'qwen-vl'
             else 'Robot RGB (GroundingDINO)') if uses_open_vocabulary else 'Robot RGB (Isaac GT)'),
            model_only=(uses_open_vocabulary and self.show_semantic_visuals
                        and rgb_sensor_name == 'camera'),
            detection_frame=debug_frame,
            draw_boxes=(self.show_semantic_visuals and rgb_sensor_name == 'camera'),
        )
        third_person_frame = self._render_camera(
            step,
            state_name,
            robot_observation,
            'overview_camera',
            'Third-person',
            draw_boxes=False,
        )
        map_frame = self._render_map(step, state_name, interaction_observation, mapping_runtime, target_context, view='geometry')
        pointcloud_frame = self._render_pointcloud(
            step, state_name, interaction_observation, mapping_runtime,
            robot_observation,
        )
        pointcloud_map_frame = np.concatenate((pointcloud_frame, map_frame), axis=1)
        if self.show_semantic_visuals:
            semantic_frame = self._render_map(step, state_name, interaction_observation, mapping_runtime, target_context, view='semantic')
            overlay_frame = self._render_map(step, state_name, interaction_observation, mapping_runtime, target_context, view='overlay')
        panel_w, panel_h = self.combined_panel_size
        first_person_panel = cv2.resize(
            rgb_frame, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR
        )
        third_person_panel = cv2.resize(
            third_person_frame, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR
        )
        map_panel = cv2.resize(
            map_frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA
        )
        pointcloud_panel = cv2.resize(
            pointcloud_frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA
        )
        combined = np.concatenate((
            np.concatenate((first_person_panel, third_person_panel), axis=1),
            np.concatenate((map_panel, pointcloud_panel), axis=1),
        ), axis=0)
        self._writers['robot_rgb'].write(rgb_frame)
        self._writers['third_person'].write(third_person_frame)
        self._writers['map_topdown'].write(map_frame)
        self._writers['pointcloud_map'].write(pointcloud_map_frame)
        if self.show_semantic_visuals:
            self._writers['semantic_map'].write(semantic_frame)
            self._writers['map_overlay'].write(overlay_frame)
        self._writers['combined'].write(combined)
        self._video_frames.write(json.dumps({
            'frame_index': self.frame_count,
            'step': step,
            'state': state_name,
        }) + '\n')
        self._last_frames = {
            'robot_rgb': rgb_frame,
            'third_person': third_person_frame,
            'map_topdown': map_frame,
            'pointcloud_map': pointcloud_map_frame,
            'combined': combined,
        }
        if self.show_semantic_visuals:
            self._last_frames['semantic_map'] = semantic_frame
            self._last_frames['map_overlay'] = overlay_frame
        if self.record_robot_topdown:
            topdown_frame = self._render_camera(
                step,
                state_name,
                robot_observation,
                'tp_camera',
                'Robot top-down',
            )
            self._writers['robot_topdown'].write(topdown_frame)
            self._last_frames['robot_topdown'] = topdown_frame
        self.frame_count += 1

    def _render_pointcloud(
        self, step, state, interaction_observation, mapping_runtime, robot_observation,
    ):
        """Render a perspective point cloud with a camera following the robot."""
        fused_map = mapping_runtime.map
        frame = np.empty((self.map_size[1], self.map_size[0], 3), dtype=np.uint8)
        frame[:] = (249, 249, 249)
        frame[:, self.map_plot_width:] = (250, 250, 250)
        left, top, draw_width, draw_height = self._map_viewport
        cv2.rectangle(
            frame, (left, top), (left + draw_width - 1, top + draw_height - 1),
            (205, 205, 205), 1,
        )

        points = fused_map.voxels.points()
        z_min, z_max = 0.1, 0.8
        point_count = 0
        robot = np.asarray(interaction_observation.robot_position, dtype=np.float64)
        yaw = getattr(interaction_observation, 'robot_yaw', None)
        if yaw is None:
            orientation_data = robot_observation.get('orientation')
            if orientation_data is None:
                yaw = 0.0
            else:
                w, x, y, z = np.asarray(orientation_data, dtype=np.float64).reshape(4)
                yaw = np.arctan2(
                    2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
                )
        yaw = float(yaw)
        cosine, sine = np.cos(yaw), np.sin(yaw)
        if len(points):
            delta = points.astype(np.float64) - robot
            # Egocentric coordinates: x forward, y left, z up. The view
            # follows both robot translation and yaw, like an RViz camera
            # targeting base_link, while retaining the accumulated world cloud.
            local = np.column_stack((
                delta[:, 0] * cosine + delta[:, 1] * sine,
                -delta[:, 0] * sine + delta[:, 1] * cosine,
                delta[:, 2],
            ))
            valid = (
                np.isfinite(local).all(axis=1)
                & (points[:, 2] >= z_min)
                & (points[:, 2] <= z_max)
                & (np.abs(local[:, 0]) <= 7.0)
                & (np.abs(local[:, 1]) <= 6.0)
                & (local[:, 2] >= -0.8)
                & (local[:, 2] <= 2.2)
            )
            local = local[valid]
            point_heights = points[valid, 2]
            point_count = len(local)
            if len(local) > 40000:
                sample = np.linspace(0, len(local) - 1, 40000, dtype=int)
                local = local[sample]
                point_heights = point_heights[sample]

            # Raise the trailing camera to show more of the floor plan and
            # furniture tops while keeping the robot centered in the view.
            camera = np.array([-3.2, 0.0, 2.7], dtype=np.float64)
            look_at = np.array([1.2, 0.0, 0.45], dtype=np.float64)
            view_forward = look_at - camera
            view_forward /= np.linalg.norm(view_forward)
            view_right = np.cross(view_forward, np.array([0.0, 0.0, 1.0]))
            view_right /= np.linalg.norm(view_right)
            view_up = np.cross(view_right, view_forward)
            relative = local - camera
            depth = relative @ view_forward
            focal = (draw_height * 0.5) / np.tan(np.deg2rad(72.0) * 0.5)
            safe_depth = np.maximum(depth, 1e-3)
            screen_x = left + draw_width * 0.5 + (relative @ view_right) * focal / safe_depth
            screen_y = top + draw_height * 0.5 - (relative @ view_up) * focal / safe_depth
            visible = (
                (depth > 0.15)
                & (screen_x >= left)
                & (screen_x < left + draw_width)
                & (screen_y >= top)
                & (screen_y < top + draw_height)
            )
            columns = screen_x[visible].astype(int)
            rows = screen_y[visible].astype(int)
            point_heights = point_heights[visible]
            height_values = np.clip(
                ((point_heights - z_min) / (z_max - z_min) * 255).astype(np.uint8),
                0, 255,
            ).reshape(-1, 1)
            colors = cv2.applyColorMap(height_values, cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
            # Far points are painted first so nearer returns remain visible.
            order = np.argsort(depth[visible])[::-1]
            sorted_rows = rows[order]
            sorted_columns = columns[order]
            sorted_colors = colors[order]
            # A filled radius-2 disk makes sparse returns readable at the
            # final four-panel video size.
            stamp_offsets = [
                (dx, dy)
                for dy in range(-2, 3)
                for dx in range(-2, 3)
                if dx * dx + dy * dy <= 4
            ]
            for dx, dy in stamp_offsets:
                stamp_x = sorted_columns + dx
                stamp_y = sorted_rows + dy
                inside_view = (
                    (stamp_x >= left)
                    & (stamp_x < left + draw_width)
                    & (stamp_y >= top)
                    & (stamp_y < top + draw_height)
                )
                frame[stamp_y[inside_view], stamp_x[inside_view]] = sorted_colors[inside_view]

            def project_robot(local_points):
                relative_points = local_points - camera
                point_depth = relative_points @ view_forward
                point_depth = np.maximum(point_depth, 1e-3)
                px = left + draw_width * 0.5 + (relative_points @ view_right) * focal / point_depth
                py = top + draw_height * 0.5 - (relative_points @ view_up) * focal / point_depth
                return np.column_stack((px, py)).astype(int), point_depth

            # Draw a small robot footprint and forward arrow as a visual
            # reference for the egocentric camera.
            body = np.array([
                [-0.26, -0.18, 0.02], [0.30, -0.18, 0.02],
                [0.30, 0.18, 0.02], [-0.26, 0.18, 0.02],
                [-0.26, -0.18, 0.24], [0.30, -0.18, 0.24],
                [0.30, 0.18, 0.24], [-0.26, 0.18, 0.24],
            ])
            body_pixels, body_depth = project_robot(body)
            edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                     (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
            for start, end in edges:
                if body_depth[start] > 0.15 and body_depth[end] > 0.15:
                    cv2.line(frame, tuple(body_pixels[start]), tuple(body_pixels[end]),
                             (59, 140, 77), 2, cv2.LINE_AA)
            arrow_pixels, arrow_depth = project_robot(np.array([[0.0, 0.0, 0.28], [0.75, 0.0, 0.28]]))
            if np.all(arrow_depth > 0.15):
                cv2.arrowedLine(frame, tuple(arrow_pixels[0]), tuple(arrow_pixels[1]),
                                (59, 140, 77), 2, cv2.LINE_AA, tipLength=0.25)

        self._header(frame, f'Robot-following 3D LiDAR cloud | step {step} | {state}')
        sidebar_x = self.map_plot_width + 16
        cv2.putText(frame, 'FOLLOWING 3D VIEW', (sidebar_x, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (45, 45, 45), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Center: robot base', (sidebar_x, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (45, 45, 45), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Camera follows robot yaw', (sidebar_x, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (45, 45, 45), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Local range: 7 m', (sidebar_x, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (45, 45, 45), 1, cv2.LINE_AA)
        cv2.putText(frame, f'Points: {point_count}', (sidebar_x, 132),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (45, 45, 45), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Point height z (m)', (sidebar_x, 169),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (45, 45, 45), 1, cv2.LINE_AA)
        for index, z in enumerate(np.linspace(z_min, z_max, 5)):
            color = cv2.applyColorMap(
                np.array([[int((z - z_min) / (z_max - z_min) * 255)]], dtype=np.uint8),
                cv2.COLORMAP_VIRIDIS,
            )[0, 0].tolist()
            y = 200 + index * 25
            cv2.circle(frame, (sidebar_x + 6, y - 4), 5, color, -1, cv2.LINE_AA)
            cv2.putText(frame, f'{z:.2f} m', (sidebar_x + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (45, 45, 45), 1, cv2.LINE_AA)
        return frame

    def record_perception(self, step: int, robot_observation: dict, mapping_runtime):
        """Record each actual detector query with its exact camera frame."""

        mode = getattr(mapping_runtime, 'semantic_detection_mode', None)
        if not getattr(mode, 'uses_open_vocabulary', False):
            return
        perception = getattr(mapping_runtime, 'open_vocabulary_perception', None)
        debug_frame = getattr(perception, 'last_debug_frame', None)
        if (
            not isinstance(debug_frame, dict)
            or debug_frame.get('step') != step
            or self._last_perception_step == step
        ):
            return
        self._last_perception_step = step
        if self._perception_events is None:
            self._perception_events = (self.output_dir / 'groundingdino_detections.jsonl').open(
                'w', encoding='utf-8', buffering=1
            )
            self._writers['groundingdino'] = self._writer('groundingdino.mp4', self.rgb_size)
        self._perception_events.write(json.dumps({
            **debug_frame, 'video_frame_index': self._perception_frame_count,
        }, ensure_ascii=False) + '\n')
        frame = self._render_camera(
            step,
            'perception',
            robot_observation,
            'camera',
            'GroundingDINO (model output)',
            model_only=True,
            detection_frame=debug_frame,
        )
        self._writers['groundingdino'].write(frame)
        self._last_frames['groundingdino'] = frame
        self._perception_frame_count += 1

    def close(self):
        if self._perception_events is not None:
            self._perception_events.close()
        self._video_frames.close()
        for writer in self._writers.values():
            writer.release()
        if self._capture_paths:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            crf = os.environ.get('INTER_NAV_VIDEO_CRF', '16')
            pixel_format = 'yuv444p' if crf == '0' else 'yuv420p'
            for filename, capture_path in self._capture_paths.items():
                output_path = self.output_dir / filename
                temporary_path = output_path.with_name(f'{output_path.stem}_encoded.mp4')
                subprocess.run(
                    [
                        ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                        '-i', str(capture_path), '-an', '-c:v', 'libx264',
                        '-preset', 'veryfast', '-crf', crf, '-pix_fmt', pixel_format,
                        '-threads', os.environ.get('INTER_NAV_VIDEO_THREADS', '4'),
                        '-movflags', '+faststart', str(temporary_path),
                    ],
                    check=True,
                    stdin=subprocess.DEVNULL,
                )
                temporary_path.replace(output_path)
                capture_path.unlink()
        if self._semantic_grid is not None:
            self._semantic_grid.save(self.output_dir)
        for name, frame in self._last_frames.items():
            cv2.imwrite(str(self.output_dir / f'{name}_preview.png'), frame)

    def _render_camera(
        self,
        step: int,
        state: str,
        robot_observation: dict,
        sensor_name: str,
        title: str,
        model_only: bool = False,
        detection_frame=None,
        draw_boxes: bool = True,
    ):
        camera = robot_observation.get('sensors', {}).get(sensor_name, {})
        rgba = camera.get('rgba')
        if rgba is None:
            frame = np.zeros((self.rgb_size[1], self.rgb_size[0], 3), dtype=np.uint8)
        else:
            image = np.asarray(rgba)
            if image.ndim != 3 or image.shape[2] < 3:
                frame = np.zeros((self.rgb_size[1], self.rgb_size[0], 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(image[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
                interpolation = (
                    cv2.INTER_AREA
                    if image.shape[1] >= self.rgb_size[0] and image.shape[0] >= self.rgb_size[1]
                    else cv2.INTER_CUBIC
                )
                frame = cv2.resize(frame, self.rgb_size, interpolation=interpolation)
                if draw_boxes and model_only:
                    self._draw_groundingdino_boxes(
                        frame, detection_frame, image.shape[1], image.shape[0]
                    )
                elif draw_boxes:
                    self._draw_camera_boxes(frame, camera, image.shape[1], image.shape[0])
        self._header(frame, f'{title} | step {step} | {state}')
        if model_only:
            status = 'not queried at this step'
            if detection_frame is not None:
                status = (
                    f"{detection_frame['status']} | raw {len(detection_frame['boxes'])}"
                    f" | mapped {len(detection_frame['mapped'])}"
                )
            cv2.putText(
                frame, f'DINO: {status}', (8, self.rgb_size[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA,
            )
        return frame

    def _draw_groundingdino_boxes(self, frame, detection_frame, source_width: int, source_height: int):
        if detection_frame is None:
            return
        scale_x = self.rgb_size[0] / source_width
        scale_y = self.rgb_size[1] / source_height
        mapped = detection_frame.get('classified') or detection_frame.get('mapped', ())
        target = canonical_semantic_label(detection_frame.get('target_query', ''))
        for box in detection_frame.get('boxes', ()):
            coords = box.get('bbox_xyxy')
            if coords is None:
                continue
            x_min, y_min, x_max, y_max = coords
            classification = next((item for item in mapped
                                   if item.get('proposal_source') == box.get('proposal_source')
                                   and len(item.get('bbox_xyxy', ())) == 4
                                   and np.allclose(item['bbox_xyxy'], coords)), None)
            final_label = None if classification is None else classification.get('label')
            color = (0, 0, 255) if final_label and canonical_semantic_label(final_label) == target else ((220, 60, 230) if box.get('above_threshold') else (150, 150, 150))
            label = box['label'] if final_label is None or final_label == box['label'] else f"{box['label']} -> {final_label}"

            start = (int(x_min * scale_x), int(y_min * scale_y))
            end = (int(x_max * scale_x), int(y_max * scale_y))
            cv2.rectangle(frame, start, end, color, 2)
            cv2.putText(
                frame,
                f"{label} {box['confidence']:.2f}",
                (start[0], max(42, start[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
            )

    def _draw_camera_boxes(self, frame, camera, source_width: int, source_height: int):
        bounding_boxes = camera.get('bounding_box_2d_tight')
        if not isinstance(bounding_boxes, dict):
            return
        label_lookup = bounding_boxes.get('info', {}).get('idToLabels', {})
        scale_x = self.rgb_size[0] / source_width
        scale_y = self.rgb_size[1] / source_height
        candidates = []
        for row in bounding_boxes.get('data', []):
            values = tuple(row.tolist()) if hasattr(row, 'tolist') else tuple(row)
            if len(values) < 5:
                continue
            semantic_id, x_min, y_min, x_max, y_max = values[:5]
            labels = label_lookup.get(str(int(semantic_id)), label_lookup.get(str(semantic_id), {}))
            label = labels.get('class', 'object') if isinstance(labels, dict) else 'object'
            label = str(label).strip().lower().split('/', 1)[0]
            if label in {
                'ceiling',
                'floor',
                'g1',
                'go2',
                'ground',
                'other',
                'robot',
                'unitree_go2',
                'wall',
            }:
                continue
            if x_max <= 0 or y_max <= 0 or x_min >= source_width or y_min >= source_height:
                continue
            x_min = max(0, min(source_width - 1, float(x_min)))
            x_max = max(x_min + 1, min(source_width, float(x_max)))
            y_min = max(0, min(source_height - 1, float(y_min)))
            y_max = max(y_min + 1, min(source_height, float(y_max)))
            area = (x_max - x_min) * (y_max - y_min)
            if area < source_width * source_height * 0.002:
                continue
            if len(values) > 5 and float(values[5]) > 0.85:
                continue
            candidates.append((area, label, x_min, y_min, x_max, y_max))
        for _, label, x_min, y_min, x_max, y_max in sorted(candidates, reverse=True)[:20]:
            start = (int(x_min * scale_x), int(y_min * scale_y))
            end = (int(x_max * scale_x), int(y_max * scale_y))
            cv2.rectangle(frame, start, end, (0, 220, 255), 1)
            cv2.putText(
                frame,
                label,
                (start[0], max(28, start[1] - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )

    def _render_map(
        self, step: int, state: str, interaction_observation,
        mapping_runtime, target_context=None, view: str = 'geometry',
    ):
        if view not in ('geometry', 'semantic', 'overlay'):
            raise ValueError(f'Unknown map view: {view}')
        fused_map = mapping_runtime.map
        occupancy = fused_map.occupancy
        observed = occupancy.observed
        occupied = occupancy.occupied_mask()
        inflated = occupancy.inflated_mask()
        safety_layer = inflated & ~occupied
        grid = np.empty((*observed.shape, 3), dtype=np.uint8)
        grid[:] = self._MAP_COLORS['unknown']
        grid[observed] = self._MAP_COLORS['free']
        if view == 'semantic':
            grid[occupied] = (206, 206, 206)
        else:
            grid[safety_layer] = self._MAP_COLORS['inflation']
            grid[occupied] = self._MAP_COLORS['obstacle']

        semantic_legend = {}
        if view in ('semantic', 'overlay'):
            semantic_legend = self._paint_semantic_evidence(
                grid, fused_map, mapping_runtime,
                alpha=0.78 if view == 'semantic' else 0.52,
            )

        height, width = observed.shape
        scale = min(self.map_plot_width / width, self.map_size[1] / height)
        draw_width = max(1, int(round(width * scale)))
        draw_height = max(1, int(round(height * scale)))
        left = (self.map_plot_width - draw_width) // 2
        top = (self.map_size[1] - draw_height) // 2
        self._map_viewport = (left, top, draw_width, draw_height)
        frame = np.empty((self.map_size[1], self.map_size[0], 3), dtype=np.uint8)
        frame[:] = self._MAP_COLORS['unknown']
        frame[:, self.map_plot_width:] = (250, 250, 250)
        frame[top:top + draw_height, left:left + draw_width] = cv2.resize(
            np.flipud(grid), (draw_width, draw_height), interpolation=cv2.INTER_NEAREST
        )

        if view in ('geometry', 'overlay'):
            topology = getattr(mapping_runtime, 'semantic_voronoi', None)
            if topology is not None:
                # Refresh against the current occupancy revision before
                # drawing frontiers, so points cleared by the latest sensor
                # update disappear in this frame.
                snapshot = topology.update()
                for edge in snapshot.edges:
                    if len(edge.cells) < 2:
                        continue
                    points = np.asarray(
                        [self._world_to_pixel(occupancy.cell_to_world(cell), fused_map)
                         for cell in edge.cells],
                        dtype=np.int32,
                    )
                    cv2.polylines(frame, [points], False, (185, 105, 45), 1, cv2.LINE_AA)
                for node in snapshot.nodes:
                    if node.kind == 'junction':
                        cv2.circle(frame, self._world_to_pixel(node.position, fused_map),
                                   3, (185, 105, 45), -1, cv2.LINE_AA)
                selected_frontier = getattr(
                    getattr(mapping_runtime, 'exploration_decision', None),
                    'selected_frontier_id', None,
                )
                for position, selected in _frontier_display_markers(
                    snapshot, occupancy, interaction_observation.robot_position,
                    selected_frontier_id=selected_frontier,
                ):
                    pixel = self._world_to_pixel(position, fused_map)
                    color = (118, 59, 196)  # RGB #C43B76
                    cv2.drawMarker(frame, pixel, (35, 42, 48),
                                   cv2.MARKER_DIAMOND, 22 if selected else 18, 3, cv2.LINE_AA)
                    cv2.drawMarker(frame, pixel, color,
                                   cv2.MARKER_DIAMOND, 18 if selected else 14, 2, cv2.LINE_AA)

            for index, (start, finish) in enumerate(
                zip(self._trajectory, self._trajectory[1:]), start=1
            ):
                pixel = self._world_to_pixel(finish[:2], fused_map)
                if index in self._trajectory_breaks:
                    cv2.drawMarker(frame, pixel, (145, 95, 145),
                                   cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
                    continue
                cv2.line(frame, self._world_to_pixel(start[:2], fused_map),
                         pixel, (59, 140, 77), 3, cv2.LINE_AA)
            planned_path = mapping_runtime._planned_path or ()
            if planned_path:
                path_points = [tuple(interaction_observation.robot_position[:2])] + [
                    point[:2] for point in planned_path
                ]
                for start, finish in zip(path_points, path_points[1:]):
                    cv2.line(frame, self._world_to_pixel(start, fused_map),
                             self._world_to_pixel(finish, fused_map),
                             (150, 105, 95), 2, cv2.LINE_AA)
            robot_pixel = self._world_to_pixel(
                interaction_observation.robot_position[:2], fused_map
            )
            cv2.circle(frame, robot_pixel, 10, (59, 140, 77), -1, cv2.LINE_AA)

        if view in ('semantic', 'overlay'):
            target_node = getattr(target_context, 'target_node', None)
            target_cue = getattr(target_context, 'target_cue', None)
            if target_node is None and target_cue is not None:
                pixel = self._world_to_pixel(target_cue.position[:2], fused_map)
                color = (170, 70, 170)
                cv2.drawMarker(frame, pixel, color, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
                semantic_legend['target cue'] = color
            if target_node is not None:
                pixel = self._world_to_pixel(target_node.position[:2], fused_map)
                color = (72, 125, 185)
                cv2.drawMarker(frame, pixel, color, cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA)
                semantic_legend['target'] = color

        titles = {
            'geometry': 'Geometry map + Voronoi',
            'semantic': 'Semantic segmentation map',
            'overlay': 'Geometry + segmentation overlay',
        }
        self._header(frame, f'{titles[view]} | step {step} | {state}')
        self._draw_map_sidebar(
            frame,
            view=view,
            stats={
                'lidar_frames': fused_map.lidar_frames,
                'rgb_frames': fused_map.rgb_frames,
                'plans': fused_map.plan_count,
            },
            occupied_count=int(occupied.sum()),
            safety_count=int(safety_layer.sum()),
            inflation_radius=fused_map.config.obstacle_inflation_radius or fused_map.config.robot_radius,
            static_inflation_radius=fused_map.config.static_obstacle_inflation_radius
                                    or fused_map.config.obstacle_inflation_radius
                                    or fused_map.config.robot_radius,
            exploration_state=getattr(mapping_runtime, 'exploration_state', ''),
            semantic_legend=semantic_legend,
        )
        return frame

    def _paint_semantic_evidence(self, grid, fused_map, mapping_runtime, alpha: float):
        semantic_grid = getattr(self, '_semantic_grid', None)
        if semantic_grid is None:
            return {}
        class_ids, labels = semantic_grid.class_grid()
        if class_ids.shape != grid.shape[:2]:
            return {}
        legend = {}
        for index, label in enumerate(labels):
            mask = class_ids == index
            if not np.any(mask):
                continue
            color = np.asarray(self._semantic_color(label), dtype=np.float32)
            grid[mask] = np.clip(
                (1.0 - alpha) * grid[mask].astype(np.float32) + alpha * color,
                0, 255,
            ).astype(np.uint8)
            legend[label] = tuple(int(value) for value in color)
        return legend

    # Occupancy colors are OpenCV BGR values. Keep topology overlays separate.
    _MAP_COLORS = {
        'unknown': (228, 224, 216),   # RGB #D8E0E4
        'free': (252, 251, 250),      # RGB #FAFBFC
        'inflation': (211, 197, 177), # RGB #B1C5D3
        'obstacle': (100, 78, 52),    # RGB #344E64
    }

    # Stable category colors keep the same class recognizable across runs.
    _SEMANTIC_COLOR_HEX = {
        'cabinet': '#4E79A7', 'chair': '#59A14F', 'curtain': '#B07AA1',
        'couch': '#E15759', 'bed': '#9C755F', 'table': '#F28E2B',
        'teatable': '#76B7B2', 'refrigerator': '#2F8F9D', 'tv': '#7F7F7F',
        'person': '#D37295', 'plant': '#367C62', 'decoration': '#C9A227',
        'blanket': '#6F84B7', 'pillow': '#BD7B60', 'cup': '#A2A43A',
        'plate': '#5B9BD5', 'bowl': '#CC6677', 'nightstand': '#8A6FA0',
        'desk': '#A05A6D', 'door': '#4F8499', 'window': '#81A683',
        'faucet': '#76839A', 'electriccooker': '#B67945', 'light': '#C3A53A',
        'clothes': '#728C4A', 'pot': '#C0746B', 'chair,sofachair': '#7C6EA9',
        'floor': '#C8BFB3', 'ground': '#BFC8BC', 'wall': '#A8B6C2',
        'ceiling': '#CDD2D5',
    }

    @classmethod
    def _semantic_color(cls, label):
        hex_color = cls._SEMANTIC_COLOR_HEX.get(label)
        if hex_color is not None:
            return tuple(int(hex_color[index:index + 2], 16) for index in (5, 3, 1))
        digest = hashlib.sha1(label.encode('utf-8')).digest()
        hsv = np.uint8([[[digest[0] % 180, 120, 175]]])
        return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])

    def _draw_map_sidebar(
        self, frame, *, view, stats, occupied_count, safety_count,
        inflation_radius, static_inflation_radius, exploration_state,
        semantic_legend,
    ):
        x = self.map_plot_width
        cv2.rectangle(frame, (x, 35), (frame.shape[1] - 1, frame.shape[0] - 1),
                      (250, 250, 250), -1)
        cv2.line(frame, (x, 35), (x, frame.shape[0] - 1), (190, 190, 190), 1)
        title_color = (66, 70, 72)
        text_color = (62, 62, 62)

        def heading(label, y):
            cv2.putText(frame, label, (x + 16, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, title_color, 1, cv2.LINE_AA)

        def line(label, y):
            cv2.putText(frame, label, (x + 16, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, text_color, 1, cv2.LINE_AA)

        if view in ('geometry', 'overlay'):
            heading('OCCUPANCY', 61)
            for index, (label, key) in enumerate((
                ('Unknown', 'unknown'), ('Free', 'free'),
                ('Inflation', 'inflation'), ('Obstacle', 'obstacle'),
            )):
                row = 79 + 22 * index
                cv2.rectangle(frame, (x + 18, row), (x + 39, row + 15),
                              self._MAP_COLORS[key], -1)
                cv2.rectangle(frame, (x + 18, row), (x + 39, row + 15),
                              (110, 105, 100), 1)
                cv2.putText(frame, label, (x + 49, row + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, text_color, 1, cv2.LINE_AA)
            heading('GEOMETRY', 201)
            for index, (label, color, symbol) in enumerate((
                ('Voronoi', (185, 105, 45), 'line'),
                ('Frontier', (118, 59, 196), 'diamond'),
                ('Trajectory', (59, 140, 77), 'line'),
                ('Robot', (59, 140, 77), 'circle'),
            )):
                row = 225 + index * 24
                if symbol == 'line':
                    cv2.line(frame, (x + 18, row - 5), (x + 40, row - 5),
                             color, 1 if label == 'Voronoi' else 2, cv2.LINE_AA)
                elif symbol == 'diamond':
                    cv2.drawMarker(frame, (x + 29, row - 5), color,
                                   cv2.MARKER_DIAMOND, 14, 2, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (x + 29, row - 5), 7, color, -1)
                cv2.putText(frame, label, (x + 49, row),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1, cv2.LINE_AA)
            if view == 'geometry':
                heading('RUN', 354)
                line(f"LiDAR {stats['lidar_frames']} | RGB {stats['rgb_frames']}", 378)
                line(f"Occupied {occupied_count} | buffer {safety_count}", 400)
                line(f"Inflation {inflation_radius:.2f}/{static_inflation_radius:.2f} m", 422)
                line(f"Plans {stats['plans']} | {exploration_state or '-'}", 444)

        if view in ('semantic', 'overlay'):
            top = 61 if view == 'semantic' else 351
            heading('SEMANTIC SEGMENTATION', top)
            line('Camera pixels + depth (<=5 m)', top + 20)
            if not semantic_legend:
                line('No labeled pixels yet', top + 49)
            for index, label in enumerate(sorted(semantic_legend, key=str.casefold)[:30]):
                column, row = divmod(index, 16)
                item_x = x + 17 + column * 136
                item_y = top + 48 + row * 20
                color = semantic_legend[label]
                cv2.circle(frame, (item_x + 6, item_y - 4), 6,
                           (35, 42, 48), -1, cv2.LINE_AA)
                cv2.circle(frame, (item_x + 6, item_y - 4), 4,
                           color, -1, cv2.LINE_AA)
                cv2.putText(frame, label[:16], (item_x + 17, item_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, text_color, 1, cv2.LINE_AA)

    @staticmethod
    def _put_label(frame, text, org, color, label_boxes, font_scale: float = 0.42):
        """Draw text only when its box does not overlap an existing label."""
        if not text:
            return
        (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        x0, y0 = int(org[0]), int(org[1]) - height - baseline
        x1, y1 = int(org[0]) + width, int(org[1]) + baseline
        for bx0, by0, bx1, by1 in label_boxes:
            if x0 < bx1 and bx0 < x1 and y0 < by1 and by0 < y1:
                return
        label_boxes.append((x0, y0, x1, y1))
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

    def _draw_voxels(self, frame, fused_map):
        points = fused_map.voxels.points()
        colors = fused_map.voxels.colors()
        if len(points) == 0:
            return
        mask = (points[:, 2] >= 0.1) & (points[:, 2] <= 1.8)
        points = points[mask]
        colors = colors[mask]
        if len(points) > 50000:
            indices = np.linspace(0, len(points) - 1, 50000, dtype=int)
            points = points[indices]
            colors = colors[indices]
        x_min, x_max = fused_map.config.x_limits
        y_min, y_max = fused_map.config.y_limits
        left, top, draw_width, draw_height = self._map_viewport
        columns = left + ((points[:, 0] - x_min) / (x_max - x_min) * (draw_width - 1)).astype(int)
        rows = top + ((y_max - points[:, 1]) / (y_max - y_min) * (draw_height - 1)).astype(int)
        valid = (
            (columns >= left)
            & (columns < left + draw_width)
            & (rows >= top)
            & (rows < top + draw_height)
        )
        observed_pixels = cv2.resize(
            np.flipud(fused_map.occupancy.observed.astype(np.uint8)),
            (draw_width, draw_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        inside = np.flatnonzero(valid)
        valid[inside] = observed_pixels[rows[inside] - top, columns[inside] - left]
        target = frame[rows[valid], columns[valid]].astype(np.uint16)
        bgr = colors[valid][:, ::-1].astype(np.uint16)
        frame[rows[valid], columns[valid]] = ((4 * target + bgr) // 5).astype(np.uint8)

    def _world_to_pixel(self, xy, fused_map):
        x_min, x_max = fused_map.config.x_limits
        y_min, y_max = fused_map.config.y_limits
        left, top, draw_width, draw_height = self._map_viewport
        x = left + int((float(xy[0]) - x_min) / (x_max - x_min) * (draw_width - 1))
        y = top + int((y_max - float(xy[1])) / (y_max - y_min) * (draw_height - 1))
        return (
            int(np.clip(x, left, left + draw_width - 1)),
            int(np.clip(y, top, top + draw_height - 1)),
        )

    @staticmethod
    def _header(frame, text: str):
        scale = frame.shape[1] / 1280.0
        bar_height = max(25, int(35 * scale))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_height), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (8, int(bar_height * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.48, 0.72 * scale),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _text(frame, text: str, y: int):
        scale = frame.shape[1] / 1280.0
        position = (8, int(y * max(1.0, scale * 1.4)))
        font_scale = max(0.42, 0.58 * scale)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (250, 250, 250), 1, cv2.LINE_AA)


def _quaternion_matrix(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quaternion_multiply(left, right):
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
