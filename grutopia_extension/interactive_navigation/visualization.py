"""MP4 visualizations for RGB observations and fused navigation maps."""

import json
from pathlib import Path

import cv2
import numpy as np

from grutopia_extension.interactive_navigation.mapping import canonical_semantic_label


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
        prim_path='/World/FollowingRobotCamera',
    ):
        from omni.isaac.sensor import Camera

        self.offset = np.asarray(offset, dtype=np.float64)
        self.pitch_degrees = float(pitch_degrees)
        self.smoothing = 0.20
        self._smoothed_position = None
        self._smoothed_yaw = None
        self.enable_depth = bool(enable_depth)
        self.horizontal_fov_degrees = float(horizontal_fov_degrees)
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
        self.camera.add_bounding_box_2d_tight_to_frame()
        if self.enable_depth:
            self.camera.add_distance_to_image_plane_to_frame()

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
        frame = self.camera.get_current_frame()
        position, orientation = self.camera.get_world_pose(camera_axes='world')
        data = {
            'rgba': self.camera.get_rgba(),
            'rendering_frame': frame.get('rendering_frame'),
            'horizontal_fov_degrees': self.horizontal_fov_degrees,
            'bounding_box_2d_tight': frame.get('bounding_box_2d_tight'),
            'position': np.asarray(position, dtype=np.float32),
            'orientation': np.asarray(orientation, dtype=np.float32),
        }
        if self.enable_depth:
            depth = frame.get('distance_to_image_plane')
            data['depth'] = depth
            if depth is None:
                data['pointcloud'] = None
            else:
                depth_array = np.asarray(depth)
                world_points = np.asarray(self.camera.get_pointcloud()).reshape((*depth_array.shape, 3))
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
        record_robot_topdown: bool = True,
    ):
        if capture_interval <= 0 or fps <= 0:
            raise ValueError('capture_interval and fps must be positive')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.capture_interval = capture_interval
        self.fps = fps
        self.show_scene_graph = show_scene_graph
        # Only H1 robots carry a tp_camera; other platforms would record an
        # all-black stream, so callers without that sensor disable this track.
        self.record_robot_topdown = bool(record_robot_topdown)
        self.rgb_size = (640, 360)
        # The map pane doubles the camera resolution; the combined layout
        # stacks both camera views on the left and the enlarged map right.
        self.map_size = (1280, 720)
        self.combined_size = (self.rgb_size[0] + self.map_size[0], self.map_size[1])
        self._writers = {
            'robot_rgb': self._writer('robot_rgb.mp4', self.rgb_size),
            'third_person': self._writer('third_person.mp4', self.rgb_size),
            'map_topdown': self._writer('map_topdown.mp4', self.map_size),
            'combined': self._writer('combined.mp4', self.combined_size),
        }
        if self.record_robot_topdown:
            self._writers['robot_topdown'] = self._writer('robot_topdown.mp4', self.rgb_size)
        self.frame_count = 0
        self._trajectory = []
        self._trajectory_breaks = set()
        self._last_frames = {}
        self._perception_events = None
        self._last_perception_step = None
        self._perception_frame_count = 0
        self._video_frames = (self.output_dir / 'video_frames.jsonl').open(
            'w', encoding='utf-8', buffering=1
        )

    def _writer(self, filename: str, size):
        writer = cv2.VideoWriter(
            str(self.output_dir / filename),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            size,
        )
        if not writer.isOpened():
            raise RuntimeError(f'OpenCV could not open MP4 writer for {filename}')
        return writer

    def capture(self, step: int, state, robot_observation: dict, interaction_observation, mapping_runtime, target_context=None):
        if step % self.capture_interval != 0:
            return
        position = tuple(float(value) for value in interaction_observation.robot_position)
        if self._trajectory and np.linalg.norm(np.asarray(position[:2]) - np.asarray(self._trajectory[-1][:2])) > 0.75:
            self._trajectory_breaks.add(len(self._trajectory))
        self._trajectory.append(position)
        state_name = state.value if hasattr(state, 'value') else str(state)
        mode = getattr(mapping_runtime, 'semantic_detection_mode', None)
        uses_open_vocabulary = bool(getattr(mode, 'uses_open_vocabulary', False))
        perception = getattr(mapping_runtime, 'open_vocabulary_perception', None)
        debug_frame = getattr(perception, 'last_debug_frame', None)
        if not isinstance(debug_frame, dict) or debug_frame.get('step') != step:
            debug_frame = None
        rgb_frame = self._render_camera(
            step,
            state_name,
            robot_observation,
            'camera',
            ('Robot RGB (DINO + Qwen-VL)' if getattr(getattr(perception, 'config', None), 'semantic_classifier', '') == 'qwen-vl'
             else 'Robot RGB (GroundingDINO)') if uses_open_vocabulary else 'Robot RGB (Isaac GT)',
            model_only=uses_open_vocabulary,
            detection_frame=debug_frame,
        )
        third_person_frame = self._render_camera(
            step,
            state_name,
            robot_observation,
            'overview_camera',
            'Third-person (Isaac GT)',
        )
        map_frame = self._render_map(step, state_name, interaction_observation, mapping_runtime, target_context)
        camera_column = np.concatenate((rgb_frame, third_person_frame), axis=0)
        combined = np.concatenate((camera_column, map_frame), axis=1)
        self._writers['robot_rgb'].write(rgb_frame)
        self._writers['third_person'].write(third_person_frame)
        self._writers['map_topdown'].write(map_frame)
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
            'combined': combined,
        }
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
                frame = cv2.resize(frame, self.rgb_size, interpolation=cv2.INTER_NEAREST)
                if model_only:
                    self._draw_groundingdino_boxes(
                        frame, detection_frame, image.shape[1], image.shape[0]
                    )
                else:
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

    def _render_map(self, step: int, state: str, interaction_observation, mapping_runtime, target_context=None):
        fused_map = mapping_runtime.map
        occupancy = fused_map.occupancy
        observed = occupancy.observed
        occupied = occupancy.occupied_mask()
        inflated = occupancy.inflated_mask()
        safety_layer = inflated & ~occupied
        grid = np.zeros((*observed.shape, 3), dtype=np.uint8)
        grid[:] = (28, 28, 32)
        grid[observed] = (205, 205, 205)
        grid[safety_layer] = (120, 190, 235)
        grid[occupied] = (20, 20, 220)
        grid = np.flipud(grid)
        frame = cv2.resize(grid, self.map_size, interpolation=cv2.INTER_NEAREST)
        topology = getattr(mapping_runtime, 'semantic_voronoi', None)
        if topology is not None:
            self._draw_regions(frame, topology, observed.shape)
        self._draw_voxels(frame, fused_map)
        label_boxes = []
        if topology is not None:
            snapshot = topology.cached_snapshot()
            nodes = {node.node_id: node for node in snapshot.nodes}
            for edge in snapshot.edges:
                # Draw the true skeleton path; straight node-to-node lines cut
                # through obstacles on curved corridors.
                points = np.asarray(
                    [
                        self._world_to_pixel(occupancy.cell_to_world(cell), fused_map)
                        for cell in edge.cells
                    ],
                    dtype=np.int32,
                )
                cv2.polylines(frame, [points], False, (200, 90, 20), 2, cv2.LINE_AA)
            for node in snapshot.nodes:
                if node.kind != 'junction':
                    continue
                cv2.circle(frame, self._world_to_pixel(node.position, fused_map), 5, (30, 140, 230), -1)
            for doorway in getattr(snapshot, 'doorways', ()):
                pixel = self._world_to_pixel(doorway.position, fused_map)
                cv2.drawMarker(frame, pixel, (255, 255, 255), cv2.MARKER_DIAMOND, 14, 2)
            selected_frontier = getattr(
                getattr(mapping_runtime, 'exploration_decision', None),
                'selected_frontier_id',
                None,
            )
            for frontier in snapshot.frontiers:
                pixel = self._world_to_pixel(frontier.centroid, fused_map)
                selected = frontier.frontier_id == selected_frontier
                color = (0, 255, 255) if selected else (255, 200, 0)
                cv2.circle(frame, pixel, 11 if selected else 7, (0, 0, 0), -1)
                cv2.circle(frame, pixel, 9 if selected else 5, color, -1)
            for attachment in snapshot.semantics:
                node = nodes.get(attachment.node_id)
                if node is None:
                    continue
                pixel = self._world_to_pixel(node.position, fused_map)
                self._put_label(
                    frame,
                    attachment.label[:12],
                    (pixel[0] + 6, pixel[1] + 16),
                    (150, 40, 190),
                    label_boxes,
                )
        for index, (left, right) in enumerate(zip(self._trajectory, self._trajectory[1:]), start=1):
            if index in self._trajectory_breaks:
                pixel = self._world_to_pixel(right[:2], fused_map)
                cv2.drawMarker(frame, pixel, (220, 0, 220), cv2.MARKER_TILTED_CROSS, 16, 2)
                cv2.putText(
                    frame,
                    'RECOVER',
                    (pixel[0] + 7, pixel[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (180, 0, 180),
                    1,
                    cv2.LINE_AA,
                )
                continue
            cv2.line(
                frame,
                self._world_to_pixel(left[:2], fused_map),
                self._world_to_pixel(right[:2], fused_map),
                (20, 160, 20),
                3,
            )

        planned_path = mapping_runtime._planned_path or ()
        if planned_path:
            path_points = [tuple(interaction_observation.robot_position[:2])] + [point[:2] for point in planned_path]
            for left, right in zip(path_points, path_points[1:]):
                cv2.line(
                    frame,
                    self._world_to_pixel(left, fused_map),
                    self._world_to_pixel(right, fused_map),
                    (255, 120, 0),
                    3,
                )
            for point in planned_path:
                cv2.circle(frame, self._world_to_pixel(point[:2], fused_map), 6, (255, 120, 0), -1)

        if self.show_scene_graph:
            graph_nodes = sorted(
                fused_map.scene_graph.object_nodes(),
                key=lambda node: node.observations,
                reverse=True,
            )
            for node in graph_nodes[:50]:
                pixel = self._world_to_pixel(node.position[:2], fused_map)
                cv2.circle(frame, pixel, 6, (0, 200, 255), -1)
                self._put_label(
                    frame,
                    node.label[:14],
                    (pixel[0] + 6, pixel[1] - 6),
                    (0, 140, 210),
                    label_boxes,
                )

        robot_pixel = self._world_to_pixel(interaction_observation.robot_position[:2], fused_map)
        cv2.circle(frame, robot_pixel, 11, (20, 230, 20), -1)
        target_node = getattr(target_context, 'target_node', None)
        target_cue = getattr(target_context, 'target_cue', None)
        if target_node is None and target_cue is not None:
            cue_pixel = self._world_to_pixel(target_cue.position[:2], fused_map)
            cv2.drawMarker(frame, cue_pixel, (255, 80, 210), cv2.MARKER_DIAMOND, 23, 2, cv2.LINE_AA)
            query = getattr(target_context.config, 'target_query', '')
            self._put_label(frame, f'CUE: {query}',
                            (cue_pixel[0] + 12, cue_pixel[1] - 8), (255, 80, 210), label_boxes)
        if target_node is not None:
            query = getattr(getattr(target_context, 'config', None), 'target_query', target_node.label)
            target_pixel = self._world_to_pixel(target_node.position[:2], fused_map)
            confirmed = bool(getattr(target_context, 'target_confirmed', False))
            color = (0, 255, 255) if confirmed else (0, 180, 255)
            cv2.circle(frame, target_pixel, 13, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.drawMarker(frame, target_pixel, color, cv2.MARKER_CROSS, 23, 3, cv2.LINE_AA)
            label = f'{"TARGET" if confirmed else "CANDIDATE"}: {query}'
            width, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            label_x = int(np.clip(target_pixel[0] + 16, 4, max(4, frame.shape[1] - width - 8)))
            label_y = int(np.clip(target_pixel[1] - 10, 100, frame.shape[0] - 8))
            cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        stats = {
            'lidar_frames': fused_map.lidar_frames,
            'rgb_frames': fused_map.rgb_frames,
            'voxel_count': fused_map.voxels.voxel_count,
            'plans': fused_map.plan_count,
        }
        self._header(frame, f'Point-cloud occupancy grid | step {step} | {state}')
        self._text(
            frame,
            f"LiDAR {stats['lidar_frames']}  RGB {stats['rgb_frames']}  "
            f"voxels {stats['voxel_count']}",
            42,
        )
        inflation_radius = fused_map.config.obstacle_inflation_radius or fused_map.config.robot_radius
        static_inflation_radius = fused_map.config.static_obstacle_inflation_radius or inflation_radius
        self._text(
            frame,
            f"occupied {int(occupied.sum())}  safety {int(safety_layer.sum())}  "
            f"inflate {inflation_radius:.2f}/{static_inflation_radius:.2f}m  plans {stats['plans']}",
            59,
        )
        exploration_state = getattr(mapping_runtime, 'exploration_state', '')
        if exploration_state:
            decision = getattr(mapping_runtime, 'exploration_decision', None)
            selected = getattr(decision, 'selected_frontier_id', None)
            used_model = getattr(decision, 'used_model', False)
            self._text(
                frame,
                f'mode {exploration_state}  frontier {selected or "-"}  qwen {bool(used_model)}',
                76,
            )
        return frame

    # Distinct muted colors for region shading (BGR).
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

    def _draw_regions(self, frame, topology, grid_shape):
        """Tint traversable space per hierarchical region."""
        assignment = getattr(topology, 'region_assignment', lambda: None)()
        if not assignment:
            return
        labels_grid, region_ids = assignment
        if labels_grid.shape != grid_shape or not region_ids:
            return
        palette = self._REGION_PALETTE[np.arange(len(region_ids)) % len(self._REGION_PALETTE)]
        colored = palette[np.clip(labels_grid, 0, None)]
        colored[labels_grid < 0] = 0
        colored = cv2.resize(
            np.flipud(colored),
            self.map_size,
            interpolation=cv2.INTER_NEAREST,
        )
        mask = cv2.resize(
            np.flipud((labels_grid >= 0).astype(np.uint8)),
            self.map_size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        blended = cv2.addWeighted(frame, 0.65, colored, 0.35, 0.0)
        frame[mask] = blended[mask]

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
        columns = ((points[:, 0] - x_min) / (x_max - x_min) * (self.map_size[0] - 1)).astype(int)
        rows = ((y_max - points[:, 1]) / (y_max - y_min) * (self.map_size[1] - 1)).astype(int)
        valid = (
            (columns >= 0)
            & (columns < self.map_size[0])
            & (rows >= 0)
            & (rows < self.map_size[1])
        )
        # Blend instead of overwrite so sparse voxel colors read as texture
        # rather than salt-and-pepper noise over the occupancy layers.
        target = frame[rows[valid], columns[valid]].astype(np.uint16)
        bgr = colors[valid][:, ::-1].astype(np.uint16)
        frame[rows[valid], columns[valid]] = ((2 * target + 3 * bgr) // 5).astype(np.uint8)

    def _world_to_pixel(self, xy, fused_map):
        x_min, x_max = fused_map.config.x_limits
        y_min, y_max = fused_map.config.y_limits
        x = int((float(xy[0]) - x_min) / (x_max - x_min) * (self.map_size[0] - 1))
        y = int((y_max - float(xy[1])) / (y_max - y_min) * (self.map_size[1] - 1))
        return (
            int(np.clip(x, 0, self.map_size[0] - 1)),
            int(np.clip(y, 0, self.map_size[1] - 1)),
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
