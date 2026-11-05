"""Lifecycle wrapper for point-navigation review cameras and video writers."""

from types import SimpleNamespace

from grutopia_extension.interactive_navigation.visualization import (
    FixedOverviewCamera,
    FollowingRobotCamera,
    InteractionVideoRecorder,
)


class NavigationRecordingSession:
    def __init__(
        self,
        output_dir: str,
        capture_interval: int,
        fps: float,
        overview_position,
        overview_look_at,
        overview_focal_length: float = 18.0,
        show_scene_graph: bool = True,
        follow_robot: bool = False,
        third_person_offset=(-0.70, 0.0, 1.55),
        third_person_pitch_degrees: float = 50.0,
        third_person_fov_degrees: float = 80.0,
    ):
        self.capture_interval = capture_interval
        # Navigation platforms (Go2/G1) have no tp_camera; recording that
        # stream would only produce a black video.
        self.recorder = InteractionVideoRecorder(
            output_dir,
            capture_interval=capture_interval,
            fps=fps,
            show_scene_graph=show_scene_graph,
            record_robot_topdown=False,
        )
        self.follow_robot = bool(follow_robot)
        if self.follow_robot:
            self.overview_camera = FollowingRobotCamera(
                resolution=(640, 360),
                offset=third_person_offset,
                pitch_degrees=third_person_pitch_degrees,
                horizontal_fov_degrees=third_person_fov_degrees,
                prim_path='/World/FollowingThirdPersonCamera',
            )
        else:
            self.overview_camera = FixedOverviewCamera(
                position=overview_position,
                look_at=overview_look_at,
                focal_length=overview_focal_length,
            )
        self._closed = False

    def prime(self, robot_observation: dict):
        if self.follow_robot:
            self.update_pose(robot_observation)

    def update_pose(self, robot_observation: dict):
        if self.follow_robot:
            self.overview_camera.update_pose(
                robot_observation['position'],
                robot_observation['orientation'],
            )

    def capture(
        self,
        step: int,
        robot_observation: dict,
        navigation_component,
        state: str = 'navigate_to_goal',
    ):
        if step % self.capture_interval == 0:
            robot_observation.setdefault('sensors', {})['overview_camera'] = self.overview_camera.get_data()
        position = tuple(float(value) for value in robot_observation['position'])
        self.recorder.capture(
            step,
            state,
            robot_observation,
            SimpleNamespace(robot_position=position),
            navigation_component.mapping,
            target_context=navigation_component,
        )

    def record_perception(self, step: int, robot_observation: dict, mapping_runtime):
        self.recorder.record_perception(step, robot_observation, mapping_runtime)

    def close(self):
        if not self._closed:
            self.recorder.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
