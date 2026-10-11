from omni.isaac.core.scenes import Scene

from grutopia.core.runtime.task_runtime import TaskRuntime
from grutopia.core.task import BaseTask


@BaseTask.register('SingleInferenceTask')
class SimpleInferenceTask(BaseTask):
    def __init__(self, runtime: TaskRuntime, scene: Scene):
        super().__init__(runtime, scene)

    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)
        task_settings = self.runtime.task_settings
        settings = task_settings.get('navigation_scene') if isinstance(task_settings, dict) else None
        if settings is not None:
            from omni.isaac.core.utils.stage import get_current_stage
            from grutopia_extension.interactive_navigation.scene_paths import prepare_navigation_scene

            root = self.runtime.root_path + self.runtime.scene_root_path
            prepare_navigation_scene(get_current_stage(), root, settings['material_mode'])

    def calculate_metrics(self) -> dict:
        pass

    def is_done(self) -> bool:
        return False

    def individual_reset(self):
        for name, metric in self.metrics.items():
            metric.reset()
