# Adapted from https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/__init__.py
# Original source SHA256: 2b61e4b16274bdad8174e7a11f49a6cde647b015b0acbe0c0861bd9c0b935546
from omni.isaac.lab.envs.mdp import *  # noqa: F401, F403
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403

# Lab 1.4 exposes the applied torque in articulation data.
def joint_effort(env, asset_cfg=SceneEntityCfg("robot")):
    return env.scene[asset_cfg.name].data.applied_torque[:, asset_cfg.joint_ids]
