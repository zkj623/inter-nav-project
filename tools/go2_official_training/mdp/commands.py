# Adapted from https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/commands/velocity_command.py
# Original source SHA256: 87358cad63a01f6131fa4ba9bbbe0a177cfcb03f142fb3ac74fd356df57cc042
from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.envs.mdp import UniformVelocityCommandCfg
from omni.isaac.lab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
