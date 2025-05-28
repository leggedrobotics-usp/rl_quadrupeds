"""
angular_velocity.py

Reward functions for tracking angular velocity commands.

Available functions:
- track_ang_vel_z_exp: Reward tracking of angular velocity commands (yaw) using exponential kernel.
- track_ang_vel_z_exp_sigma_squared: Reward tracking of angular velocity commands (yaw) using exponential kernel with squared standard deviation.
"""

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Reward tracking of angular velocity commands (yaw) using exponential kernel.
    Taken from the paper "Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior"

    $$r_{\omega^{cmd}_z} = \exp(-(\omega_z - \omega_z^{cmd})^2/\sigma_{\omega_z}) $$
    
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        asset.data.root_ang_vel_b[:, 2] - env.command_manager.get_command(command_name)[:, 2]
    )
    return torch.exp(-ang_vel_error / std)

def track_ang_vel_z_exp_sigma_squared(
    env: ManagerBasedRLEnv,
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Reward tracking of angular velocity commands (yaw) using exponential kernel.
    Taken from the paper "Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior"

    $$r_{\omega^{cmd}_z} = \exp(-(\omega_z - \omega_z^{cmd})^2/\sigma_{\omega_z}^2) $$
    
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        asset.data.root_ang_vel_b[:, 2] - env.command_manager.get_command(command_name)[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)