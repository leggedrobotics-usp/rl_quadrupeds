"""
linear_velocity.py

Reward functions for tracking linear velocity commands.

Available functions:
- track_lin_vel_xy_exp: Reward tracking of linear velocity commands (xy axes) using exponential kernel.
- track_lin_vel_xy_exp_sigma_squared: Reward tracking of linear velocity commands (xy axes) using exponential kernel with squared standard deviation.
- track_foot_contact_schedule_velocities: Tracks the velocities xy of the feet and penalizes the deviation from the desired contact states.
- track_joint_vel_l2: Tracks the magnitude of the joint velocities.
- track_feet_contact_velocity: Computes a reward based on the velocity of the feet when they are close to the ground.
"""

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils.math import quat_apply_yaw

def track_lin_vel_xy_exp(
    env,
    std: float,
    command_name: str,
    asset_cfg=None
) -> torch.Tensor:
    """Bonus: Track linear velocity commands (xy) — returns [0, 1]."""
    asset = env.scene[asset_cfg.name]
    base_lin_vel_body = asset.data.root_lin_vel_b
    cmd_lin_vel_body = env.command_manager.get_command(command_name)
    lin_vel_error = torch.sum(torch.square(base_lin_vel_body[:, :2] - cmd_lin_vel_body[:, :2]), dim=1)
    bonus = torch.exp(-lin_vel_error / std)
    return torch.clip(bonus, 0.0, 1.0)

def track_lin_vel_xy_exp_sigma_squared(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Reward tracking of linear velocity commands (xy axes) using exponential kernel.
    It uses the squared standard deviation instead of the standard deviation.
    It sums the difference between the x and y components of the linear velocity and the command.
    Taken from Isaac Lab standard implementation.
    
    $$ r_{v_{x,y}^{cmd}} = \exp(-|v_{xy} - v_{xy}^{cmd}|^2/\sigma_{v_{xy}}^2)$$
    
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(
            asset.data.root_lin_vel_b[:, :2] - env.command_manager.get_command(command_name)[:, :2]
        ),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)

def track_foot_contact_schedule_velocities(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Tracks the velocities xy of the feet and penalizes the
    deviation from the desired contact states.

    
    $$-\frac{1}{4} \sum_{i=0}^{3} \left(1 - D_i\right) \left(1 - \exp\left(-\frac{v_{{xy}_i}^2}{\sigma}\right)\right)$$
    
    
    ATTENTION: to use this reward function, the environment must
    contain the quadrupeds_assets.gaits.WTWCommandFootStates inside
    the command_manager. This is because this function requires
    the desired_contact_states variable.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # Get the magnitude of the velocities acting on each feet
    feet_velocities = torch.norm(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2],
        dim=-1
    )

    exp_term = torch.exp(-(feet_velocities ** 2) / std)
    return torch.mean(
        -(1 - env.desired_contact_states) * (1 - exp_term), 
        dim=1
    )

def track_joint_vel_l2(env, asset_cfg=SceneEntityCfg("robot"), k: float = 5.0):
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    penalty_val = torch.sum(torch.square(joint_vel), dim=1)
    penalty = -torch.exp(-k * penalty_val)  # [-1, 0]
    return penalty

def track_feet_contact_velocity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Computes a reward based on the velocity of the feet
    when they are close to the ground.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Approximate contact by checking if foot is near the ground 
    # (z-position close to 0)
    near_ground = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2] < 0.03
    foot_velocities = torch.square(
        torch.norm(
            asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 0:3],
            dim=2
        )
    )
    return torch.sum(near_ground * foot_velocities, dim=1)

def lin_vel_z_penalty(env, asset_cfg=None, k: float = 5.0) -> torch.Tensor:
    """Penalty: z-axis linear velocity — returns [-1, 0]."""
    asset = env.scene[asset_cfg.name]
    penalty_val = torch.square(asset.data.root_lin_vel_b[:, 2])
    penalty = -torch.exp(-k * penalty_val) + 1  # shift so 0 = perfect
    return torch.clip(-torch.exp(-k * penalty_val) + 1, -1.0, 0.0)

def ang_vel_xy_penalty(env, asset_cfg=None, k: float = 5.0) -> torch.Tensor:
    """Penalty: xy-axis angular velocity — returns [-1, 0]."""
    asset = env.scene[asset_cfg.name]
    penalty_val = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    return torch.clip(-torch.exp(-k * penalty_val) + 1, -1.0, 0.0)
    
def joint_acc_penalty(env, asset_cfg=None, k: float = 5.0) -> torch.Tensor:
    """Penalty: joint accelerations — returns [-1, 0]."""
    asset = env.scene[asset_cfg.name]
    penalty_val = torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)
    return torch.clip(-torch.exp(-k * penalty_val) + 1, -1.0, 0.0)