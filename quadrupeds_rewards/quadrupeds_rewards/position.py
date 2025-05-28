"""
position.py

Reward functions for tracking position commands.

Available classes:
- RaibertHeuristic: Computes the Raibert heuristic 
    reward for a quadruped robot.
"""
from collections.abc import Sequence

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import RayCaster

from quadrupeds_assets.utils import (
    batch_quaternion_inverse,
    batch_quaternion_rotate,
    convert_batch_foot_to_local_frame,
    quat_from_angle_axis,
    quat_mul
)

class RaibertHeuristic(ManagerTermBase):
    """
    Computes the Raibert heuristic reward for a quadruped robot.

    ATTENTION: to use this reward function, the environment must
    contain the quadrupeds_assets.gaits.WTWCommandFootStates inside
    the command_manager. This is because the Raibert heuristic
    requires the phase of each foot to be computed.
    """

    def __init__(
        self, 
        cfg: RewardTermCfg, 
        env: ManagerBasedRLEnv
    ):
        super().__init__(cfg, env)

        self.env = env

        self.gait_stance_distances_cmd = cfg.params.get("gait_stance_distances_cmd")
        robot_cfg = cfg.params.get("robot_cfg")
        self.robot: RigidObject = env.scene[robot_cfg.name]

        self.neutral_y_pos_l = torch.zeros(
            (env.num_envs, 4),
            device=env.device
        )
        self.neutral_x_pos_l = torch.zeros(
            (env.num_envs, 4),
            device=env.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the manager term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        stance_distances_cmd = self.env.command_manager.get_command(
            self.gait_stance_distances_cmd
        )
        self.neutral_y_pos_l[env_ids, :] = stance_distances_cmd[env_ids, 0].unsqueeze(1) / 2.
        self.neutral_y_pos_l[env_ids, 1:4:2] *= -1.0
        self.neutral_x_pos_l[env_ids, :] = stance_distances_cmd[env_ids, 1].unsqueeze(1) / 2.
        self.neutral_x_pos_l[env_ids, 2:4] *= -1.0
    
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        gait_stance_distances_cmd: str,
        gait_step_freq_cmd: str,
        robot_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        cmd = env.command_manager.get_command(command_name)
        step_freq_cmd = env.command_manager.get_command(gait_step_freq_cmd)
        stance_distances_cmd = env.command_manager.get_command(
            gait_stance_distances_cmd
        )

        x_base_vel_cmd = cmd[:, 0].unsqueeze(1)
        z_base_ang_vel_cmd = cmd[:, 2].unsqueeze(1)

        # $$ \left|1 - 2t_{foot}\right| - 0.5$$
        # Converts phase from [-1, 1] to [-0.5, 0.5]
        phases = torch.abs(1.0 - (env.foot_indices * 2.0)) * 1.0 - 0.5

        # $$ y_{cmd} = \dot{\theta} d_{length}/2$$
        y_base_vel_cmd = z_base_ang_vel_cmd * \
            (stance_distances_cmd[:, 1].unsqueeze(1))/2


        """ $$ \begin{cases} \Delta x = t_{foot} \frac{\dot{x}}{2f} \\ 
        \Delta y = t_{foot} \frac{\dot{\theta}}{2f}\end{cases}$$
        
        Translation in Y is ignored.
        """
        xs_foot_offset_cmd = phases*x_base_vel_cmd*(0.5/step_freq_cmd)
        ys_foot_offset_cmd = phases*y_base_vel_cmd*(0.5/step_freq_cmd)
        ys_foot_offset_cmd[:, 2:4] *= -1.0
        x_foot_pos_cmd = self.neutral_x_pos_l + xs_foot_offset_cmd
        y_foot_pos_cmd = self.neutral_y_pos_l + ys_foot_offset_cmd
        foot_pos_cmd = torch.stack(
            [
                x_foot_pos_cmd,
                y_foot_pos_cmd,
            ],
            dim=2,
        )

        # robot_cfg.body_ids have the ids of the feet inside the 
        # robot's body list because they were explicitly selected.
        foot_pos_w = self.robot.data.body_pos_w[:, robot_cfg.body_ids]
        base_pos_w = self.robot.data.root_pos_w[:]
        base_quat_w = self.robot.data.root_quat_w[:]
        foot_pos_l = convert_batch_foot_to_local_frame(
            foot_pos_w,
            base_pos_w,
            base_quat_w,
        )
        error = torch.abs(foot_pos_cmd - foot_pos_l[:, :, :2])

        # The reward is the sum of the squared error for each foot
        # and for each environment.
        return torch.sum(
            torch.square(error), dim=(1, 2)
        )

def track_joint_positions_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Tracks the amplitude of the joint positions in relation
    to the default position of the joints.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(
        torch.square(joint_pos - joint_default_pos),
        dim=1
    )

def track_footswing_height(
    env: ManagerBasedRLEnv,
    foot_height_cmd: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Tracks the footswing height of the feet and penalizes the
    deviation from the desired foot clearance.

    ATTENTION: to use this reward function, the environment must
    contain the quadrupeds_assets.gaits.WTWCommandFootStates inside
    the command_manager. This is because this function requires
    the desired_contact_states and foot_indices variables.
    """
    desired_foot_height = env.command_manager.get_command(foot_height_cmd)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # $$\text{phases} = 1 - \left| 1 - \min\left( \max\left(2 \cdot t_{foot} - 1,\ 0\right),\ 1 \right) \cdot 2 \right|$$
    # Converts the foot indice to indicate only the swing phase of the foot.
    # If t_foot is in [0, 0.5], then phases = 0
    # If t_foot is in [0.5, 0.75], then phase increases linearly from 0 to 1
    # If t_foot is in [0.75, 1], then phase decreases linearly from 1 to 0
    phases = 1 - torch.abs(
        1.0 - torch.clip(
            (env.foot_indices * 2.0) - 1.0, 0.0, 1.0
        ) * 2.0
    )
    target_height = desired_foot_height * phases + 0.02  # offset for foot radius 2cm
    return torch.sum(
        torch.square(
            (
                target_height - \
                asset.data.body_pos_w[:, asset_cfg.body_ids, 2])
        ) * (1 - env.desired_contact_states),
        dim=1
    )

def track_base_orientation(
    env: ManagerBasedRLEnv,
    orientation_cmd: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Tracks the base orientation of the robot and penalizes the
    deviation from the desired orientation.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    desired_orientation = env.command_manager.get_command(orientation_cmd)

    quat_pitch = quat_from_angle_axis(
        -desired_orientation[:, 1],
        torch.tensor([1., 0., 0.], device=env.device).expand(env.num_envs, -1)
    )
    quat_roll = quat_from_angle_axis(
        -desired_orientation[:, 2],
        torch.tensor([0., 1., 0.], device=env.device).expand(env.num_envs, -1)
    )
    desired_base_quat = quat_mul(quat_pitch, quat_roll)
    desired_base_quat_inv = batch_quaternion_inverse(desired_base_quat)
    desired_projected_gravity = batch_quaternion_rotate(
        desired_base_quat_inv,
        asset.data.projected_gravity_b.unsqueeze(1)
    ).squeeze(1)
    return torch.sum(
        torch.square(
            asset.data.projected_gravity_b[:, :2] - desired_projected_gravity[:, :2]
        ),
        dim=1
    )

def track_base_height_l2_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Tracks the base height of the robot and penalizes the
    deviation from the desired height. The desired height
    is determined by a command in the command manager.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target_height = env.command_manager.get_command(command_name)[:, 0]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height

    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)