"""
position.py

Reward functions for tracking position commands.

Available functions:
- track_joint_positions_l2: Tracks the amplitude of the joint positions in relation to the default position of the joints.
- track_footswing_height: Tracks the footswing height of the feet and penalizes the deviation from the desired foot clearance.
- track_base_orientation: Tracks the base orientation of the robot and penalizes the deviation from the desired orientation.
- track_base_height_l2_cmd: Tracks the base height of the robot and penalizes the deviation from the desired height (indicated by command).

Available classes:
- RaibertHeuristic: Computes the Raibert heuristic reward for a quadruped robot.
"""
from collections.abc import Sequence
from typing import List

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import RayCaster

from quadrupeds_mdp.utils import (
    batch_quaternion_inverse,
    batch_quaternion_rotate,
    convert_batch_foot_to_local_frame,
    quat_from_angle_axis,
    quat_mul,
    quat_to_yaw
)

class RaibertHeuristic(ManagerTermBase):
    """
    Computes the Raibert heuristic reward for a quadruped robot.

    ATTENTION: to use this reward function, the environment must
    contain the quadrupeds_mdp.assets.gaits.WTWCommandFootStates inside
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

class FootDeviationPenalty(ManagerTermBase):
    """Penalizes deviation of feet from default local positions. [-1, 0]"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env = env
        self.robot_cfg = cfg.params.get("robot_cfg")
        self.robot = env.scene[self.robot_cfg.name]
        self.env.default_foot_pos_l = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = range(self.env.num_envs)

        foot_pos_w = self.robot.data.body_pos_w[:, self.robot_cfg.body_ids]
        base_pos_w = self.robot.data.root_pos_w[:]
        base_quat_w = self.robot.data.root_quat_w[:]
        foot_pos_l = convert_batch_foot_to_local_frame(
            foot_pos_w, base_pos_w, base_quat_w
        )

        if self.env.default_foot_pos_l is None:
            self.env.default_foot_pos_l = torch.zeros_like(foot_pos_l)
        self.env.default_foot_pos_l[env_ids] = foot_pos_l[env_ids]

    def __call__(self, env, robot_cfg, k: float = 5.0) -> torch.Tensor:
        foot_pos_w = self.robot.data.body_pos_w[:, self.robot_cfg.body_ids]
        base_pos_w = self.robot.data.root_pos_w[:]
        base_quat_w = self.robot.data.root_quat_w[:]
        foot_pos_l = convert_batch_foot_to_local_frame(
            foot_pos_w, base_pos_w, base_quat_w
        )

        deviation = torch.sum(torch.square(foot_pos_l - self.env.default_foot_pos_l), dim=(1, 2))
        penalty = -torch.exp(-k * deviation)  # range [-1, 0]
        return penalty



def track_joint_positions_bonus(env, asset_cfg=None, k: float = 5.0) -> torch.Tensor:
    """Bonus: track default joint positions — returns [0, 1]."""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    penalty_val = torch.sum(torch.square(joint_pos - joint_default_pos), dim=1)
    bonus = torch.exp(-k * penalty_val)
    return torch.clip(bonus, 0.0, 1.0)

def penalize_hip_movement(env, asset_cfg=None, k: float = 5.0) -> torch.Tensor:
    """Penalty: hip movement away from default — returns [-1, 0]."""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    penalty_val = torch.sum(torch.square(joint_pos - joint_default_pos), dim=1)
    return torch.clip(-torch.exp(-k * penalty_val) + 1, -1.0, 0.0)
    
def track_footswing_height(
    env,
    foot_height_cmd: str,
    command_name: str,
    asset_cfg=None,
    k: float = 5.0,
) -> torch.Tensor:
    """Bonus: Track foot swing height — returns [0, 1]."""
    desired_robot_velocity = env.command_manager.get_command(command_name)
    desired_foot_height = env.command_manager.get_command(foot_height_cmd)
    asset = env.scene[asset_cfg.name]
    measured_phase = env.measured_foot_phase

    phases = 1 - torch.abs(1.0 - torch.clip((measured_phase * 2.0) - 1.0, 0.0, 1.0) * 2.0)
    target_height = desired_foot_height * phases + 0.02
    deviation = torch.sum(
        torch.square(target_height - asset.data.body_pos_w[:, asset_cfg.body_ids, 2])
        * (1 - env.desired_contact_states)
        * torch.linalg.norm(desired_robot_velocity, dim=1, keepdim=True),
        dim=1,
    )
    bonus = torch.exp(-k * deviation)
    return torch.clip(bonus, 0.0, 1.0)

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

def track_base_height_l2_cmd_no_sensor(env, command_name, asset_cfg=SceneEntityCfg("robot"), k: float = 5.0):
    asset = env.scene[asset_cfg.name]
    target_height = env.command_manager.get_command(command_name)[:, 0]
    current_height = asset.data.root_pos_w[:, 2]
    height_error = torch.square(current_height - target_height)
    bonus = torch.exp(-k * height_error)  # [0, 1]
    return bonus

def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)

def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()

def heading_command_error_cosine(env, command_name: str) -> torch.Tensor:
    """
    Reward correct heading tracking.
    Heading error is in radians; reward is highest at 0 error and decreases with absolute error.
    """
    command = env.command_manager.get_command(command_name)
    heading_error = command[:, 3].abs()
    # Use cosine so 0 rad error = 1.0 reward, pi/2 = 0 reward, pi = -1 reward
    return torch.cos(heading_error)

def position_command_error_threshold(
    env: ManagerBasedRLEnv,
    std: float,
    position_threshold: float,
    heading_threshold: float,
    command_name: str
) -> torch.Tensor:
    """
    Return 1.0 if both position and heading errors are below given thresholds, else 0.0.
    """
    position_reward = position_command_error_tanh(env, std=std, command_name=command_name)
    heading_error = heading_command_error_abs(env, command_name=command_name)

    reached = (position_reward >= position_threshold) & (heading_error <= heading_threshold)
    return reached.float()

def viewpoint_action_rate_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Computes a weighted L2 norm of the action rate of the robot viewpoints (pos + orientation).
    Explicitly penalizes abrupt signal changes ("jumps") in x (0), y (1), and heading (3).
    Continuous changes in the same direction are penalized less than sudden flips/jumps.
    """
    processed = env.action_manager._terms["viewpoint_action"].processed_actions
    last = env.action_manager._terms["viewpoint_action"].last_actions

    diffs = processed - last

    # Stronger weights for x (0), y (1), heading (3)
    weights = torch.tensor([10.0, 10.0, 1.0, 8.0], device=diffs.device)

    # Base penalty (L2 norm with weights)
    base_penalty = torch.sum(torch.square(weights * diffs), dim=1)

    # Detect jumps: sign flip or large sudden change compared to magnitude
    jump_mask = (
        torch.sign(processed) != torch.sign(last)
    ) & (torch.abs(diffs) > 0.2)  # 0.2 = threshold for "big" change

    # Apply extra jump penalty only to indices of interest (0,1,3)
    jump_indices = torch.tensor([0, 1, 3], device=diffs.device)
    jump_penalty = torch.sum(
        torch.square(diffs[:, jump_indices]) * 2. * jump_mask[:, jump_indices],
        dim=1,
    )

    return base_penalty + jump_penalty

@torch.no_grad()
def viewpoint_towards_objects(
    env: ManagerBasedRLEnv,
    objects_of_interest: List[str],
    max_angle: float = torch.pi / 12,  # 15 degrees
    distance_weight: float = 0.25      # 1/(1 + k*d) proximity scaling; set 0 to disable
) -> torch.Tensor:
    """
    Reward the robot for facing toward any object in `objects_of_interest`.
    """
    device = env.device
    num_envs = env.num_envs
    if len(objects_of_interest) == 0:
        return torch.zeros(num_envs, device=device)

    # Robot pose in local env frame (XY)
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    robot_pos = torch.nan_to_num(robot_pos, nan=0.0)

    robot_rot = env.scene["robot"].data.root_link_state_w[:, 3:7]
    robot_rot = torch.nan_to_num(robot_rot, nan=0.0)

    # Convert quaternion to yaw
    yaw = quat_to_yaw(robot_rot)
    yaw = torch.nan_to_num(yaw, nan=0.0)

    forward_vec = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)  # [N, 2]
    forward_vec = torch.nan_to_num(forward_vec, nan=0.0)

    # Collect object positions
    obj_positions = []
    for name in objects_of_interest:
        data = env.scene[name].data
        if hasattr(data, "root_pos_w"):
            pos = data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
        else:
            pos = data.default_root_state[:, :2]
        obj_positions.append(torch.nan_to_num(pos, nan=0.0))

    # [M, N, 2]
    obj_pos = torch.stack(obj_positions, dim=0)

    # Vectors from robot to each object
    vec = obj_pos - robot_pos.unsqueeze(0)
    vec = torch.nan_to_num(vec, nan=0.0)
    dist = torch.norm(vec, dim=-1, keepdim=True)
    dist = torch.nan_to_num(dist, nan=1e-6)

    dir_unit = vec / (dist + 1e-6)
    dir_unit = torch.nan_to_num(dir_unit, nan=0.0)

    # Cosine similarity
    cos_sim = (dir_unit * forward_vec.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
    cos_sim = torch.nan_to_num(cos_sim, nan=0.0)

    cos_thr = torch.cos(torch.as_tensor(max_angle, device=device, dtype=cos_sim.dtype))
    angle_score = torch.relu(cos_sim - cos_thr) / (1.0 - cos_thr + 1e-6)
    angle_score = torch.nan_to_num(angle_score, nan=0.0)

    # Proximity scaling
    if distance_weight > 0.0:
        dist_factor = 1.0 / (1.0 + distance_weight * dist.squeeze(-1))
        dist_factor = torch.nan_to_num(dist_factor, nan=0.0)
        score = angle_score * dist_factor
    else:
        score = angle_score

    # Best object score
    reward = score.max(dim=0).values
    reward = torch.nan_to_num(reward, nan=1.0, posinf=1.0, neginf=0.0)

    return reward

def viewpoint_action_l2(
    env: ManagerBasedRLEnv
):
    return torch.sum(
        torch.square(
            env.action_manager._terms["viewpoint_action"].processed_actions
        ), dim=1
    )

def viewpoint_action_flip_penalty(env: ManagerBasedRLEnv, threshold: float = 0.1):
    """
    Penalize viewpoint action sign flips (positive↔negative) between consecutive steps,
    except when both actions are near zero.
    """
    actions = env.action_manager._terms["viewpoint_action"].processed_actions  # (batch, action_dim)
    
    # Compute difference across time: compare t with t-1
    prev_actions = torch.roll(actions, shifts=1, dims=0)
    
    # Detect sign flips: sign product < 0 means sign flipped
    sign_flip_mask = (actions * prev_actions) < 0
    
    # Allow small near-zero changes (e.g. -0.01 ↔ 0.02)
    near_zero_mask = (torch.abs(actions) < threshold) & (torch.abs(prev_actions) < threshold)
    
    # Penalize flips only when not near zero
    effective_flips = sign_flip_mask & (~near_zero_mask)
    
    # Optionally k penalty by magnitude of the change
    penalty = torch.sum(effective_flips.float(), dim=1)
    
    return penalty

class DiagonalFootBase(ManagerTermBase):
    """Base class providing shared circular helpers and setup."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env
        self.robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg")
        self.robot: RigidObject = env.scene[self.robot_cfg.name]
        # Expected order: [FL, FR, RL, RR]
        self.FL, self.FR, self.RL, self.RR = 0, 1, 2, 3

    # -------------------------------
    # Circular math helpers
    # -------------------------------
    def _circular_diff(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Minimal phase difference on unit circle (range [0, 0.5])."""
        raw = torch.abs(a - b)
        return torch.minimum(raw, 1.0 - raw)

    def _circular_mean(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Mean of two phases on the circle."""
        angles = 2 * torch.pi * torch.stack([a, b], dim=-1)
        sin_mean = torch.mean(torch.sin(angles), dim=-1)
        cos_mean = torch.mean(torch.cos(angles), dim=-1)
        mean_angle = torch.atan2(sin_mean, cos_mean) / (2 * torch.pi)
        return torch.remainder(mean_angle, 1.0)

    def _circular_variance(self, phases: torch.Tensor) -> torch.Tensor:
        """How spread out foot phases are on the unit circle."""
        angles = 2 * torch.pi * phases
        sin_mean = torch.mean(torch.sin(angles), dim=-1)
        cos_mean = torch.mean(torch.cos(angles), dim=-1)
        R = torch.sqrt(sin_mean ** 2 + cos_mean ** 2 + 1e-8)
        return 1 - R

class DiagonalFootTrotPositive(DiagonalFootBase):
    """Positive bonus for correct trot gait. [0, 1]"""

    def __call__(self, env, robot_cfg, k: float = 2.0) -> torch.Tensor:
        phases = torch.nan_to_num(env.measured_foot_phase[:, :4], nan=0.0)
        diff_FL_RR = self._circular_diff(phases[:, self.FL], phases[:, self.RR])
        diff_FR_RL = self._circular_diff(phases[:, self.FR], phases[:, self.RL])
        diagonal_sync = 1.0 - (diff_FL_RR + diff_FR_RL)

        mean_FLRR = self._circular_mean(phases[:, self.FL], phases[:, self.RR])
        mean_FRRL = self._circular_mean(phases[:, self.FR], phases[:, self.RL])
        inter_pair_diff = self._circular_diff(mean_FLRR, mean_FRRL)
        trot_offset = 1.0 - torch.abs(inter_pair_diff - 0.5)

        variance = self._circular_variance(phases)
        base_score = 2.0 * diagonal_sync + 4.0 * trot_offset + variance
        bonus = torch.sigmoid(k * base_score)  # [0, 1]
        return bonus

    def _circular_diff(self, a, b):
        return torch.remainder(a - b + 0.5, 1.0) - 0.5

    def _circular_mean(self, a, b):
        return (a + b) / 2.0

    def _circular_variance(self, x):
        sin_term = torch.sin(2 * torch.pi * x)
        cos_term = torch.cos(2 * torch.pi * x)
        return 1 - torch.sqrt(torch.square(sin_term.mean(dim=1)) + torch.square(cos_term.mean(dim=1)))


# ------------------------------------------------
# Diagonal Foot Trot Negative (penalty ∈ [-1, 0])
# ------------------------------------------------
class DiagonalFootTrotNegative(DiagonalFootBase):
    """Penalty for undesired gait. [-1, 0]"""

    def __call__(self, env, robot_cfg, k: float = 5.0) -> torch.Tensor:
        phases = torch.nan_to_num(env.measured_foot_phase[:, :4], nan=0.0)
        diff_FL_FR = self._circular_diff(phases[:, self.FL], phases[:, self.FR])
        diff_RL_RR = self._circular_diff(phases[:, self.RL], phases[:, self.RR])
        pacing_error = torch.abs(diff_FL_FR - 0.5) + torch.abs(diff_RL_RR - 0.5)

        diff_FL_RL = self._circular_diff(phases[:, self.FL], phases[:, self.RL])
        diff_FR_RR = self._circular_diff(phases[:, self.FR], phases[:, self.RR])
        bounding_error = torch.abs(diff_FL_RL - 0.5) + torch.abs(diff_FR_RR - 0.5)

        variance = self._circular_variance(phases)
        pronking_penalty = 1.0 - variance

        combined_error = pacing_error + bounding_error + pronking_penalty
        penalty = -torch.exp(-k * combined_error)  # [-1, 0]
        return penalty

    def _circular_diff(self, a, b):
        return torch.remainder(a - b + 0.5, 1.0) - 0.5

    def _circular_variance(self, x):
        sin_term = torch.sin(2 * torch.pi * x)
        cos_term = torch.cos(2 * torch.pi * x)
        return 1 - torch.sqrt(torch.square(sin_term.mean(dim=1)) + torch.square(cos_term.mean(dim=1)))


# ------------------------------------------------
# Diagonal Joint Symmetry (bonus ∈ [0, 1])
# ------------------------------------------------
class DiagonalJointSymmetryReward(ManagerTermBase):
    """Bonus for symmetry between diagonal legs. [0, 1]"""

    def __call__(self, env, robot_cfg, k: float = 5.0) -> torch.Tensor:
        robot = env.scene[robot_cfg.name]
        joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
        n = joint_pos.shape[1] // 4

        FL, FR, RL, RR = slice(0, n), slice(n, 2*n), slice(2*n, 3*n), slice(3*n, 4*n)
        diff_FL_RR = torch.sum(torch.square(joint_pos[:, FL] - joint_pos[:, RR]), dim=1)
        diff_FR_RL = torch.sum(torch.square(joint_pos[:, FR] - joint_pos[:, RL]), dim=1)
        total_diff = (diff_FL_RR + diff_FR_RL) / n

        bonus = torch.exp(-k * total_diff)  # [0, 1]
        return bonus


# ------------------------------------------------
# Diagonal Motion Balance (bonus ∈ [0, 1])
# ------------------------------------------------
class DiagonalMotionBalanceReward(ManagerTermBase):
    """Encourages balanced diagonal motion. [0, 1]"""

    def __call__(self, env, robot_cfg, k: float = 2.0) -> torch.Tensor:
        robot = env.scene[robot_cfg.name]
        joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
        n = joint_pos.shape[1] // 4
        FL, FR, RL, RR = slice(0, n), slice(n, 2*n), slice(2*n, 3*n), slice(3*n, 4*n)

        energy_FL = torch.sum(torch.square(joint_pos[:, FL]), dim=1)
        energy_FR = torch.sum(torch.square(joint_pos[:, FR]), dim=1)
        energy_RL = torch.sum(torch.square(joint_pos[:, RL]), dim=1)
        energy_RR = torch.sum(torch.square(joint_pos[:, RR]), dim=1)

        balance_error = torch.abs((energy_FL + energy_RR) - (energy_FR + energy_RL))

        if hasattr(env, "measured_foot_phase"):
            phases = torch.nan_to_num(env.measured_foot_phase[:, :4], nan=0.0)
            phase_diff = torch.abs((phases[:, 0] + phases[:, 3]) / 2 - (phases[:, 1] + phases[:, 2]) / 2)
        else:
            phase_diff = 0.0

        combined_error = balance_error + phase_diff
        bonus = torch.exp(-k * combined_error)  # [0, 1]
        return bonus


# ------------------------------------------------
# Stillness Penalty (penalty ∈ [-1, 0])
# ------------------------------------------------
class NoMotionWhenStationary(ManagerTermBase):
    """Penalizes motion when commanded velocity ≈ 0. [-1, 0]"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env = env
        self.robot_cfg = cfg.params.get("robot_cfg")
        self.robot = env.scene[self.robot_cfg.name]

    def __call__(self, env, robot_cfg, command_name, torque_weight, velocity_threshold, k: float = 5.0):
        cmd = env.command_manager.get_command(command_name)
        linear_cmd, yaw_cmd = cmd[:, :2], cmd[:, 2:3]
        cmd_mag = torch.linalg.norm(torch.cat([linear_cmd, yaw_cmd], dim=1), dim=1)
        still_factor = torch.exp(-10.0 * torch.square(cmd_mag / velocity_threshold))

        joint_vel = self.robot.data.joint_vel[:, robot_cfg.joint_ids]
        joint_torque = self.robot.data.applied_torque[:, robot_cfg.joint_ids]

        vel_pen = torch.sum(torch.square(joint_vel), dim=1)
        torque_pen = torch.sum(torch.square(joint_torque), dim=1)
        total_penalty = still_factor * (vel_pen + torque_weight * torque_pen)

        penalty = -torch.exp(-k * total_penalty)  # [-1, 0]
        return penalty


# ------------------------------------------------
# All Feet Off Ground Penalty (penalty ∈ [-1, 0])
# ------------------------------------------------
class AllFeetOffGroundPenalty(ManagerTermBase):
    """Penalizes all-feet-off-ground events. [-1, 0]"""

    def __call__(self, env, sensor_cfg, contact_force_threshold=1.0, k: float = 10.0):
        contact_sensor = env.scene[sensor_cfg.name]
        recent_forces = contact_sensor.data.net_forces_w_history[:, -1, :, :]
        magnitudes = torch.linalg.norm(recent_forces, dim=-1)
        in_contact = (magnitudes > contact_force_threshold).float()
        num_contact = torch.sum(in_contact, dim=1)
        all_off = (num_contact == 0).float()

        penalty = -torch.exp(-k * all_off)  # [-1, 0]
        return penalty

def joint_pos_limits_penalty(env, asset_cfg=None, k: float = 5.0) -> torch.Tensor:
    """Penalty: joint positions beyond soft limits — returns [-1, 0]."""
    asset = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    penalty_val = torch.sum(out_of_limits, dim=1)
    return torch.clip(-torch.exp(-k * penalty_val) + 1, -1.0, 0.0)

def flat_orientation_bonus(env, asset_cfg=None, k: float = 5.0) -> torch.Tensor:
    """Bonus: keep base orientation flat — returns [0, 1]."""
    asset = env.scene[asset_cfg.name]
    penalty_val = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    bonus = torch.exp(-k * penalty_val)
    return torch.clip(bonus, 0.0, 1.0)