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
    """
    Penalizes the agent if its feet deviate too far from their
    default (neutral) positions relative to the robot's base.

    The default foot positions are recorded during reset().
    """

    def __init__(
        self,
        cfg,
        env: ManagerBasedRLEnv
    ):
        super().__init__(cfg, env)
        self.env = env
        self.robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg")
        self.robot: RigidObject = env.scene[self.robot_cfg.name]

        # Store default local foot positions (initialized at reset)
        self.env.default_foot_pos_l = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Capture default local foot positions during environment reset."""
        if env_ids is None:
            env_ids = range(self.env.num_envs)

        # Get current positions of feet and base (default stance)
        foot_pos_w = self.robot.data.body_pos_w[:, self.robot_cfg.body_ids]
        base_pos_w = self.robot.data.root_pos_w[:]
        base_quat_w = self.robot.data.root_quat_w[:]

        # Convert to local (base) frame
        foot_pos_l = convert_batch_foot_to_local_frame(
            foot_pos_w, base_pos_w, base_quat_w
        )

        # Cache as the default foot positions
        if self.env.default_foot_pos_l is None:
            self.env.default_foot_pos_l = torch.zeros_like(foot_pos_l)
        self.env.default_foot_pos_l[env_ids] = foot_pos_l[env_ids]

    def __call__(self, env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
        """Compute penalty for deviation from default foot positions."""
        # Get current foot positions
        foot_pos_w = self.robot.data.body_pos_w[:, self.robot_cfg.body_ids]
        base_pos_w = self.robot.data.root_pos_w[:]
        base_quat_w = self.robot.data.root_quat_w[:]

        # Convert to base frame
        foot_pos_l = convert_batch_foot_to_local_frame(
            foot_pos_w, base_pos_w, base_quat_w
        )

        # Compute deviation
        foot_deviation = torch.sum(
            torch.square(foot_pos_l - self.env.default_foot_pos_l),
            dim=(1, 2),
        )

        # print("foot_pos_l:", foot_pos_l[0])
        # print("default_foot_pos_l:", self.env.default_foot_pos_l[0])
        # print("foot_deviation:", foot_deviation[0])

        # Return negative reward (penalty)
        return foot_deviation


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

def penalize_hip_movement(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalizes excessive motion in the hip joints.
    This helps reduce unnecessary lateral leg spreading or swinging.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # L2 distance from default hip positions
    hip_deviation = torch.sum(torch.square(joint_pos - joint_default_pos), dim=1)
    
    return hip_deviation

def track_footswing_height(
    env: ManagerBasedRLEnv,
    foot_height_cmd: str,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Tracks the foot swing height based on the robot's actual gait phase.

    This function now uses the measured gait phase (from contact events)
    instead of the desired commanded phase, for more realistic tracking.

    Requirements:
        - The environment must include the WTWCommandFootStates in the command manager,
          since it provides the measured_foot_phase and desired_contact_states variables.
    """

    desired_robot_velocity = env.command_manager.get_command(command_name)
    desired_foot_height = env.command_manager.get_command(foot_height_cmd)
    asset: RigidObject = env.scene[asset_cfg.name]

    # --- Use measured gait phase instead of desired foot_indices ---
    measured_phase = env.measured_foot_phase

    # Convert measured phase to swing amplitude scaling:
    #   0 → stance
    #   0.5 → mid-swing (highest point)
    #   1 → stance again
    phases = 1 - torch.abs(
        1.0 - torch.clip(
            (measured_phase * 2.0) - 1.0, 0.0, 1.0
        ) * 2.0
    )

    # Desired target height during swing (with small offset for foot radius)
    target_height = desired_foot_height * phases + 0.02

    # Penalize deviation from target height, weighted by velocity magnitude
    return torch.sum(
        torch.square(
            (target_height - asset.data.body_pos_w[:, asset_cfg.body_ids, 2])
        )
        * (1 - env.desired_contact_states)
        * torch.linalg.norm(desired_robot_velocity, dim=1, keepdim=True),
        dim=1,
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

def track_base_height_l2_cmd_no_sensor(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Tracks the robot base height and penalizes deviation from
    the desired (commanded) height, without using any sensors.

    Args:
        env: The RL environment.
        command_name: Name of the command providing the target base height.
        asset_cfg: Configuration for the robot asset (default: "robot").

    Returns:
        torch.Tensor: The squared L2 penalty for each environment.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Commanded (desired) base height
    target_height = env.command_manager.get_command(command_name)[:, 0]

    # Current base height from simulation data
    current_height = asset.data.root_pos_w[:, 2]

    # Compute L2 penalty (squared deviation)
    height_error = torch.square(current_height - target_height)

    return height_error

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

class DiagonalFootBase(ManagerTermBase):
    """Base class providing shared circular helpers and setup."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env
        self.robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg")
        self.robot: RigidObject = env.scene[self.robot_cfg.name]
        self.scale = cfg.params.get("scale", 10.0)
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
    """
    Positive, unbounded reward for correct trot gait patterns based on actual foot phases:
      • Diagonal feet synchronized (FL–RR and FR–RL)
      • Diagonal pairs 180° out of phase (true trot)
      • High global phase variance (spread out phases)
    """

    def __call__(self, env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, scale: float = None) -> torch.Tensor:
        scale = scale or self.scale

        # Use measured gait phase instead of desired
        phases = torch.nan_to_num(env.measured_foot_phase[:, :4], nan=0.0)

        # --- Diagonal synchronization (FL–RR, FR–RL) → smaller diff = higher reward
        diff_FL_RR = self._circular_diff(phases[:, self.FL], phases[:, self.RR])
        diff_FR_RL = self._circular_diff(phases[:, self.FR], phases[:, self.RL])
        diagonal_sync_reward = 1.0 - (diff_FL_RR + diff_FR_RL)  # higher when more in sync

        # --- Diagonal pairs 180° out of phase
        mean_FLRR = self._circular_mean(phases[:, self.FL], phases[:, self.RR])
        mean_FRRL = self._circular_mean(phases[:, self.FR], phases[:, self.RL])
        inter_pair_diff = self._circular_diff(mean_FLRR, mean_FRRL)
        trot_offset_reward = 1.0 - torch.abs(inter_pair_diff - 0.5)  # higher when near 0.5

        # --- Global variance (spread-out foot phases)
        global_variance = self._circular_variance(phases)
        variance_reward = global_variance  # more spread → higher reward

        # --- Combine components with positive weights
        reward = (
            2.0 * diagonal_sync_reward +
            4.0 * trot_offset_reward +
            1.0 * variance_reward
        )

        # Unbounded, purely positive
        return torch.clamp(reward, min=0.0)

class DiagonalFootTrotNegative(DiagonalFootBase):
    """
    Returns a negative penalty for undesired gaits based on actual measured phases:
      • Lateral synchronization (pacing)
      • Fore–hind synchronization (bounding)
      • Full synchronization (pronking)
    More negative → worse gait.
    """

    def __call__(self, env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, scale: float = None) -> torch.Tensor:
        scale = scale or self.scale

        # Use measured gait phase instead of desired
        phases = torch.nan_to_num(env.measured_foot_phase[:, :4], nan=0.0)

        # --- Lateral synchronization (FL–FR, RL–RR)
        diff_FL_FR = self._circular_diff(phases[:, self.FL], phases[:, self.FR])
        diff_RL_RR = self._circular_diff(phases[:, self.RL], phases[:, self.RR])
        pacing_error = torch.abs(diff_FL_FR - 0.5) + torch.abs(diff_RL_RR - 0.5)

        # --- Fore–hind synchronization (FL–RL, FR–RR)
        diff_FL_RL = self._circular_diff(phases[:, self.FL], phases[:, self.RL])
        diff_FR_RR = self._circular_diff(phases[:, self.FR], phases[:, self.RR])
        bounding_error = torch.abs(diff_FL_RL - 0.5) + torch.abs(diff_FR_RR - 0.5)

        # --- Global synchronization (pronking)
        global_variance = self._circular_variance(phases)
        pronking_penalty = 1.0 - global_variance  # higher when feet move together

        # --- Combine all errors
        combined_error = pacing_error + bounding_error + pronking_penalty

        return combined_error

class DiagonalJointSymmetryReward(DiagonalFootBase):
    """
    Rewards symmetric joint motion between diagonal legs (FL–RR and FR–RL).
    Encourages both diagonal pairs to move in phase and with similar amplitude.

    • Compares corresponding joint angles of each diagonal pair.
    • Higher value (less negative) → more symmetric.
    • Unbounded: scaling is handled externally.
    """

    def __call__(self, env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, scale: float = None) -> torch.Tensor:
        robot: RigidObject = env.scene[robot_cfg.name]

        # Get joint positions for all four legs
        joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]  # (num_envs, num_joints)

        # --- Identify joint indices per leg
        n_per_leg = joint_pos.shape[1] // 4
        FL_idx = slice(0 * n_per_leg, 1 * n_per_leg)
        FR_idx = slice(1 * n_per_leg, 2 * n_per_leg)
        RL_idx = slice(2 * n_per_leg, 3 * n_per_leg)
        RR_idx = slice(3 * n_per_leg, 4 * n_per_leg)
        
        # --- Compute diagonal differences (L2)
        diff_FL_RR = torch.sum(torch.square(joint_pos[:, FL_idx] - joint_pos[:, RR_idx]), dim=1)
        diff_FR_RL = torch.sum(torch.square(joint_pos[:, FR_idx] - joint_pos[:, RL_idx]), dim=1)

        # --- Combine and normalize by number of joints per leg
        total_diff = (diff_FL_RR + diff_FR_RL) / n_per_leg

        # --- Return as unbounded reward (smaller diff → higher reward)
        reward = -total_diff
        return reward

class DiagonalMotionBalanceReward(DiagonalFootBase):
    """
    Encourages both diagonal pairs (FL–RR and FR–RL) to move with similar amplitude and timing.
    Prevents one diagonal pair from dominating the motion while the other remains passive.

    Components:
      • Joint amplitude balance: both diagonals should move with similar joint-space energy.
      • Phase synchronization balance: both diagonals should have similar phase activity.

    Reward is higher (less negative) when both diagonals contribute equally to locomotion.
    """

    def __call__(self, env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
        robot: RigidObject = env.scene[robot_cfg.name]

        # --- Joint position data (shape: [num_envs, num_joints])
        joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]

        # --- Determine per-leg joint slices
        n_per_leg = joint_pos.shape[1] // 4
        FL_idx = slice(0 * n_per_leg, 1 * n_per_leg)
        FR_idx = slice(1 * n_per_leg, 2 * n_per_leg)
        RL_idx = slice(2 * n_per_leg, 3 * n_per_leg)
        RR_idx = slice(3 * n_per_leg, 4 * n_per_leg)

        # --- Compute per-leg motion magnitude (sum of squared joint angles)
        energy_FL = torch.sum(torch.square(joint_pos[:, FL_idx]), dim=1)
        energy_FR = torch.sum(torch.square(joint_pos[:, FR_idx]), dim=1)
        energy_RL = torch.sum(torch.square(joint_pos[:, RL_idx]), dim=1)
        energy_RR = torch.sum(torch.square(joint_pos[:, RR_idx]), dim=1)

        # --- Diagonal total motion energy
        energy_diag1 = energy_FL + energy_RR  # FL–RR
        energy_diag2 = energy_FR + energy_RL  # FR–RL

        # --- Energy balance (penalize imbalance between diagonals)
        energy_balance = -torch.abs(energy_diag1 - energy_diag2)

        # --- Phase-based balance (if measured phases are available)
        if hasattr(env, "measured_foot_phase"):
            phases = torch.nan_to_num(env.measured_foot_phase[:, :4], nan=0.0)
            mean_diag1 = self._circular_mean(phases[:, self.FL], phases[:, self.RR])
            mean_diag2 = self._circular_mean(phases[:, self.FR], phases[:, self.RL])
            phase_diff = self._circular_diff(mean_diag1, mean_diag2)
            phase_balance = -torch.abs(phase_diff)
        else:
            phase_balance = 0.0

        # --- Combine both components
        reward = energy_balance + phase_balance

        return reward

class NoMotionWhenStationary(ManagerTermBase):
    """
    Rewards the agent for keeping joints still when the commanded velocity is near zero.

    This term discourages unnecessary actuation when the robot is supposed to be idle.

    Components:
      • Uses the linear + angular velocity command (x, y, yaw) to detect "stationary" mode.
      • Penalizes nonzero joint velocities and torques when the command is near zero.
      • Scales penalty by how close the command is to zero — continuous, not binary.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env
        self.robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg")
        self.robot: RigidObject = env.scene[self.robot_cfg.name]

    def __call__(self, env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, command_name: str, torque_weight: float, velocity_threshold: float) -> torch.Tensor:
        # Retrieve commanded velocity [vx, vy, yaw_rate]
        cmd = env.command_manager.get_command(command_name)
        linear_cmd = cmd[:, :2]
        yaw_rate_cmd = cmd[:, 2:3]

        # Combined magnitude of commanded motion
        cmd_magnitude = torch.linalg.norm(torch.cat([linear_cmd, yaw_rate_cmd], dim=1), dim=1)

        # Compute "stationary factor" → 1 when near zero, 0 when moving
        still_factor = torch.exp(-10.0 * torch.square(cmd_magnitude / velocity_threshold))
        # Shape: [num_envs]

        # Joint velocities and torques
        joint_vel = self.robot.data.joint_vel[:, robot_cfg.joint_ids]
        joint_torque = self.robot.data.applied_torque[:, robot_cfg.joint_ids]

        # Energy-like penalties
        vel_penalty = torch.sum(torch.square(joint_vel), dim=1)
        torque_penalty = torch.sum(torch.square(joint_torque), dim=1)

        # Weighted combined penalty, scaled by stillness
        total_penalty = still_factor * (vel_penalty + torque_weight * torque_penalty)

        # Return as reward
        return total_penalty