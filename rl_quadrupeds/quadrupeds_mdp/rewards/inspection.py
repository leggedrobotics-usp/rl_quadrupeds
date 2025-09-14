from typing import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

from quadrupeds_mdp.utils import (
    quat_to_yaw
)

def get_inspection_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Computes when the agent decides to inspect the environment.
    Returns 1 if the inspection action is taken, otherwise returns 0
    for each environment.
    """
    actions = env.action_manager._terms["capture_feat_action"].processed_actions.squeeze(-1)
    # Sanitize to avoid propagating NaN or Inf
    return torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
    
def get_overall_inspection_coverage(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Computes the overall inspection coverage for each environment.
    It sums the coverage across all objects in the environment.
    """
    return torch.sum(env.coverage, dim=1)

class MaxCoverageGainReward(ManagerTermBase):
    """
    Rewards the agent only when the overall inspection coverage increases
    and exceeds the maximum coverage observed so far for each environment.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env
        self.max_coverage = torch.zeros(env.num_envs, device=env.device)
        self.max_coverage = torch.nan_to_num(self.max_coverage, nan=0.0)
        self.prev_coverage = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is not None:
            self.max_coverage[env_ids] = 0.0
            self.prev_coverage[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        current_coverage = torch.nan_to_num(torch.sum(env.coverage, dim=1), nan=0.0)

        diff = (current_coverage - self.prev_coverage).clamp_min(0.0)
        diff = torch.nan_to_num(diff, nan=0.0)

        reward_mask = current_coverage > self.max_coverage
        reward = diff * reward_mask.float()
        reward = torch.nan_to_num(reward, nan=0.0)

        self.max_coverage = torch.maximum(self.max_coverage, current_coverage)
        self.max_coverage = torch.nan_to_num(self.max_coverage, nan=0.0)

        self.prev_coverage = current_coverage.clone()

        return reward

def get_if_inspection_done(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Returns a float tensor indicating whether the inspection is done 
    for each environment (1 if done, 0 otherwise).
    
    It checks the environment's _inspection_done
    attribute, which is updated based on the coverage by a Termination
    Term.
    """
    return env._inspection_done.float() if hasattr(env, "_inspection_done") \
        else torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

def get_robot_stuck_if_inspection_not_done(
    env: ManagerBasedRLEnv,
    velocity_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalizes the robot for being stuck (not moving) when inspection is not yet done.
    Returns a positive value (penalty weight will be applied externally).

    Args:
        env: The RL environment.
        velocity_threshold: Minimum velocity magnitude to be considered as moving.
        asset_cfg: Configuration for the robot entity.

    Returns:
        torch.Tensor: Positive values indicating penalty when stuck and inspection not done.
    """
    # Get inspection completion status (1 if done, 0 otherwise)
    inspection_done = get_if_inspection_done(env)

    # Get robot linear velocity magnitude in xy-plane
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_xy = asset.data.root_lin_vel_b[:, :2]
    speed_xy = torch.norm(lin_vel_xy, dim=1)

    # Robot is considered stuck if speed is below threshold
    is_stuck = (speed_xy < velocity_threshold).float()

    # Penalize only when inspection is not done
    penalty = is_stuck * (1.0 - inspection_done)

    return penalty

class MilestoneCoverageReward(ManagerTermBase):
    """
    Rewards the agent once per milestone when the overall inspection
    coverage surpasses predefined thresholds.

    Example milestones: 30%, 50%, 60%, 70%, 80%, 90%
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env

        self.milestones = cfg.params.get(
            "milestones", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        self.reward_per_milestone = cfg.params.get(
            "reward_per_milestone", 1.0
        )

        self.prev_coverage = torch.zeros(env.num_envs, device=env.device)
        self.milestone_tensor = torch.tensor(
            self.milestones, device=env.device
        ).view(1, -1)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the previous coverage for given environments."""
        if env_ids is None:
            self.prev_coverage[:] = 0.0
        else:
            self.prev_coverage[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Computes the milestone-based reward for the current step."""
        # Sanitize coverage to avoid NaNs
        current_coverage = torch.sum(
            torch.nan_to_num(env.coverage, nan=0.0), dim=1
        )
        prev_cov = torch.nan_to_num(self.prev_coverage, nan=0.0)

        # Compare current vs previous to detect milestone crossings
        prev_flags = prev_cov.unsqueeze(1) >= self.milestone_tensor
        current_flags = current_coverage.unsqueeze(1) >= self.milestone_tensor

        new_crossings = current_flags & ~prev_flags
        reward = new_crossings.sum(dim=1).float() * self.reward_per_milestone

        # Ensure reward is finite
        reward = torch.nan_to_num(reward, nan=0.0)

        # Update previous coverage
        self.prev_coverage = current_coverage.clone()

        return reward

def get_unknown_inspection_points(env):
    """
    Returns a 1D tensor of shape (num_envs,) indicating, for each environment,
    the total normalized number of unknown inspection points across all objects.

    A point is considered unknown if its confidence < 1e-3.
    Each object's unknown count is normalized by its number of contour points.
    """
    confidence = env.confidence  # shape: (num_envs, num_objects, num_points)
    unknown_mask = (confidence < 0.6)  # shape: (num_envs, num_objects, num_points)
    unknown_count = unknown_mask.sum(dim=-1).float()  # (num_envs, num_objects)

    num_points = confidence.shape[-1]
    normalized_unknowns = unknown_count / num_points  # (num_envs, num_objects)

    per_env_total = normalized_unknowns.sum(dim=-1)  # (num_envs,)
    return per_env_total

def get_known_inspection_points(env):
    """
    Returns a 1D tensor of shape (num_envs,) indicating, for each environment,
    the total normalized number of known inspection points across all objects.

    A point is considered known if its confidence > 0.5
    Each object's known count is normalized by its number of contour points.
    """
    confidence = env.confidence  # shape: (num_envs, num_objects, num_points)
    
    # Known points mask (inverse of unknown)
    known_mask = (confidence > 0.5)  # shape: (num_envs, num_objects, num_points)
    
    # Count known points per object
    known_count = known_mask.sum(dim=-1).float()  # (num_envs, num_objects)
    
    # Normalize by number of points per object
    num_points = confidence.shape[-1]
    normalized_knowns = known_count / num_points  # (num_envs, num_objects)
    
    # Sum normalized counts per environment
    per_env_total = normalized_knowns.sum(dim=-1)  # (num_envs,)
    return per_env_total

class KnownInspectionPointsGainReward(ManagerTermBase):
    """
    Tracks known inspection points per environment and gives rewards only
    when the number of known points increases.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env
        self.threshold = cfg.params.get("threshold", 0.5)
        self.last_knowns = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        confidence = torch.nan_to_num(env.confidence, nan=0.0)

        known_mask = confidence > self.threshold
        known_count = known_mask.sum(dim=-1).float()

        total_knowns = known_count.sum(dim=-1)
        total_knowns = torch.nan_to_num(total_knowns, nan=0.0)

        reward = torch.clamp(total_knowns - self.last_knowns, min=0.0)
        reward = torch.nan_to_num(reward, nan=0.0)

        self.last_knowns = total_knowns

        return reward

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """
        Reset the tracker for the specified environments.

        Args:
            env_ids: List of environment indices to reset. If None, resets all.
        """
        if env_ids is None:
            self.last_knowns.zero_()
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
            self.last_knowns[env_ids_tensor] = 0.0

@torch.no_grad()
def get_robot_closeness_to_ideal_inspection_pose(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Computes a dual-scale reward based on the *proposed action command* 
    (processed target position/orientation) instead of the robot's current state.
    - Uses agent's processed actions (x, y, heading) as target pose.
    - Coarse reward: larger sigmas (3m, pi rad)
    - Fine reward: smaller sigmas (0.5m, 0.5 rad)
    - If ideal pose is NaN, reward = 0.
    """
    device = env.device
    E = env.num_envs

    if not hasattr(env, "best_robot_pose"):
        return torch.zeros(E, device=device)

    ideal_pose = env.best_robot_pose  # (E, 3): x, y, yaw

    # --- Get agent-proposed target pose from processed actions ---
    if "viewpoint_action" not in env.action_manager._terms:
        return torch.zeros(E, device=device)

    pos_command_w = env.action_manager._terms["viewpoint_action"].pos_command_w
    heading_command_w = env.action_manager._terms["viewpoint_action"].heading_command_w
    processed = torch.cat([pos_command_w, heading_command_w.unsqueeze(-1)], dim=-1)
    # processed: (E,4) → [x_norm, y_norm, dummy, heading_norm]
    # We'll use indices [0,1,3]
    target_pos = processed[:, :2]    # x, y (normalized local)
    target_yaw = processed[:, 3]     # heading (normalized)

    target_pos = torch.nan_to_num(target_pos, nan=0.0)
    target_yaw = torch.nan_to_num(target_yaw, nan=0.0)

    # --- Mask invalid ideal poses ---
    valid_mask = ~torch.isnan(ideal_pose).any(dim=1)
    if not valid_mask.any():
        return torch.zeros(E, device=device)

    # --- Differences (target command vs. ideal pose) ---
    pos_diff = target_pos - ideal_pose[:, :2]
    dist_pos = torch.norm(pos_diff, dim=1)

    yaw_diff = torch.atan2(
        torch.sin(target_yaw - ideal_pose[:, 2]),
        torch.cos(target_yaw - ideal_pose[:, 2]),
    ).abs()

    # --- Dual-scale sigmas ---
    sigma_pos_coarse = 3.0
    sigma_yaw_coarse = torch.pi
    sigma_pos_fine = 0.5
    sigma_yaw_fine = 0.5

    # Coarse reward
    reward_pos_coarse = torch.exp(-(dist_pos**2) / (2 * sigma_pos_coarse**2))
    reward_yaw_coarse = torch.exp(-(yaw_diff**2) / (2 * sigma_yaw_coarse**2))
    reward_coarse = reward_pos_coarse * reward_yaw_coarse

    # Fine reward
    reward_pos_fine = torch.exp(-(dist_pos**2) / (2 * sigma_pos_fine**2))
    reward_yaw_fine = torch.exp(-(yaw_diff**2) / (2 * sigma_yaw_fine**2))
    reward_fine = reward_pos_fine * reward_yaw_fine

    # Print dists, diffs and rewards
    # print("Dist Pos (m):", dist_pos[valid_mask])
    # print("Diff Yaw (rad):", yaw_diff[valid_mask])
    # print("Reward Coarse:", reward_coarse[valid_mask])
    # print("Reward Fine:", reward_fine[valid_mask])

    reward = reward_coarse + reward_fine

    reward[~valid_mask] = 0.0
    return reward