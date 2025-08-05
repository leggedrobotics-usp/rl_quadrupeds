from typing import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

def get_inspection_action(
    env: ManagerBasedRLEnv
) -> torch.Tensor:
    """
    Computes when the agent decides to inspect the environment.
    Returns 1 if the inspection action is taken, otherwise returns 0
    for each environment.
    
    It is used to penalize the inspection action.
    """
    return env.action_manager._terms["capture_feat_action"].processed_actions

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

    def __init__(
        self,
        cfg: RewardTermCfg,
        env: ManagerBasedRLEnv,
    ):
        super().__init__(cfg, env)
        self.env = env

        # Track the maximum coverage per environment
        self.max_coverage = torch.zeros(env.num_envs, device=env.device)

        # Track the previous coverage per environment
        self.prev_coverage = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset stored coverage values for given environments."""
        if env_ids is not None:
            self.max_coverage[env_ids] = 0.0
            self.prev_coverage[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Computes the reward for exceeding the previous max coverage."""
        # Current overall coverage per environment
        current_coverage = torch.sum(env.coverage, dim=1)

        # Compute coverage difference (only positive changes matter)
        diff = (current_coverage - self.prev_coverage).clamp_min(0.0)

        # Reward only if coverage increased AND exceeded previous max
        reward_mask = current_coverage > self.max_coverage
        reward = diff * reward_mask.float()

        # Update the maximum coverage per environment
        self.max_coverage = torch.maximum(self.max_coverage, current_coverage)

        # Update previous coverage for next step
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

class MilestoneCoverageReward(ManagerTermBase):
    """
    Rewards the agent once per milestone when the overall inspection
    coverage surpasses predefined thresholds.

    Example milestones: 30%, 50%, 60%, 70%, 80%, 90%
    """
    def __init__(
        self,
        cfg: RewardTermCfg,
        env: ManagerBasedRLEnv,
    ):
        super().__init__(cfg, env)
        self.env = env

        # Load params or use defaults
        self.milestones = cfg.params.get(
            "milestones", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        self.reward_per_milestone = cfg.params.get(
            "reward_per_milestone", 1.0
        )

        # Store previous coverage per environment
        self.prev_coverage = torch.zeros(
            env.num_envs, device=env.device
        )

        # Convert milestones to tensor for vectorized comparison
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
        # Current overall coverage per environment
        current_coverage = torch.sum(env.coverage, dim=1)

        # Compare current vs previous to detect milestone crossings
        prev_flags = self.prev_coverage.unsqueeze(1) >= self.milestone_tensor
        current_flags = current_coverage.unsqueeze(1) >= self.milestone_tensor

        # Milestone is crossed if current >= milestone and prev < milestone
        new_crossings = (current_flags & ~prev_flags)

        # Reward per new milestone crossed
        reward = new_crossings.sum(dim=1).float() * self.reward_per_milestone

        # Update previous coverage for next step
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

        # Load params or use defaults
        self.threshold = cfg.params.get("threshold", 0.5)

        # Track last known normalized counts per environment
        self.last_knowns = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """
        Compute the reward based on updated confidence values.

        Returns:
            reward: 1D tensor of shape (num_envs,) with rewards.
        """
        confidence = env.confidence  # (num_envs, num_objects, num_points)

        # Determine known points
        known_mask = confidence > self.threshold  # (num_envs, num_objects, num_points)

        # Count known points per object and normalize
        known_count = known_mask.sum(dim=-1).float()  # (num_envs, num_objects)

        # Sum per environment
        total_knowns = known_count.sum(dim=-1)  # (num_envs,)

        # Compute reward: only positive increase is rewarded
        reward = torch.clamp(total_knowns - self.last_knowns, min=0.0)

        # Update state
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