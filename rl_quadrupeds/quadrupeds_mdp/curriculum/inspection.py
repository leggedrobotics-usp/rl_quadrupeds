from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def inspection_levels(
    env: "ManagerBasedRLEnv", 
    env_ids: Sequence[int], 
) -> torch.Tensor:
    """Curriculum based on inspection_done flag and coverage.

    Levels up if inspection is done; levels down if coverage is low but above eps.
    """
    device = env.device

    # Ensure coverage exists
    if not hasattr(env, "coverage"):
        return -1.0 * torch.ones(env.num_envs, device=device)

    # Get coverage for the selected envs
    coverage = env.coverage[env_ids].view(-1)

    # Fetch the observation term that holds env_levels
    object_coverage_idx = env.observation_manager._group_obs_term_names["policy"].index("object_coverage")
    coverage_obs_term = env.observation_manager._group_obs_term_cfgs["policy"][object_coverage_idx].func

    # Minimum coverage to consider for leveling down
    eps = 0.1

    # Level up: inspection is done
    if hasattr(env, "_inspection_done"):
        move_up = env._inspection_done[env_ids]
    else:
        move_up = torch.zeros(len(env_ids), dtype=torch.bool, device=device)

    # Level down: coverage low but not too low
    move_down = (coverage > eps) & (coverage < 0.25)

    # Update levels
    coverage_obs_term.update_env_levels(env_ids, move_up, move_down)

    return torch.mean(coverage_obs_term.levels.float())