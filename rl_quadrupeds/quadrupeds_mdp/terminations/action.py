import torch
from isaaclab.envs import ManagerBasedRLEnv

def terminate_if_raw_action_outside_limits(
    env: ManagerBasedRLEnv,
    action_term: str,
    min_limit: float = -5.0,
    max_limit: float = 5.0
) -> torch.Tensor:
    """
    Terminates episodes where raw actions exceed specified limits.
    Returns a boolean tensor of shape (num_envs,) indicating termination.
    """
    raw_actions = env.action_manager._terms[action_term].raw_actions

    # Replace NaN/Inf to avoid comparison errors
    raw_actions = torch.nan_to_num(raw_actions, nan=0.0, posinf=0.0, neginf=0.0)

    # Out-of-bounds mask per element
    out_of_bounds = (raw_actions < min_limit) | (raw_actions > max_limit)

    # Any action component out of range means terminate
    terminate_mask = out_of_bounds.any(dim=-1)

    # Optional: Export termination reason to env for debugging/logging
    env._action_oob_termination = terminate_mask

    return terminate_mask