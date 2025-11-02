import torch
from isaaclab.envs import ManagerBasedRLEnv

def penalize_raw_action_saturation(
    env: ManagerBasedRLEnv,
    action_term: str,
    threshold: float = 0.95,   # consider actions beyond |0.95| as saturated
    epsilon: float = 1e-6
):
    # Retrieve raw actions
    raw_actions = env.action_manager._terms[action_term].raw_actions

    # Replace NaN/Inf with 0.0 to avoid invalid comparison effects
    raw_actions = torch.nan_to_num(raw_actions, nan=0.0, posinf=0.0, neginf=0.0)

    # Check for saturation: values very close to -1 or 1
    saturated_mask = torch.abs(raw_actions) >= threshold

    # Count number of saturated dimensions per env
    saturated_count = saturated_mask.sum(dim=-1).float()

    # Optionally weight by how close to -1/1 the action is (stronger penalty near exact saturation)
    saturation_strength = torch.clamp(torch.abs(raw_actions) - threshold, min=0.0)
    saturation_strength = saturation_strength.sum(dim=-1)

    # Combine count and strength into final penalty
    penalties = saturated_count + saturation_strength

    # Ensure penalties are finite
    penalties = torch.nan_to_num(penalties, nan=0.0)

    return penalties

def action_rate_penalty(env, k: float = 5.0) -> torch.Tensor:
    """Penalty: large changes in actions — returns [-1, 0]."""
    penalty_val = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return torch.exp(-k * penalty_val) - 1