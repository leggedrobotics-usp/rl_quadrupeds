import torch
from isaaclab.envs import ManagerBasedRLEnv

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

def get_overall_inspection_coverage_gain(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Computes the overall inspection coverage gain for each environment.
    It calculates the difference in coverage from the previous step.
    """
    return torch.sum(env.coverage - env.coverage_prev, dim=1)

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