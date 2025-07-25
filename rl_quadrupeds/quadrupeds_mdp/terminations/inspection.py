import torch
from isaaclab.envs import ManagerBasedRLEnv

def check_if_inspection_done(
    env: ManagerBasedRLEnv,
    threshold_per_env: float = 0.9
) -> torch.Tensor:
    """
    Checks if the inspection is done for each environment.
    Returns a boolean tensor where each element is True if the inspection is done,
    otherwise False.

    It exports "_inspection_done" to the env so termination reason can be evaluated
    when calculating rewards.
    """
    if not hasattr(env, "_inspection_done"):
        env._inspection_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    done = torch.sum(env.coverage, dim=1) >= threshold_per_env * env.coverage.shape[1]
    env._inspection_done[:] = done
    return done