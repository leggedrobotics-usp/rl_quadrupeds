import torch
from isaaclab.envs import ManagerBasedRLEnv

def check_if_inspection_done(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Checks if the inspection is done for each environment.
    Returns the number of environments that have finished inspecting.

    It also exports "_inspection_done" to the env so termination reason
    can be evaluated when calculating rewards.
    """
    if not hasattr(env, "_inspection_done"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    return env._inspection_done