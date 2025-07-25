import torch

from isaaclab.envs import ManagerBasedRLEnv

def reset_on_start(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Reset only once at the beginning of training. Use env's internal state to track this.
    """
    if not hasattr(env, "_startup_reset_done"):
        env._startup_reset_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # First call returns True (reset) for all envs
    if not env._startup_reset_done.any():
        env._startup_reset_done[:] = True
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)