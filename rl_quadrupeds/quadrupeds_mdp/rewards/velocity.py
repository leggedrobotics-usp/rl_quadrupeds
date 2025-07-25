import torch
from isaaclab.envs import ManagerBasedRLEnv

def vel_action_rate_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Computes the L2 norm of the action rate of the robot velocities.
    It considers that the first three components of the action
    space correspond to the velocities (vx, vy and wz).

    It can be used when the action space has other components
    that cannot be penalized by the action rate.
    """
    return torch.sum(
        torch.square(
            env.action_manager._terms["hl_vel"].processed_actions - env.action_manager._terms["hl_vel"]._last_processed_actions
        ), dim=1
    )