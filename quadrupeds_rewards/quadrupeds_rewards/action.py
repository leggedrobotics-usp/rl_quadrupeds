"""
action.py

Reward functions for tracking actions.

Available functions:
- action_smoothness_l1: tracks the smoothness of the target 
    joint positions
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

def action_smoothness_l1(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Tracks the smoothness of the target joint positions in
    relation to the previous target joint positions using
    the L1 norm.
    """
    action_smoothness = torch.square(
        env.action_manager.action - \
        env.action_manager.prev_action
    )
    action_smoothness *= env.action_manager.prev_action != 0
    return torch.sum(
        action_smoothness,
        dim=1
    )

class ActionSmoothnessL2(ManagerTermBase):
    """
    Tracks the smoothness of the target joint positions in
    relation to the previous target joint positions using
    the L2 norm.
    """
    def __init__(
        self, 
        cfg: RewardTermCfg, 
        env: ManagerBasedRLEnv
    ):
        super().__init__(cfg, env)
        self.prev_prev_action = torch.zeros_like(
            env.action_manager.prev_action,
            device=env.device
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
    ) -> torch.Tensor:
        action_smoothness = torch.square(
            self.prev_prev_action - \
            2*env.action_manager.prev_action + \
            env.action_manager.action
        )
        action_smoothness *= env.action_manager.prev_action != 0
        action_smoothness *= self.prev_prev_action != 0
        
        self.prev_prev_action = env.action_manager.prev_action
        return torch.sum(
            action_smoothness,
            dim=1
        )