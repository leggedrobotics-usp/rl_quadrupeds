import torch
from isaaclab.envs import ManagerBasedEnv

def last_action(env: ManagerBasedEnv, action_name: str) -> torch.Tensor:
    """
    The last input action to the environment.
    """
    return env.action_manager.get_term(action_name).processed_actions