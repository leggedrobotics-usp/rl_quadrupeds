import torch

def get_env_exploration_percentage(env):
    """
    Returns a 1D tensor of shape (num_envs,) indicating the percentage of
    viewpoints of the environment that has been visited by the agent.
    """
    return env.env_exploration_proportion if hasattr(env, "env_exploration_proportion") \
        else torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

def check_if_current_robot_viewpoint_not_visited(env):
    """
    Returns a 1D tensor of shape (num_envs,) indicating whether the robot's
    current viewpoint has been visited or not. 
    """
    return env.current_viewpoint_not_visited.float() if hasattr(env, "current_viewpoint_not_visited") \
        else torch.zeros(env.num_envs, dtype=torch.float, device=env.device)