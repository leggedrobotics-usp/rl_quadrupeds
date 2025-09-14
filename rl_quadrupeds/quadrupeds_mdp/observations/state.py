from isaaclab.envs.manager_based_env import ManagerBasedEnv

def robot_is_ready_for_new_command(
    env: ManagerBasedEnv
):
    """
    Checks the state of the env variable is_ready_for_new_command.
    """
    if not hasattr(env, "is_ready_for_new_command"):
        raise ValueError("The environment does not have 'is_ready_for_new_command' attribute.")
    
    return env.is_ready_for_new_command.float()