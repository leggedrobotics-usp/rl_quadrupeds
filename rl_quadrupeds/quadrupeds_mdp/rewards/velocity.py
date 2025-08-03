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

def velocity_aligned_with_goal_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: str = "robot",
    normalize: bool = True,
    clip_min: float = -10.0,
    clip_max: float = 10.0
) -> torch.Tensor:
    """
    Compute a velocity-alignment reward: dot(current_velocity, goal_direction).

    The goal position is retrieved from env.command_manager.pos_command_w.

    Args:
        env: RL environment
        robot_cfg: Robot name in the scene
        normalize: If True, normalize velocity and goal direction before dot
        clip_min: Min reward value
        clip_max: Max reward value

    Returns:
        torch.Tensor of shape (num_envs,) with alignment reward
    """
    device = env.device
    robot: RigidObject = env.scene[robot_cfg]

    # --- 1) Retrieve the goal position from world commands
    goal_positions = env.command_manager._terms["pose_command"].pos_command_w[:, :2].to(device)  # (num_envs, 2)

    # --- 2) Robot positions and velocities in XY plane
    positions = robot.data.root_pos_w[:, :2]       # (num_envs, 2)
    velocities = robot.data.root_lin_vel_w[:, :2]  # (num_envs, 2)

    # --- 3) Compute goal direction (normalized)
    goal_direction = goal_positions - positions  # (num_envs, 2)
    goal_norm = torch.norm(goal_direction, dim=-1, keepdim=True) + 1e-8
    goal_direction = goal_direction / goal_norm

    # --- 4) Optionally normalize velocity
    if normalize:
        vel_norm = torch.norm(velocities, dim=-1, keepdim=True) + 1e-8
        velocities = velocities / vel_norm

    # --- 5) Forward progress = dot(velocity, goal_direction)
    forward_progress = torch.sum(velocities * goal_direction, dim=-1)

    # --- 6) Clip reward
    return torch.clamp(forward_progress, clip_min, clip_max)