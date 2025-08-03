from typing import List

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .utils import point_to_segment_distance

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

def get_min_distance_from_objects_and_walls(
    env: ManagerBasedRLEnv,
    objects: List[str],
    walls: List[str],
    robot_cfg: str = "robot",
    clip_min: float = 0.001,
    clip_max: float = 10.0
):
    """
    Compute clipped minimum distance from the robot to the closest object or wall per environment.

    Args:
        env: RL environment
        objects: List of str objects representing point obstacles
        walls: List of str objects representing wall segments
        robot_cfg: Robot name in the scene
        clip_min: Minimum distance to avoid zeros
        clip_max: Maximum distance to clip extreme values

    Returns:
        torch.Tensor of shape (num_envs,) with minimum distances
    """
    device = env.device

    # --- 1) Robot position (num_envs, 2)
    robot: RigidObject = env.scene[robot_cfg]
    robot_positions = robot.data.root_pos_w[:, :2]

    # --- 2) Point obstacle distances
    if objects:
        obj_positions = torch.stack(
            [env.scene[obj].data.root_pos_w[:, :2] for obj in objects], dim=1
        )  # (num_envs, num_objects, 2)
        point_distances = torch.norm(robot_positions.unsqueeze(1) - obj_positions, dim=-1)
        point_min_dist = point_distances.min(dim=1).values
    else:
        point_min_dist = torch.full((env.num_envs,), float("inf"), device=device)

    # --- 3) Wall distances
    if walls:
        num_walls = len(walls)
        wall_start = torch.empty((env.num_envs, num_walls, 2), device=device)
        wall_end = torch.empty((env.num_envs, num_walls, 2), device=device)

        for i, wall_name in enumerate(walls):
            wall_obj: RigidObject = env.scene[wall_name]
            centers = wall_obj.data.root_pos_w[:, :2]  # (num_envs, 2)
            size = torch.as_tensor(wall_obj.cfg.spawn.size, device=device)

            # Choose axis and compute start/end points
            if size[0] >= size[1]:  # horizontal
                direction = torch.tensor([1.0, 0.0], device=device)
                half_len = size[0] / 2.0
            else:  # vertical
                direction = torch.tensor([0.0, 1.0], device=device)
                half_len = size[1] / 2.0

            dir_expanded = direction.unsqueeze(0).expand(env.num_envs, -1)
            wall_start[:, i] = centers - half_len * dir_expanded
            wall_end[:, i] = centers + half_len * dir_expanded

        # Expand robot positions to (num_envs, num_walls, 2)
        robot_expanded = robot_positions.unsqueeze(1).expand(-1, num_walls, -1)
        wall_distances = point_to_segment_distance(robot_expanded, wall_start, wall_end)
        wall_min_dist = wall_distances.min(dim=1).values
    else:
        wall_min_dist = torch.full((env.num_envs,), float("inf"), device=device)

    # --- 4) Combine and clip distances
    min_distances = torch.minimum(point_min_dist, wall_min_dist)
    return torch.clamp(min_distances, clip_min, clip_max)