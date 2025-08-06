from typing import List, Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

from isaaclab.utils.math import quat_apply_yaw

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

class MinDistanceToObjectsAndWalls(ManagerTermBase):
    """
    Computes a nonlinear minimum distance reward from the robot to the closest object or wall per environment.
    - Far from obstacles -> ~0
    - Moderate distances -> small negative reward
    - Near obstacles -> strong negative reward (exponential)
    """
    def __init__(
        self,
        cfg: RewardTermCfg,
        env: ManagerBasedRLEnv
    ):
        super().__init__(cfg, env)
        self.env = env

        # --- Parameters
        self.objects: list[str] = cfg.params.get("objects", [])
        self.walls: list[str] = cfg.params.get("walls", [])
        self.robot_cfg: str = cfg.params.get("robot_cfg", "robot")
        self.clip_min: float = cfg.params.get("clip_min", 0.001)
        self.clip_max: float = cfg.params.get("clip_max", 20.0)

        # Nonlinear shaping parameters
        self.safe_distance: float = cfg.params.get("safe_distance", 1.0)   # No penalty beyond this
        self.near_distance: float = cfg.params.get("near_distance", 0.5)   # Strong penalty below this
        self.sharpness: float = cfg.params.get("sharpness", 10.0)          # Controls exponential growth

        # --- Cached references
        self.robot: RigidObject = env.scene[self.robot_cfg]

        # --- Preallocate inf tensor for when no objects/walls exist
        self.inf_tensor = torch.full((self.env.num_envs,), float("inf"), device=self.env.device)

        # --- Precompute wall endpoints (static)
        if self.walls:
            self.wall_start, self.wall_end = self._precompute_wall_segments()
        else:
            self.wall_start = None
            self.wall_end = None

    def _precompute_wall_segments(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute the start and end points of all wall segments for all envs.
        Returns:
            wall_start, wall_end: (num_envs, num_walls, 2)
        """
        num_walls = len(self.walls)
        wall_start = torch.empty((self.env.num_envs, num_walls, 2), device=self.env.device)
        wall_end = torch.empty((self.env.num_envs, num_walls, 2), device=self.env.device)

        centers_list = []
        dirs_2d_list = []
        half_lengths = []

        for wall_name in self.walls:
            wall_obj: RigidObject = self.env.scene[wall_name]
            centers = wall_obj.data.root_pos_w[:, :2]  # (num_envs, 2)
            rot = wall_obj.data.root_quat_w            # (num_envs, 4)
            size = torch.as_tensor(wall_obj.cfg.spawn.size, device=self.env.device)

            # Determine wall orientation in local frame
            if size[0] >= size[1]:  # horizontal
                local_dir = torch.tensor([1.0, 0.0, 0.0], device=self.env.device)
                half_len = size[0] / 2.0
            else:                   # vertical
                local_dir = torch.tensor([0.0, 1.0, 0.0], device=self.env.device)
                half_len = size[1] / 2.0

            # Rotate to world yaw for all envs
            dirs_2d = quat_apply_yaw(
                rot,
                local_dir.unsqueeze(0).expand(self.env.num_envs, -1)
            )[:, :2]

            centers_list.append(centers)
            dirs_2d_list.append(dirs_2d)
            half_lengths.append(half_len)

        centers = torch.stack(centers_list, dim=1)        # (num_envs, num_walls, 2)
        dirs_2d = torch.stack(dirs_2d_list, dim=1)        # (num_envs, num_walls, 2)
        half_lengths = torch.tensor(half_lengths, device=self.env.device).view(1, num_walls, 1)

        wall_start = centers - half_lengths * dirs_2d
        wall_end = centers + half_lengths * dirs_2d

        return wall_start, wall_end

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """No reset needed for static geometry."""
        pass

    def __call__(
        self, 
        env: ManagerBasedRLEnv, 
        objects: List[str], 
        walls: List[str],
        safe_distance: float = 1.0,
        near_distance: float = 0.5,
        sharpness: float = 10.0,
    ) -> torch.Tensor:
        """
        Compute nonlinear minimum distance reward from the robot to the closest object or wall per environment.
        """
        # --- 1) Robot positions (num_envs, 2)
        robot_positions = self.robot.data.root_pos_w[:, :2]

        # --- 2) Point-object distances
        if self.objects:
            obj_positions = torch.stack(
                [env.scene[obj].data.root_pos_w[:, :2] for obj in self.objects],
                dim=1
            )  # (num_envs, num_objects, 2)
            point_min_dist = torch.norm(
                robot_positions.unsqueeze(1) - obj_positions, dim=-1
            ).min(dim=1).values
        else:
            point_min_dist = self.inf_tensor

        # --- 3) Wall distances (vectorized)
        if self.walls:
            robot_expanded = robot_positions.unsqueeze(1)  # (num_envs, 1, 2)
            wall_min_dist = point_to_segment_distance(
                robot_expanded, self.wall_start, self.wall_end
            ).min(dim=1).values
        else:
            wall_min_dist = self.inf_tensor

        # --- 4) Combine and clip
        min_distances = torch.minimum(point_min_dist, wall_min_dist)
        min_distances = torch.clamp(min_distances, self.clip_min, self.clip_max)

        # --- 5) Nonlinear penalty shaping
        # If distance >= safe_distance -> reward = 0
        # If near_distance < distance < safe_distance -> small negative linear penalty
        # If distance <= near_distance -> exponential negative penalty
        reward = torch.zeros_like(min_distances)

        # far_mask = min_distances >= self.safe_distance
        mid_mask = (min_distances < self.safe_distance) & (min_distances >= self.near_distance)
        near_mask = min_distances < self.near_distance

        # Mid range: linear negative reward
        reward[mid_mask] = (self.safe_distance - min_distances[mid_mask]) / self.safe_distance

        # Near range: strong exponential negative reward
        reward[near_mask] = torch.exp(-self.sharpness * (min_distances[near_mask] - self.near_distance))

        return reward