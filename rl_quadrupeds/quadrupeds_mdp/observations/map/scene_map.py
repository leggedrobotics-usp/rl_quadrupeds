from typing import (
    List, 
    Sequence, 
    Tuple
)

import torch
import torch.nn.functional as F

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg

from quadrupeds_mdp.utils import (
    batch_quaternion_rotate,
    quat_to_yaw
)

import matplotlib.pyplot as plt

class SceneGroundTruthMap(ManagerTermBase):
    """
    An observation that constructs a ground truth map of the scene.
    Tracks visited positions and viewpoints (position + orientation).
    """

    def __init__(
        self, 
        cfg: ObservationTermCfg, 
        env: ManagerBasedEnv
    ):
        super().__init__(cfg, env)
        self.env = env
        
        self.map_size = cfg.params.get("map_size", 8)  # meters
        self.resolution = cfg.params.get("resolution", 0.3)  # meters per cell
        self.objects = cfg.params.get("objects", None)
        if self.objects is None:
            raise ValueError("Objects must be specified in the configuration.")
        
        self.map_dim = int(self.map_size / self.resolution)
        self.map_total_cells = self.map_dim * self.map_dim
        self.grid_origin = -self.map_size / 2.0  # centered map

        self.map_tensor, self.occupied_counts = self._construct_map(self.objects)  # [num_envs, H, W]
        self.free_cells = self.map_total_cells - self.occupied_counts  # [num_envs]
        self.env_indices = torch.arange(env.num_envs, device=env.device)  # [num_envs]
        self.visited_counts = torch.zeros(env.num_envs, device=env.device)  # [num_envs]
        self.index = 0

        # Viewpoint tracking (position + orientation)
        self.num_yaw_bins = 8
        env.env_exploration_proportion = torch.zeros(
            env.num_envs,
            dtype=torch.float32,
            device=env.device
        )
        env.visited_viewpoints = torch.zeros(
            (env.num_envs, self.map_dim, self.map_dim, self.num_yaw_bins),
            dtype=torch.bool,
            device=env.device
        )
        env.current_viewpoint_not_visited = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device
        )

    def _construct_map(
        self,
        objects: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_envs = self.env.num_envs
        device = self.env.device
        map_tensor = torch.ones((num_envs, self.map_dim, self.map_dim), device=device)

        for object_ in objects:
            state = self.env.scene[object_].data.default_root_state  # shape [B, D]
            size = torch.tensor(self.env.scene[object_].cfg.spawn.size, device=device)  # [3]
            pos = state[:, :3]  # [B, 3]
            rot = state[:, 3:7]  # [B, 4] quaternion
            pos = batch_quaternion_rotate(rot, pos.unsqueeze(1)).squeeze(1)

            half_size = size[:2] / 2.0  # XY half size for 2D map
            half_size = half_size / self.resolution  # convert to map units

            map_coords = ((pos[:, :2] - self.grid_origin) / self.resolution).long()  # [B, 2]

            for i in range(num_envs):
                cx, cy = map_coords[i]
                sx = int(half_size[0])
                sy = int(half_size[1])

                x_min = torch.clamp(cx - sx, 0, self.map_dim - 1)
                x_max = torch.clamp(cx + sx + 1, 0, self.map_dim)
                y_min = torch.clamp(cy - sy, 0, self.map_dim - 1)
                y_max = torch.clamp(cy + sy + 1, 0, self.map_dim)

                map_tensor[i, x_min:x_max, y_min:y_max] = 0.0  # mark object on map

        occupied_counts = (map_tensor == 0.0).sum(dim=(1, 2))
        return map_tensor, occupied_counts

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is not None:
            self.map_tensor[env_ids][self.map_tensor[env_ids] == 0.5] = 1.0
            self.env.env_exploration_proportion[env_ids] = 0.0
            self.env.current_viewpoint_not_visited[env_ids] = True
            self.visited_counts[env_ids] = 0.0
            self.env.visited_viewpoints[env_ids] = False

    def save_map_to_img(self):
        if self.index % 10 == 0:
            plt.figure()
            plt.imshow(self.map_tensor[0].cpu().numpy(), cmap='gray', origin='lower')
            plt.title("Ground Truth Map")
            plt.colorbar()
            plt.savefig(f"/rl_quadrupeds/ground_truth_map_{self.index}.png")
            plt.close()

    def __call__(self, env: ManagerBasedRLEnv, objects: List[str]) -> torch.Tensor:
        # Get robot 2D positions and rotations relative to scene origin
        robot_pos = env.scene["robot"].data.root_link_state_w[:, :2] - env.scene.env_origins[:, :2]
        robot_rot = env.scene["robot"].data.root_link_state_w[:, 3:7]
        yaw = quat_to_yaw(robot_rot)

        # Discretize position and orientation
        map_coords = ((robot_pos - self.grid_origin) / self.resolution).long()  # [B, 2]
        yaw_bins = ((yaw + torch.pi) / (2 * torch.pi) * self.num_yaw_bins).long() % self.num_yaw_bins

        cx, cy = map_coords[:, 0], map_coords[:, 1]
        mask = (cx >= 0) & (cx < self.map_dim) & (cy >= 0) & (cy < self.map_dim)

        valid_envs = self.env_indices[mask]
        valid_cx = cx[mask]
        valid_cy = cy[mask]
        valid_yaw = yaw_bins[mask]

        visited_viewpoints = env.visited_viewpoints
        already_visited = visited_viewpoints[valid_envs, valid_cx, valid_cy, valid_yaw]

        # Update only the valid environments
        new_visit = ~already_visited
        env.current_viewpoint_not_visited[:] = False
        env.current_viewpoint_not_visited[valid_envs] = new_visit

        self.visited_counts[valid_envs] += new_visit.float()
        visited_viewpoints[valid_envs, valid_cx, valid_cy, valid_yaw] = True

        # Optional: mark cells for visualization
        self.map_tensor[valid_envs, valid_cx, valid_cy] = 0.5
        env.env_exploration_proportion[:] = self.visited_counts / (self.free_cells * self.num_yaw_bins)

        # self.index += 1
        # self.save_map_to_img()

        return env.env_exploration_proportion.unsqueeze(-1)