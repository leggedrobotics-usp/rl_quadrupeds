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

from quadrupeds_mdp.utils import batch_quaternion_rotate

import matplotlib.pyplot as plt

class SceneGroundTruthMap(ManagerTermBase):
    """
    An observation that constructs a ground truth map of the scene.
    "Ground truth" refers to the actual positions of objects in the scene.
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
        self.env_indices = torch.arange(env.num_envs, device=env.device) # [num_envs]
        self.visited_counts = torch.zeros(env.num_envs, device=env.device) # [num_envs]
        self.index = 0

        env.env_exploration_proportion = torch.zeros(
            env.num_envs,
            dtype=torch.float32,
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

            # Convert world positions to map indices
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

        # Count occupied cells (where value == 0) per environment
        occupied_counts = (map_tensor == 0.0).sum(dim=(1, 2))

        return map_tensor, occupied_counts

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is not None:
            self.map_tensor[self.map_tensor == 0.5] = 1.0
            self.env.env_exploration_proportion[env_ids] = 0.0
    
    def save_map_to_img(self):
        if self.index % 10 == 0:
            plt.figure()
            plt.imshow(self.map_tensor[0].cpu().numpy(), cmap='gray', origin='lower')
            plt.title("Ground Truth Map")
            plt.colorbar()
            plt.savefig(f"/rl_quadrupeds/ground_truth_map_{self.index}.png")
            plt.close()

    def __call__(self, env: ManagerBasedRLEnv, objects: List[str]) -> torch.Tensor:
        robot_pos_env = env.scene["robot"].data.root_link_state_w[:, :2] - env.scene.env_origins[:, :2]
        robot_map_coords = ((robot_pos_env - self.grid_origin) / self.resolution).long()

        cx, cy = robot_map_coords[:, 0], robot_map_coords[:, 1]
        mask = (cx >= 0) & (cx < self.map_dim) & (cy >= 0) & (cy < self.map_dim)

        valid_envs = self.env_indices[mask]

        # Get previous values before setting to 0.5
        prev_vals = self.map_tensor[valid_envs, cx[mask], cy[mask]]

        # Increment visited count where the cell wasn't already visited
        self.visited_counts[valid_envs] += (prev_vals != 0.5).float()

        # Mark the cells as visited
        self.map_tensor[valid_envs, cx[mask], cy[mask]] = 0.5

        # Update exploration proportion
        env.env_exploration_proportion[:] = self.visited_counts / self.free_cells

        # self.index += 1
        # self.save_map_to_img()

        return self.map_tensor.view(env.num_envs, -1)