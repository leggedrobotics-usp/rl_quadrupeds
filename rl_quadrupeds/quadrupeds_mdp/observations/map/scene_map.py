from typing import List, Sequence, Tuple
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from quadrupeds_mdp.utils import quat_to_yaw


class SceneGroundTruthMap(ManagerTermBase):
    """
    Ground truth map tracking visited positions and viewpoints.
    Optimized to avoid slowdowns after many iterations.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.env = env

        self.map_size = cfg.params.get("map_size", 8)
        self.resolution = cfg.params.get("resolution", 0.3)
        self.objects = cfg.params.get("objects", None)
        if self.objects is None:
            raise ValueError("Objects must be specified in the configuration.")

        # Map configuration
        self.map_dim = int(self.map_size / self.resolution)
        self.map_total_cells = self.map_dim * self.map_dim
        self.grid_offset = self.map_size / 2.0  # shift to make coordinates non-negative

        self.env_indices = torch.arange(env.num_envs, device=env.device)

        # Create static map (obstacles)
        self.map_tensor, self.occupied_counts = self._construct_map(self.objects)
        self.free_cells = self.map_total_cells - self.occupied_counts
        self.visited_counts = torch.zeros(env.num_envs, device=env.device)

        # Store visited states in a compressed 2D tensor: [num_envs, total_cells * yaw_bins]
        self.num_yaw_bins = 8
        self.flat_cells = self.map_dim * self.map_dim * self.num_yaw_bins
        env.visited_flat = torch.zeros((env.num_envs, self.flat_cells), dtype=torch.bool, device=env.device)

        env.env_exploration_proportion = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        env.current_viewpoint_not_visited = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        self.index = 0

    def _construct_map(self, objects: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        num_envs = self.env.num_envs
        device = self.env.device
        map_tensor = torch.ones((num_envs, self.map_dim, self.map_dim), device=device)

        for object_ in objects:
            state = self.env.scene[object_].data.default_root_state
            size = torch.tensor(self.env.scene[object_].cfg.spawn.size, device=device)
            pos = state[:, :2]  # XY plane in local env coordinates

            # Shift coordinates to be non-negative
            map_coords = ((pos + self.grid_offset) / self.resolution).to(torch.int32)
            half_size = (size[:2] / (2.0 * self.resolution)).to(torch.int32)

            cx, cy = map_coords[:, 0], map_coords[:, 1]
            sx, sy = half_size[0].item(), half_size[1].item()

            x_min = torch.clamp(cx - sx, 0, self.map_dim - 1)
            x_max = torch.clamp(cx + sx + 1, 0, self.map_dim)
            y_min = torch.clamp(cy - sy, 0, self.map_dim - 1)
            y_max = torch.clamp(cy + sy + 1, 0, self.map_dim)

            for i in range(num_envs):
                map_tensor[i, y_min[i]:y_max[i], x_min[i]:x_max[i]] = 0.0

        occupied_counts = (map_tensor == 0.0).sum(dim=(1, 2))
        return map_tensor, occupied_counts

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is not None:
            self.env.env_exploration_proportion[env_ids] = 0.0
            self.env.current_viewpoint_not_visited[env_ids] = True
            self.visited_counts[env_ids] = 0.0
            self.env.visited_flat[env_ids] = False
        else:
            self.env.env_exploration_proportion.zero_()
            self.env.current_viewpoint_not_visited.zero_()
            self.visited_counts.zero_()
            self.env.visited_flat.zero_()

    @torch.no_grad()
    def __call__(self, env: ManagerBasedRLEnv, objects: List[str]) -> torch.Tensor:
        # Robot positions and rotations (local env coordinates)
        robot_pos = env.scene["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
        robot_rot = env.scene["robot"].data.root_link_state_w[:, 3:7]
        yaw = quat_to_yaw(robot_rot)

        # Convert to non-negative map coordinates
        map_coords = ((robot_pos + self.grid_offset) / self.resolution).to(torch.int32)
        yaw_bins = (((yaw + torch.pi) / (2 * torch.pi) * self.num_yaw_bins).to(torch.int32)) % self.num_yaw_bins

        cx, cy = map_coords[:, 0], map_coords[:, 1]
        mask = (cx >= 0) & (cx < self.map_dim) & (cy >= 0) & (cy < self.map_dim)

        valid_envs = self.env_indices[mask]
        valid_cx, valid_cy, valid_yaw = cx[mask], cy[mask], yaw_bins[mask]

        # Flatten 3D index -> 1D index per env
        flat_idx = (valid_yaw * (self.map_dim * self.map_dim) + valid_cy * self.map_dim + valid_cx)

        visited_flat = env.visited_flat
        already_visited = visited_flat[valid_envs, flat_idx]
        new_visit = ~already_visited

        # Reset flags first
        env.current_viewpoint_not_visited[:] = False
        env.current_viewpoint_not_visited[valid_envs] = new_visit

        # Update visited map
        new_envs = valid_envs[new_visit]
        new_indices = flat_idx[new_visit]
        visited_flat[new_envs, new_indices] = True

        # Update visited counts correctly per environment
        self.visited_counts.index_add_(0, new_envs, torch.ones_like(new_envs, dtype=torch.float32))

        # Compute exploration proportion for all envs
        env.env_exploration_proportion[:] = self.visited_counts / ((self.free_cells * self.num_yaw_bins) + 1e-6)

        self.index += 1
        return env.env_exploration_proportion.unsqueeze(-1)