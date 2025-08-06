from dataclasses import MISSING
from typing import List, Sequence

import torch

from isaaclab.envs.mdp.commands import UniformPose2dCommand, UniformPose2dCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

class ObstaclesPose2dCommand(UniformPose2dCommand):
    """
    Command generator that samples pose commands while avoiding collisions with obstacles.
    This version does not access terrain information.
    """

    cfg: "ObstaclesPose2dCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env = env

        self.obstacles = [env.scene[obstacle] for obstacle in cfg.obstacles]

        self.obstacle_positions = torch.stack(
            [obstacle.data.root_link_state_w[:, :2] for obstacle in self.obstacles],
            dim=1  # shape: (num_envs, num_obstacles, 2)
        ).to(env.device)

        self.obstacle_radii = self._calculate_collision_radius()  # [num_obstacles]
        self.num_obstacles = self.obstacle_positions.shape[1]  # Correct dimension: obstacles along dim=1

        # Preallocate buffers
        self._candidate_buf = None

    def _calculate_collision_radius(self):
        radii = []
        for node in self.obstacles:
            size = torch.tensor(node.cfg.spawn.size[:2], dtype=torch.float32, device=self.env.device)
            radius = torch.norm(size / 2)
            radii.append(radius)
        return torch.tensor(radii, device=self.env.device)  # [num_obstacles]

    def _resample_command(self, env_ids: Sequence[int]):
        device = self.env.device
        num_envs = len(env_ids)
        max_attempts = 50

        # Preallocate candidate buffer if needed
        if self._candidate_buf is None or self._candidate_buf.shape[0] < num_envs:
            self._candidate_buf = torch.empty((num_envs, 3), device=device)

        invalid_mask = torch.ones(num_envs, dtype=torch.bool, device=device)

        # Cache environment-specific data once
        base_positions = self._env.scene.env_origins[env_ids]
        z_positions = self.robot.data.default_root_state[env_ids, 2]
        obstacle_positions = self.obstacle_positions[env_ids]  # [num_envs, num_obstacles, 2]

        for _ in range(max_attempts):
            if not invalid_mask.any():
                break

            active_indices = torch.nonzero(invalid_mask, as_tuple=False).squeeze(-1)
            active_count = active_indices.numel()

            # Sample offsets for active envs only
            x_offsets = torch.empty(active_count, device=device).uniform_(*self.cfg.ranges.pos_x)
            y_offsets = torch.empty(active_count, device=device).uniform_(*self.cfg.ranges.pos_y)

            candidates = torch.zeros((active_count, 3), device=device)
            candidates[:, 0] = base_positions[active_indices, 0] + x_offsets
            candidates[:, 1] = base_positions[active_indices, 1] + y_offsets
            candidates[:, 2] = z_positions[active_indices]

            # Compute distances to obstacles [active_count, num_obstacles]
            delta = candidates[:, None, :2] - obstacle_positions[active_indices]
            dists = torch.norm(delta, dim=-1)

            collision_mask = (dists < self.obstacle_radii.unsqueeze(0)).any(dim=1)

            # Update commands only for active envs
            self.pos_command_w[env_ids[active_indices]] = candidates
            invalid_mask[active_indices] = collision_mask

        else:
            print("[Warning] Some positions could not avoid obstacles after max attempts.")

        # Compute heading
        if self.cfg.simple_heading:
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            curr_heading = self.robot.data.heading_w[env_ids]
            curr_to_target = wrap_to_pi(target_direction - curr_heading).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - curr_heading).abs()

            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            self.heading_command_w[env_ids] = torch.empty(num_envs, device=device).uniform_(*self.cfg.ranges.heading)

@configclass
class ObstaclesPose2dCommandCfg(UniformPose2dCommandCfg):
    """Configuration for the terrain-based position command generator."""

    class_type = ObstaclesPose2dCommand

    obstacles: List[str] = MISSING