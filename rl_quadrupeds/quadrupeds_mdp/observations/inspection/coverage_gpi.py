from typing import Sequence
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg

from quadrupeds_mdp.rewards.inspection import (
    get_inspection_action
)
from quadrupeds_mdp.observations.ray_caster import lidar_scan
from .gpi import OnlineGaussianProcessRFF

class ObjectInspectionCoverage(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.env = env

        self.valid_object_ids = torch.tensor([4], device=env.device)
        self.valid_object_names = ["block1"]
        self.valid_objects_rigid_objects = [
            env.scene[object_name] for object_name in self.valid_object_names
        ]

        # CONFIG: fixed shapes for compilation
        self.max_hits_per_step = 96

        # Fast RFF GP
        self.gp = OnlineGaussianProcessRFF(
            max_points=500,
            input_dim=2,
            num_envs=env.num_envs,
            num_objects=self.valid_object_ids.shape[0],
            device=env.device,
            lengthscale=0.01,
            variance=2.5,
            noise=1e-2,
            num_features=96,
            dtype=torch.float32,
            jitter=1e-6,
            max_chol_tries=5,
            decay=0.995,
        )

        # Coverage and confidence
        env.coverage = torch.zeros(
            (env.num_envs, self.valid_object_ids.shape[0]),
            dtype=torch.float32, device=env.device
        )
        env.coverage_prev = torch.zeros_like(env.coverage)

        self._create_contour()
        env.confidence = torch.zeros(
            (env.num_envs, self.valid_object_ids.shape[0], self.num_contour_points),
            dtype=torch.float32, device=env.device
        )

        # Cache contour points to avoid recomputation
        self._cached_positions = None
        self._cached_contour_points = None

        # Threshold for skipping high-confidence objects
        self.confidence_lock_threshold_per_point = 0.9
        self.coverage_threshold = 0.95

        env._inspection_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # -----------------------------------------------------------

    def _create_contour(self):
        side = 0.5
        half = side / 2.0
        points_per_edge = 7

        corners = torch.tensor(
            [[-half, -half], [half, -half], [half, half], [-half, half]],
            device=self.env.device,
        )

        t = torch.linspace(0, 1, steps=points_per_edge, device=self.env.device).unsqueeze(-1)
        edges = [(1 - t) * corners[i] + t * corners[(i + 1) % 4] for i in range(4)]

        self.local_contour = torch.cat(edges, dim=0)  # [D, 2]
        self.num_contour_points = self.local_contour.shape[0]

    # -----------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is not None:
            self.gp.reset(env_ids)
            self.env.coverage[env_ids] = 0.0
            self.env.coverage_prev[env_ids] = 0.0
            self.env.confidence[env_ids] = 0.0
            self.env._inspection_done[env_ids] = False
            self._cached_positions = None
            self._cached_contour_points = None

    # -----------------------------------------------------------

    @torch.no_grad()
    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        # Use lidar_scan: only take the center 3 rays
        lidar_data = lidar_scan(
            env=env,
            sensor_cfg=sensor_cfg,
            fill_value=20.0,
            num_rays=6,
            flatten=False,
            return_hits=True,
            return_labels=True
        )

        lidar_hits = lidar_data[:, :, 0:2]             # [E, 3, 2] -> x, y
        lidar_labels = lidar_data[:, :, 2].long()      # [E, 3]   -> labels

        print("Coverage:", env.coverage)
        print("Confidence:", env.confidence)
        print("Inspection Done:", env._inspection_done)

        capture_mask = env.current_viewpoint_not_visited.bool() & get_inspection_action(env).bool()
        if not capture_mask.any():
            confidence_flat = env.confidence.view(env.num_envs, -1)
            return torch.cat((env.coverage.view(env.num_envs, -1), confidence_flat), dim=1)

        # Process lidar hits
        lidar_hits_tensor, y_tensor, gp_update_mask = self._process_lidar_hits(
            env, lidar_hits, lidar_labels, capture_mask
        )

        if not gp_update_mask.any():
            confidence_flat = env.confidence.view(env.num_envs, -1)
            return torch.cat((env.coverage.view(env.num_envs, -1), confidence_flat), dim=1)

        # Update GP normally (no confidence mask here)
        self.gp.update(x_new=lidar_hits_tensor, y_new=y_tensor, object_mask=gp_update_mask)

        # Compute coverage (mask applied inside)
        coverage_score, confidence = self._compute_coverage_score(env, gp_update_mask)

        done = torch.sum(self.env.coverage >= self.coverage_threshold - 0.05, dim=1) >= self.env.coverage.shape[1]
        self.env._inspection_done[:] = done

        confidence_flat = confidence.view(env.num_envs, -1)
        obs = torch.cat((coverage_score.view(env.num_envs, -1), confidence_flat), dim=1)
        return obs


    # -----------------------------------------------------------
    @torch.no_grad()
    def _process_lidar_hits(self, env, lidar_hits, lidar_labels, capture_mask):
        """
        Select lidar hits strictly for valid objects (center 8 rays only).
        """
        E, N, _ = lidar_hits.shape
        M = self.valid_object_ids.shape[0]
        T = min(self.max_hits_per_step, N)
        device = env.device

        # Step 1: Distance filter (cap to a radius)
        sensor_pos = env.scene.sensors["ray_caster"].data.pos_w[:, :2]  # [E, 2]
        distances = torch.norm(lidar_hits - sensor_pos.unsqueeze(1), dim=-1)  # [E, N]
        distance_mask = distances <= 3.0  # keep near hits

        # Step 2: Label mask + capture mask + distance mask
        labels_eq = (
            (lidar_labels.unsqueeze(-1) == self.valid_object_ids.view(1, 1, M))
            & capture_mask.view(E, 1, 1)
        )  # [E, N, M]
        labels_eq = labels_eq.transpose(1, 2)  # [E, M, N]

        # Combine all masks
        mask_emn = labels_eq & distance_mask.view(E, 1, N)  # [E, M, N]

        # Step 3: Count hits and select objects with at least one valid hit
        counts = mask_emn.sum(dim=2)  # [E, M]
        gp_update_mask = counts > 0   # [E, M]

        # Step 4: Sort rays, putting valid ones first (stable sort)
        sort_keys = (~mask_emn).float()  # valid=True→0, invalid=False→1
        sort_idx = torch.argsort(sort_keys, dim=2, stable=True)[..., :T]  # [E, M, T]

        # Step 5: Gather hits
        hits_sorted = torch.gather(
            lidar_hits.unsqueeze(1).expand(E, M, N, 2), 2,
            sort_idx.unsqueeze(-1).expand(E, M, T, 2)
        )  # [E, M, T, 2]

        # Step 6: Validity mask per object to zero-out padded gathers
        valid_point_mask = torch.arange(T, device=device).view(1, 1, T) < counts.unsqueeze(-1)  # [E, M, T]

        # Final y_tensor (just 1.0 for valid hits, 0.0 otherwise)
        y_tensor = valid_point_mask.float()  # [E, M, T]

        return hits_sorted, y_tensor, gp_update_mask

    # -----------------------------------------------------------

    @torch.no_grad()
    def _compute_coverage_score(self, env, gp_update_mask):
        E = env.num_envs
        M = self.valid_object_ids.shape[0]

        # Skip objects already above coverage threshold
        high_cov_mask = env.coverage > self.coverage_threshold
        effective_mask = gp_update_mask & (~high_cov_mask)

        # If nothing new to update, just return stored values
        if not effective_mask.any():
            return env.coverage, env.confidence

        # Cache contour points
        current_positions = torch.stack(
            [obj.data.root_pos_w[:, :2] for obj in self.valid_objects_rigid_objects],
            dim=1,
        )  # [E, M, 2]
        if self._cached_positions is None or not torch.allclose(self._cached_positions, current_positions):
            self._cached_positions = current_positions.clone()
            self._cached_contour_points = current_positions.unsqueeze(2) + self.local_contour.view(
                1, 1, self.num_contour_points, 2
            )

        # Test mask for GP prediction
        test_mask = effective_mask.unsqueeze(-1).expand(E, M, self.num_contour_points)

        _, var_diag = self.gp.predict(
            X_test=self._cached_contour_points,
            test_mask=test_mask,
            return_diag_only=True,
        )

        # Compute new confidence
        confidence_new = 1.0 - torch.clamp(var_diag, min=0.0, max=1.0)

        # Lock points already above threshold
        high_conf_mask_points = env.confidence >= self.confidence_lock_threshold_per_point
        confidence_new = torch.where(high_conf_mask_points, torch.ones_like(confidence_new), confidence_new)

        coverage_score_new = confidence_new.mean(dim=2)

        # Update only effective entries
        env.coverage_prev[effective_mask] = env.coverage[effective_mask]
        env.coverage[effective_mask] = coverage_score_new[effective_mask]
        env.confidence[effective_mask] = confidence_new[effective_mask]

        return env.coverage, env.confidence