from typing import Sequence
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg

from .gpi import OnlineGaussianProcessRFF
from quadrupeds_mdp.observations.ray_caster import lidar_scan_hits_labels


class ObjectInspectionCoverage(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.env = env

        self.valid_object_ids = torch.tensor([4, 5], device=env.device)
        self.valid_object_names = ["block1", "block2"]
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
            variance=5.0,
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
        self.coverage_threshold = 0.95

    # -----------------------------------------------------------

    def _create_contour(self):
        side = 0.5
        half = side / 2.0
        points_per_edge = 20

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
            self._cached_positions = None
            self._cached_contour_points = None

    # -----------------------------------------------------------

    @torch.no_grad()
    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        lidar_data = lidar_scan_hits_labels(env, sensor_cfg=sensor_cfg)  # [E, N, 4+]
        lidar_hits = lidar_data[:, :, 0:2]  # x, y
        lidar_labels = lidar_data[:, :, 2].long()  # label

        capture_mask = env.capture_feat_action.bool()
        if not capture_mask.any():
            # Skip GP update entirely
            return torch.cat([env.coverage.unsqueeze(-1), env.confidence], dim=-1).view(env.num_envs, -1)

        lidar_weights = self._compute_lidar_weights(env, lidar_hits)
        lidar_hits_tensor, y_tensor, gp_update_mask = self._process_lidar_hits(
            env, lidar_hits, lidar_labels, lidar_weights, capture_mask
        )

        if not gp_update_mask.any():
            # Skip GP update if no new hits
            return torch.cat([env.coverage.unsqueeze(-1), env.confidence], dim=-1).view(env.num_envs, -1)

        # Update GP
        self.gp.update(x_new=lidar_hits_tensor, y_new=y_tensor, object_mask=gp_update_mask)

        # Compute coverage only for objects below threshold
        coverage_score, confidence = self._compute_coverage_score(env, gp_update_mask)

        return torch.cat([coverage_score.unsqueeze(-1), confidence], dim=-1).view(env.num_envs, -1)

    # -----------------------------------------------------------

    @torch.no_grad()
    def _compute_lidar_weights(self, env, lidar_hits):
        E, N = lidar_hits.shape[:2]
        angles = torch.linspace(-1, 1, N, device=env.device)
        weights = 1.0 - torch.abs(angles)
        return weights.view(1, N).expand(E, N)

    # -----------------------------------------------------------

    @torch.no_grad()
    def _process_lidar_hits(self, env, lidar_hits, lidar_labels, lidar_weights, capture_mask):
        E, N, _ = lidar_hits.shape
        M = self.valid_object_ids.shape[0]
        T = min(self.max_hits_per_step, N)

        labels_eq = (lidar_labels.unsqueeze(-1) == self.valid_object_ids.view(1, 1, M)) & capture_mask.view(E, 1, 1)
        mask_emn = labels_eq.transpose(1, 2)

        counts = mask_emn.sum(dim=2)
        gp_update_mask = counts > 0

        sort_keys = (~mask_emn).float()
        sort_idx = torch.argsort(sort_keys, dim=2, stable=True)[..., :T]

        hits_sorted = torch.gather(
            lidar_hits.unsqueeze(1).expand(E, M, N, 2), 2,
            sort_idx.unsqueeze(-1).expand(E, M, T, 2)
        )
        weights_sorted = torch.gather(
            lidar_weights.unsqueeze(1).expand(E, M, N), 2, sort_idx
        )

        valid_point_mask = torch.arange(T, device=env.device).view(1, 1, T) < counts.unsqueeze(-1)
        y_tensor = weights_sorted * valid_point_mask
        return hits_sorted, y_tensor, gp_update_mask

    # -----------------------------------------------------------

    @torch.no_grad()
    def _compute_coverage_score(self, env, gp_update_mask):
        E = env.num_envs
        M = self.valid_object_ids.shape[0]

        # Skip high coverage objects
        high_cov_mask = env.coverage > self.coverage_threshold
        effective_mask = gp_update_mask & (~high_cov_mask)

        if not effective_mask.any():
            return env.coverage, env.confidence

        # Cache contour points
        current_positions = torch.stack(
            [obj.data.root_pos_w[:, :2] for obj in self.valid_objects_rigid_objects],
            dim=1,
        )
        if self._cached_positions is None or not torch.allclose(self._cached_positions, current_positions):
            self._cached_positions = current_positions.clone()
            self._cached_contour_points = current_positions.unsqueeze(2) + self.local_contour.view(1, 1, self.num_contour_points, 2)

        test_mask = effective_mask.unsqueeze(-1).expand(E, M, self.num_contour_points)

        _, var_diag = self.gp.predict(
            X_test=self._cached_contour_points,
            test_mask=test_mask,
            return_diag_only=True,
        )

        confidence = 1.0 - torch.clamp(var_diag, min=0.0, max=1.0)
        coverage_score = confidence.mean(dim=2)

        env.coverage_prev[effective_mask] = env.coverage[effective_mask]
        env.coverage[effective_mask] = coverage_score[effective_mask]
        env.confidence[effective_mask] = confidence[effective_mask]

        return env.coverage, env.confidence