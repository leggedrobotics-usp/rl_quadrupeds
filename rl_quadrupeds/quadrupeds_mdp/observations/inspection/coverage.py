from typing import Sequence

from matplotlib import pyplot as plt
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg

from .gpi import OnlineGaussianProcess
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

        self.gp = OnlineGaussianProcess(
            max_points=500,
            input_dim=2,
            num_envs=env.num_envs,
            num_objects=self.valid_object_ids.shape[0],
            device=env.device,
            lengthscale=0.5/20.,
            variance=1.,
        )

        env.coverage = torch.zeros(
            (env.num_envs, self.valid_object_ids.shape[0]),
            dtype=torch.float32,
            device=env.device
        )
        env.coverage_prev = torch.zeros_like(env.coverage)
        self._create_contour()
        env.confidence = torch.zeros(
            (env.num_envs, self.valid_object_ids.shape[0], self.num_contour_points),
            dtype=torch.float32,
            device=env.device
        )

    def _create_contour(self):
        side = 0.5
        half = side / 2.0
        points_per_edge = 20

        corners = torch.tensor([
            [-half, -half],
            [half, -half],
            [half, half],
            [-half, half],
        ], device=self.env.device)

        contour = []
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            for t in torch.linspace(0, 1, steps=points_per_edge, device=self.env.device):
                point = (1 - t) * start + t * end
                contour.append(point)
        self.local_contour = torch.stack(contour, dim=0)
        self.num_contour_points = self.local_contour.shape[0]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is not None:
            self.gp.reset(env_ids)
            self.env.coverage[env_ids] = 0.0
            self.env.coverage_prev[env_ids] = 0.0
            self.env.confidence[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        """
        Returns the coverage score and confidence for each object in each environment.

        The coverage score is a measure of how well the agent has covered the area around each object,
        while the confidence is a measure of how certain the agent is about the coverage score.
        """
        lidar_data = lidar_scan_hits_labels(env, sensor_cfg=sensor_cfg)
        lidar_hits = lidar_data[:, :, 0:2]  # x, y
        lidar_labels = lidar_data[:, :, 3].long()  # label

        capture_mask = env.capture_feat_action.bool()  # [E]
        # TODO: when testing coverage, uncomment the next line
        # capture_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        lidar_weights = self._compute_lidar_weights(env, lidar_hits)

        (
            lidar_hits_tensor,
            lidar_hits_mask,
            weights_tensor,
            gp_update_mask
        ) = self._process_lidar_hits(env, lidar_hits, lidar_labels, lidar_weights, capture_mask)

        y_new = lidar_hits_mask.float() * weights_tensor
        self.gp.update(x_new=lidar_hits_tensor, y_new=y_new, object_mask=gp_update_mask)

        coverage_score, confidence = self._compute_coverage_score(env, gp_update_mask)
        # coverage_score is [E, M]
        # confidence is [E, M, D], where D is the number of contour points
        return torch.cat(
            [coverage_score.unsqueeze(-1), confidence],
            dim=-1
        ).view(env.num_envs, -1)

    def _compute_lidar_weights(self, env, lidar_hits):
        E, N = lidar_hits.shape[:2]
        angles = torch.linspace(-1, 1, N, device=env.device)
        weights = 1.0 - torch.abs(angles)
        return weights.view(1, N).expand(E, N)

    def _process_lidar_hits(self, env, lidar_hits, lidar_labels, lidar_weights, capture_mask):
        E = env.num_envs
        M = self.valid_object_ids.shape[0]
        gp_update_mask = torch.zeros((E, M), dtype=torch.bool, device=env.device)

        lidar_hits_per_object = []
        lidar_hits_mask_per_object = []
        weights_per_object = []
        max_hits = 0

        for i, prim_id in enumerate(self.valid_object_ids):
            object_hits, object_masks, object_weights = [], [], []

            object_mask = lidar_labels == prim_id

            for env_id in range(E):
                if not capture_mask[env_id]:
                    object_hits.append(torch.empty((0, 2), device=env.device))
                    object_masks.append(torch.empty((0,), dtype=torch.bool, device=env.device))
                    object_weights.append(torch.empty((0,), device=env.device))
                    continue

                env_mask = object_mask[env_id]
                hits = lidar_hits[env_id][env_mask]
                weights = lidar_weights[env_id][env_mask]
                mask = torch.ones(hits.shape[0], dtype=torch.bool, device=env.device)

                if hits.shape[0] > 0:
                    gp_update_mask[env_id, i] = True

                max_hits = max(max_hits, hits.shape[0])
                object_hits.append(hits)
                object_masks.append(mask)
                object_weights.append(weights)

            lidar_hits_per_object.append(object_hits)
            lidar_hits_mask_per_object.append(object_masks)
            weights_per_object.append(object_weights)

        for i in range(M):
            for env_id in range(E):
                hits = lidar_hits_per_object[i][env_id]
                mask = lidar_hits_mask_per_object[i][env_id]
                weights = weights_per_object[i][env_id]

                pad = max_hits - hits.shape[0]
                if pad > 0:
                    hits = torch.cat([hits, torch.zeros((pad, 2), device=env.device)])
                    mask = torch.cat([mask, torch.zeros(pad, dtype=torch.bool, device=env.device)])
                    weights = torch.cat([weights, torch.zeros(pad, device=env.device)])

                lidar_hits_per_object[i][env_id] = hits
                lidar_hits_mask_per_object[i][env_id] = mask
                weights_per_object[i][env_id] = weights

            lidar_hits_per_object[i] = torch.stack(lidar_hits_per_object[i], dim=0)       # [E, T, 2]
            lidar_hits_mask_per_object[i] = torch.stack(lidar_hits_mask_per_object[i], dim=0)  # [E, T]
            weights_per_object[i] = torch.stack(weights_per_object[i], dim=0)            # [E, T]

        lidar_hits_tensor = torch.stack(lidar_hits_per_object, dim=1)       # [E, M, T, 2]
        lidar_hits_mask = torch.stack(lidar_hits_mask_per_object, dim=1)    # [E, M, T]
        weights_tensor = torch.stack(weights_per_object, dim=1)             # [E, M, T]

        return lidar_hits_tensor, lidar_hits_mask, weights_tensor, gp_update_mask

    def _compute_coverage_score(self, env, gp_update_mask):
        E = env.num_envs
        M = self.valid_object_ids.shape[0]

        object_positions = torch.stack(
            [obj.data.root_pos_w[:, :2] for obj in self.valid_objects_rigid_objects],
            dim=1
        )

        contour_points = object_positions.unsqueeze(2) + self.local_contour.view(1, 1, self.num_contour_points, 2)

        # Only consider environments and objects where we actually have data
        gp_has_data = self.gp.masks.sum(dim=2) > 0
        valid_updates = gp_update_mask & gp_has_data

        if valid_updates.any():
            mu, cov = self.gp.predict(X_test=contour_points, test_mask=valid_updates.unsqueeze(2))

            uncertainty = cov.diagonal(dim1=-2, dim2=-1)
            confidence = 1.0 - (uncertainty / 1.0)
            confidence = torch.clamp(confidence, min=0.0, max=1.0)
            coverage_score = confidence.mean(dim=2)  # [E, M]

            # Only update where data is valid
            self.env.coverage_prev[valid_updates] = self.env.coverage[valid_updates]
            self.env.coverage[valid_updates] = coverage_score[valid_updates]
            self.env.confidence[valid_updates] = confidence[valid_updates]

        return self.env.coverage, self.env.confidence