from typing import Sequence, Tuple

from matplotlib import pyplot as plt
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg

from .gpi import OnlineGaussianProcessRFF
from quadrupeds_mdp.observations.ray_caster import lidar_scan_hits_labels

def _maybe_compile(fn):
    """Wrap a function with torch.compile if available (PyTorch >= 2.0)."""
    try:
        import torch._dynamo
        return torch.compile(fn, mode="default", fullgraph=False)  # type: ignore
    except Exception:
        return fn

class ObjectInspectionCoverage(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.env = env

        self.valid_object_ids = torch.tensor([4, 5], device=env.device)
        self.valid_object_names = ["block1", "block2"]
        self.valid_objects_rigid_objects = [
            env.scene[object_name] for object_name in self.valid_object_names
        ]

        # CONFIG: keep shapes fixed (avoid recompiles)
        self.max_hits_per_step = 128  # cap hits per (env,obj) per step to a fixed size

        # Fast RFF GP
        self.gp = OnlineGaussianProcessRFF(
            max_points=500,  # unused, kept for compatibility
            input_dim=2,
            num_envs=env.num_envs,
            num_objects=self.valid_object_ids.shape[0],
            device=env.device,
            lengthscale=0.01,
            variance=5.0,
            noise=1e-2,
            num_features=128,    # <-- matches your change
            dtype=torch.float32,
            jitter=1e-6,
            max_chol_tries=5,
            decay=0.995,         # stronger forgetting to avoid ill-conditioning
        )

        env.coverage = torch.zeros(
            (env.num_envs, self.valid_object_ids.shape[0]),
            dtype=torch.float32,
            device=env.device,
        )
        env.coverage_prev = torch.zeros_like(env.coverage)
        self._create_contour()
        env.confidence = torch.zeros(
            (env.num_envs, self.valid_object_ids.shape[0], self.num_contour_points),
            dtype=torch.float32,
            device=env.device,
        )

    # -----------------------------------------------------------

    def _create_contour(self):
        side = 0.5
        half = side / 2.0
        points_per_edge = 20

        corners = torch.tensor(
            [
                [-half, -half],
                [half, -half],
                [half, half],
                [-half, half],
            ],
            device=self.env.device,
        )

        t = torch.linspace(0, 1, steps=points_per_edge, device=self.env.device).unsqueeze(-1)

        edges = []
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            edges.append((1 - t) * start + t * end)
        self.local_contour = torch.cat(edges, dim=0)  # [D, 2]
        self.num_contour_points = self.local_contour.shape[0]

    # -----------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is not None:
            self.gp.reset(env_ids)
            self.env.coverage[env_ids] = 0.0
            self.env.coverage_prev[env_ids] = 0.0
            self.env.confidence[env_ids] = 0.0

    # -----------------------------------------------------------

    @torch.no_grad()
    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        lidar_data = lidar_scan_hits_labels(env, sensor_cfg=sensor_cfg)  # [E, N, 4+]
        lidar_hits = lidar_data[:, :, 0:2]  # x, y
        lidar_labels = lidar_data[:, :, 2].long()  # label

        capture_mask = env.capture_feat_action.bool()  # [E]
        # capture_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)  # for debugging
        lidar_weights = self._compute_lidar_weights(env, lidar_hits)

        lidar_hits_tensor, y_tensor, gp_update_mask = self._process_lidar_hits(
            env, lidar_hits, lidar_labels, lidar_weights, capture_mask
        )

        # Update GP
        self.gp.update(
            x_new=lidar_hits_tensor,  # [E, M, T, 2]
            y_new=y_tensor,           # [E, M, T]
            object_mask=gp_update_mask,
        )

        coverage_score, confidence = self._compute_coverage_score(env, gp_update_mask)

        # return coverage_score
        # Return shape: [E, M + M*D] flattened across M
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
        """
        Vectorized, fixed-size (T = max_hits_per_step) output.
        Returns:
            lidar_hits_tensor: [E, M, T, 2]
            y_tensor:          [E, M, T]
            gp_update_mask:    [E, M]
        """
        E, N, _ = lidar_hits.shape
        M = self.valid_object_ids.shape[0]

        T = min(self.max_hits_per_step, N)

        # [E, N, M]: matches
        labels_eq = lidar_labels.unsqueeze(-1) == self.valid_object_ids.view(1, 1, M)

        # Apply capture mask
        labels_eq = labels_eq & capture_mask.view(E, 1, 1)

        # [E, M, N]
        mask_emn = labels_eq.transpose(1, 2)

        hits_full = lidar_hits.unsqueeze(1).expand(E, M, N, 2)
        weights_full = lidar_weights.unsqueeze(1).expand(E, M, N)

        # Bring True to the front along N
        sort_keys = (~mask_emn).float()
        sort_idx = torch.argsort(sort_keys, dim=2, stable=True)

        hits_sorted = torch.gather(hits_full, 2, sort_idx.unsqueeze(-1).expand(E, M, N, 2))
        weights_sorted = torch.gather(weights_full, 2, sort_idx)

        counts = mask_emn.sum(dim=2)  # [E, M]
        gp_update_mask = counts > 0

        # Fixed-size slice
        lidar_hits_tensor = hits_sorted[:, :, :T, :]  # [E, M, T, 2]
        y_tensor = weights_sorted[:, :, :T]          # [E, M, T]

        # Valid points inside that slice
        arange_t = torch.arange(T, device=env.device).view(1, 1, T)
        valid_point_mask = arange_t < counts.unsqueeze(-1)  # [E, M, T]

        # Zero out invalids
        y_tensor = y_tensor * valid_point_mask

        return lidar_hits_tensor, y_tensor, gp_update_mask

    # -----------------------------------------------------------

    @torch.no_grad()
    def _compute_coverage_score(self, env, gp_update_mask):
        E = env.num_envs
        M = self.valid_object_ids.shape[0]

        object_positions = torch.stack(
            [obj.data.root_pos_w[:, :2] for obj in self.valid_objects_rigid_objects],
            dim=1,
        )  # [E, M, 2]

        contour_points = (
            object_positions.unsqueeze(2) + self.local_contour.view(1, 1, self.num_contour_points, 2)
        )  # [E, M, D, 2]

        test_mask = gp_update_mask.unsqueeze(-1).expand(E, M, self.num_contour_points)

        # Use the fast path: only the diagonal of the covariance
        mu, var_diag = self.gp.predict(
            X_test=contour_points,
            test_mask=test_mask,
            return_diag_only=True,
        )

        confidence = 1.0 - torch.clamp(var_diag, min=0.0, max=1.0)  # [E, M, D]
        coverage_score = confidence.mean(dim=2)  # [E, M]

        env.coverage_prev[gp_update_mask] = env.coverage[gp_update_mask]
        env.coverage[gp_update_mask] = coverage_score[gp_update_mask]
        env.confidence[gp_update_mask] = confidence[gp_update_mask]

        return env.coverage, env.confidence