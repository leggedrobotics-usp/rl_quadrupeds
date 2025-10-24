from typing import Optional

import matplotlib.pyplot as plt
import torch

from cfg import Cfg
from object_interest import ObjectInterestBuilder
from ray_caster import RayCasterBuilder
from robot import RobotBuilder

class ObjectCoverageCfg(Cfg):
    def _define_defaults(self):
        self.num_center_rays_coverage = 7  # rays used for coverage update (odd number, centered)

        self.hit_tolerance = 0.1  # meters, radius around contour point to consider "observed"
        self.max_hit_distance = 2.0  # meters, max distance to consider a hit valid

        # --- Randomization parameters ---
        self.min_clusters = 0
        self.max_clusters = 5
        self.min_cluster_radius = 0.05
        self.max_cluster_radius = 0.2
        self.force_closest_face_prob = 0.1  # probability of forcing closest face to be fully inspected

        self.noise_confidence_std = 0.05  # stddev of noise added to confidence observation

class ObjectCoverage:
    def __init__(
        self,
        cfg: ObjectCoverageCfg,
        ray_caster_builder: RayCasterBuilder,
        robot_builder: RobotBuilder,
        object_interest_builder: ObjectInterestBuilder,
        device: torch.device,
        num_envs: int = 1,
    ):
        self.cfg = cfg
        self.ray_caster_builder = ray_caster_builder
        self.robot_builder = robot_builder
        self.object_interest_builder = object_interest_builder
        self.device = device
        self.num_envs = num_envs
        self.build()

    def build(self):
        E = self.num_envs
        M = self.object_interest_builder.cfg.num_objects_per_env
        P = self.object_interest_builder.cfg.contour_points_per_edge * 4

        # Confidence buffers
        self.observed_confidence = torch.zeros((E, M, P), device=self.device)
        self.randomized_confidence = torch.zeros((E, M, P), device=self.device)
        self.confidence = torch.zeros((E, M, P), device=self.device)
        self.coverage = torch.zeros((E, M), device=self.device)

    @torch.no_grad()
    def update(self, lidar_scan=None, lidar_pose=None, store_results: bool = True):
        """Vectorized update of observed confidence and coverage based on lidar hits."""
        E = self.num_envs
        M = self.object_interest_builder.cfg.num_objects_per_env
        contour = self.object_interest_builder.obj_prop["contour"]  # (E, M, P, 2)
        device = self.device

        # --- use provided lidar scan or compute new one ---
        if lidar_scan is None:
            lidar_scan = self.ray_caster_builder.get_ray_data(
                return_hits=True,
                return_labels=True,
                flatten=False,
                normalize=False,
                num_rays=self.cfg.num_center_rays_coverage,
            )

        if lidar_pose is None:
            lidar_pose = self.ray_caster_builder._ray_tracing_results["sensor_pos"][:, :2]  # (E, 2)

        hits = lidar_scan[:, :, 0:2]  # (E, N, 2)
        labels = lidar_scan[:, :, 2].long()  # (E, N)

        # --- Broadcast hits per object ---
        object_mask = (torch.arange(M, device=device).view(1, M, 1) == labels.unsqueeze(1))  # (E, M, N)
        dist_to_sensor = torch.norm(
            hits.unsqueeze(1) - lidar_pose.unsqueeze(1).unsqueeze(2), dim=-1
        )  # (E, M, N)
        valid_mask = object_mask & (dist_to_sensor <= self.cfg.max_hit_distance)

        if valid_mask.sum() == 0:
            if not store_results:
                return self.observed_confidence.clone(), self.confidence.clone(), self.coverage.clone()
            return

        # --- Compute distances from contour points to hits ---
        hits_exp = hits.unsqueeze(1).expand(E, M, hits.shape[1], 2)
        contour_exp = contour.unsqueeze(3)  # (E, M, P, 1, 2)
        diff = contour_exp - hits_exp.unsqueeze(2)  # (E, M, P, N, 2)
        dist2 = torch.sum(diff * diff, dim=-1)  # (E, M, P, N)
        dist2 = torch.nan_to_num(dist2, nan=1e6)

        hit_mask = dist2 <= (self.cfg.hit_tolerance**2)  # (E, M, P, N)
        observed_now = (hit_mask & valid_mask.unsqueeze(2)).any(dim=-1).float()  # (E, M, P)

        # --- Update confidence and coverage ---
        observed_confidence = torch.maximum(self.observed_confidence, observed_now)
        confidence = observed_confidence[:]
        coverage = confidence.mean(dim=2)

        if not store_results:
            return observed_confidence, confidence, coverage

        self.observed_confidence[:] = observed_confidence
        self.confidence[:] = confidence
        self.coverage[:] = coverage
        self._calculate_face_confidence()

    @torch.no_grad()
    def randomize_confidence(self):
        """Simulate previous inspections with clusters of points and optionally mark the closest face as fully inspected."""
        contour = self.object_interest_builder.obj_prop["contour"]  # (E,M,P,2)
        face_indices = self.object_interest_builder.obj_prop["face_indices"]  # (E,M,F,pts_per_edge)
        E, M, P, _ = contour.shape
        F = face_indices.shape[2]

        num_clusters = torch.randint(
            self.cfg.min_clusters, self.cfg.max_clusters + 1, (E, M), device=self.device
        )
        self.randomized_confidence[:] = 0.0

        # --- Normal randomized clusters ---
        for k in range(self.cfg.max_clusters):
            mask = num_clusters > k
            if not mask.any():
                continue
            center_idx = torch.randint(0, P, (E, M), device=self.device)
            centers = contour[torch.arange(E)[:, None], torch.arange(M)[None, :], center_idx]
            radius = torch.empty((E, M), device=self.device).uniform_(
                self.cfg.min_cluster_radius, self.cfg.max_cluster_radius
            )
            dists = torch.norm(contour - centers.unsqueeze(2), dim=-1)
            cluster_mask = (dists <= radius.unsqueeze(-1)) & mask.unsqueeze(-1)
            self.randomized_confidence[:] = torch.maximum(self.randomized_confidence, cluster_mask.float())

        if self.cfg.force_closest_face_prob > 0.0:
            robot_xy = self.robot_builder.viewpoint[:, :2].unsqueeze(1).unsqueeze(2)  # (E,1,1,2)
            face_pts = torch.gather(
                contour.unsqueeze(2).expand(-1, -1, F, -1, -1),  # [E,M,F,P,2]
                3,
                face_indices.unsqueeze(-1).expand(-1, -1, -1, -1, 2)
            )  # [E,M,F,pts_per_edge,2]
            centroids = face_pts.mean(dim=3)  # [E,M,F,2]
            dists = torch.norm(centroids - robot_xy, dim=-1)  # [E,M,F]

            closest_idx = dists.view(E, -1).argmin(dim=1)  # [E]
            m_idx = closest_idx // F
            f_idx = closest_idx % F

            # --- Apply probability mask (vectorized) ---
            mask_apply = (torch.rand(E, device=self.device) < self.cfg.force_closest_face_prob)  # [E]

            if mask_apply.any():
                batch_ids = torch.arange(E, device=self.device)[mask_apply]
                obj_ids = m_idx[mask_apply]
                face_ids = f_idx[mask_apply]

                pts_idx = face_indices[batch_ids, obj_ids, face_ids]  # [E_sel, pts_per_edge]
                self.randomized_confidence[batch_ids.unsqueeze(-1), obj_ids.unsqueeze(-1), pts_idx] = 1.0

        # --- Update combined confidence ---
        self.confidence[:] = torch.maximum(self.observed_confidence, self.randomized_confidence)
        self.coverage[:] = self.confidence.mean(dim=2)
        self._calculate_face_confidence()

    def get_observation(self, noise: bool = False):
        obs_confidence = self.observed_confidence.clone()
        if noise:
            obs_confidence += torch.randn_like(obs_confidence) * self.cfg.noise_confidence_std
            obs_confidence.clamp_(0.0, 1.0)

        confidence = torch.maximum(obs_confidence, self.randomized_confidence)
        coverage = confidence.mean(dim=2)
        return {
            "confidence": confidence,
            "observed_confidence": obs_confidence,
            "randomized_confidence": self.randomized_confidence.clone(),
            "coverage": coverage,
        }

    def is_fully_inspected(self, threshold: float = 0.95, any_env: bool = False):
        inspected = self.coverage >= threshold  # (O, E)

        # Check for each env whether all objects are inspected
        per_env_done = inspected.all(dim=0)  # (E,)

        if any_env:
            # True if at least one env has all objects inspected
            return per_env_done.any()
        else:
            # True if every env has all objects inspected
            return per_env_done.all()

    def plot(self, env_id: int = 0, ax: Optional[plt.Axes] = None):
        """
        Plot rays and object coverage for one environment.
        - Observed this step (lidar) = red
        - Randomized (previous) = blue
        - Uninspected = gray
        """
        import numpy as np

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        else:
            fig = None

        env_id = int(env_id)

        # --- Get ray results from ray caster ---
        if not hasattr(self.ray_caster_builder, "_ray_tracing_results"):
            ax.set_aspect("equal", adjustable="box")
            if fig is not None:
                plt.show()
            return fig, ax

        results = self.ray_caster_builder._ray_tracing_results

        hits = results["hits"][env_id]  # (R,2)
        dists = results["dists"][env_id]  # (R,)
        sensor_pos = results["sensor_pos"][env_id].cpu().numpy()  # (2,)
        dirs = results["dirs"][env_id].cpu().numpy()  # (R,2)
        R_total = hits.shape[0]

        # --- Select only the central rays used for coverage ---
        k = self.cfg.num_center_rays_coverage
        assert k % 2 == 1, "num_center_rays_coverage should be odd"
        center = R_total // 2
        half_k = k // 2
        ray_indices = np.arange(center - half_k, center + half_k + 1)

        hits_sel = hits[ray_indices].cpu().numpy()
        dists_sel = dists[ray_indices].cpu().numpy()
        dirs_sel = dirs[ray_indices]

        # --- Plot only those rays in red ---
        for r in range(k):
            d = dists_sel[r]
            d_plot = (
                self.ray_caster_builder.max_distance
                if not np.isfinite(d) or d > self.ray_caster_builder.max_distance
                else d
            )
            end = sensor_pos + dirs_sel[r] * d_plot
            ax.plot(
                [sensor_pos[0], end[0]],
                [sensor_pos[1], end[1]],
                ":",
                linewidth=0.8,
                c="red",
            )

        # --- Plot object contours and coverage ---
        M = self.object_interest_builder.cfg.num_objects_per_env
        contour = self.object_interest_builder.obj_prop["contour"][env_id]  # (M,P,2)

        for m in range(M):
            pts = contour[m].cpu().numpy()
            conf_obs = self.observed_confidence[env_id, m].cpu().numpy()
            conf_rand = self.randomized_confidence[env_id, m].cpu().numpy()

            # masks
            observed_mask = conf_obs > 0.5
            randomized_mask = (conf_rand > 0.5) & ~observed_mask
            uninspected_mask = ~(observed_mask | randomized_mask)

            # scatter
            ax.scatter(
                pts[uninspected_mask, 0],
                pts[uninspected_mask, 1],
                c="lightgray",
                s=40,
                edgecolor="k",
                alpha=0.3,
                label="uninspected" if m == 0 else "",
            )
            ax.scatter(
                pts[observed_mask, 0],
                pts[observed_mask, 1],
                c="red",
                s=40,
                edgecolor="k",
                label="observed" if m == 0 else "",
            )
            ax.scatter(
                pts[randomized_mask, 0],
                pts[randomized_mask, 1],
                c="blue",
                s=40,
                edgecolor="k",
                label="randomized" if m == 0 else "",
            )

            ax.plot(pts[:, 0], pts[:, 1], "k-", linewidth=0.5, alpha=0.5)

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Env {env_id} Coverage: {self.coverage[env_id].cpu().numpy()}")
        ax.legend()
        if fig is not None:
            plt.show()
        return fig, ax

    def _calculate_face_confidence(self):
        face_indices = self.object_interest_builder.obj_prop["face_indices"]  # (E,M,F,pts_per_edge)
        F = face_indices.shape[2]

        # --- Compute mean confidence per face and store ---
        face_conf = torch.gather(
            self.confidence.unsqueeze(2).expand(-1, -1, F, -1),  # [E,M,F,P]
            3,
            face_indices
        ).mean(dim=-1)  # [E,M,F]
        self.face_confidence = face_conf  # Store for ranking later