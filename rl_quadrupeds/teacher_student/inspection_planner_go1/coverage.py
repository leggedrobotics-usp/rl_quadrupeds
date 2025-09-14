from typing import Optional

import matplotlib.pyplot as plt
import torch

from object_interest import ObjectInterestBuilder
from ray_caster import RayCasterBuilder
from robot import RobotBuilder

class ObjectCoverageCfg:
    def __init__(self):
        self.num_center_rays_coverage = 7  # rays used for coverage update (odd number, centered)

        self.hit_tolerance = 0.1  # meters, radius around contour point to consider "observed"
        self.max_hit_distance = 2.0  # meters, max distance to consider a hit valid

        # --- Randomization parameters ---
        self.min_clusters = 0
        self.max_clusters = 5
        self.min_cluster_radius = 0.05
        self.max_cluster_radius = 0.2

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
        P = self.object_interest_builder.cfg.contour_points_per_edge * 4  # full contour

        # Keep components separate
        self.observed_confidence = torch.zeros((E, M, P), device=self.device)
        self.randomized_confidence = torch.zeros((E, M, P), device=self.device)

        # Combined
        self.confidence = torch.zeros((E, M, P), device=self.device)
        self.coverage = torch.zeros((E, M), device=self.device)

    @torch.no_grad()
    def update(self):
        """Vectorized update of observed confidence and total coverage based on lidar hits."""
        E = self.num_envs
        M = self.object_interest_builder.cfg.num_objects_per_env
        contour = self.object_interest_builder.obj_prop["contour"]  # (E, M, P, 2)

        # Lidar scan
        scan = self.ray_caster_builder.get_ray_data(
            return_hits=True,
            return_labels=True,
            flatten=False,
            normalize=False,
            num_rays=self.cfg.num_center_rays_coverage,
        )
        hits = scan[:, :, 0:2]  # (E, N, 2)
        labels = scan[:, :, 2].long()  # (E, N)

        # Sensor positions
        sensor_pos = self.ray_caster_builder._ray_tracing_results["sensor_pos"][:, :2]  # (E, 2)

        cell_radius = self.cfg.hit_tolerance
        max_hit_distance = self.cfg.max_hit_distance

        # --- Broadcast hits per object ---
        P = contour.shape[2]  # number of points per object
        object_mask = (
            torch.arange(M, device=self.device).view(1, M, 1) == labels.unsqueeze(1)
        )  # (E, M, N)

        dist_to_sensor = torch.norm(
            hits.unsqueeze(1) - sensor_pos.unsqueeze(1).unsqueeze(2), dim=-1
        )  # (E, 1, N)
        valid_distance_mask = dist_to_sensor <= max_hit_distance  # (E, 1, N)
        valid_mask = object_mask & valid_distance_mask  # (E, M, N)

        counts = valid_mask.sum(dim=2)  # (E, M)
        if counts.sum() == 0:
            return

        # --- Compute distances from contour points to hits ---
        hits_exp = hits.unsqueeze(1).expand(E, M, hits.shape[1], 2)
        contour_exp = contour.unsqueeze(3)  # (E, M, P, 1, 2)
        diff = contour_exp - hits_exp.unsqueeze(2)  # (E, M, P, N, 2)
        dist2 = torch.sum(diff * diff, dim=-1)  # (E, M, P, N)
        dist2 = torch.nan_to_num(dist2, nan=1e6)

        hit_mask = dist2 <= (cell_radius**2)  # (E, M, P, N)

        valid_mask_exp = valid_mask.unsqueeze(2)  # (E, M, 1, N)
        observed = (hit_mask & valid_mask_exp).any(dim=-1).float()  # (E, M, P)

        # Update observed confidence
        self.observed_confidence[:] = torch.maximum(self.observed_confidence, observed)

        # Update combined confidence and coverage
        self.confidence[:] = torch.maximum(
            self.observed_confidence, self.randomized_confidence
        )
        self.coverage[:] = self.confidence.mean(dim=2)

    @torch.no_grad()
    def randomize_confidence(self):
        """Simulate previous inspections by marking clusters of points as observed."""
        contour = self.object_interest_builder.obj_prop["contour"]  # (E,M,P,2)
        E, M, P, _ = contour.shape

        # number of clusters per object
        num_clusters = torch.randint(
            self.cfg.min_clusters, self.cfg.max_clusters + 1, (E, M), device=self.device
        )

        for k in range(self.cfg.max_clusters):
            mask = num_clusters > k  # (E,M)
            if not mask.any():
                continue

            # pick random centers among contour points
            center_idx = torch.randint(0, P, (E, M), device=self.device)
            centers = contour[
                torch.arange(E)[:, None], torch.arange(M)[None, :], center_idx
            ]  # (E,M,2)

            # random radius
            radius = torch.empty((E, M), device=self.device).uniform_(
                self.cfg.min_cluster_radius, self.cfg.max_cluster_radius
            )

            # distances to all points
            dists = torch.norm(contour - centers.unsqueeze(2), dim=-1)  # (E,M,P)

            # mark as randomized where inside radius
            cluster_mask = dists <= radius.unsqueeze(-1)  # (E,M,P)
            cluster_mask &= mask.unsqueeze(-1)  # apply only for active objs

            self.randomized_confidence = torch.maximum(
                self.randomized_confidence, cluster_mask.float()
            )

        # Update combined confidence and coverage
        self.confidence[:] = torch.maximum(
            self.observed_confidence, self.randomized_confidence
        )
        self.coverage[:] = self.confidence.mean(dim=2)

    def get_observation(self, noise: bool = False):
        """Return object coverage as observation."""

        if noise:
            self.observed_confidence += torch.randn_like(self.observed_confidence) * self.cfg.noise_confidence_std
            self.observed_confidence = torch.clamp(self.observed_confidence, 0.0, 1.0)
            self.confidence = torch.maximum(self.observed_confidence, self.randomized_confidence)
            self.coverage = self.confidence.mean(dim=2)
            
        return {
            "confidence": self.confidence.clone(),
            "observed_confidence": self.observed_confidence.clone(),
            "randomized_confidence": self.randomized_confidence.clone(),
            "coverage": self.coverage.clone(),
        }

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