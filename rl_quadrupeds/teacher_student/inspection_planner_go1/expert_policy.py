from typing import Optional

import matplotlib.pyplot as plt
import torch

from coverage import ObjectCoverage
from object_interest import ObjectInterestBuilder
from robot import RobotBuilder

class ExpertPolicyCfg:
    def __init__(self):
        self.offset_from_face = 0.8  # meters, distance to stand back from face centroid
        self.max_action_xy = (1.5, 1.5) # x,y max distance per step (m)
        self.max_action_heading = 2*torch.pi  # max heading change per step (rad)

class ExpertPolicy:
    def __init__(
        self,
        cfg: ExpertPolicyCfg,
        object_coverage: ObjectCoverage,
        object_interest_builder: ObjectInterestBuilder,
        robot_builder: RobotBuilder,
        device: str,
        num_envs: int = 1,
    ):
        self.cfg = cfg
        self.object_coverage = object_coverage
        self.object_interest_builder = object_interest_builder
        self.robot_builder = robot_builder
        self.device = device
        self.num_envs = num_envs

    def build(self):
        pass

    def update(self):
        self.current_best_pose_b = self.compute_best_robot_pose()
        self.inspection_action = self.compute_inspection_action()

    def get_action(self, normalized: bool = True) -> torch.Tensor:
        if not normalized:
            return {
                "robot_pose": self.current_best_pose_b,  # (E, M, 3)
                "inspection_action": self.inspection_action,  # (E,)
            }

        normalized_action = self.current_best_pose_b.clone()
        normalized_action[:, 0] /= self.cfg.max_action_xy[0]
        normalized_action[:, 1] /= self.cfg.max_action_xy[1]
        normalized_action[:, 2] /= self.cfg.max_action_heading
        return {
            "robot_pose": normalized_action,  # (E, M, 3)
            "inspection_action": self.inspection_action,  # (E,)
        }

    @torch.no_grad()
    def compute_best_robot_pose(self) -> torch.Tensor:
        """
        Compute the best robot pose in the robot frame (dx, dy, dθ).
        Returns: (E, 3) relative actions
        """
        # --- world-frame target pose ---
        self.current_best_pose_w = self._rank_possible_viewpoints(
            self._generate_possible_viewpoints()
        )  # (E, 3)

        self.current_best_pose_w = self._apply_action_limits(
            self.current_best_pose_w
        ) # (E, 3)

        # --- relative transform: world → robot frame ---
        robot_pose_world = self.robot_builder.viewpoint
        dx = self.current_best_pose_w[:, 0] - robot_pose_world[:, 0]
        dy = self.current_best_pose_w[:, 1] - robot_pose_world[:, 1]
        dθ = self.current_best_pose_w[:, 2] - robot_pose_world[:, 2]

        cos_r = torch.cos(-robot_pose_world[:, 2])
        sin_r = torch.sin(-robot_pose_world[:, 2])
        dx_local = cos_r * dx - sin_r * dy
        dy_local = sin_r * dx + cos_r * dy

        relative_action = torch.stack([dx_local, dy_local, dθ], dim=-1)  # (E, 3)
        return relative_action

    @torch.no_grad()
    def compute_inspection_action(self) -> torch.Tensor:
        """
        Decide whether the current best pose is a good inspection pose.
        Reuses the existing robot_builder instead of creating a new one.
        """
        E = self.num_envs
        device = self.device
        threshold = self.object_interest_builder.cfg.contour_points_per_edge // 2

        # --- backup current viewpoint ---
        old_viewpoint = self.robot_builder.viewpoint.clone()

        # --- set candidate pose as robot viewpoint ---
        self.robot_builder.viewpoint = self.current_best_pose_w

        # --- cast rays and get raw labels ---
        data = self.robot_builder.ray_caster_builder.cast_rays(store_results=False)
        ray_data = self.robot_builder.ray_caster_builder.get_ray_data(
            data=data,
            return_hits=True,
            return_labels=True,
            flatten=False,
            normalize=False,
            num_rays=self.object_coverage.cfg.num_center_rays_coverage,
        )  # (E, R, 2): [dist, obj_id]

        obj_ids = ray_data[..., 1].long()  # (E, R)
        M = self.object_interest_builder.cfg.num_objects_per_env

        # --- count visible rays per object ---
        counts = torch.zeros((E, M), device=device, dtype=torch.long)
        for m in range(M):
            counts[:, m] = (obj_ids == m).sum(dim=1)

        inspect_mask = (counts >= threshold).any(dim=1).float()  # (E,)
        action_sign = (2 * inspect_mask - 1).unsqueeze(-1)       # (E,1)

        # --- restore old viewpoint ---
        self.robot_builder.viewpoint = old_viewpoint

        return action_sign

    def plot(
        self,
        env_id: int = 0,
        ax: Optional[plt.Axes] = None,
        scale: float = 0.5,
    ):
        """
        Plot object contours and the best robot poses (as arrows) for one environment.
        - Object contours drawn in black.
        - Robot pose arrows drawn in orange.
        - Inspection action drawn as a green (inspect=1) or red (skip=-1) dot at robot pose.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        else:
            fig = None

        env_id = int(env_id)

        # --- Plot object contours ---
        M = self.object_interest_builder.cfg.num_objects_per_env
        contour = self.object_interest_builder.obj_prop["contour"][env_id]  # (M,P,2)

        for m in range(M):
            pts = contour[m].cpu().numpy()
            ax.plot(pts[:, 0], pts[:, 1], "k-", linewidth=0.8, alpha=0.6)

        # --- Get poses (E,M,3) ---
        poses = self.current_best_pose_w
        if poses is None:
            poses = self.compute_best_robot_pose()
        poses_env = poses[env_id]  # (M,3)

        xs = poses_env[..., 0].reshape(-1).cpu()
        ys = poses_env[..., 1].reshape(-1).cpu()
        yaws = poses_env[..., 2].reshape(-1).cpu()

        dx = torch.cos(yaws) * scale
        dy = torch.sin(yaws) * scale

        # --- Plot robot poses as arrows ---
        ax.quiver(xs, ys, dx, dy,
                  angles="xy", scale_units="xy", scale=1,
                  color="orange", width=0.004, label="Expert Pose")

        # --- Overlay inspection action ---
        if self.inspection_action is None:
            action = self.compute_inspection_action()
        else:
            action = self.inspection_action
        action_env = action[env_id].item()  # scalar {-1, 1}

        color = "green" if action_env > 0 else "red"
        ax.scatter(xs.mean(), ys.mean(), s=120, c=color, marker="o", 
                   edgecolors="k", linewidths=0.8,
                   label=f"Inspection: {'YES' if action_env > 0 else 'NO'}")

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Env {env_id} Best Inspection Poses")
        ax.legend()

        if fig is not None:
            plt.show()
        return fig, ax

    @torch.no_grad()
    def _apply_action_limits(self, target_pose_w: torch.Tensor) -> torch.Tensor:
        """
        Clamp the target world-frame pose relative to the robot's current world pose,
        then push it outside objects if it lies inside or too close.
        """
        robot_pose_w = self.robot_builder.viewpoint  # [E, 3]

        # --- compute difference in world frame ---
        dx = target_pose_w[:, 0] - robot_pose_w[:, 0]
        dy = target_pose_w[:, 1] - robot_pose_w[:, 1]
        dθ = target_pose_w[:, 2] - robot_pose_w[:, 2]

        # --- clamp differences ---
        dx = torch.clamp(dx, -self.cfg.max_action_xy[0], self.cfg.max_action_xy[0])
        dy = torch.clamp(dy, -self.cfg.max_action_xy[1], self.cfg.max_action_xy[1])
        dθ = torch.clamp(dθ, -self.cfg.max_action_heading, self.cfg.max_action_heading)

        # --- reconstruct clamped world-frame pose ---
        clamped_pose_w = torch.stack([
            robot_pose_w[:, 0] + dx,
            robot_pose_w[:, 1] + dy,
            robot_pose_w[:, 2] + dθ,
        ], dim=-1)

        # --- push outside objects if needed ---
        pushed_pose_w = self._push_outside_objects(clamped_pose_w)
        return pushed_pose_w

    @torch.no_grad()
    def _push_outside_objects(self, pose_w: torch.Tensor, min_dist: float = 1.0) -> torch.Tensor:
        """
        Apply a repulsive push to keep poses outside the object contours.

        Args:
            pose_w: (E, 3) world-frame poses
            min_dist: minimum allowed distance to any object contour (meters)

        Returns:
            adjusted poses (E, 3)
        """
        device = pose_w.device
        E = pose_w.shape[0]

        # contour: (E, M, P, 2)
        contour = self.object_interest_builder.obj_prop["contour"]
        if contour is None:
            return pose_w  # nothing to do

        # ensure device
        contour = contour.to(device)

        # shapes
        # If there are no objects or no contour points, just return
        if contour.numel() == 0:
            return pose_w

        _, M, P, _ = contour.shape
        total_pts = M * P
        if total_pts == 0:
            return pose_w

        # flatten contour per environment: (E, M*P, 2)
        contour_flat = contour.view(E, total_pts, 2)  # (E, M*P, 2)

        # pose positions: (E, 1, 2) -> broadcast to (E, M*P, 2)
        pos = pose_w[:, :2].unsqueeze(1)  # (E,1,2)

        # compute squared distances: (E, M*P)
        dist2 = ((pos - contour_flat) ** 2).sum(dim=-1)  # (E, M*P)

        # get min distances and indices
        min_dist2, min_idx = dist2.min(dim=1)  # both (E,)

        # safe-cast index to long on correct device
        min_idx = min_idx.to(device=device, dtype=torch.long)

        # clamp indices to valid range (defensive)
        min_idx = torch.clamp(min_idx, 0, total_pts - 1)

        # gather nearest points: (E, 2)
        batch_idx = torch.arange(E, device=device)
        nearest_pts = contour_flat[batch_idx, min_idx]  # (E,2)

        # actual distance values
        min_dist_val = torch.sqrt(min_dist2 + 1e-12)  # (E,)

        # direction from nearest contour point to pose: pose - nearest_pt
        dir_vec = pos.squeeze(1) - nearest_pts  # (E,2)

        # if dir_vec is very small (on top), choose outward normal approx:
        dir_norm = torch.norm(dir_vec, dim=-1, keepdim=True)  # (E,1)
        small_mask = (dir_norm.squeeze(-1) < 1e-6)  # (E,)

        # when tiny, pick a default direction away from object's centroid:
        if small_mask.any():
            # compute object centroids (E,2) as average of contour points (safe fallback)
            centroids = contour_flat.mean(dim=1)  # (E,2)
            fallback_dir = (pos.squeeze(1) - centroids)  # (E,2)
            # normalize fallback_dir
            fb_norm = torch.norm(fallback_dir, dim=-1, keepdim=True) + 1e-8
            fallback_unit = fallback_dir / fb_norm  # (E,2)
            dir_vec[small_mask] = fallback_unit[small_mask]

            # recompute dir_norm for those entries
            dir_norm = torch.norm(dir_vec, dim=-1, keepdim=True)

        # normalize direction
        dir_unit = dir_vec / (dir_norm + 1e-12)  # (E,2)

        # compute push magnitude for those below min_dist
        shortage = (min_dist - min_dist_val).clamp(min=0.0)  # (E,)
        need_push_mask = (shortage > 0.0)  # (E,)

        # nothing to do if no one needs push
        if not need_push_mask.any():
            return pose_w

        push_mag = shortage.unsqueeze(-1)  # (E,1)

        # new positions: apply push only where needed
        adjusted_xy = pose_w[:, :2].clone()
        adjusted_xy[need_push_mask] = (
            adjusted_xy[need_push_mask] + push_mag[need_push_mask] * dir_unit[need_push_mask]
        )

        adjusted_pose = torch.cat([adjusted_xy, pose_w[:, 2:]], dim=-1)  # (E,3)
        return adjusted_pose

    @torch.no_grad()
    def _rank_possible_viewpoints(self, possible_viewpoints: torch.Tensor) -> torch.Tensor:
        """
        For each environment, select a single viewpoint across all objects.
        Strategy:
          1. Find faces with lowest confidence across all objects in env.
          2. If multiple faces tie, pick the closest one to the robot.
        
        Args:
            possible_viewpoints: [E, M, F, 3]
        Returns:
            best_viewpoints: [E, 3]
        """
        confidence = self.object_coverage.confidence          # [E, M, P]
        face_indices = self.object_interest_builder.obj_prop["face_indices"]  # [E, M, F, pts_per_edge]
        E, M, F, pts_per_edge = face_indices.shape

        # --- Flatten face indices ---
        face_indices_flat = face_indices.view(E, M, F, -1)  # [E, M, F, pts_per_edge]

        # --- Gather confidence per face ---
        face_confidence = torch.gather(
            confidence.unsqueeze(2).expand(-1, -1, F, -1),  # [E, M, F, P]
            3,
            face_indices_flat
        )  # [E, M, F, pts_per_edge]

        # --- Average confidence per face ---
        face_confidence_mean = face_confidence.mean(dim=-1)  # [E, M, F]

        # --- Get global min confidence per environment ---
        min_conf, _ = face_confidence_mean.view(E, -1).min(dim=1, keepdim=True)  # [E,1]
        mask = (face_confidence_mean == min_conf.view(E, 1, 1))  # [E,M,F]

        # --- Compute distances to robot ---
        robot_pos = self.robot_builder.viewpoint[:, :2]  # [E,2]
        robot_pos_exp = robot_pos.view(E, 1, 1, 2)
        pos_xy = possible_viewpoints[..., :2]  # [E, M, F, 2]
        dist2 = ((pos_xy - robot_pos_exp) ** 2).sum(dim=-1)  # [E,M,F]

        # Apply mask: only consider min-conf faces
        dist2_masked = dist2.clone()
        dist2_masked[~mask] = float("inf")

        # --- Pick closest among min-conf faces globally ---
        flat_idx = dist2_masked.view(E, -1).argmin(dim=1)  # [E]
        m_idx = flat_idx // F
        f_idx = flat_idx % F

        best_viewpoints = possible_viewpoints[
            torch.arange(E, device=possible_viewpoints.device),
            m_idx,
            f_idx
        ]  # [E,3]

        return best_viewpoints

    @torch.no_grad()
    def _generate_possible_viewpoints(self) -> torch.Tensor:
        """
        Compute suggested robot XY and yaw to inspect each face of each object
        in every environment, in a vectorized way. Handles object heading (rotation).
        
        Returns:
            poses: (E, M, 4, 3) tensor [x, y, yaw]
                where E=num_envs, M=num_objects, 4 faces
        """
        device = self.device
        contour = self.object_interest_builder.obj_prop["contour"]   # (E,M,P,2)
        face_indices = self.object_interest_builder.obj_prop["face_indices"]  # (E,M,4,pts_per_edge)
        heading = self.object_interest_builder.obj_prop["heading"]   # (E,M)

        E, M, P, _ = contour.shape
        _, _, F, pts_per_edge = face_indices.shape  # F=4

        # --- gather face points ---
        idx = face_indices.unsqueeze(-1).expand(-1, -1, -1, -1, 2)  # (E, M, 4, 7, 2)

        # Temporarily unsqueeze contour at axis=2 to align dims
        contour_exp = contour.unsqueeze(2)  # (E, M, 1, P, 2)

        # Gather along P (dim=3 now)
        face_pts = torch.take_along_dim(contour_exp, idx, dim=3)  # (E, M, 4, 7, 2)

        # --- centroids per face ---
        centroids = face_pts.mean(dim=3)  # (E,M,4,2)

        # --- outward normals in canonical (object-local) frame ---
        base_normals = torch.tensor(
            [[0.0, -1.0],   # bottom
            [1.0,  0.0],   # right
            [0.0,  1.0],   # top
            [-1.0, 0.0]],  # left
            device=device
        )  # (4,2)

        # rotate normals by object heading
        cos_h = torch.cos(heading)  # (E,M)
        sin_h = torch.sin(heading)  # (E,M)
        rot = torch.stack([
            torch.stack([cos_h, -sin_h], dim=-1),
            torch.stack([sin_h,  cos_h], dim=-1)
        ], dim=-2)  # (E,M,2,2)

        # apply rotation: (E,M,4,2) = (E,M,4,2) @ (E,M,2,2)
        normals = torch.einsum(
            "fk,emkj->emfj",
            base_normals, rot
        )  # (E,M,4,2)

        # --- robot position offset ---
        pos_xy = centroids + normals * self.cfg.offset_from_face  # (E,M,4,2)

        # --- yaw facing centroid ---
        vec_to_target = centroids - pos_xy  # (E,M,4,2)
        yaw = torch.atan2(vec_to_target[..., 1], vec_to_target[..., 0])  # (E,M,4)

        # --- stack results ---
        return torch.cat([pos_xy, yaw.unsqueeze(-1)], dim=-1)  # (E,M,4,3)