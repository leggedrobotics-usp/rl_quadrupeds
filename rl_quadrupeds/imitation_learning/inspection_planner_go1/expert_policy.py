from typing import Optional

import matplotlib.pyplot as plt
import torch

from cfg import Cfg
from coverage import ObjectCoverage
from object_interest import ObjectInterestBuilder
from robot import RobotBuilder

class ExpertPolicyCfg(Cfg):
    def _define_defaults(self):
        self.offset_from_face = 0.8  # meters, distance to stand back from face centroid
        self.max_action_xy = (0.5, 0.5) # x,y max distance per step (m)
        self.max_action_heading = torch.pi  # max heading change per step (rad)

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

        self.ranges = {
            'x': self.cfg.max_action_xy[0],
            'y': self.cfg.max_action_xy[1],
            'heading': self.cfg.max_action_heading,
        }

    def build(self):
        pass

    def update(self):
        self.current_best_pose_b = self.compute_best_robot_pose()
        self.inspection_action = self.compute_inspection_action()

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        normalized_action = action.clone()
        normalized_action[:, 0] /= self.ranges['x']
        normalized_action[:, 1] /= self.ranges['y']
        normalized_action[:, 2] /= self.ranges['heading']
        return normalized_action

    def denormalize(self, normalized_action: torch.Tensor) -> torch.Tensor:
        action = normalized_action.clone()
        action[:, 0] *= self.ranges['x']
        action[:, 1] *= self.ranges['y']
        action[:, 2] *= self.ranges['heading']
        return action

    def get_action(self, normalized: bool = True) -> torch.Tensor:
        if not normalized:
            return {
                "robot_pose": self.current_best_pose_b,  # (E, M, 3)
                "inspection_action": self.inspection_action,  # (E,)
            }

        normalized_action = self.normalize(self.current_best_pose_b)
        return {
            "robot_pose": normalized_action,  # (E, M, 3)
            "inspection_action": self.inspection_action,  # (E,)
        }

    @torch.no_grad()
    def compute_best_robot_pose(self) -> torch.Tensor:
        """
        Compute the best robot pose in the robot frame (dx, dy, dθ),
        now ensuring the heading points toward the selected face.
        Returns: (E, 3) relative actions
        """
        # --- world-frame target pose (unconstrained) ---
        self.possible_viewpoints_w = self._generate_possible_viewpoints()
        self.unconstrained_best_pose_w, self.selected_face_centroids = self._rank_possible_viewpoints(
            self.possible_viewpoints_w
        )  # (E, 3), (E, 2)

        # --- apply constraints ---
        self.current_best_pose_w, self.unpushed_best_pose_w = self._apply_action_limits(
            self.unconstrained_best_pose_w,
            self.selected_face_centroids
        )  # (E, 3)

        # --- delta in world-frame ---
        robot_pose_world = self.robot_builder.viewpoint
        dx = self.current_best_pose_w[:, 0] - robot_pose_world[:, 0]
        dy = self.current_best_pose_w[:, 1] - robot_pose_world[:, 1]
        dθ = self.current_best_pose_w[:, 2] - robot_pose_world[:, 2]

        return torch.stack([dx, dy, dθ], dim=-1)  # (E, 3)

    @torch.no_grad()
    def compute_inspection_action(self) -> torch.Tensor:
        """
        Decide whether the current robot pose yields *new* coverage
        that comes from observed_confidence (not preexisting randomized_confidence).
        Returns +1 if observed_confidence increases coverage, else -1.
        """
        obs_conf = self.object_coverage.observed_confidence  # (E,M,P)
        rand_conf = self.object_coverage.randomized_confidence  # (E,M,P)

        # "New gain" happens only if obs_conf marks points that rand_conf did not already cover
        new_points = (obs_conf > 0.5) & (rand_conf <= 0.5)   # (E,M,P)

        # Coverage gain exists if ANY new point is added
        gain = new_points.any(dim=2).float()  # (E,M)
        has_gain = gain.any(dim=1).float()    # (E,)

        action_sign = (2 * has_gain - 1).unsqueeze(-1)  # (E,1), values ∈ {-1,1}
        return action_sign

    def get_unpushed_action(self, normalized: bool = True) -> torch.Tensor:
        """
        Get the unpushed best robot pose.
        Used for debugging and visualization.
        """
        robot_pose_world = self.robot_builder.viewpoint
        dmove = self.unpushed_best_pose_w - robot_pose_world  # (E,3)

        if normalized:
            dmove = self.normalize(dmove)

        return dmove

    def plot(
        self,
        env_id: int = 0,
        ax: Optional[plt.Axes] = None,
        scale: float = 0.5,
    ):
        """
        Plot object contours and the best robot poses (as arrows) for one environment.
        - Object contours drawn in black.
        - Unconstrained best pose drawn in blue arrows (lighter).
        - Constrained best pose drawn in orange arrows.
        - All possible viewpoints drawn in purple arrows (semi-transparent).
        - Inspection action drawn as a green (inspect=1) or red (skip=-1) dot.
        """
        import matplotlib.lines as mlines

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

        # --- Ensure we have both unconstrained + constrained poses ---
        if not hasattr(self, "unconstrained_best_pose_w") or self.unconstrained_best_pose_w is None:
            self.compute_best_robot_pose()

        poses_robot = self.robot_builder.viewpoint
        poses_constrained = self.current_best_pose_w
        poses_unconstrained = self.unconstrained_best_pose_w

        poses_env_r = poses_robot[env_id]          # (3,)
        poses_env_c = poses_constrained[env_id]  # (3,)
        poses_env_u = poses_unconstrained[env_id]  # (3,)

        # --- Constrained pose (orange solid) ---
        xs_c, ys_c, yaws_c = poses_env_c[0].cpu(), poses_env_c[1].cpu(), poses_env_c[2].cpu()
        dx_c, dy_c = torch.cos(yaws_c) * scale, torch.sin(yaws_c) * scale
        ax.quiver(xs_c, ys_c, dx_c, dy_c,
                angles="xy", scale_units="xy", scale=1,
                color="orange", width=0.004, label="Constrained Pose")

        # --- Unconstrained pose (blue, semi-transparent) ---
        xs_u, ys_u, yaws_u = poses_env_u[0].cpu(), poses_env_u[1].cpu(), poses_env_u[2].cpu()
        dx_u, dy_u = torch.cos(yaws_u) * scale, torch.sin(yaws_u) * scale
        ax.quiver(xs_u, ys_u, dx_u, dy_u,
                angles="xy", scale_units="xy", scale=1,
                color="blue", alpha=0.6, width=0.004, label="Unconstrained Pose")

        # --- All possible viewpoints (purple semi-transparent) ---
        # if hasattr(self, "possible_viewpoints_w") and self.possible_viewpoints_w is not None:
        #     poses_possible = self.possible_viewpoints_w[env_id]  # (M,F,3)
        #     for m in range(poses_possible.shape[0]):
        #         for f in range(poses_possible.shape[1]):
        #             pose = poses_possible[m, f]
        #             xs_p, ys_p, yaws_p = pose[0].cpu(), pose[1].cpu(), pose[2].cpu()
        #             dx_p, dy_p = torch.cos(yaws_p) * scale, torch.sin(yaws_p) * scale
        #             ax.quiver(xs_p, ys_p, dx_p, dy_p,
        #                       angles="xy", scale_units="xy", scale=1,
        #                       color="purple", width=0.003,
        #                       alpha=0.5)

        #     # Add legend entry with dashed line style (using Line2D)
        #     proxy = mlines.Line2D([], [], color="purple", linestyle="--", alpha=0.5,
        #                           label="Possible Viewpoints")
        #     ax.legend(handles=ax.get_legend_handles_labels()[0] + [proxy])

        # --- Overlay inspection action ---
        if self.inspection_action is None:
            action = self.compute_inspection_action()
        else:
            action = self.inspection_action
        action_env = action[env_id].item()  # scalar {-1, 1}

        color = "green" if action_env > 0 else "red"
        xs_r, ys_r, yaws_r = poses_env_r[0].cpu(), poses_env_r[1].cpu(), poses_env_r[2].cpu()
        ax.scatter(xs_r, ys_r, s=120, c=color, marker="o",
                edgecolors="k", linewidths=0.8,
                label=f"Inspection: {'YES' if action_env > 0 else 'NO'}")

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Env {env_id} Best Inspection Poses")
        ax.legend()

        if fig is not None:
            plt.show()
        return fig, ax

    def _transform_axes(self, viewpoint: torch.Tensor, apply_to_heading: bool = False) -> torch.Tensor:
        """
        Adapt axes from this simulator (x is right, y is up, heading=0 is x-axis)
        to the IsaacSim pattern (x is down, y is right, heading=0 is x-axis).

        apply_to_heading: if True, rotate heading by +90 degrees. It should not be applied
        if the viewpoint is an action delta (dx, dy, dheading).
        """
        x_old, y_old, h_old = viewpoint[:, 0], viewpoint[:, 1], viewpoint[:, 2]
        x_new = -y_old
        y_new = x_old

        h_new = h_old
        if apply_to_heading:
            raise ValueError("apply_to_heading should be False for action deltas.")
        return torch.stack([x_new, y_new, h_new], dim=-1)

    def _detransform_axes(self, viewpoint: torch.Tensor, apply_to_heading: bool = False) -> torch.Tensor:
        """
        Inverse of _transform_axes.
        """
        x_new, y_new, h_new = viewpoint[:, 0], viewpoint[:, 1], viewpoint[:, 2]
        x_old = y_new
        y_old = -x_new

        h_old = h_new
        if apply_to_heading:
            raise ValueError("apply_to_heading should be False for action deltas.")
        return torch.stack([x_old, y_old, h_old], dim=-1)

    @torch.no_grad()
    def _check_inside_or_too_close(self, adjusted_xy, contour, edge_start, edge_end, min_dist, eps=1e-12):
        """Helper: return mask of violators (inside or < min_dist)."""
        E = adjusted_xy.shape[0]
        px = adjusted_xy[:, 0].unsqueeze(1).unsqueeze(2)
        py = adjusted_xy[:, 1].unsqueeze(1).unsqueeze(2)
        x0, y0 = edge_start[..., 0], edge_start[..., 1]
        x1, y1 = edge_end[..., 0], edge_end[..., 1]

        cond1 = (y0 > py) != (y1 > py)
        denom = (y1 - y0).clone()
        denom[denom.abs() < 1e-9] = 1e-9
        x_inter = x0 + (py - y0) * (x1 - x0) / denom
        cond2 = px < x_inter
        crossings = (cond1 & cond2).to(torch.int64)
        cross_count = crossings.sum(dim=-1)
        inside_any = (cross_count % 2 == 1).any(dim=1)  # (E,)

        # distances
        p = adjusted_xy[:, None, None, :]
        ab = edge_end - edge_start
        ap = p - edge_start
        ab2 = (ab * ab).sum(dim=-1).clamp(min=eps)
        t = ((ap * ab).sum(dim=-1) / ab2).clamp(0.0, 1.0)[..., None]
        closest = edge_start + t * ab
        dist2 = ((p - closest) ** 2).sum(dim=-1)
        min_dist2, _ = dist2.view(E, -1).min(dim=1)
        too_close = torch.sqrt(min_dist2 + eps) < (min_dist - 1e-6)

        return inside_any | too_close

    @torch.no_grad()
    def _rank_possible_viewpoints(
        self, possible_viewpoints: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized selection of viewpoints using soft utility weighting.

        Returns:
            best_viewpoints: [E, 3] world-frame pose per environment
            face_centroids: [E, 2] centroid of the selected face
        """
        E, M, F, _ = possible_viewpoints.shape
        device = possible_viewpoints.device
        eps = 1e-8

        # --- Face centroids ---
        contour = self.object_interest_builder.obj_prop["contour"]        # [E,M,P,2]
        face_indices = self.object_interest_builder.obj_prop["face_indices"]  # [E,M,F,pts]

        face_pts = torch.gather(
            contour.unsqueeze(2).expand(-1, -1, F, -1, -1),               # [E,M,F,P,2]
            3,
            face_indices.unsqueeze(-1).expand(-1, -1, -1, -1, 2)          # [E,M,F,pts,2]
        )
        centroids = face_pts.mean(dim=3)  # [E,M,F,2]

        # --- Distances robot → face centroid ---
        robot_pose = self.robot_builder.viewpoint[:, :2].unsqueeze(1).unsqueeze(2)  # [E,1,1,2]
        dist = torch.norm(centroids - robot_pose, dim=-1).clamp(min=1e-6)           # [E,M,F]

        # --- Utility ---
        face_conf = self.object_coverage.face_confidence  # [E,M,F]
        remaining = (1.0 - face_conf).clamp(min=0.0)
        utility = remaining / (dist + eps)  # [E,M,F]

        # --- Flatten candidates ---
        C = M * F
        utility_flat = utility.view(E, C)            # [E,C]
        candidates_flat = possible_viewpoints.view(E, C, 3)  # [E,C,3]
        centroids_flat = centroids.view(E, C, 2)     # [E,C,2]

        # --- Softmax weighting ---
        beta = getattr(self.cfg, "selection_beta", 10.0)
        util_norm = utility_flat - utility_flat.max(dim=1, keepdim=True).values
        weights = torch.softmax(beta * util_norm, dim=1)  # [E,C]

        # --- Hard selection (argmax) ---
        topidx = utility_flat.argmax(dim=1)   # [E]
        hard_viewpoints = candidates_flat[torch.arange(E, device=device), topidx]  # [E,3]
        hard_centroids = centroids_flat[torch.arange(E, device=device), topidx]    # [E,2]

        # --- Soft selection (weighted average) ---
        w = weights.unsqueeze(-1)  # [E,C,1]

        # XY avg
        avg_xy = (w * candidates_flat[:, :, :2]).sum(dim=1)  # [E,2]

        # Yaw avg (via atan2 of weighted sin/cos)
        yaws = candidates_flat[:, :, 2]  # [E,C]
        avg_sin = (weights * torch.sin(yaws)).sum(dim=1)
        avg_cos = (weights * torch.cos(yaws)).sum(dim=1)
        avg_yaw = torch.atan2(avg_sin, avg_cos)  # [E]

        soft_viewpoints = torch.cat([avg_xy, avg_yaw.unsqueeze(-1)], dim=1)  # [E,3]
        soft_centroids = (w * centroids_flat).sum(dim=1)  # [E,2]

        # --- Confidence-based switch (vectorized) ---
        hard_threshold = getattr(self.cfg, "hard_select_threshold", 0.9)
        topw = weights.max(dim=1).values  # [E]
        hard_mask = topw > hard_threshold  # [E] bool

        best_viewpoints = torch.where(
            hard_mask.unsqueeze(-1), hard_viewpoints, soft_viewpoints
        )
        selected_centroids = torch.where(
            hard_mask.unsqueeze(-1), hard_centroids, soft_centroids
        )

        return best_viewpoints, selected_centroids

    @torch.no_grad()
    def _generate_possible_viewpoints(self) -> torch.Tensor:
        """
        Generate candidate robot poses for inspection.
        Ensures the robot is positioned in front of each face centroid,
        oriented directly towards the face centroid.

        Improvements:
         - Add clearance to offset_from_face so initial candidates are less likely to overlap.
         - Returns [E, M, F, 3].
        """
        device = self.device
        contour = self.object_interest_builder.obj_prop["contour"]   # [E,M,P,2]
        face_indices = self.object_interest_builder.obj_prop["face_indices"]  # [E,M,F,pts_per_edge]

        E, M, P, _ = contour.shape
        _, _, F, pts_per_edge = face_indices.shape

        # --- Compute face centroids ---
        face_pts = torch.gather(
            contour.unsqueeze(2).expand(-1, -1, F, -1, -1),  # [E,M,F,P,2]
            3,
            face_indices.unsqueeze(-1).expand(-1, -1, -1, -1, 2)
        )  # [E,M,F,pts_per_edge,2]
        centroids = face_pts.mean(dim=3)  # [E,M,F,2]

        # --- Compute outward normals: centroid to object center ---
        obj_center = contour.mean(dim=2).unsqueeze(2)  # [E,M,1,2]
        normals = centroids - obj_center               # points from object center to face centroid
        norm_mag = torch.norm(normals, dim=-1, keepdim=True).clamp(min=1e-8)
        normals = normals / norm_mag                   # normalized outward vectors

        # --- Clearance: combine configured offset + min_dist or robot radius if available ---
        # Use cfg.offset_from_face as base and add a safety term (min_dist)
        min_clearance = getattr(self.cfg, "min_clearance", 0.2)  # optional additional clearance
        effective_offset = (self.cfg.offset_from_face + min_clearance)

        # --- Candidate robot positions: offset along normal ---
        pos_xy = centroids + normals * effective_offset  # [E,M,F,2]

        # --- Yaw: point towards face centroid ---
        vec_to_face = centroids - pos_xy
        yaw = torch.atan2(vec_to_face[...,1], vec_to_face[...,0])  # [E,M,F]

        possible_viewpoints = torch.cat([pos_xy, yaw.unsqueeze(-1)], dim=-1)  # [E,M,F,3]
        return possible_viewpoints


    @torch.no_grad()
    def _apply_action_limits(self, target_pose_w: torch.Tensor, face_centroids: torch.Tensor) -> torch.Tensor:
        """
        Clamp the target world-frame pose relative to the robot's current world pose.
        Ensure final pose is valid (not overlapping objects).
        If invalid, push outward from objects.
        Also: orient heading toward selected face while respecting heading limits.
        Returns tuple (pushed_pose_w, clipped_pose_w)
        """
        robot_pose_w = self.robot_builder.viewpoint  # [E,3]

        # --- raw displacement ---
        dx = target_pose_w[:, 0] - robot_pose_w[:, 0]
        dy = target_pose_w[:, 1] - robot_pose_w[:, 1]
        dθ = target_pose_w[:, 2] - robot_pose_w[:, 2]

        # normalize heading diff to [-pi, pi]
        dθ = (dθ + torch.pi) % (2 * torch.pi) - torch.pi

        # --- clip displacement ---
        dx_clipped = dx.clamp(-self.cfg.max_action_xy[0], self.cfg.max_action_xy[0])
        dy_clipped = dy.clamp(-self.cfg.max_action_xy[1], self.cfg.max_action_xy[1])
        dθ_clipped = dθ.clamp(-self.cfg.max_action_heading, self.cfg.max_action_heading)

        clipped_pose_w = torch.stack([
            robot_pose_w[:, 0] + dx_clipped,
            robot_pose_w[:, 1] + dy_clipped,
            robot_pose_w[:, 2] + dθ_clipped,
        ], dim=-1)

        # --- orient heading toward selected face but respect heading limits ---
        # compute desired heading from clipped position toward face_centroids
        vec_to_face = face_centroids - clipped_pose_w[:, :2]  # [E,2]
        desired_heading = torch.atan2(vec_to_face[:, 1], vec_to_face[:, 0])  # absolute heading

        # heading change relative to robot's current heading - we clamp the delta
        robot_heading = robot_pose_w[:, 2]
        heading_diff = (desired_heading - robot_heading + torch.pi) % (2 * torch.pi) - torch.pi
        heading_diff_clamped = heading_diff.clamp(-self.cfg.max_action_heading, self.cfg.max_action_heading)
        clipped_pose_w[:, 2] = robot_heading + heading_diff_clamped

        pushed_pose_w = self._push_outside_objects(
            pose_w=clipped_pose_w,
            robot_xy=robot_pose_w[:, :2]
        )

        return pushed_pose_w, clipped_pose_w


    @torch.no_grad()
    def _push_outside_objects(
        self,
        pose_w: torch.Tensor,
        robot_xy: Optional[torch.Tensor] = None,
        min_dist: float = 0.8,
        max_push_iters: int = 10,
        push_step: float = 0.05,
    ) -> torch.Tensor:
        """
        Project candidate poses back into the feasible set:
        - Inside axis-aligned action box (|dx|<=Lx, |dy|<=Ly).
        - Outside every object polygon (inflated by min_dist).
        - If a candidate is inside/too close, iteratively push outward along the face normal
          until it is free or max iterations reached.
        Returns adjusted poses (E,3).
        """
        device, dtype = pose_w.device, pose_w.dtype
        E = pose_w.shape[0]

        if robot_xy is None:
            robot_xy = self.robot_builder.viewpoint[:, :2]  # (E,2)

        max_x, max_y = self.cfg.max_action_xy
        # step box bounds
        lo = robot_xy - torch.tensor([max_x, max_y], device=device, dtype=dtype)
        hi = robot_xy + torch.tensor([max_x, max_y], device=device, dtype=dtype)

        # clamp candidate to box (xy only)
        cand_xy = pose_w[:, :2].clamp(lo, hi)  # (E,2)

        # handle polygon edges
        contour = self.object_interest_builder.obj_prop["contour"]  # (E,M,P,2)
        E_c, M, P, _ = contour.shape
        assert E_c == E, "Environment count mismatch"

        # Build edge start / end arrays for _check_inside_or_too_close
        # edges: shape (E, M, P, 2) and (E, M, P, 2) where last edge closes polygon
        edge_start = contour
        edge_end = torch.cat([contour[:, :, 1:, :], contour[:, :, :1, :]], dim=2)

        # quick test: which cand_xy are violating (inside polygon or too close)
        viol_mask = self._check_inside_or_too_close(
            adjusted_xy=cand_xy,
            contour=contour,
            edge_start=edge_start,
            edge_end=edge_end,
            min_dist=min_dist
        )  # (E,) boolean mask

        # if none violate, return final poses with original headings
        if not viol_mask.any():
            adjusted = torch.cat([cand_xy, pose_w[:, 2:3]], dim=-1)
            return adjusted

        # For violators, attempt iterative outward push along the nearest outward normal
        # Compute object centers to get normal directions per object (E,M,1,2)
        obj_centers = contour.mean(dim=2, keepdim=True)  # (E,M,1,2)

        # We'll compute per-env the nearest object (for simplicity), then push away from that object center.
        # Compute distance from cand_xy to each object center
        px = cand_xy[:, None, None, :]  # (E,1,1,2)
        centers = obj_centers  # (E,M,1,2)
        dcent = (px - centers).squeeze(2)  # (E,M,2)
        dcent_norm = torch.norm(dcent, dim=-1) + 1e-9  # (E,M)
        nearest_obj_idx = dcent_norm.argmin(dim=1)  # (E,)

        adjusted_xy = cand_xy.clone()
        for it in range(max_push_iters):
            # recompute viol_mask with current adjusted_xy
            viol_mask = self._check_inside_or_too_close(
                adjusted_xy=adjusted_xy,
                contour=contour,
                edge_start=edge_start,
                edge_end=edge_end,
                min_dist=min_dist
            )
            if not viol_mask.any():
                break

            # for each violating env, compute push direction: from corresponding obj_center to point
            idx = torch.nonzero(viol_mask, as_tuple=False).squeeze(-1)
            # pick centers for those envs & their nearest object
            nearest_idx = nearest_obj_idx[idx]  # (K,)
            centers_for_idx = obj_centers[idx, nearest_idx, 0, :]  # (K,2)
            pts = adjusted_xy[idx]  # (K,2)
            push_dirs = pts - centers_for_idx  # away from object center
            push_norm = torch.norm(push_dirs, dim=-1, keepdim=True).clamp(min=1e-8)
            push_dirs = push_dirs / push_norm

            # apply small step outward
            adjusted_xy[idx] = adjusted_xy[idx] + push_dirs * push_step

            # also enforce bounds inside action box
            adjusted_xy = adjusted_xy.clamp(lo, hi)

        # final assemble
        adjusted_pose = torch.cat([adjusted_xy, pose_w[:, 2:3]], dim=-1)
        return adjusted_pose