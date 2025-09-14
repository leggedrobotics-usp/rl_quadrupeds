from typing import Sequence
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg

from quadrupeds_mdp.rewards.inspection import get_inspection_action
from quadrupeds_mdp.observations.ray_caster import lidar_scan


class ObjectInspectionCoverage(ManagerTermBase):
    """Refactored, vectorized manager that tracks inspection coverage of object contour points.

    Curriculum levels (per-env):
      1..4 -> inspect the 1 closest face, 2nd closest, ... , farthest (single face)
      5    -> inspect the 2 closest faces
      6    -> inspect the 3 closest faces
      7    -> inspect all 4 faces

    This implementation initializes all per-env buffers up-front, keeps a cached
    world-space contour, maintains a per-env randomized multi-face sequence for
    multi-face levels (5..7), and suggests a best robot pose per-env to approach
    the currently targeted face.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.env = env
        device = env.device

        # ----------------------- static object selection -----------------------
        self.valid_object_ids = torch.tensor([0], device=device, dtype=torch.long)
        self.valid_object_names = ["block1"]
        self.valid_objects_rigid_objects = [env.scene[name] for name in self.valid_object_names]

        # ----------------------- hyper-parameters -------------------------------
        self.max_hits_per_step = 96
        self.cell_radius = 0.2
        self.max_hit_distance = 2.0
        self.coverage_threshold = 0.95

        # ----------------------- contour geometry ------------------------------
        self._create_contour()

        # Face grouping and masks
        self.face_groups = self._assign_points_to_faces()  # list of 4 tensors
        self.num_faces = 4
        self.face_point_mask = torch.zeros((self.num_faces, self.num_contour_points), dtype=torch.bool, device=device)
        for f, idx in enumerate(self.face_groups):
            self.face_point_mask[f, idx] = True
        self.face_point_mask_f32 = self.face_point_mask.float()
        self.face_point_counts = self.face_point_mask_f32.sum(dim=1).clamp_min(1.0)  # (4,)

        self._level_pos_mask = torch.tensor(
            [
                [0, 0, 0, 0],
                # [1, 0, 0, 0],
                # [1, 1, 0, 0],
                # [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=torch.bool,
            device=device,
        )
        self.env.num_levels = self._level_pos_mask.shape[0] - 1  # exclude level 0

        # ----------------------- per-env buffers (initialized once) -------------
        E = env.num_envs
        M = self.valid_object_ids.shape[0]
        D = self.num_contour_points

        env.coverage = torch.zeros((E, M), dtype=torch.float32, device=device)
        env.coverage_prev = torch.zeros_like(env.coverage)

        env.confidence = torch.zeros((E, M, D), dtype=torch.float32, device=device)
        env.face_confidence = torch.zeros((env.num_envs, M, self.num_faces), device=env.device, dtype=torch.float32)
        env.uninspected_distances = torch.zeros((E, M, D), dtype=torch.float32, device=device)
        env.inference_points = self.local_contour.clone()  # local contour (D, 2)
        env._inspection_done = torch.zeros(E, dtype=torch.bool, device=device)

        env.best_robot_pose = torch.full((E, 3), float("nan"), device=device)

        # Caches
        self._cached_positions = torch.zeros((E, M, 2), device=device)
        self._cached_contour_points = torch.zeros((E, M, D, 2), device=device)
        self._cached_face_order = torch.zeros((E, self.num_faces), dtype=torch.long, device=device)
        self._cached_directions = torch.ones((E,), dtype=torch.long, device=device)  # +1/-1 preserved

        # Curriculum state per-environment
        env.env_levels = torch.ones(E, dtype=torch.long, device=device)  # default level 1

        # Per-env multi-face sequences & pointers
        # sequence entries are face indices (0..3) or -1 if unused
        self._multi_face_sequence = -torch.ones((E, self.num_faces), dtype=torch.long, device=device)
        self._multi_face_pos = torch.zeros((E,), dtype=torch.long, device=device)

        # ----------------------- other derived attributes ----------------------
        # keep these for quick access
        self.num_contour_points = D

        env._level_completion_counts = torch.zeros(
            (E, self.env.num_levels + 1),  # +1 because levels are 1..num_levels
            dtype=torch.long,
            device=device,
        )

    # ----------------------- public properties --------------------------------
    @property
    def levels(self) -> torch.Tensor:
        """Tensor view of current levels for all environments (shape: [E], dtype long)."""
        return self.env.env_levels

    # ----------------------- contour helpers ---------------------------------
    def _create_contour(self) -> None:
        device = self.env.device
        side = 0.5
        half = side / 2.0
        points_per_edge = 7

        corners = torch.tensor(
            [[-half, -half], [half, -half], [half, half], [-half, half]],
            device=device,
        )
        t = torch.linspace(0.0, 1.0, steps=points_per_edge, device=device).unsqueeze(-1)
        edges = [(1 - t) * corners[i] + t * corners[(i + 1) % 4] for i in range(4)]

        self.local_contour = torch.cat(edges, dim=0)  # (D, 2)
        self.num_contour_points = self.local_contour.shape[0]

    def _assign_points_to_faces(self):
        pts_per_edge = self.num_contour_points // 4
        groups = []
        for i in range(4):
            start = i * pts_per_edge
            end = start + pts_per_edge
            groups.append(torch.arange(start, end, device=self.env.device))
        return groups

    # ----------------------- lifecycle: reset --------------------------------
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # clear per-env buffers (but DO NOT zero cached world contour/positions)
        self.env.coverage[env_ids] = 0.0
        self.env.coverage_prev[env_ids] = 0.0
        self.env.confidence[env_ids] = 0.0
        self.env.face_confidence[env_ids] = 0.0
        self.env.uninspected_distances[env_ids] = 0.0
        self.env._inspection_done[env_ids] = False

        # Recompute the world-space contour for just these envs (force refresh)
        self._ensure_cached_contour_world(self.env, env_ids=env_ids, force=True)

        # compute and cache face order (based on robot position at reset)
        self._cached_face_order = self._compute_face_order_at_reset(env_ids)

        # initialize multi-face sequences/pointers for these envs
        self._init_multi_face_sequence(env_ids)

        # compute initial best robot pose suggestions
        self._compute_best_robot_pose(env_ids)

    # ----------------------- compute face order & best pose -------------------
    @torch.no_grad()
    def _compute_face_order_at_reset(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute face order such that:
          - Start from the closest face to the robot.
          - Then go sequentially through adjacent faces (wrap around).
          - If the environment already completed level 4 at least once,
            randomize the order direction (clockwise or counter-clockwise).
        """
        device = self.device
        E = self.env.num_envs

        # ensure cached contour world is ready
        self._ensure_cached_contour_world(self.env)

        # robot xy from the ray caster sensor
        robot_pos = self.env.scene.sensors["ray_caster"].data.pos_w[:, :2]  # (E,2)
        contour_w = self._cached_contour_points[:, 0, :, :]  # (E, D, 2)

        # distances per contour point
        dist_ED = torch.norm(contour_w - robot_pos.unsqueeze(1), dim=-1)  # (E, D)

        # mean distance per face: (E,4)
        face_mean_dist = (dist_ED @ self.face_point_mask_f32.t()) / self.face_point_counts

        # closest face index per env
        closest_face = torch.argmin(face_mean_dist, dim=1)  # (E,)

        # prebuild clockwise and counter-clockwise orderings starting from each face
        clockwise_orders = torch.stack(
            [torch.roll(torch.arange(4, device=device), -i) for i in range(4)], dim=0
        )  # (4,4)
        counter_orders = torch.stack(
            [torch.roll(torch.arange(3, -1, -1, device=device), -i) for i in range(4)], dim=0
        )  # (4,4)

        # check which envs already finished level 4 at least once
        completed_lvl4 = self.env._level_completion_counts[:, -1] > 0  # (E,)

        # random choice of direction for those envs
        rand_dirs = torch.randint(0, 2, (E,), device=device, dtype=torch.bool)  # True = clockwise, False = counter
        use_clockwise = ~completed_lvl4 | rand_dirs  # environments that must use clockwise

        # gather correct orders
        order_clockwise = clockwise_orders[closest_face]     # (E,4)
        order_counter = counter_orders[closest_face]         # (E,4)
        sorted_faces = torch.where(use_clockwise.unsqueeze(1), order_clockwise, order_counter)  # (E,4)

        # write into cached tensor for requested envs
        self._cached_face_order[env_ids] = sorted_faces[env_ids]
        return self._cached_face_order

    @torch.no_grad()
    def _compute_best_robot_pose(self, env_ids: torch.Tensor) -> None:
        """Compute a suggested robot XY and yaw for each env in `env_ids`.

        Uses the face id pointed to by the per-env pointer inside self._multi_face_sequence.
        If pointer is invalid, falls back to the first valid entry in the sequence.
        Only updates self.env.best_robot_pose for envs that have a valid target face.
        """
        if env_ids.numel() == 0:
            return

        device = self.device
        E_all = self.env.num_envs

        # face normals in object frame (axis-aligned)
        face_normals = torch.tensor(
            [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            device=device,
        )

        # Ensure caches
        self._ensure_cached_contour_world(self.env)
        if self._cached_face_order is None:
            self._cached_face_order = self._compute_face_order_at_reset(torch.arange(E_all, device=device))

        # Normalize env_ids to device/shape
        env_ids = env_ids.to(device=device, dtype=torch.long).view(-1)

        # Pull per-env sequences and pointers (K x 4) and (K,)
        seq = self._multi_face_sequence[env_ids]  # (K, 4) - face ids or -1
        ptrs = self._multi_face_pos[env_ids]      # (K,)

        # Valid entries and counts
        valid_mask = seq >= 0                     # (K, 4)
        valid_counts = valid_mask.sum(dim=1)      # (K,) long

        # Clamp pointers to last valid index when needed (prevent out-of-range)
        ptrs_clamped = torch.minimum(ptrs, (valid_counts - 1).clamp_min(0))

        # Row index to perform row-wise gather
        row_idx = torch.arange(seq.shape[0], device=device)

        # Select face id at the (possibly clamped) pointer
        seq_faces = seq[row_idx, ptrs_clamped]    # (K,)

        # For rows where the selected face is invalid but there exists a valid face, fallback to first valid
        first_valid_idx = valid_mask.int().argmax(dim=1)  # gives index of first True or 0 if none
        fallback_faces = seq[row_idx, first_valid_idx]    # (K,)
        seq_faces = torch.where((seq_faces < 0) & (valid_counts > 0), fallback_faces, seq_faces)

        # Now apply only to envs that actually have a valid face id
        good = seq_faces >= 0
        if not good.any():
            return

        good_rows = torch.nonzero(good, as_tuple=True)[0]    # indices into `seq` / `env_ids`
        chosen_envs = env_ids[good_rows]                    # global env indices to update
        chosen_faces = seq_faces[good_rows]                 # face ids for those envs

        # World-space contour for those envs
        contour_w = self._cached_contour_points[chosen_envs, 0, :, :]  # (Kg, D, 2)

        # Per-face point mask (Kg, D)
        chosen_mask = self.face_point_mask[chosen_faces]   # (Kg, D)
        chosen_mask_f = chosen_mask.float()
        counts = chosen_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)  # (Kg, 1)

        # Centroid of targeted face (Kg, 2)
        centroid = (contour_w * chosen_mask_f.unsqueeze(-1)).sum(dim=1) / counts

        # Desired robot offset (move 1.0m along face normal, then face the centroid)
        normals = face_normals[chosen_faces]               # (Kg, 2)
        offset = 1.5
        pos_xy = centroid + normals * offset              # (Kg, 2)

        vec_to_target = centroid - pos_xy
        yaws = torch.atan2(vec_to_target[:, 1], vec_to_target[:, 0])

        pose_out = torch.stack([pos_xy[:, 0], pos_xy[:, 1], yaws], dim=1)  # (Kg, 3)

        # Write back only for the chosen envs
        self.env.best_robot_pose[chosen_envs] = pose_out

    # ----------------------- curriculum update --------------------------------
    @torch.no_grad()
    def update_env_levels(self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor) -> None:
        """Update levels, re-seed the sequence according to the new level mask, and refresh best pose."""
        if move_up.any():
            idx = env_ids[move_up]
            self.env.env_levels[idx] = torch.clamp(self.env.env_levels[idx] + 1, 1, self.env.num_levels)
        if move_down.any():
            idx = env_ids[move_down]
            self.env.env_levels[idx] = torch.clamp(self.env.env_levels[idx] - 1, 1, self.env.num_levels)

        if env_ids.numel() > 0:
            self._ensure_cached_contour_world(self.env)
            if self._cached_face_order is None:
                self._cached_face_order = self._compute_face_order_at_reset(torch.arange(self.env.num_envs, device=self.device))

            # Rebuild the (distance-ordered) per-env sequence based on the new level's active faces
            self._init_multi_face_sequence(env_ids)
            # And compute the best pose for the first face in the (possibly new) sequence
            self._compute_best_robot_pose(env_ids)

    # ----------------------- main step call ----------------------------------
        # ----------------------- main step call ----------------------------------
    @torch.no_grad()
    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        lidar_data = lidar_scan(
            env=env,
            sensor_cfg=sensor_cfg,
            fill_value=5.,
            num_rays=7,
            flatten=False,
            return_hits=True,
            return_labels=True,
            normalize=True,
            dist_norm=1.0,
            label_norm=1.0
        )
        lidar_hits = lidar_data[:, :, 0:2]
        lidar_labels = lidar_data[:, :, 2].long()

        # print("Coverage:", env.coverage)
        # print("Env Levels:", self.env.env_levels)
        # print("Multi-face seq:", self._multi_face_sequence)

        capture_mask = get_inspection_action(env).bool()  # (E,)
        if not capture_mask.any():
            conf_flat = env.confidence.view(env.num_envs, -1)
            return conf_flat

        hits_sorted, valid_hits_mask, update_mask = self._process_lidar_hits(env, lidar_hits, lidar_labels, capture_mask)
        if not update_mask.any():
            conf_flat = env.confidence.view(env.num_envs, -1)
            return conf_flat

        # Update occupancy & coverage
        self._update_occupancy_and_coverage(env, hits_sorted, valid_hits_mask, update_mask)

        # Determine if inspection is done (uses env.coverage internally)
        done = (env.coverage >= self.coverage_threshold).sum(dim=1) >= env.coverage.shape[1]

        # newly_done = those that are done now but weren't before
        newly_done = done & ~env._inspection_done  # true only on first frame of completion
        if newly_done.any():
            env._level_completion_counts[newly_done, env.env_levels[newly_done]] += 1

        # update the stored done flags
        env._inspection_done[:] = done

        self._update_uninspected_distances(env)
        self._advance_multi_face_if_needed(env)

        # Return confidence
        conf_flat = env.confidence.view(env.num_envs, -1)
        return conf_flat

        # conf_flat = env.confidence.view(env.num_envs, -1)
        # inspected_flat = (env.confidence > 0).float().view(env.num_envs, -1)
        # return torch.cat((conf_flat, inspected_flat), dim=1)

    # ----------------------- lidar processing --------------------------------
    @torch.no_grad()
    def _process_lidar_hits(self, env, lidar_hits, lidar_labels, capture_mask):
        """
        Process raw lidar hits and return:
        - hits_sorted: (E, M, T, 2) top-k hit coordinates per env/object
        - valid_hits_mask: (E, M, T) boolean indicating which of the top-k are valid
        - update_mask: (E, M) boolean indicating which env/object pairs had >=1 valid hit
        """
        E, N, _ = lidar_hits.shape
        M = self.valid_object_ids.shape[0]
        T = min(self.max_hits_per_step, N)
        device = self.device

        # Ignore invalid rays (label == -1)
        valid_label_mask = lidar_labels != -1  # (E, N)

        # sensor position (E,2)
        sensor_pos = env.scene.sensors["ray_caster"].data.pos_w[:, :2]

        # distances to hits
        distances = torch.norm(lidar_hits - sensor_pos.unsqueeze(1), dim=-1)
        distances = torch.nan_to_num(distances, nan=1e6)
        distance_mask = distances <= self.max_hit_distance  # (E, N)

        # Match labels with valid object IDs
        labels_eq = (lidar_labels.unsqueeze(-1) == self.valid_object_ids).permute(0, 2, 1)  # (E, M, N)

        # Ensure capture_mask has shape (E,1,1) for broadcast
        capture_b = capture_mask.view(E, 1, 1)

        # Valid rays only if: label matches, within distance, label != -1, and capture action is active
        valid_mask_emn = labels_eq & distance_mask.unsqueeze(1) & valid_label_mask.unsqueeze(1) & capture_b  # (E, M, N)

        # Count valid rays per env/object
        counts = valid_mask_emn.sum(dim=2)  # (E, M)
        update_mask = counts > 0  # (E, M)

        # Create score to prioritize valid hits in topk
        # Large positive for valid hits, large negative for invalid
        score_base = torch.arange(N, device=device).float().view(1, 1, N)
        sort_score = valid_mask_emn.float() * (N * 2 - score_base) - score_base  # (E, M, N)

        topk_idx = torch.topk(sort_score, k=T, dim=2).indices  # (E, M, T)

        # Gather hits into sorted array
        hits_exp = lidar_hits.unsqueeze(1).expand(E, M, N, 2)  # (E, M, N, 2)
        hits_sorted = torch.gather(hits_exp, 2, topk_idx.unsqueeze(-1).expand(E, M, T, 2))

        # Mark valid hits for top-k selection
        arange_t = torch.arange(T, device=device)
        valid_hits_mask = arange_t.view(1, 1, T) < counts.unsqueeze(-1)

        return hits_sorted, valid_hits_mask, update_mask

    # ----------------- occupancy & coverage update ---------------------
    @torch.no_grad()
    def _update_occupancy_and_coverage(self, env, hits_sorted, valid_hits_mask, update_mask):
        """
        Vectorized update:
          - update per-point env.confidence
          - compute per-face mean confidence -> env.face_confidence (E, M, 4)
          - compute env.coverage (E, M) as mean of active face confidences
        """
        self._ensure_cached_contour_world(env)
        contour_w = self._cached_contour_points  # (E, M, D, 2)

        E, M, T, _ = hits_sorted.shape
        D = self.num_contour_points
        device = self.device

        # Expand update mask first -> (E, M, D)
        mask_expanded = update_mask.unsqueeze(-1).expand(E, M, D)

        # Distances between contour points and hits (vectorized)
        # contour_w: (E, M, D, 2), hits_sorted: (E, M, T, 2)
        diff = contour_w.unsqueeze(3) - hits_sorted.unsqueeze(2)  # (E, M, D, T, 2)
        dist2 = torch.sum(diff * diff, dim=-1)                    # (E, M, D, T)
        dist2 = torch.nan_to_num(dist2, nan=1e6)

        hit_mask = dist2 <= (self.cell_radius ** 2)                # (E, M, D, T)
        hit_mask &= valid_hits_mask.unsqueeze(2)                  # only consider valid top-k

        # ---------------- orientation gating ----------------
        # robot yaw quaternion -> yaw angle
        robot_quat = env.scene.sensors["ray_caster"].data.quat_w.to(device)  # (E,4)
        w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))     # (E,)
        fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)          # (E,2)

        # Build per-point face ids tensor once (D,)
        face_ids_per_point = torch.empty(D, dtype=torch.long, device=device)
        for f, idx in enumerate(self.face_groups):
            face_ids_per_point[idx] = f

        # face normals and per-point normals
        face_normals = torch.tensor(
            [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            device=device
        )
        normals_per_point = face_normals[face_ids_per_point]  # (D, 2)
        normals_per_point = normals_per_point.unsqueeze(0).expand(E, D, 2)  # (E, D, 2)

        # Dot between robot forward and per-point face normal -> (E, D)
        dot = (fwd.unsqueeze(1) * normals_per_point).sum(-1)
        ORTHO_THRESHOLD = 0.95
        orientation_mask = dot.abs() >= ORTHO_THRESHOLD   # (E, D)
        orientation_mask = orientation_mask.unsqueeze(1).expand(E, M, D)  # (E, M, D)

        # Observed now if any hit (over T) AND orientation ok AND update_mask active
        observed_now = hit_mask.any(dim=-1).float()  # (E, M, D)
        observed_now = observed_now * mask_expanded.float() * orientation_mask.float()

        # Update per-point confidence (only where update_mask is True)
        # Use elementwise maximum to keep cumulative detection
        # Avoid indexing large masks repeatedly: do an in-place maximum over full tensors.
        # Create a temporary candidate tensor for places that changed (for speed)
        # We still compute candidate array but write only where mask_expanded is True to avoid needless writes.
        prior = env.confidence  # (E, M, D)
        # compute element-wise maximum (vectorized)
        new_conf = torch.maximum(prior, observed_now)
        # write back
        env.confidence[:] = new_conf
        env.confidence.clamp_(0.0, 1.0)

        # ---------------- compute per-face confidence (vectorized) ----------------
        # env.confidence: (E, M, D)
        # self.face_point_mask_f32: (4, D)
        # we want sums per face: (E, M, 4) = einsum('emd,fd->emf')
        face_sums = torch.einsum('emd,fd->emf', env.confidence, self.face_point_mask_f32)  # (E, M, 4)
        counts = self.face_point_counts.to(device=device).view(1, 1, -1)                    # (1,1,4)
        env.face_confidence[:] = face_sums / counts                                       # (E, M, 4)

        # ---------------- compute coverage per-object using active faces ----------------
        # active masks: (E, M, D) from _compute_active_masks -> convert to per-face active mask (E, M, 4)
        active_pts = self._compute_active_masks(env)  # (E, M, D) bool
        # convert to per-face boolean: sum mask over points per face -> if >0 then face active
        active_face_counts = torch.einsum('emd,fd->emf', active_pts.float(), self.face_point_mask_f32)  # (E, M, 4)
        active_face_bool = active_face_counts > 0.0  # (E, M, 4)

        # To avoid dividing by zero, compute mean over the active faces only. If no active faces, coverage -> 0.
        # Sum of confidences over active faces:
        active_face_conf_sum = (env.face_confidence * active_face_bool.float()).sum(dim=2)  # (E, M)
        num_active_faces = active_face_bool.float().sum(dim=2).clamp_min(1.0)                # (E, M)
        env.coverage_prev[:] = env.coverage
        env.coverage[:] = active_face_conf_sum / num_active_faces

        # Keep coverage in [0,1]
        env.coverage.clamp_(0.0, 1.0)

    # ----------------------- uninspected distances ---------------------------
    @torch.no_grad()
    def _update_uninspected_distances(self, env):
        """
        Maintain uninspected_distances consistent with active points and per-point confidence.
        """
        if self._cached_contour_points is None:
            return

        # uninspected points = confidence < 1.0
        uninspected_mask = (env.confidence < 1.0).float()  # (E, M, D)

        # Active mask (E, M, D)
        active_mask = self._compute_active_masks(env).float()

        # Only mark active points; inactive points get 0.0 in uninspected_distances
        env.uninspected_distances[:] = uninspected_mask * active_mask

    # ----------------------- multi-face sequence management ------------------
    @torch.no_grad()
    def _init_multi_face_sequence(self, env_ids: torch.Tensor):
        """Initialize per-env face sequences based on current level and cached face order.

        - Uses the cached distance-ordered faces (closest->farthest).
        - Keeps only the positions active for the current level (preserving distance order).
        - Writes that list into self._multi_face_sequence (padded with -1) and sets pointer to 0.
        """
        device = self.device
        E = self.env.num_envs

        # Ensure we have a cached face order
        if self._cached_face_order is None:
            self._cached_face_order = self._compute_face_order_at_reset(torch.arange(E, device=device))

        sorted_faces = self._cached_face_order            # (E, 4) - face ids sorted by distance
        pos_keep_all = self._level_pos_mask[self.env.env_levels]  # (E, 4) bools per level

        # Normalize env_ids and iterate (explicit loop keeps semantics simple and safe)
        env_ids = env_ids.to(device=device, dtype=torch.long).view(-1)
        for idx in env_ids.tolist():
            keep = pos_keep_all[idx]              # (4,) booleans selecting which *positions* (closest..farthest) are active
            faces_ordered = sorted_faces[idx]     # (4,) face ids in distance order
            active_faces = faces_ordered[keep]    # (k,) selected face ids (closest-first)
            k = int(active_faces.numel())

            # Reset row and fill first k entries with the selected face ids
            self._multi_face_sequence[idx] = -1
            if k > 0:
                self._multi_face_sequence[idx, :k] = active_faces
                self._multi_face_pos[idx] = 0
            else:
                # No active faces (shouldn't normally happen given masks) — keep pointer 0 and sequence -1
                self._multi_face_pos[idx] = 0
    
    @torch.no_grad()
    def _advance_multi_face_if_needed(self, env):
        """Advance per-env pointer when the current face is fully inspected.

        Uses precomputed env.face_confidence (E, M, 4) to decide completion.
        """
        if self._cached_face_order is None or self._cached_contour_points is None:
            return

        E = env.num_envs
        device = self.device

        # Consider only envs that are not globally done
        active_envs = ~env._inspection_done  # (E,)
        if not active_envs.any():
            return

        idx_envs = torch.nonzero(active_envs, as_tuple=True)[0]  # (K,)
        if idx_envs.numel() == 0:
            return

        # Pull sequences and pointers for these envs (K, 4) and (K,)
        seq = self._multi_face_sequence[idx_envs]    # (K, 4)
        ptrs = self._multi_face_pos[idx_envs]        # (K,)

        # clamp pointers to available cols
        ptrs_clamped = ptrs.clamp(min=0, max=seq.shape[1] - 1)
        row_idx = torch.arange(seq.shape[0], device=device)

        # current face ids (K,)
        current_face_ids = seq[row_idx, ptrs_clamped]

        # only keep rows with valid current_face_id
        valid_mask = current_face_ids >= 0
        if not valid_mask.any():
            return

        valid_rows = torch.nonzero(valid_mask, as_tuple=True)[0]
        valid_env_idx = idx_envs[valid_rows]                       # global env indices
        valid_face_ids = current_face_ids[valid_rows]              # face ids for those envs

        # Get per-face confidence for these envs & objects (E_valid, M, 4)
        # env.face_confidence: (E, M, 4)
        # We need the confidence of the current face for each env: gather on dim=2
        # Build gather index: (E_valid, M, 1) with face id repeated
        E_valid = valid_env_idx.shape[0]
        M = self.valid_object_ids.shape[0]
        face_idx_expand = valid_face_ids.view(E_valid, 1, 1).expand(E_valid, M, 1)  # (E_valid, M, 1)
        cur_face_conf = torch.gather(env.face_confidence[valid_env_idx], 2, face_idx_expand).squeeze(2)  # (E_valid, M)

        # If you want completion per-object, check per-object face confidence separately.
        # We'll require the face confidence for the object(s) to exceed threshold to advance.
        FACE_COMPLETE_THRESHOLD = 0.95
        # completed_mask: (E_valid, M)
        completed_mask = cur_face_conf >= FACE_COMPLETE_THRESHOLD

        # We only advance the pointer for envs where the face is completed for ALL objects
        # (if M>1 we demand every object had that face completed)
        advance_per_env = completed_mask.all(dim=1)  # (E_valid,)

        if not advance_per_env.any():
            return

        # global env indices to advance
        adv_envs = valid_env_idx[advance_per_env]  # (A,)
        if adv_envs.numel() == 0:
            return

        # Advance pointer but clamp to last valid index in the sequence (entries >=0)
        seq_adv = self._multi_face_sequence[adv_envs]     # (A, 4)
        valid_counts = (seq_adv >= 0).sum(dim=1)         # (A,)
        cur_ptrs = self._multi_face_pos[adv_envs]        # (A,)
        new_ptrs = torch.minimum(cur_ptrs + 1, valid_counts - 1)
        self._multi_face_pos[adv_envs] = new_ptrs

        # Immediately recompute best robot pose for advanced envs
        self._compute_best_robot_pose(adv_envs)

    # ----------------------- active mask computation -------------------------
    @torch.no_grad()
    def _compute_active_masks(self, env) -> torch.Tensor:
        """Return boolean mask (E, M, D) of contour points active under current levels."""
        E = env.num_envs
        M = self.valid_object_ids.shape[0]
        D = self.num_contour_points

        if self._cached_face_order is None:
            self._cached_face_order = self._compute_face_order_at_reset(torch.arange(E, device=self.device))

        sorted_faces = self._cached_face_order  # (E, 4)
        pos_keep = self._level_pos_mask[self.env.env_levels]  # (E, 4)  <-- levels are already 1..7

        # Build per-env mask of active contour points
        active_points = torch.zeros((E, D), dtype=torch.bool, device=self.device)
        for f in range(self.num_faces):
            face_ids = sorted_faces[:, f]  # (E,)
            keep_mask = pos_keep[:, f]     # (E,)
            if keep_mask.any():
                # gather the mask rows corresponding to the chosen face ids
                active_points[keep_mask] |= self.face_point_mask[face_ids[keep_mask]]

        # Expand to (E, M, D)
        active_points_expanded = active_points.unsqueeze(1).expand(E, M, D)
        return active_points_expanded

    # ----------------------- world contour cache ------------------------------
    @torch.no_grad()
    def _ensure_cached_contour_world(self, env, env_ids: torch.Tensor | None = None, force: bool = False):
        """
        Ensure self._cached_contour_points (E,M,D,2) matches the current object world positions.
        If `env_ids` is provided, updates only those envs; if `force` is True, always recomputes
        for the specified envs (or all envs if `env_ids` is None).
        """
        # Current object XY positions per env (E,M,2)
        current_positions = torch.stack(
            [obj.data.root_pos_w[:, :2] for obj in self.valid_objects_rigid_objects], dim=1
        )
        current_positions = torch.nan_to_num(current_positions, nan=0.0)

        # Helper to (re)build world contours from positions
        def rebuild_for_indices(idx: torch.Tensor):
            # idx: (K,)
            pos = current_positions[idx]  # (K,M,2)
            # Broadcast add: (K,M,1,2) + (1,1,D,2) -> (K,M,D,2)
            self._cached_positions[idx] = pos
            self._cached_contour_points[idx] = pos.unsqueeze(2) + self.local_contour.view(1, 1, -1, 2)

        if env_ids is not None:
            # Normalize to a 1D LongTensor on the correct device
            env_ids = env_ids.to(device=self.device, dtype=torch.long).view(-1)
            if force:
                rebuild_for_indices(env_ids)
            else:
                # Only update envs whose positions changed meaningfully
                changed = ~torch.isclose(
                    self._cached_positions[env_ids], current_positions[env_ids], atol=1e-6, rtol=0.0
                ).all(dim=(1, 2))
                if changed.any():
                    rebuild_for_indices(env_ids[changed])
        else:
            # No specific envs: handle all
            if force or (self._cached_positions.shape != current_positions.shape) or (
                not torch.allclose(self._cached_positions, current_positions, atol=1e-6, rtol=0.0)
            ):
                self._cached_positions = current_positions.clone()
                self._cached_contour_points = current_positions.unsqueeze(2) + self.local_contour.view(1, 1, -1, 2)
