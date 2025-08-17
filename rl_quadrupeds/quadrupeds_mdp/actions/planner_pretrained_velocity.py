from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file
from isaaclab.utils.math import wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RobotPlannerActionTrainedNavigation(ActionTerm):
    r"""Planner ActionTerm that converts target positions to body-frame velocity commands
    using a simple local planner (no navigation JIT policy), then uses a locomotion model
    to convert those velocities into joint actions.

    Optimizations:
    - Batched storage for intermediate waypoints (tensor, not lists)
    - Vectorized waypoint advancement and velocity computation across envs
    - No per-env Python loops inside `apply_actions`
    """

    cfg: RobotPlannerActionTrainedNavigationCfg

    def __init__(self, cfg: RobotPlannerActionTrainedNavigationCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._env = env
        self.robot: Articulation = env.scene[cfg.asset_name]

        # ---- Load the locomotion policy (velocity -> joint actions) ----
        if not check_file_path(cfg.locomotion_policy_path):
            raise FileNotFoundError(f"Locomotion policy '{cfg.locomotion_policy_path}' not found.")
        loco_bytes = read_file(cfg.locomotion_policy_path)
        self.locomotion_policy = torch.jit.load(loco_bytes).to(env.device).eval()

        # ---- Buffers ----
        # raw_actions: [x_b, y_b, z_b, heading_b] (z_b unused by locomotion, kept for compatibility/visualization)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim + 1, device=self.device)  # [x, y, z, heading]
        self.last_action = torch.zeros_like(self._raw_actions)

        # Velocity commands produced by the local planner (body-frame vx, vy, yaw_rate)
        self._nav_actions = torch.zeros(self.num_envs, 3, device=self.device)

        # Locomotion action term & buffer (applies joint actions to the robot)
        self._loc_action_term: ActionTerm = cfg.locomotion_actions.class_type(cfg.locomotion_actions, env)
        self._loc_actions = torch.zeros(self.num_envs, self._loc_action_term.action_dim, device=self.device)

        # ---- Observation wiring ----
        def loc_last_action():
            if hasattr(env, "episode_length_buf"):
                self._loc_actions[env.episode_length_buf == 0, :] = 0
            return self._loc_actions

        cfg.locomotion_observations.actions.func = lambda _: loc_last_action()
        cfg.locomotion_observations.actions.params = dict()
        cfg.locomotion_observations.velocity_commands.func = lambda _: self.nav_actions
        cfg.locomotion_observations.velocity_commands.params = dict()
        self._loco_obs_manager = ObservationManager({"loco_policy": cfg.locomotion_observations}, env)

        self._loco_counter = 0
        self.observation_reset_done = False

        # ---- Command buffers ----
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_b = torch.zeros(self.num_envs, device=self.device)

        self.half_range_pos_x = (self.cfg.ranges.pos_x[1] - self.cfg.ranges.pos_x[0]) / 2
        self.half_range_pos_y = (self.cfg.ranges.pos_y[1] - self.cfg.ranges.pos_y[0]) / 2
        self.half_range_heading = (self.cfg.ranges.heading[1] - self.cfg.ranges.heading[0]) / 2

        # Readiness flag for accepting a new high-level command
        self._env.is_ready_for_new_command = torch.ones((self.num_envs, 1), dtype=torch.bool, device=self.device)

        # Tolerances
        self.position_tolerance = 0.5
        self.heading_tolerance = 0.1
        self.waypoint_tolerance = 0.2  # slightly tighter to progress across intermediate points

        # Obstacles (world-frame XY positions). Extend this list as needed.
        self.obstacles = torch.stack(
            [env.scene[object_name].data.root_pos_w[:, :2] for object_name in ["block1"]],
            dim=1
        ).to(env.device)  # (N, M, 2)

        # ---- Intermediate waypoints (batched) ----
        # Fixed K steps for the local planner rollout (vectorized over envs)
        self.K = getattr(cfg, "num_waypoints", 5)
        # Waypoints tensor: (N, K, 3) -> (x, y, heading)
        self._waypoints = torch.zeros(self.num_envs, self.K, 3, device=self.device)
        # Active mask: whether a given env currently has waypoints assigned
        self._wp_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Per-env current waypoint index [0..K-1]
        self._waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Controller gains for converting waypoint error -> velocity command
        self.k_pos = cfg.k_pos
        self.k_yaw = cfg.k_yaw

    # ---------------------- Properties ----------------------
    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def last_actions(self) -> torch.Tensor:
        return self.last_action

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    @property
    def nav_actions(self) -> torch.Tensor:
        """Body-frame velocity command [vx, vy, yaw_rate] for the locomotion policy."""
        return self._nav_actions

    # ---------------------- Reset ----------------------
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            return
        self._raw_actions[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
        self._nav_actions[env_ids] = 0.0
        self._loc_actions[env_ids] = 0.0
        self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
        self.heading_command_w[env_ids] = 0.0
        self.pos_command_b[env_ids] = 0.0
        self.heading_command_b[env_ids] = 0.0
        self._loco_counter = 0
        self._loco_obs_manager.reset(env_ids=env_ids)
        self._env.is_ready_for_new_command[env_ids] = True
        self._waypoint_idx[env_ids] = 0
        self._waypoints[env_ids] = 0.0
        self._wp_active[env_ids] = False

    # ---------------------- Local planner ----------------------
    def simple_local_planner(
        self,
        robot_pos: torch.Tensor,
        robot_yaw: torch.Tensor,
        goal_pos: torch.Tensor,
        obstacles: torch.Tensor,
        step_size: float = 1.0,
        influence_radius: float = 2.0
    ) -> torch.Tensor:
        """
        Simple potential-field-based local planner.
        Accepts both single (2,) inputs or batched (N,2) inputs.
        Returns the next XY position(s) in world frame. Shape: (N,2)
        """
        device = robot_pos.device
        eps = 1e-6

        # Ensure batched form
        if robot_pos.ndim == 1:
            robot_pos = robot_pos.unsqueeze(0)
            goal_pos = goal_pos.unsqueeze(0)
            robot_yaw = robot_yaw.unsqueeze(0)
            if obstacles.ndim == 2:
                obstacles = obstacles.unsqueeze(0)

        # Attractive force to goal (normalized)
        force_goal = goal_pos - robot_pos                          # (N,2)
        dist_goal = torch.norm(force_goal, dim=1, keepdim=True).clamp_min(eps)
        force_goal = force_goal / dist_goal

        # Repulsive force from obstacles inside influence radius
        if obstacles.numel() > 0:
            # robot_pos: (N,2) -> (N,1,2)
            # obstacles : (N,M,2)
            obs_vec = robot_pos.unsqueeze(1) - obstacles           # (N,M,2)
            dist_obs = torch.norm(obs_vec, dim=2, keepdim=True).clamp_min(eps)  # (N,M,1)
            mask = (dist_obs < influence_radius).float()           # (N,M,1)
            repulsion = (obs_vec / (dist_obs ** 2)) * mask         # (N,M,2)
            force_obs = repulsion.sum(dim=1)                       # (N,2)
        else:
            force_obs = torch.zeros_like(force_goal)

        force_total = force_goal + force_obs
        norm_total = torch.norm(force_total, dim=1, keepdim=True).clamp_min(eps)
        force_total = force_total / norm_total

        next_pos = robot_pos + step_size * force_total             # (N,2)
        return next_pos

    # ---------------------- High-level command processing (vectorized) ----------------------
    def process_actions(self, actions: torch.Tensor):
        """Interpret high-level XY/heading command into a stack of intermediate waypoints (batched)."""
        if not self.observation_reset_done:
            self._reset_observations()
            self.observation_reset_done = True

        device = self.device
        current_pos = self.robot.data.root_pos_w[:, :3]                     # (N,3)
        current_heading = self.robot.data.heading_w                          # (N,)
        default_z = self.robot.data.default_root_state[:, 2]                 # (N,)

        # Are we at the previously commanded final goal?
        pos_error_sq = (self.pos_command_w - current_pos).pow_(2).sum(dim=1)
        pos_tol_sq = self.position_tolerance * self.position_tolerance
        heading_error_abs = wrap_to_pi(self.heading_command_w - current_heading).abs()

        goal_reached = (pos_error_sq < pos_tol_sq) & (heading_error_abs < self.heading_tolerance)
        first_step = self._raw_actions.abs().max(dim=1).values == 0
        ready_mask = goal_reached | first_step
        self._env.is_ready_for_new_command[:, 0] = ready_mask

        # Map normalized [-1,1] actions to world-frame final goal and heading
        xy = torch.tanh(actions[:, :2].clamp(min=-1.0))                      # (N,2)
        heading = torch.tanh(actions[:, 2].clamp(min=-1.0))                  # (N,)

        new_global_goal_w = self._env.scene.env_origins.clone()              # (N,3)
        new_global_goal_w[:, 0] += xy[:, 0] * self.half_range_pos_x
        new_global_goal_w[:, 1] += xy[:, 1] * self.half_range_pos_y
        new_global_goal_w[:, 2] = default_z

        new_heading_command_w = heading * self.half_range_heading            # (N,)

        # ---- Build intermediate waypoints for all "ready" envs in a batched rollout ----
        ready_idx = torch.nonzero(ready_mask, as_tuple=False).squeeze(-1)    # (R,)
        if ready_idx.numel() > 0:
            R = ready_idx.numel()
            K = self.K

            sim_pos = current_pos[ready_idx, :2].clone()                     # (R,2)
            sim_heading = current_heading[ready_idx]                         # (R,)
            goal_xy = new_global_goal_w[ready_idx, :2]                       # (R,2)
            obs = self.obstacles[ready_idx]                                  # (R,M,2)

            # Rollout K steps with the simple local planner (vectorized across R)
            waypoints_xy = []
            for _ in range(K):
                sim_pos = self.simple_local_planner(sim_pos, sim_heading, goal_xy, obs).to(device)
                waypoints_xy.append(sim_pos.unsqueeze(1))                    # [(R,1,2)] * K
            waypoints_xy = torch.cat(waypoints_xy, dim=1)                    # (R,K,2)

            # Headings between successive waypoints; final one uses commanded heading
            deltas = waypoints_xy[:, 1:, :] - waypoints_xy[:, :-1, :]        # (R,K-1,2)
            seg_headings = torch.atan2(deltas[..., 1], deltas[..., 0])       # (R,K-1)
            last_hdg = new_heading_command_w[ready_idx].unsqueeze(-1)        # (R,1)
            headings = torch.cat([seg_headings, last_hdg], dim=1)            # (R,K)

            waypoints_xyz_hdg = torch.cat(
                [waypoints_xy, headings.unsqueeze(-1)], dim=2
            )                                                                 # (R,K,3) -> (x,y,heading)

            # Persist per-env waypoints and reset progression index (batched)
            self._waypoints[ready_idx] = waypoints_xyz_hdg
            self._waypoint_idx[ready_idx] = 0
            self._wp_active[ready_idx] = True

            # Set final commanded pose (for visualization & goal checks)
            last_xy = waypoints_xy[:, -1, :]                                  # (R,2)
            z_vals = default_z[ready_idx]                                     # (R,)
            self.pos_command_w[ready_idx, 0] = last_xy[:, 0]
            self.pos_command_w[ready_idx, 1] = last_xy[:, 1]
            self.pos_command_w[ready_idx, 2] = z_vals
            self.heading_command_w[ready_idx] = new_heading_command_w[ready_idx]

        # Update body-frame deltas (for visualization only)
        target_vec = self.pos_command_w - current_pos                         # (N,3)
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - current_heading)

        self.last_action.copy_(self._raw_actions)
        self._raw_actions[:, :3] = self.pos_command_b
        self._raw_actions[:, 3] = self.heading_command_b

    # ---------------------- Apply actions: local planner -> velocities -> locomotion ----------------------
    def apply_actions(self):
        """Compute body-frame velocity commands to chase intermediate waypoints (batched), then use locomotion policy."""
        linear_min, linear_max = self.cfg.ranges.v_linear
        angular_min, angular_max = self.cfg.ranges.v_angular

        current_pos = self.robot.data.root_pos_w[:, :3]           # (N,3)
        current_heading = self.robot.data.heading_w               # (N,)
        root_quat_w = self.robot.data.root_quat_w                # (N,4)

        N = self.num_envs
        K = self.K

        # If an env is inactive (no waypoints), command zeros
        self._nav_actions.zero_()
        if not self._wp_active.any():
            # No active waypoints anywhere; still run locomotion below
            pass
        else:
            active_mask = self._wp_active.clone()                 # (N,)
            idx = self._waypoint_idx.clamp_(0, K - 1)            # (N,)

            # Gather current waypoint (x,y,heading) for all envs
            # Build indices for torch.gather: (N,1,3)
            idx_expanded = idx.view(N, 1, 1).expand(-1, 1, 3)
            wp_now = torch.gather(self._waypoints, 1, idx_expanded).squeeze(1)  # (N,3)
            wp_xy = wp_now[:, :2]                                              # (N,2)
            wp_hdg = wp_now[:, 2]                                              # (N,)

            # Advance waypoints for envs that are within tolerance.
            # We may need to advance multiple steps; do it in at most K-1 vectorized passes.
            # Only consider active envs and those not already at the last index.
            for _ in range(K - 1):
                # Recompute distance to current waypoint
                dist = torch.norm(wp_xy - current_pos[:, :2], dim=1)           # (N,)
                can_advance = active_mask & (idx < (K - 1)) & (dist < self.waypoint_tolerance)
                if not torch.any(can_advance):
                    break
                # Increment indices where we can advance
                idx = idx + can_advance.to(idx.dtype)
                # Refresh gathered waypoint for advanced envs
                idx_expanded = idx.view(N, 1, 1).expand(-1, 1, 3)
                wp_now = torch.gather(self._waypoints, 1, idx_expanded).squeeze(1)
                wp_xy = wp_now[:, :2]
                wp_hdg = wp_now[:, 2]

            # Save updated indices
            self._waypoint_idx.copy_(idx)

            # World-frame XY error to waypoint (z=0)
            err_w = torch.zeros(N, 3, device=self.device)
            err_w[:, :2] = wp_xy - current_pos[:, :2]

            # Convert to body-frame XY using current yaw (batched)
            err_b = quat_rotate_inverse(yaw_quat(root_quat_w), err_w)         # (N,3)

            # Proportional controller for linear velocity (vx, vy)
            v_b_xy = (self.k_pos * err_b[:, :2]).clamp(min=linear_min, max=linear_max)  # (N,2)

            # Heading error and proportional yaw rate
            yaw_err = wrap_to_pi(wp_hdg - current_heading)                     # (N,)
            yaw_rate = (self.k_yaw * yaw_err).clamp(min=angular_min, max=angular_max)    # (N,)

            # Write command [vx, vy, yaw_rate] only for active envs; zeros elsewhere
            self._nav_actions[active_mask, 0:2] = v_b_xy[active_mask]
            self._nav_actions[active_mask, 2] = yaw_rate[active_mask]

        # ---- Locomotion step ----
        if self._loco_counter % self.cfg.locomotion_decimation == 0:
            loco_obs = self._loco_obs_manager.compute_group("loco_policy")
            self._loc_actions[:] = self.locomotion_policy(loco_obs)
            self._loc_action_term.process_actions(self._loc_actions)
            self._loco_counter = 0
        self._loc_action_term.apply_actions()
        self._loco_counter += 1

    # ---------------------- Debug Visualization ----------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            if not hasattr(self, "waypoints_visualizer"):
                self.waypoints_visualizer = VisualizationMarkers(self.cfg.intermediate_pose_visualizer_cfg.replace(
                    prim_path="/Visuals/Command/waypoints"
                ))
            self.goal_pose_visualizer.set_visibility(True)
            self.waypoints_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "waypoints_visualizer"):
                self.waypoints_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Visualize final goal pose
        self.goal_pose_visualizer.visualize(
            translations=self.pos_command_w[:, :3],
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
        )

        # Visualize intermediate waypoints (batched)
        active = self._wp_active
        if not torch.any(active):
            return

        N, K = self.num_envs, self.K
        # Build (N*K, 3) translations with z from default root state
        z_vals = self.robot.data.default_root_state[:, 2]                     # (N,)
        # Repeat z along K and stack with x,y
        xy = self._waypoints[:, :, :2]                                        # (N,K,2)
        z = z_vals.view(N, 1).expand(-1, K)                                   # (N,K)
        pts = torch.cat([xy, z.unsqueeze(-1)], dim=2)                         # (N,K,3)

        # Keep only active envs
        pts = pts[active]                                                     # (Na,K,3)
        hdg = self._waypoints[active, :, 2]                                   # (Na,K)

        all_waypoints = pts.reshape(-1, 3)                                    # (Na*K,3)
        all_orientations = quat_from_euler_xyz(
            torch.zeros_like(hdg).reshape(-1),
            torch.zeros_like(hdg).reshape(-1),
            hdg.reshape(-1),
        )

        self.waypoints_visualizer.visualize(
            translations=all_waypoints,    # (Na*K, 3)
            orientations=all_orientations  # (Na*K, 4)
        )

    # ---------------------- Internal helpers ----------------------
    def _reset_observations(self):
        self._loco_obs_manager.reset(env_ids=range(self.num_envs))


@configclass
class RobotPlannerActionTrainedNavigationCfg(ActionTermCfg):
    """Configuration for Planner ActionTerm using a local planner for velocity commands and a Locomotion model."""
    class_type: type[ActionTerm] = RobotPlannerActionTrainedNavigation
    asset_name: str = MISSING

    # Path to locomotion policy (velocity -> joint actions)
    locomotion_policy_path: str = MISSING

    # Locomotion manager interface
    locomotion_decimation: int = 4
    locomotion_actions: ActionTermCfg = MISSING
    locomotion_observations: ObservationGroupCfg = MISSING  # must include "actions" and "velocity_commands" groups

    @configclass
    class Ranges:
        # Command ranges for mapping normalized high-level actions and for velocity clamps
        pos_x: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING
        heading: tuple[float, float] = MISSING
        v_linear: tuple[float, float] = MISSING     # [min, max] linear velocity component clamp in body frame
        v_angular: tuple[float, float] = MISSING    # [min, max] yaw rate clamp

    ranges: Ranges = MISSING

    # Controller gains for turning waypoint error into velocity commands
    k_pos: float = 1.0   # proportional gain on position error (m/s per m)
    k_yaw: float = 2.0   # proportional gain on heading error (rad/s per rad)

    # Number of intermediate waypoints simulated per command (kept small for speed)
    num_waypoints: int = 5

    # Debug visualization configuration
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal"
    )
    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)
    intermediate_pose_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal"
    )
    intermediate_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)