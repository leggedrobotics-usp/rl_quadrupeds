from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file
from isaaclab.utils.math import wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RobotPlannerActionTrainedNavigation(ActionTerm):
    r"""Planner ActionTerm mapping NN outputs to relative movement commands in body-frame,
    converts to world-frame, clips to arena bounds, and generates velocity commands."""

    cfg: "RobotPlannerActionTrainedNavigationCfg"

    def __init__(self, cfg: "RobotPlannerActionTrainedNavigationCfg", env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._env = env
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Load locomotion policy
        if not check_file_path(cfg.locomotion_policy_path):
            raise FileNotFoundError(f"Locomotion policy '{cfg.locomotion_policy_path}' not found.")
        loco_bytes = read_file(cfg.locomotion_policy_path)
        self.locomotion_policy = torch.jit.load(loco_bytes).to(env.device).eval()

        # Buffers
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim + 1, device=self.device)
        self.last_action = torch.zeros_like(self._processed_actions)
        self._nav_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._loc_action_term: ActionTerm = cfg.locomotion_actions.class_type(cfg.locomotion_actions, env)
        self._loc_actions = torch.zeros(self.num_envs, self._loc_action_term.action_dim, device=self.device)

        # Locomotion obs manager
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

        # Command buffers
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_b = torch.zeros(self.num_envs, device=self.device)

        self._last_pos_error_b = torch.zeros_like(self.pos_command_b)
        self._last_heading_error = torch.zeros_like(self.heading_command_b)

        self._env.is_ready_for_new_command = torch.ones((self.num_envs, 1), dtype=torch.bool, device=self.device)

        self.position_tolerance = 0.5
        self.heading_tolerance = 0.2

        # Controller gains
        self.kp_pos = cfg.k_pos
        self.kd_pos = getattr(cfg, "kd_pos", 0.0)
        self.kp_yaw = cfg.k_yaw
        self.kd_yaw = getattr(cfg, "kd_yaw", 0.0)
        self.K = getattr(cfg, "num_waypoints", 1)

    # ---------------------- Properties ----------------------
    @property
    def action_dim(self) -> int:
        return 3  # [dx, dy, dheading]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def last_actions(self) -> torch.Tensor:
        return self.last_action

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def nav_actions(self) -> torch.Tensor:
        return self._nav_actions

    # ---------------------- Reset ----------------------
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            return
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
        self._nav_actions[env_ids] = 0.0
        self._loc_actions[env_ids] = 0.0
        self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
        self.heading_command_w[env_ids] = 0.0
        self.pos_command_b[env_ids] = 0.0
        self.heading_command_b[env_ids] = 0.0
        self._last_pos_error_b[env_ids] = 0.0
        self._last_heading_error[env_ids] = 0.0
        self._loco_counter = 0
        self._loco_obs_manager.reset(env_ids=env_ids)
        self._env.is_ready_for_new_command[env_ids] = True

    # ---------------------- High-level command processing ----------------------
    def process_actions(self, actions: torch.Tensor):
        """Interpret NN outputs as relative movement in body-frame, clip to arena (per-env), convert to world-frame."""
        if self.cfg.manual_cmd is not None:
            self._nav_actions[:] = self._env.command_manager.get_command(self.cfg.manual_cmd)
            return

        self._raw_actions = actions
        if not self.observation_reset_done:
            self._reset_observations()
            self.observation_reset_done = True

        device = self.device

        # --- world-frame quantities from the simulator (global / sim frame) ---
        current_pos_w = self.robot.data.root_pos_w[:, :3]  # (N,3) in simulation/global frame
        current_heading = self.robot.data.heading_w        # (N,)
        root_quat_w = self.robot.data.root_quat_w          # (N,4)
        default_z_w = self.robot.data.default_root_state[:, 2]  # (N,)

        # env origins (world frame) — used to convert between global and per-env (local) coordinates
        env_origins_w = self._env.scene.env_origins  # (N,3), in global frame

        # Step resolution tensor
        res = torch.tensor([self.cfg.resolution_x,
                            self.cfg.resolution_y,
                            self.cfg.resolution_yaw], device=device)

        # Relative delta in body-frame
        delta_body = actions * res  # (N,3) [dx, dy, dyaw]

        # Make delta_body a 3D vector [dx, dy, 0] for rotation
        delta_body_3d = torch.zeros(self.num_envs, 3, device=device)
        delta_body_3d[:, :2] = delta_body[:, :2]  # copy x, y, set z=0

        # Rotate from body to world frame (delta expressed in world frame)
        delta_world_3d = quat_rotate_inverse(yaw_quat(root_quat_w), delta_body_3d)

        # Proposed new position in WORLD/GLOBAL frame (before clipping)
        proposed_pos_w = current_pos_w + delta_world_3d
        proposed_pos_w[:, 2] = default_z_w

        # --- Convert proposed position to per-env LOCAL coordinates for clipping ---
        # local = global - env_origin
        proposed_pos_local = proposed_pos_w - env_origins_w
        # Clip to arena bounds which are defined in per-env (local) coordinates
        proposed_pos_local[:, 0] = torch.clamp(proposed_pos_local[:, 0],
                                            self.cfg.ranges.pos_x[0], self.cfg.ranges.pos_x[1])
        proposed_pos_local[:, 1] = torch.clamp(proposed_pos_local[:, 1],
                                            self.cfg.ranges.pos_y[0], self.cfg.ranges.pos_y[1])

        # Convert clipped local back to world/global frame
        new_pos_w = proposed_pos_local + env_origins_w
        new_pos_w[:, 2] = default_z_w

        # Heading update (relative) — headings are frame-independent (angles), remain in world heading
        new_heading_w = current_heading + delta_body[:, 2]
        new_heading_w = wrap_to_pi(new_heading_w)

        # Store world-frame commands
        self.pos_command_w[:] = new_pos_w
        self.heading_command_w[:] = new_heading_w

        # Convert to body-frame displacement for controller (target_vec in world frame)
        target_vec_w = self.pos_command_w - current_pos_w
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(root_quat_w), target_vec_w)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - current_heading)

        # For visualization, store processed actions (normalized) (use local ranges for normalization)
        self.last_action.copy_(self._processed_actions)
        max_pos_x = max(abs(self.cfg.ranges.pos_x[0]), abs(self.cfg.ranges.pos_x[1]))
        max_pos_y = max(abs(self.cfg.ranges.pos_y[0]), abs(self.cfg.ranges.pos_y[1]))
        max_heading = max(abs(self.cfg.ranges.heading[0]), abs(self.cfg.ranges.heading[1]))

        self._processed_actions[:, 0] = self.pos_command_b[:, 0] / (max_pos_x + 1e-8)
        self._processed_actions[:, 1] = self.pos_command_b[:, 1] / (max_pos_y + 1e-8)
        self._processed_actions[:, 2] = 0
        self._processed_actions[:, 3] = self.heading_command_b / (max_heading + 1e-8)

    # ---------------------- Apply actions ----------------------
    def apply_actions(self):
        linear_min, linear_max = self.cfg.ranges.v_linear
        angular_min, angular_max = self.cfg.ranges.v_angular

        dt = getattr(self._env, "sim_dt", None) or getattr(self._env, "dt", None) or 0.02
        nav_old = self._nav_actions.clone()
        nav_new = torch.zeros_like(self._nav_actions)

        err_pos_b = self.pos_command_b[:, :2]
        d_err_pos = (err_pos_b - self._last_pos_error_b[:, :2]) / dt
        d_err_heading = (self.heading_command_b - self._last_heading_error) / dt

        v_b_xy = (self.kp_pos * err_pos_b) + (self.kd_pos * d_err_pos)
        dist_to_goal = torch.norm(err_pos_b, dim=1)
        stopping_region = max(1e-3, 2.0 * self.position_tolerance)
        scale = torch.clamp(dist_to_goal / stopping_region, max=1.0)
        dot_along_error = (err_pos_b * v_b_xy).sum(dim=1)
        away_mask = dot_along_error < 0
        v_b_xy = v_b_xy * scale.unsqueeze(-1)
        v_b_xy = v_b_xy.clamp(min=linear_min, max=linear_max)
        if away_mask.any():
            v_b_xy[away_mask, :] = 0.0

        yaw_rate = (self.kp_yaw * self.heading_command_b) + (self.kd_yaw * d_err_heading)
        yaw_rate = yaw_rate * scale
        yaw_rate = yaw_rate.clamp(min=angular_min, max=angular_max)

        nav_new[:, 0:2] = v_b_xy
        nav_new[:, 2] = yaw_rate

        alpha = 0.9
        self._nav_actions[:] = alpha * nav_old + (1.0 - alpha) * nav_new
        self._last_pos_error_b[:, :2] = err_pos_b
        self._last_heading_error[:] = self.heading_command_b

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
            if not hasattr(self, "ideal_pose_visualizer"):
                self.ideal_pose_visualizer = VisualizationMarkers(
                    self.cfg.ideal_pose_visualizer_cfg.replace(prim_path="/Visuals/Command/ideal_pose")
                )
            self.goal_pose_visualizer.set_visibility(True)
            self.ideal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "ideal_pose_visualizer"):
                self.ideal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pose_visualizer.visualize(
            translations=self.pos_command_w[:, :3],
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
        )
        if hasattr(self._env, "best_robot_pose"):
            ideal_pose = self._env.best_robot_pose
            z_vals = self.robot.data.default_root_state[:, 2]
            translations = torch.stack([ideal_pose[:, 0], ideal_pose[:, 1], z_vals], dim=1)
            orientations = quat_from_euler_xyz(
                torch.zeros_like(ideal_pose[:, 2]),
                torch.zeros_like(ideal_pose[:, 2]),
                ideal_pose[:, 2],
            )
            self.ideal_pose_visualizer.visualize(translations=translations, orientations=orientations)

    def _reset_observations(self):
        self._loco_obs_manager.reset(env_ids=range(self.num_envs))


@configclass
class RobotPlannerActionTrainedNavigationCfg(ActionTermCfg):
    class_type: type[ActionTerm] = RobotPlannerActionTrainedNavigation
    asset_name: str = MISSING
    locomotion_policy_path: str = MISSING
    locomotion_decimation: int = 4
    locomotion_actions: ActionTermCfg = MISSING
    locomotion_observations: ObservationGroupCfg = MISSING

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING
        heading: tuple[float, float] = MISSING
        v_linear: tuple[float, float] = MISSING
        v_angular: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

    # Step resolutions (relative command scaling)
    resolution_x: float = 1.5
    resolution_y: float = 1.5
    resolution_yaw: float = 2*3.14

    k_pos: float = 3.
    k_yaw: float = 3.
    kd_pos: float = 0.0
    kd_yaw: float = 0.0
    num_waypoints: int = 1

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal"
    )
    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)
    intermediate_pose_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal"
    )
    intermediate_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)
    ideal_pose_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_ideal"
    )
    ideal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)

    manual_cmd: str = None