from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file
from isaaclab.utils.math import wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class RobotPlannerActionTrainedNavigation(ActionTerm):
    r"""Planner ActionTerm that uses a navigation model to convert target positions to velocities,
    and a locomotion model to convert velocities into joint actions.
    """

    cfg: RobotPlannerActionTrainedNavigationCfg

    def __init__(self, cfg: RobotPlannerActionTrainedNavigationCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._env = env
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Load policies
        if not check_file_path(cfg.nav_policy_path):
            raise FileNotFoundError(f"Navigation policy '{cfg.nav_policy_path}' not found.")
        if not check_file_path(cfg.locomotion_policy_path):
            raise FileNotFoundError(f"Locomotion policy '{cfg.locomotion_policy_path}' not found.")

        nav_bytes = read_file(cfg.nav_policy_path)
        loco_bytes = read_file(cfg.locomotion_policy_path)
        self.nav_policy = torch.jit.load(nav_bytes).to(env.device).eval()
        self.locomotion_policy = torch.jit.load(loco_bytes).to(env.device).eval()

        # Buffers
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim + 1, device=self.device)  # [x, y, z, heading] in robot frame
        self.last_action = torch.zeros_like(self._raw_actions)

        self._nav_action_term: ActionTerm = cfg.nav_actions.class_type(cfg.nav_actions, env)
        self._nav_actions = torch.zeros(self.num_envs, self._nav_action_term.action_dim, device=self.device)

        self._loc_action_term: ActionTerm = cfg.locomotion_actions.class_type(cfg.locomotion_actions, env)
        self._loc_actions = torch.zeros(self.num_envs, self._loc_action_term.action_dim, device=self.device)

        # Observation functions
        def loc_last_action():
            if hasattr(env, "episode_length_buf"):
                self._loc_actions[env.episode_length_buf == 0, :] = 0
            return self._loc_actions

        def nav_last_action():
            if hasattr(env, "episode_length_buf"):
                self._nav_actions[env.episode_length_buf == 0, :] = 0
            return self._nav_actions

        cfg.locomotion_observations.actions.func = lambda _: loc_last_action()
        cfg.locomotion_observations.actions.params = dict()
        cfg.locomotion_observations.velocity_commands.func = lambda _: self.nav_actions
        cfg.locomotion_observations.velocity_commands.params = dict()
        self._loco_obs_manager = ObservationManager({"loco_policy": cfg.locomotion_observations}, env)

        cfg.nav_observations.actions.func = lambda _: nav_last_action()
        cfg.nav_observations.actions.params = dict()
        cfg.nav_observations.pose_command.func = lambda _: self._raw_actions
        cfg.nav_observations.pose_command.params = dict()
        self._nav_obs_manager = ObservationManager({"nav_policy": cfg.nav_observations}, env)

        self._nav_counter = 0
        self._loco_counter = 0
        self.observation_reset_done = False

        # Buffers for world-frame commands
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_b = torch.zeros(self.num_envs, device=self.device)

        self.half_range_pos_x = (self.cfg.ranges.pos_x[1] - self.cfg.ranges.pos_x[0]) / 2
        self.half_range_pos_y = (self.cfg.ranges.pos_y[1] - self.cfg.ranges.pos_y[0]) / 2
        self.half_range_heading = (self.cfg.ranges.heading[1] - self.cfg.ranges.heading[0]) / 2

        # Add readiness flag
        self._env.is_ready_for_new_command = torch.ones((self.num_envs, 1), dtype=torch.bool, device=self.device)

        # Position and heading tolerances
        self.position_tolerance = 1.0  # meters
        self.heading_tolerance = 0.5   # radians (~5.7 degrees)

    # ---------------------- Properties ----------------------

    @property
    def action_dim(self) -> int:
        return 3  # [x, y, heading]

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
        return self._nav_actions

    # ---------------------- Operations ----------------------
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for specified environments."""
        if env_ids is None:
            return

        # Reset action buffers
        self._raw_actions[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
        self._nav_actions[env_ids] = 0.0
        self._loc_actions[env_ids] = 0.0

        # Reset command buffers
        self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
        self.heading_command_w[env_ids] = 0.0
        self.pos_command_b[env_ids] = 0.0
        self.heading_command_b[env_ids] = 0.0

        # Reset counters
        self._nav_counter = 0
        self._loco_counter = 0

        # Reset observation managers
        self._loco_obs_manager.reset(env_ids=env_ids)
        self._nav_obs_manager.reset(env_ids=env_ids)

        # Reset readiness flags
        self._env.is_ready_for_new_command[env_ids] = True

    def nav_process_actions(self, actions: torch.Tensor):
        """Map nav policy outputs to linear/angular velocities."""
        # Linear velocities (x and y)
        linear_min, linear_max = self.cfg.ranges.v_linear
        linear_actions = torch.tanh(actions[:, :2].clamp(min=-1.0))
        self._nav_actions[:, :2] = linear_actions * (linear_max - linear_min) / 2 + (linear_max + linear_min) / 2

        # Angular velocity (yaw rate)
        angular_min, angular_max = self.cfg.ranges.v_angular
        angular_action = torch.tanh(actions[:, 2].clamp(min=-1.0))
        self._nav_actions[:, 2] = angular_action * (angular_max - angular_min) / 2 + (angular_max + angular_min) / 2

    def process_actions(self, actions: torch.Tensor):
        if not self.observation_reset_done:
            self._reset_observations()
            self.observation_reset_done = True

        # --- Step 0: Check if previous command is achieved ---
        current_pos = self.robot.data.root_pos_w[:, :3]
        current_heading = self.robot.data.heading_w

        # Squared position error
        pos_error_sq = (self.pos_command_w - current_pos).pow_(2).sum(dim=1)
        pos_tol_sq = self.position_tolerance * self.position_tolerance

        # Heading error
        heading_error_abs = wrap_to_pi(self.heading_command_w - current_heading).abs()

        goal_reached = (pos_error_sq < pos_tol_sq) & (heading_error_abs < self.heading_tolerance)
        first_step = self._raw_actions.abs().max(dim=1).values == 0

        # Update readiness
        ready_mask = goal_reached | first_step
        self._env.is_ready_for_new_command[:, 0] = ready_mask

        # --- Step 1: Convert network outputs to world-frame position commands ---
        xy = torch.tanh(actions[:, :2].clamp(min=-1.0))
        heading = torch.tanh(actions[:, 2].clamp(min=-1.0))

        # Precompute new commands
        new_pos_command_w = self.pos_command_w.clone()
        new_pos_command_w[:] = self._env.scene.env_origins
        new_pos_command_w[:, 0] += xy[:, 0] * self.half_range_pos_x
        new_pos_command_w[:, 1] += xy[:, 1] * self.half_range_pos_y
        new_pos_command_w[:, 2] += self.robot.data.default_root_state[:, 2]

        new_heading_command_w = heading * self.half_range_heading

        # --- Step 2: Only update commands if the robot is ready ---
        self.pos_command_w[ready_mask] = new_pos_command_w[ready_mask]
        self.heading_command_w[ready_mask] = new_heading_command_w[ready_mask]

        # --- Step 3: Transform world-frame commands into robot-frame commands ---
        target_vec = self.pos_command_w - current_pos
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - current_heading)

        # Update buffers in-place
        self.last_action.copy_(self._raw_actions)
        self._raw_actions[:, :3] = self.pos_command_b
        self._raw_actions[:, 3] = self.heading_command_b

    def apply_actions(self):
        """Run navigation and locomotion policies to generate joint actions."""
        # Navigation step
        if self._nav_counter % self.cfg.nav_decimation == 0:
            nav_obs = self._nav_obs_manager.compute_group("nav_policy")
            self._nav_actions[:] = self.nav_policy(nav_obs)
            self.nav_process_actions(self._nav_actions)
            self._nav_counter = 0
        self._nav_counter += 1

        # Locomotion step
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
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pose_visualizer.visualize(
            translations=self.pos_command_w[:, :3],
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
        )

    # ---------------------- Internal helpers ----------------------
    def _reset_observations(self):
        self._nav_obs_manager.reset(env_ids=range(self.num_envs))
        self._loco_obs_manager.reset(env_ids=range(self.num_envs))


@configclass
class RobotPlannerActionTrainedNavigationCfg(ActionTermCfg):
    """Configuration for Planner ActionTerm using pre-trained Navigation and Locomotion models."""
    class_type: type[ActionTerm] = RobotPlannerActionTrainedNavigation
    asset_name: str = MISSING

    # Paths to policies
    nav_policy_path: str = MISSING  # predicts velocities from positions
    locomotion_policy_path: str = MISSING  # predicts joint actions from velocities

    locomotion_decimation: int = 4
    locomotion_actions: ActionTermCfg = MISSING
    locomotion_observations: ObservationGroupCfg = MISSING

    nav_decimation: int = 4
    nav_actions: ActionTermCfg = MISSING
    nav_observations: ObservationGroupCfg = MISSING

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING
        heading: tuple[float, float] = MISSING
        v_linear: tuple[float, float] = MISSING
        v_angular: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

    # Debug visualization configuration
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal"
    )
    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)