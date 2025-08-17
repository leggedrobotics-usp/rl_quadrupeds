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
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim + 1, device=self.device)  # [x, y, z, heading]
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

        # Buffers for commands
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
        self.position_tolerance = 0.2
        self.heading_tolerance = 0.1

        # Store obstacles as tensor
        self.obstacles = torch.stack(
            [env.scene[object_name].data.root_pos_w[:, :2] for object_name in ["block1"]],
            dim=1
        ).to(env.device)

        # Storage for intermediate waypoints (for debug visualization)
        self._intermediate_waypoints = [[] for _ in range(self.num_envs)]

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
        return self._nav_actions

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
        self._nav_counter = 0
        self._loco_counter = 0
        self._loco_obs_manager.reset(env_ids=env_ids)
        self._nav_obs_manager.reset(env_ids=env_ids)
        self._env.is_ready_for_new_command[env_ids] = True
        for i in env_ids:
            self._intermediate_waypoints[i] = []

    def nav_process_actions(self, actions: torch.Tensor):
        linear_min, linear_max = self.cfg.ranges.v_linear
        linear_actions = torch.tanh(actions[:, :2].clamp(min=-1.0))
        self._nav_actions[:, :2] = linear_actions * (linear_max - linear_min) / 2 + (linear_max + linear_min) / 2

        angular_min, angular_max = self.cfg.ranges.v_angular
        angular_action = torch.tanh(actions[:, 2].clamp(min=-1.0))
        self._nav_actions[:, 2] = angular_action * (angular_max - angular_min) / 2 + (angular_max + angular_min) / 2

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
        """
        device = robot_pos.device
        eps = 1e-6

        # Ensure batched form
        if robot_pos.ndim == 1:
            robot_pos = robot_pos.unsqueeze(0)
            goal_pos = goal_pos.unsqueeze(0)
            robot_yaw = robot_yaw.unsqueeze(0)
            obstacles = obstacles.unsqueeze(0) if obstacles.ndim == 2 else obstacles

        # Attractive force
        force_goal = goal_pos - robot_pos
        dist_goal = torch.norm(force_goal, dim=1, keepdim=True).clamp_min(eps)
        force_goal = force_goal / dist_goal

        # Repulsive force
        if obstacles.numel() > 0:
            obs_vec = robot_pos.unsqueeze(1) - obstacles
            dist_obs = torch.norm(obs_vec, dim=2, keepdim=True).clamp_min(eps)
            mask = (dist_obs < influence_radius).float()
            repulsion = (obs_vec / (dist_obs ** 2)) * mask
            force_obs = repulsion.sum(dim=1)
        else:
            force_obs = torch.zeros_like(force_goal)

        force_total = force_goal + force_obs
        norm_total = torch.norm(force_total, dim=1, keepdim=True).clamp_min(eps)
        force_total = force_total / norm_total

        next_pos = robot_pos + step_size * force_total
        return next_pos if next_pos.shape[0] > 1 else next_pos.squeeze(0)

    def process_actions(self, actions: torch.Tensor):
        if not self.observation_reset_done:
            self._reset_observations()
            self.observation_reset_done = True

        current_pos = self.robot.data.root_pos_w[:, :3]
        current_heading = self.robot.data.heading_w

        pos_error_sq = (self.pos_command_w - current_pos).pow_(2).sum(dim=1)
        pos_tol_sq = self.position_tolerance * self.position_tolerance
        heading_error_abs = wrap_to_pi(self.heading_command_w - current_heading).abs()

        goal_reached = (pos_error_sq < pos_tol_sq) & (heading_error_abs < self.heading_tolerance)
        first_step = self._raw_actions.abs().max(dim=1).values == 0
        ready_mask = goal_reached | first_step
        self._env.is_ready_for_new_command[:, 0] = ready_mask

        xy = torch.tanh(actions[:, :2].clamp(min=-1.0))
        heading = torch.tanh(actions[:, 2].clamp(min=-1.0))

        new_global_goal_w = self._env.scene.env_origins.clone()
        new_global_goal_w[:, 0] += xy[:, 0] * self.half_range_pos_x
        new_global_goal_w[:, 1] += xy[:, 1] * self.half_range_pos_y
        new_global_goal_w[:, 2] = self.robot.data.default_root_state[:, 2]

        new_heading_command_w = heading * self.half_range_heading

        for i in range(self.num_envs):
            if ready_mask[i]:
                # Store multiple intermediate waypoints from local planner
                waypoints = []
                sim_pos = current_pos[i, :2]
                sim_heading = current_heading[i]

                for _ in range(5):  # simulate 5 steps of the local planner
                    sim_pos = self.simple_local_planner(
                        sim_pos,
                        sim_heading,
                        new_global_goal_w[i, :2],
                        self.obstacles[i]
                    ).to(self.device)
                    waypoints.append(sim_pos.clone())

                waypoints = torch.stack(waypoints)  # shape [N, 2]

                # Compute headings for each waypoint
                headings = []
                for k in range(len(waypoints)):
                    if k < len(waypoints) - 1:
                        delta = waypoints[k + 1] - waypoints[k]
                        hdg = torch.atan2(delta[1], delta[0])
                    else:
                        hdg = new_heading_command_w[i]  # last waypoint uses final heading goal
                    headings.append(hdg)
                headings = torch.stack(headings)

                # Store as (x, y, heading)
                self._intermediate_waypoints[i] = torch.cat(
                    [waypoints, headings.unsqueeze(-1)], dim=1
                )  # shape [N, 3]

                # Final commanded position
                self.pos_command_w[i] = torch.tensor(
                    [waypoints[-1, 0], waypoints[-1, 1], self.robot.data.default_root_state[i, 2]],
                    device=self.device
                )
                self.heading_command_w[i] = new_heading_command_w[i]

        target_vec = self.pos_command_w - current_pos
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - current_heading)

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

        # Visualize intermediate waypoints
        all_waypoints = []
        all_orientations = []
        for i in range(self.num_envs):
            if len(self._intermediate_waypoints[i]) > 0:
                z_val = self.robot.data.default_root_state[i, 2]

                # split out x, y, heading
                xy = self._intermediate_waypoints[i][:, :2]
                hdg = self._intermediate_waypoints[i][:, 2]

                # add z column
                pts = torch.cat(
                    [xy, torch.full((len(xy), 1), z_val, device=self.device)],
                    dim=1
                )
                all_waypoints.append(pts)

                # convert heading -> quaternion
                orients = quat_from_euler_xyz(
                    torch.zeros_like(hdg),
                    torch.zeros_like(hdg),
                    hdg,
                )
                all_orientations.append(orients)

        if all_waypoints:
            all_waypoints = torch.cat(all_waypoints, dim=0)
            all_orientations = torch.cat(all_orientations, dim=0)
            self.waypoints_visualizer.visualize(
                translations=all_waypoints,   # (M, 3)
                orientations=all_orientations # (M, 4)
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
    intermediate_pose_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal"
    )
    intermediate_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)