# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import NormalVelocityCommandCfg, UniformVelocityCommandCfg

class UniformVelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution."""

    cfg: UniformVelocityCommandCfg

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading` "
                "parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}' "
                "but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # robot asset
        self.robot: Articulation = env.scene[cfg.asset_name]

        # command buffers
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)

        # metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

        # per-env command ranges (initialized equal to cfg)
        self.lin_vel_x_range = torch.tensor(self.cfg.ranges.lin_vel_x, device=self.device).repeat(self.num_envs, 1)
        self.lin_vel_y_range = torch.tensor(self.cfg.ranges.lin_vel_y, device=self.device).repeat(self.num_envs, 1)
        self.ang_vel_z_range = torch.tensor(self.cfg.ranges.ang_vel_z, device=self.device).repeat(self.num_envs, 1)
        if self.cfg.heading_command and self.cfg.ranges.heading is not None:
            self.heading_range = torch.tensor(self.cfg.ranges.heading, device=self.device).repeat(self.num_envs, 1)
        else:
            self.heading_range = None

    def __str__(self) -> str:
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    # ---------------------
    # Properties
    # ---------------------

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    # ---------------------
    # Public API
    # ---------------------

    def set_command_range_for_envs(self, env_ids: Sequence[int], ranges):
        """Update command ranges for specific environments.
        `ranges` should be an object or dict with attributes/keys:
        lin_vel_x, lin_vel_y, ang_vel_z, (optional) heading.
        """
        def _to_tensor(val): return torch.tensor(val, device=self.device).expand(len(env_ids), 2)
        if env_ids is None:
            env_ids = list(range(self.num_envs))
    
        self.lin_vel_x_range[env_ids] = _to_tensor(ranges["lin_vel_x"] if isinstance(ranges, dict) else ranges.lin_vel_x)
        self.lin_vel_y_range[env_ids] = _to_tensor(ranges["lin_vel_y"] if isinstance(ranges, dict) else ranges.lin_vel_y)
        self.ang_vel_z_range[env_ids] = _to_tensor(ranges["ang_vel_z"] if isinstance(ranges, dict) else ranges.ang_vel_z)
        if self.heading_range is not None and (
            (isinstance(ranges, dict) and "heading" in ranges) or hasattr(ranges, "heading")
        ):
            heading_val = ranges["heading"] if isinstance(ranges, dict) else ranges.heading
            self.heading_range[env_ids] = _to_tensor(heading_val)

    # ---------------------
    # Core Methods
    # ---------------------

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands per env using per-env ranges
        n = len(env_ids)
        r = torch.rand(n, device=self.device)

        self.vel_command_b[env_ids, 0] = (
            r * (self.lin_vel_x_range[env_ids, 1] - self.lin_vel_x_range[env_ids, 0])
            + self.lin_vel_x_range[env_ids, 0]
        )
        self.vel_command_b[env_ids, 1] = (
            r * (self.lin_vel_y_range[env_ids, 1] - self.lin_vel_y_range[env_ids, 0])
            + self.lin_vel_y_range[env_ids, 0]
        )
        self.vel_command_b[env_ids, 2] = (
            r * (self.ang_vel_z_range[env_ids, 1] - self.ang_vel_z_range[env_ids, 0])
            + self.ang_vel_z_range[env_ids, 0]
        )

        if self.cfg.heading_command and self.heading_range is not None:
            self.heading_target[env_ids] = (
                r * (self.heading_range[env_ids, 1] - self.heading_range[env_ids, 0])
                + self.heading_range[env_ids, 0]
            )
            self.is_heading_env[env_ids] = (
                torch.rand(n, device=self.device) <= self.cfg.rel_heading_envs
            )

        self.is_standing_env[env_ids] = (
            torch.rand(n, device=self.device) <= self.cfg.rel_standing_envs
        )

        if self.cfg.noise_ranges is not None:
            self.sampled_vel_command_b = self.vel_command_b.clone()

    def _update_command(self):
        if hasattr(self, "sampled_vel_command_b"):
            self.vel_command_b = self.sampled_vel_command_b.clone()
        else:
            self.vel_command_b = self.vel_command_b.clone()

        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
            )
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.ang_vel_z_range[env_ids, 0].min().item(),
                max=self.ang_vel_z_range[env_ids, 1].max().item(),
            )

        if self.cfg.noise_ranges is not None:
            env_ids = (
                torch.rand(self.num_envs, device=self.device) <= self.cfg.rel_noise_envs
            ).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                r = torch.empty(len(env_ids), device=self.device)
                self.vel_command_b[env_ids, 0] += r.uniform_(*self.cfg.noise_ranges.lin_vel_x)
                self.vel_command_b[env_ids, 1] += r.uniform_(*self.cfg.noise_ranges.lin_vel_y)
                self.vel_command_b[env_ids, 2] += r.uniform_(*self.cfg.noise_ranges.ang_vel_z)
                self.vel_command_b[env_ids, 0] = torch.clip(
                    self.vel_command_b[env_ids, 0],
                    min=self.lin_vel_x_range[env_ids, 0].min().item(),
                    max=self.lin_vel_x_range[env_ids, 1].max().item(),
                )
                self.vel_command_b[env_ids, 1] = torch.clip(
                    self.vel_command_b[env_ids, 1],
                    min=self.lin_vel_y_range[env_ids, 0].min().item(),
                    max=self.lin_vel_y_range[env_ids, 1].max().item(),
                )
                self.vel_command_b[env_ids, 2] = torch.clip(
                    self.vel_command_b[env_ids, 2],
                    min=self.ang_vel_z_range[env_ids, 0].min().item(),
                    max=self.ang_vel_z_range[env_ids, 1].max().item(),
                )

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat