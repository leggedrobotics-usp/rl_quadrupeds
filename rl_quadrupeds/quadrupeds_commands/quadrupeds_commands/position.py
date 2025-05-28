from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

from quadrupeds_assets.utils import (
    euler_to_quat,
    quat_to_yaw,
    quat_mul
)

if TYPE_CHECKING:
    from .commands_cfg import (
        QuadrupedBaseHeightOrientationCommandCfg,
        QuadrupedGaitFootswingHeightCommandCfg,   
    )

class QuadrupedGaitFootswingHeightCommand(CommandTerm):
    """
    Command generator that generates height commands for a quadruped robot.
    """

    cfg: QuadrupedGaitFootswingHeightCommandCfg


    def __init__(self, cfg: QuadrupedGaitFootswingHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.footswing_height_cmd = torch.zeros(self.num_envs, 1, device=self.device)

    def __str__(self) -> str:
        msg = "QuadrupedGaitFootswingHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired height in base frame. Shape is (num_envs, 1)."""
        return self.footswing_height_cmd

    def _update_metrics(self):
        pass
        
    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), 1, device=self.device)
        self.footswing_height_cmd[env_ids] = r.uniform_(
            self.cfg.ranges.footswing_height[0], self.cfg.ranges.footswing_height[1]
        )

    def _update_command(self):
        pass

class QuadrupedBaseHeightOrientationCommand(CommandTerm):
    """
    Command generator that generates height commands for a quadruped robot.
    """

    cfg: QuadrupedBaseHeightOrientationCommandCfg

    def __init__(self, cfg: QuadrupedBaseHeightOrientationCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        # orientation is pitch and roll
        self.base_height_orientation = torch.zeros(self.num_envs, 3, device=self.device)

    def __str__(self) -> str:
        msg = "QuadrupedBaseHeightOrientationCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired height in base frame. Shape is (num_envs, 1)."""
        return self.base_height_orientation

    def _update_metrics(self):
        pass
        
    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), 3, device=self.device)
        self.base_height_orientation[env_ids, 0] = r[:, 0].uniform_(
            self.cfg.ranges.height[0], self.cfg.ranges.height[1]
        )
        self.base_height_orientation[env_ids, 1] = r[:, 1].uniform_(
            self.cfg.ranges.pitch[0], self.cfg.ranges.pitch[1]
        )
        self.base_height_orientation[env_ids, 2] = r[:, 2].uniform_(
            self.cfg.ranges.roll[0], self.cfg.ranges.roll[1]
        )

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "height_orientation_visualizer"):
                self.height_orientation_visualizer = VisualizationMarkers(self.cfg.height_orientation_visualizer)
            # set their visibility to true
            self.height_orientation_visualizer.set_visibility(True)
        else:
            if hasattr(self, "height_orientation_visualizer"):
                self.height_orientation_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        yaw = quat_to_yaw(self.robot.data.root_quat_w)
        q_pitch_roll = euler_to_quat(
            self.base_height_orientation[:, 1],
            self.base_height_orientation[:, 2],
            yaw
        )
        q_yaw = euler_to_quat(
            torch.zeros_like(self.robot.data.root_quat_w[:, 2]),
            torch.zeros_like(self.robot.data.root_quat_w[:, 2]),
            self.robot.data.root_quat_w[:, 2]
        )
        self.height_orientation_visualizer.visualize(
            translations=torch.cat([
                self.robot.data.root_pos_w[:, :2],
                self.base_height_orientation[:, 0].unsqueeze(1)],
                dim=1,
            ),
            orientations=quat_mul(q_yaw, q_pitch_roll),
        )