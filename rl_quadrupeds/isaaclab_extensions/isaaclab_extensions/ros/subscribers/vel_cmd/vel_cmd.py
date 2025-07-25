from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import rclpy
import torch

from isaaclab.managers import CommandTerm
from isaaclab_extensions.ros.subscriber import ROSSubscriber

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from.vel_cmd_cfg import ROSVelCmdSubscriberCfg

class ROSVelCmdSubscriber(ROSSubscriber, CommandTerm):
    """
    Command generator that subscribes to a ROS topic for velocity commands.
    """
    cfg: ROSVelCmdSubscriberCfg
    
    def __init__(self, cfg: ROSVelCmdSubscriberCfg, env: ManagerBasedEnv):
        CommandTerm.__init__(self, cfg, env)
        ROSSubscriber.__init__(self, cfg)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # x vel, y vel, yaw vel
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

        # metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def callback(self, msg):
        self.vel_command_b[:, 0] = msg.twist.linear.x
        self.vel_command_b[:, 1] = msg.twist.linear.y
        self.vel_command_b[:, 2] = msg.twist.angular.z

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        rclpy.spin_once(self, timeout_sec=0)

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )