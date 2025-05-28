from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from .commands_cfg import (
        QuadrupedGaitDutyCycleCommandCfg,
        QuadrupedGaitFrequencyCommandCfg,
        QuadrupedGaitStanceDistancesCommandCfg,
        QuadrupedGaitTypeCommandCfg
    )

class QuadrupedGaitDutyCycleCommand(CommandTerm):
    """
    Command generator that generates the gait duty cycle command.
    """

    cfg: QuadrupedGaitDutyCycleCommandCfg

    def __init__(self, cfg: QuadrupedGaitDutyCycleCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.stance_duty_cycle = torch.zeros(self.num_envs, 1, device=self.device)

    def __str__(self) -> str:
        msg = "QuadrupedGaitDutyCycleCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired height in base frame. Shape is (num_envs, 1)."""
        return self.stance_duty_cycle

    def _update_metrics(self):
        pass
        
    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), 1, device=self.device)
        self.stance_duty_cycle[env_ids] = r.uniform_(
            self.cfg.ranges.duty_cycle[0], self.cfg.ranges.duty_cycle[1]
        )

    def _update_command(self):
        pass

class QuadrupedGaitFrequencyCommand(CommandTerm):
    """
    Command generator that generates gait frequencies for a quadruped robot.
    """

    cfg: QuadrupedGaitFrequencyCommandCfg

    def __init__(self, cfg: QuadrupedGaitFrequencyCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.freq_command = torch.zeros(self.num_envs, 1, device=self.device)

    def __str__(self) -> str:
        msg = "QuadrupedGaitFrequencyCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired height in base frame. Shape is (num_envs, 1)."""
        return self.freq_command

    def _update_metrics(self):
        pass
        
    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), 1, device=self.device)
        self.freq_command[env_ids] = r.uniform_(
            self.cfg.ranges.frequency[0], self.cfg.ranges.frequency[1]
        )

    def _update_command(self):
        pass

class QuadrupedGaitStanceDistancesCommand(CommandTerm):
    """
    Command generator that generates gait stance distances for a quadruped robot.
    """

    cfg: QuadrupedGaitStanceDistancesCommandCfg

    def __init__(self, cfg: QuadrupedGaitStanceDistancesCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.gait_stance_distances = torch.zeros(self.num_envs, 2, device=self.device)

    def __str__(self) -> str:
        msg = "QuadrupedGaitStanceDistancesCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired gait stance distances. Shape is (num_envs, 2)."""
        return self.gait_stance_distances

    def _update_metrics(self):
        pass
        
    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), 2, device=self.device)
        self.gait_stance_distances[env_ids, 0] = r[:, 0].uniform_(
            self.cfg.ranges.width[0], self.cfg.ranges.width[1]
        )
        self.gait_stance_distances[env_ids, 1] = r[:, 1].uniform_(
            self.cfg.ranges.length[0], self.cfg.ranges.length[1]
        )

    def _update_command(self):
        pass

class QuadrupedGaitTypeCommand(CommandTerm):
    """
    Command generator that samples gait types for a quadruped robot.
    """

    cfg: QuadrupedGaitTypeCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: QuadrupedGaitTypeCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.gait_theta = torch.zeros(self.num_envs, 3, device=self.device)
        self.desired_gait_types = torch.tensor(
            cfg.desired_gait_types, device=self.device, dtype=torch.float)

    def __str__(self) -> str:
        msg = "QuadrupedGaitTypeCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired gait offsets (theta) (num_envs, 3)."""
        return self.gait_theta

    def _update_metrics(self):
        pass
        
    def _resample_command(self, env_ids: Sequence[int]):
        sampled_indices = torch.randint(
            0, self.desired_gait_types.shape[0], (env_ids.shape[0],), 
            device=self.device
        )
        self.gait_theta[env_ids] = self.desired_gait_types[sampled_indices]

    def _update_command(self):
        pass