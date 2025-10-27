"""
gaits.py

Observation managers for the quadruped robot.

Available classes:
- WTWCommandFootStates: Computes the desired contact state of each foot from
    the phase and timing variable. Externalizes to the environment:
    - foot_indices: the phase of each foot
    - clock_inputs: the timing of each foot
    - desired_contact_states: the desired contact state of each foot
"""

"""
Defines the timing offsets between pairs of feet.
Taken from the paper "Walk These Ways: Tuning 
    Robot Control for Generalization with 
    Multiplicity of Behavior"

- Pronking is when the quadruped is "lifting 
all four feet off the ground simultaneously".
[https://en.wikipedia.org/wiki/Stotting]

- Trotting is "a two-beat diagonal horse gait 
where the diagonal pairs of legs move forward
at the same time with a moment of suspension
between each beat". 
[https://en.wikipedia.org/wiki/Trot]

- Bounding is similar to Pronking, but it has 
 a "longer duration of the aerial phase and 
 higher angle of initial launch"
[https://en.wikipedia.org/wiki/Jumping]

- Pacing is "a lateral two-beat gait. In the 
pace, the two legs on the same side of the 
horse move forward together"
[https://en.wikipedia.org/wiki/Horse_gait#Pace]
"""
GAIT_TIMING_OFFSETS = {
    "pronking": [0, 0, 0],
    "trotting": [0.5, 0, 0],
    "bounding": [0, 0.5, 0],
    "pacing": [0, 0, 0.5],
}

from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.managers import SceneEntityCfg

from .go1 import GO1_FOOT_NAMES

class WTWCommandFootStates(ManagerTermBase):
    """
    Computes both the desired and measured contact states / phases of each foot.

    Desired values come from the commanded gait pattern.
    Measured values are estimated from actual contact transitions.

    Exports to env:
    - foot_indices: desired gait phase [0,1)
    - clock_inputs: sinusoidal timing input
    - desired_contact_states: desired stance/swing indicator
    - measured_foot_phase: measured gait phase inferred from contact events
    """

    to_env_vars = [
        "foot_indices",
        "clock_inputs",
        "desired_contact_states",
        "measured_foot_phase",
    ]

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env

        # --- Parameters ---
        self.kappa = cfg.params.get("kappa", 0.05)
        self.gait_type_cmd = cfg.params.get("gait_type_cmd")
        self.contact_threshold = cfg.params.get("contact_threshold", 1.0)
        self.default_expected_period = cfg.params.get("expected_period", 0.5)  # seconds per full cycle

        # --- Von Mises smoothing approximation ---
        self.smoothing_cdf_start = torch.distributions.normal.Normal(0, self.kappa).cdf

        # --- Desired gait buffers ---
        env.foot_indices = torch.zeros((env.num_envs, 4), device=env.device)
        env.desired_contact_states = torch.zeros((env.num_envs, 4), device=env.device)
        env.clock_inputs = torch.zeros((env.num_envs, 4), device=env.device)

        # --- Measured gait buffers ---
        env.measured_foot_phase = torch.zeros((env.num_envs, 4), device=env.device)
        self.prev_contacts = torch.zeros((env.num_envs, 4), device=env.device)
    # -------------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets desired and measured gait phase variables."""
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)

        gait_type = self.env.command_manager.get_command(self.gait_type_cmd)[env_ids]

        # Desired phase initialization (from paper)
        self.env.foot_indices[env_ids, 0] = gait_type[:, 0] + gait_type[:, 2]  # FL
        self.env.foot_indices[env_ids, 1] = gait_type[:, 1] + gait_type[:, 2]  # FR
        self.env.foot_indices[env_ids, 2] = gait_type[:, 1]                    # RL
        self.env.foot_indices[env_ids, 3] = gait_type[:, 0]                    # RR
        self.env.foot_indices[env_ids] = torch.remainder(self.env.foot_indices[env_ids], 1.0)

        # Reset measured phase buffers
        self.env.measured_foot_phase[env_ids] = 0.0
        self.prev_contacts[env_ids] = 0.0

    # -------------------------------------------------------------------------

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        gait_duty_cycle_cmd: str,
        gait_step_freq_cmd: str,
        gait_type_cmd: str,
        sensor_cfg: SceneEntityCfg,
        kappa: float,
    ) -> torch.Tensor:
        """Update desired gait phase and measure actual phase from contacts."""

        # --- Commands ---
        step_frequencies = env.command_manager.get_command(gait_step_freq_cmd)  # [num_envs]
        stance_duty_cycles = env.command_manager.get_command(gait_duty_cycle_cmd)

        # ---------------------------------------------------------------------
        # 🦾 Desired Gait Phase (as in original WTW)
        # ---------------------------------------------------------------------
        env.foot_indices = torch.remainder(
            env.foot_indices + env.step_dt * step_frequencies, 1.0
        )

        stance_duty_cycles_expanded = stance_duty_cycles.expand(-1, env.foot_indices.shape[1])
        stance_mask = env.foot_indices < stance_duty_cycles_expanded
        swing_mask = env.foot_indices > stance_duty_cycles_expanded

        env.foot_indices[stance_mask] *= (0.5 / stance_duty_cycles_expanded[stance_mask])
        env.foot_indices[swing_mask] = 0.5 + (
            env.foot_indices[swing_mask] - stance_duty_cycles_expanded[swing_mask]
        ) * (0.5 / (1 - stance_duty_cycles_expanded[swing_mask]))

        env.clock_inputs[:] = torch.sin(2 * np.pi * env.foot_indices)
        env.foot_indices = torch.remainder(env.foot_indices, 1.0)

        env.desired_contact_states[:, :] = (
            self.smoothing_cdf_start(env.foot_indices)
            * (1 - self.smoothing_cdf_start(env.foot_indices - 0.5))
            + self.smoothing_cdf_start(env.foot_indices - 1)
            * (1 - self.smoothing_cdf_start(env.foot_indices - 1.5))
        )

        # ---------------------------------------------------------------------
        # 🦶 Measured Gait Phase Estimation (Contact-driven finite-state)
        # ---------------------------------------------------------------------
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, 2]  # [num_envs, 4]
        contacts = (net_contact_forces > self.contact_threshold).float()

        dt = env.step_dt
        phase_inc = dt * step_frequencies

        measured_phase = env.measured_foot_phase
        prev_contacts = self.prev_contacts

        # Detect transitions
        touchdown = (contacts == 1) & (prev_contacts == 0)
        liftoff = (contacts == 0) & (prev_contacts == 1)

        # Advance phase
        measured_phase += phase_inc

        # Reset phase at touchdown/liftoff
        measured_phase[touchdown] = 0.0
        measured_phase[liftoff] = 0.5

        measured_phase[:] = torch.remainder(measured_phase, 1.0)

        # Store for next step
        self.prev_contacts[:] = contacts
        env.measured_foot_phase[:] = measured_phase

        # print(f"Raw Actions: {env.action_manager._terms['joint_pos'].raw_actions}")
        # print(f"Processed Actions: {env.action_manager._terms['joint_pos'].processed_actions}")

        # ---------------------------------------------------------------------
        # Return sinusoidal features for consistency with lab’s expected API
        # ---------------------------------------------------------------------
        return torch.cat(
            [
                torch.sin(2 * np.pi * env.foot_indices),
                torch.cos(2 * np.pi * env.foot_indices),
            ],
            dim=1,
        )