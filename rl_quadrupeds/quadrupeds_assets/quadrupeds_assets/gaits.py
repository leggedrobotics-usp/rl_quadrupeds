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


class WTWCommandFootStates(ManagerTermBase):
    """
    Computes the desired contact state of each foot from
    the phase and timing variable.
    Taken from the paper "Walk These Ways: Tuning 
        Robot Control for Generalization with 
        Multiplicity of Behavior"
    
    This observation manager externalizes to the environment:
    - foot_indices: the phase of each foot
    - clock_inputs: the timing of each foot
    - desired_contact_states: the desired contact state of each foot
    """

    to_env_vars = [
        "foot_indices",
        "clock_inputs",
        "desired_contact_states",
    ]

    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedEnv,
    ):
        super().__init__(cfg, env)
        # To use it inside the reset function
        self.env = env
        
        # The standard deviation of the smoothing distribution $$\kappa$$
        kappa = cfg.params.get("kappa")
        # The gait type command
        self.gait_type_cmd = cfg.params.get("gait_type_cmd")
        # Used as an approximation to the Von Mises distribution used in the
        # paper "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic
        # Reward Composition"
        self.smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf
        
        env.foot_indices = torch.zeros(
            (env.num_envs, 4),
            device=env.device
        )
        env.desired_contact_states = torch.zeros(
            (env.num_envs, 4), 
            device=env.device
        )
        env.clock_inputs = torch.zeros(
            (env.num_envs, 4), 
            device=env.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the manager term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        if env_ids is None:
            return

        # $$ \theta = [\theta_1^{cmd}, \theta_2^{cmd}, \theta_3^{cmd}] $$
        gait_type = self.env.command_manager.get_command(self.gait_type_cmd)[env_ids]
        """$$
        \begin{cases}
        t = [t^{FR}, t^{FL}, t^{RR}, t^{RL}] \\
        t^{FR} = t + \theta_2^{cmd} + \theta_3^{cmd} \\
        t^{FL} = t + \theta_1^{cmd} + \theta_3^{cmd} \\
        t^{RR} = t + \theta_1^{cmd} \\
        t^{RL} = t + \theta_2^{cmd} \\
        \end{cases}
        $$"""
        self.env.foot_indices[env_ids, 0] = gait_type[:, 1] + gait_type[:, 2]
        self.env.foot_indices[env_ids, 1] = gait_type[:, 0] + gait_type[:, 2]
        self.env.foot_indices[env_ids, 2] = gait_type[:, 0]
        self.env.foot_indices[env_ids, 3] = gait_type[:, 1]
        self.env.foot_indices[env_ids] = torch.remainder(
            self.env.foot_indices[env_ids], 1.0
        )

    def __call__(
        self, 
        env: ManagerBasedRLEnv, 
        gait_duty_cycle_cmd: str,
        gait_step_freq_cmd: str,
        gait_type_cmd: str,
        kappa: float,
    ) -> torch.Tensor:
        """
        Args:
            gait_duty_cycle_cmd: The duty cycle command. It can change during 
                the episode.
            gait_step_freq_cmd: The step frequency command. It can change
                during the episode.
            gait_type_cmd: The gait type command. It SHOULD NOT change
                during the episode. It only changes when the episode restarts.
        """
        step_frequencies = env.command_manager.get_command(gait_step_freq_cmd)
        stance_duty_cycles = env.command_manager.get_command(gait_duty_cycle_cmd)

        # $$t = clip(t, 0, 1)$$
        env.foot_indices = torch.remainder(
            env.foot_indices + env.step_dt * step_frequencies, 
            1.0
        )

        """
        Normalizes the phase $$t$$ between [0, 0.5] for stance and [0.5, 1] for swing.
        $$d_{foot}$$ represents the duty cycle of the foot contact (stance).
        $$ \begin{cases}
        t_{foot} = t_{foot} * 0.5/d_{foot}, \text{stance} \\
        t_{foot} = 0.5 + (t_{foot} - d_{foot})*0.5/(1 - d_{foot}), \text{swing} \\
        \end{cases} $$
        """
        # Expand the durations to enable broadcasting with the masks
        stance_duty_cycles_expanded = stance_duty_cycles.expand(-1, env.foot_indices.shape[1])
        stance_mask = env.foot_indices < stance_duty_cycles_expanded
        swing_mask = env.foot_indices > stance_duty_cycles_expanded
        env.foot_indices[stance_mask] *= (0.5 / stance_duty_cycles_expanded[stance_mask])
        env.foot_indices[swing_mask] = 0.5 + (
            env.foot_indices[swing_mask] - stance_duty_cycles_expanded[swing_mask]
        ) * (
            0.5 / (1 - stance_duty_cycles_expanded[swing_mask])
        )

        """$$
        \mathbf{t}_t = [\sin(2\pi t^{FR}), sin(2\pi t^{FL}), \sin(2\pi t^{RR}), \sin(2\pi t^{RL})]
        $$"""
        env.clock_inputs[:,:] = torch.sin(2 * np.pi * env.foot_indices)
        
        env.foot_indices = torch.remainder(env.foot_indices, 1.0)
        env.desired_contact_states[:, :] = \
            self.smoothing_cdf_start(env.foot_indices) * \
            (1 - self.smoothing_cdf_start(env.foot_indices - 0.5)) + \
            self.smoothing_cdf_start(env.foot_indices - 1) * \
            (1 - self.smoothing_cdf_start(env.foot_indices - 1.5))

        return env.desired_contact_states