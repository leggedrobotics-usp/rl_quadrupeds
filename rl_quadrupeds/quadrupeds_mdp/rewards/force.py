"""
force.py

Reward functions related to applied forces.

Avaliable functions:
- track_feet_contact_schedule_forces: Tracks the contact forces of the feet and penalizes the deviation from the desired contact states.
- track_feet_slip: Tracks if the feet are slipping by multiplying the contact sensor data with the feet xy velocities.
- track_feet_contact_forces: Tracks the contact force between the feet and the ground.

Avaliable classes:
- TrackFeetImpactVel: Tracks the impact velocity of the feet.
"""

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor

@torch.no_grad()
def track_feet_contact_schedule_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    std: float
) -> torch.Tensor:
    """
    Tracks the contact forces of the feet and penalizes the 
    deviation from the desired contact states.

    
    $$-\frac{1}{4} \sum_{i=0}^{3} \left(1 - D_i\right) \left(1 - \exp\left(-\frac{F_i^2}{\sigma}\right)\right)$$
    
    
    ATTENTION: to use this reward function, the environment must
    contain the quadrupeds_assets.gaits.WTWCommandFootStates inside
    the command_manager. This is because this function requires
    the desired_contact_states variable.
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # Get the magnitude of the forces acting on each feet
    feet_forces = torch.norm(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :],
        dim=-1
    )

    exp_term = torch.exp(-(feet_forces ** 2) / std)
    return torch.mean(
        -(1 - env.desired_contact_states) * (1 - exp_term), 
        dim=1
    )

def track_feet_slip(env, sensor_cfg, asset_cfg, k: float = 5.0):
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_cfg.name]

    contact = torch.logical_or(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1,
        contact_sensor.data.net_forces_w_history[:, -2, sensor_cfg.body_ids, 2] > 1,
    )
    feet_velocities = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 0:2], dim=2)
    slip_penalty = torch.sum(contact * torch.square(feet_velocities), dim=1)
    penalty = -torch.exp(-k * slip_penalty)  # [-1, 0]
    return penalty

def track_feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_contact_force: float,
) -> torch.Tensor:
    """
    Tracks the contact force between the feet and the ground.
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    return torch.sum(
        (torch.norm(
            contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :],
            dim=2
        ) - max_contact_force).clip(min=0.),
        dim=1
    )

class TrackFeetImpactVel(ManagerTermBase):
    def __init__(
        self,
        cfg: RewardTermCfg, 
        env: ManagerBasedRLEnv
    ):
        self.last_feet_velocities = None
        self.asset = env.scene[cfg.params.get("asset_cfg").name]
        self.contact_sensor = env.scene[cfg.params.get("sensor_cfg").name]
    
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        if self.last_feet_velocities is None:
            self.last_feet_velocities = torch.clone(
                self.asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2],
            )
            return 0

        contact = torch.norm(
            self.contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :],
            dim=-1
        ) > 1.0
        self.last_feet_velocities = torch.clone(
            self.asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2],
        )

        return torch.sum(
            contact*torch.square(
                torch.clip(
                    self.last_feet_velocities, -100, 0
                )
            ),
            dim=1
        )

def applied_torque_limits_bonus(env, asset_cfg=None, k: float = 2.0) -> torch.Tensor:
    """Bonus: keep applied torques within limits — returns [0, 1]."""
    asset = env.scene[asset_cfg.name]
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    penalty_val = torch.sum(out_of_limits, dim=1)
    bonus = torch.exp(-k * penalty_val)
    return torch.clip(bonus, 0.0, 1.0)

def joint_torques_l2(env, asset_cfg=SceneEntityCfg("robot"), k: float = 5.0):
    asset = env.scene[asset_cfg.name]
    penalty_val = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    penalty = -torch.exp(-k * penalty_val)  # [-1, 0]
    return penalty