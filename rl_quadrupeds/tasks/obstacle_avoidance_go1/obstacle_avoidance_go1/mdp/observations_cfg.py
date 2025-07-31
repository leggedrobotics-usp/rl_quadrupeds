from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup, 
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg
)
from isaaclab.envs.mdp.observations import (
    base_lin_vel,
    generated_commands,
    last_action,
    projected_gravity
)

from quadrupeds_mdp.observations.ray_caster import lidar_scan

@configclass
class NavObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=base_lin_vel)
        projected_gravity = ObsTerm(func=projected_gravity)
        pose_command = ObsTerm(func=generated_commands, params={"command_name": "pose_command"})
        actions = ObsTerm(func=last_action, noise=Unoise(n_min=-0.01, n_max=0.01))
        lidar_scan = ObsTerm(
            func=lidar_scan,
            params={"sensor_cfg": SceneEntityCfg("ray_caster")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()