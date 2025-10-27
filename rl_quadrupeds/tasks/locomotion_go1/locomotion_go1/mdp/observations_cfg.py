from isaaclab.envs.mdp import (                 # MDP utils
    base_lin_vel,                               # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.observations.base_lin_vel
    base_ang_vel,                               # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.observations.base_ang_vel
    generated_commands,                         # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.observations.generated_commands
)
from isaaclab.envs.mdp.observations import (    # MDP observations
    joint_pos_rel,                              # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.observations.joint_pos_rel
    joint_vel_rel,                              # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.observations.joint_vel_rel
    last_action,                                # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.observations.last_action
    projected_gravity                           # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.observations.projected_gravity
)
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup, 
    ObservationTermCfg as ObsTerm
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from quadrupeds_mdp.assets.gaits import (
    WTWCommandFootStates
)
from isaaclab.managers import SceneEntityCfg

from quadrupeds_mdp.assets.go1 import (
    GO1_FOOT_NAMES
)

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Commands
        history_length = 20

        velocity_commands = ObsTerm(func=generated_commands, params={"command_name": "base_velocity"})
        # base_height_orientation = ObsTerm(func=generated_commands, params={"command_name": "body_height_orientation_cmd"})
        # step_freq = ObsTerm(func=generated_commands, params={"command_name": "gait_freq_cmd"})
        # gait_type = ObsTerm(func=generated_commands, params={"command_name": "gait_type_cmd"})
        # gait_duty_cycle = ObsTerm(func=generated_commands, params={"command_name": "duty_cycle_cmd"})
        # gait_footswing_height = ObsTerm(func=generated_commands, params={"command_name": "footswing_height_cmd"})
        # gait_stance_distances = ObsTerm(func=generated_commands, params={"command_name": "gait_stance_distances_cmd"})
        clock = ObsTerm(
            func=WTWCommandFootStates,
            params={
                "gait_duty_cycle_cmd": "duty_cycle_cmd",
                "gait_step_freq_cmd": "gait_freq_cmd",
                "gait_type_cmd": "gait_type_cmd",
                "kappa": 0.07,
                "sensor_cfg": SceneEntityCfg(
                    name="contact_forces",
                    body_names=GO1_FOOT_NAMES
                ),
            }
        )
        _projected_gravity = ObsTerm(func=projected_gravity)
        joint_pos = ObsTerm(func=joint_pos_rel)
        joint_vel = ObsTerm(func=joint_vel_rel)
        actions = ObsTerm(func=last_action)
        _base_lin_vel = ObsTerm(func=base_lin_vel)
        _base_ang_vel = ObsTerm(func=base_ang_vel)

        def __post_init__(self):
            self.enable_corruption = True       # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.ObservationGroupCfg.enable_corruption
            self.concatenate_terms = True       # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.ObservationGroupCfg.concatenate_terms

    policy: PolicyCfg = PolicyCfg()

    # Check sim2real perfomance and # uncomment to use the value observation group.
    # @configclass
    # class ValueCfg(ObsGroup):
    #     _projected_gravity = ObsTerm(func=projected_gravity, noise=Unoise(n_min=-0.1, n_max=0.1))
    #     _oi = ObsTerm(func=oi)
    # value: ValueCfg = ValueCfg()