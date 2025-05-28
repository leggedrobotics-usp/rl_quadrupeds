import math

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.rewards import (                             # MDP rewards
    applied_torque_limits,                                          # https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/envs/mdp/rewards.html#applied_torque_limits 
    lin_vel_z_l2,                                                   # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.lin_vel_z_l2 
    ang_vel_xy_l2,                                                  # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.ang_vel_xy_l2
    joint_torques_l2,                                               # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.joint_torques_l2
    joint_acc_l2,                                                   # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.joint_acc_l2
    action_rate_l2,                                                 # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.action_rate_l2
    undesired_contacts,                                             # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.undesired_contacts
    flat_orientation_l2,                                            # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.flat_orientation_l2
    joint_pos_limits                                                # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.joint_pos_limits
)

from quadrupeds_assets.go1 import GO1_FOOT_NAMES
from quadrupeds_rewards.action import (
    action_smoothness_l1,
    ActionSmoothnessL2
)
from quadrupeds_rewards.angular_velocity import (
    track_ang_vel_z_exp
)
from quadrupeds_rewards.force import (
    track_feet_contact_forces,
    track_feet_contact_schedule_forces,
    track_feet_slip,
    TrackFeetImpactVel
)
from quadrupeds_rewards.linear_velocity import (
    track_feet_contact_velocity,
    track_foot_contact_schedule_velocities,
    track_joint_vel_l2,
    track_lin_vel_xy_exp
)
from quadrupeds_rewards.position import (
    track_base_height_l2_cmd,
    track_base_orientation,
    track_footswing_height,
    track_joint_positions_l2,
    RaibertHeuristic,
)

@configclass
class RewardsCfg:
    # TASK
    _track_lin_vel_xy_exp = RewTerm(
        func=track_lin_vel_xy_exp, 
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25)
        }
    )
    _track_ang_vel_z_exp = RewTerm(
        func=track_ang_vel_z_exp,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25)
        }
    )

    # AUGMENTED AUXILIARY
    _track_feet_contact_schedule_forces = RewTerm(
        func=track_feet_contact_schedule_forces,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", 
                body_names=".*foot"
            ),
            "std": 0.25,
        }
    )
    _track_foot_contact_schedule_velocities = RewTerm(
        func=track_foot_contact_schedule_velocities,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*foot"
            ),
            "std": 0.25,
        }
    )
    # TODO: In WTW code, they use a base height + a jump height
    # For now, I am assuming the sum is 1.3.
    # TODO: it seems that this reward function has problems in rough terrain
    # https://github.com/isaac-sim/IsaacLab/issues/1698
    # _base_height_l2 = RewTerm(
    #     func=base_height_l2, 
    #     weight=-30.0,
    #     params={
    #         "target_height": 1.3,
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #     }
    # )
    _based_height_l2_cmd = RewTerm(
        func=track_base_height_l2_cmd,
        weight=0.0,
        params={
            "command_name": "body_height_orientation_cmd",
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        }
    )
    _track_base_orientation = RewTerm(
        func=track_base_orientation,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names="trunk",
                preserve_order=True
            ),
            "orientation_cmd": "body_height_orientation_cmd",
        }
    )
    raibert_footswing = RewTerm(
        func=RaibertHeuristic,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "gait_stance_distances_cmd": "gait_stance_distances_cmd",
            "gait_step_freq_cmd": "gait_freq_cmd",
            "robot_cfg": SceneEntityCfg(
                "robot", body_names=GO1_FOOT_NAMES, preserve_order=True
            )
        },
    )
    _track_footswing_height = RewTerm(
        func=track_footswing_height,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=GO1_FOOT_NAMES, 
                preserve_order=True
            ),
            "foot_height_cmd": "footswing_height_cmd",
        }
    )

    # FIXED AUXILIARY
    _lin_vel_z_l2 = RewTerm(func=lin_vel_z_l2, weight=-0.5)
    _ang_vel_xy_l2 = RewTerm(func=ang_vel_xy_l2, weight=-0.05)
    _track_feet_slip = RewTerm(
        func=track_feet_slip,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*foot"
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*foot"
            ),
        }
    )
    _undesired_contacts = RewTerm(
        func=undesired_contacts,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", 
                body_names=".*_(thigh|calf)"
            ),
            "threshold": 1.0
        }
    )
    _joint_pos_limits = RewTerm(func=joint_pos_limits, weight=-1)
    _joint_torques_l2 = RewTerm(func=joint_torques_l2, weight=-2.0e-4)
    _track_joint_vel_l2 = RewTerm(func=track_joint_vel_l2, weight=0.0)
    _joint_acc_l2 = RewTerm(func=joint_acc_l2, weight=-2.5e-7)
    _action_smoothness_l1 = RewTerm(func=action_smoothness_l1, weight=-0.01)
    _action_smoothness_l2 = RewTerm(func=ActionSmoothnessL2, weight=-0.01)

    # ONLY IN WTW CODE
    _action_rate_l2 = RewTerm(func=action_rate_l2, weight=-0.01)
    _flat_orientation_l2 = RewTerm(func=flat_orientation_l2, weight=0.0)
    _track_joint_positions_l2 = RewTerm(func=track_joint_positions_l2, weight=-0.05)
    _track_feet_contact_velocity = RewTerm(
        func=track_feet_contact_velocity,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*foot"
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*foot"
            ),
        }
    )
    _track_feet_contact_forces = RewTerm(
        func=track_feet_contact_forces,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*foot"
            ),
            "max_contact_force": 100.0,
        }
    )
    _track_feet_impact_vel = RewTerm(
        func=TrackFeetImpactVel,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*foot"
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*foot"
            ),
        }
    )

    # Lab code
    joint_torque_limit = RewTerm(
        func=applied_torque_limits,
        weight=-1.0
    )