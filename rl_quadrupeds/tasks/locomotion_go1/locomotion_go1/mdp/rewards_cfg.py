import math

from isaaclab.envs.mdp.rewards import (             # MDP rewards
    applied_torque_limits,                          # https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/envs/mdp/rewards.html#applied_torque_limits 
    lin_vel_z_l2,                                   # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.lin_vel_z_l2 
    ang_vel_xy_l2,                                  # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.ang_vel_xy_l2
    joint_acc_l2,                                   # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.joint_acc_l2
    action_rate_l2,                                 # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.action_rate_l2
    # undesired_contacts,                             # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.undesired_contacts
    flat_orientation_l2,                            # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.flat_orientation_l2
    joint_pos_limits,                               # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.joint_pos_limits
    is_terminated,                                  # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.is_terminated
    is_terminated_term
)
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from quadrupeds_mdp.assets.go1 import (
    GO1_FOOT_NAMES,
    GO1_JOINT_NAMES
)
from quadrupeds_mdp.rewards.angular_velocity import (
    track_ang_vel_z_exp
)
from quadrupeds_mdp.rewards.linear_velocity import (
    track_lin_vel_xy_exp
)
from quadrupeds_mdp.rewards.position import (
    DiagonalFootTrotPositive,
    DiagonalFootTrotNegative,
    DiagonalJointSymmetryReward,
    DiagonalMotionBalanceReward,
    FootDeviationPenalty,
    track_base_height_l2_cmd_no_sensor,
    track_footswing_height,
    track_joint_positions_l2,
    penalize_hip_movement,
    NoMotionWhenStationary,
    AllFeetOffGroundPenalty
)
from quadrupeds_mdp.rewards.contact import get_illegal_contact
from quadrupeds_mdp.rewards.force import track_feet_contact_schedule_forces
from quadrupeds_mdp.rewards.linear_velocity import track_joint_vel_l2
from isaaclab.envs.mdp.rewards import joint_torques_l2
from quadrupeds_mdp.rewards.position import RaibertHeuristic

from quadrupeds_mdp.rewards.force import track_feet_slip

@configclass
class RewardsCfg:
    # Task
    _track_lin_vel_xy_exp = RewTerm(            # In WTW paper and code
        func=track_lin_vel_xy_exp, 
        weight=8,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25)
        }
    )
    _track_ang_vel_z_exp = RewTerm(             # In WTW paper and code
        func=track_ang_vel_z_exp,
        weight=7,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25)
        }
    )
    # Augmented Auxiliary
    _track_footswing_height = RewTerm(          # In WTW paper and code
        func=track_footswing_height,
        weight=-10,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=GO1_FOOT_NAMES, 
                preserve_order=True
            ),
            "foot_height_cmd": "footswing_height_cmd",
            "command_name": "base_velocity",
        }
    )
    # Fixed Auxiliary
    _lin_vel_z_l2 = RewTerm(func=lin_vel_z_l2, weight=-20)          # In WTW paper and code
    _ang_vel_xy_l2 = RewTerm(func=ang_vel_xy_l2, weight=-0.016)         # In WTW paper and code
    # _undesired_contacts = RewTerm(                                    # In WTW paper and code
    #     func=undesired_contacts,
    #     weight=-1000,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces", 
    #             body_names=".*_(thigh|calf)"
    #         ),
    #         "threshold": 1.0
    #     }
    # )
    # _illegal_contacts = RewTerm(
    #     func=get_illegal_contact,
    #     weight=-1,
    # )
    _joint_pos_limits = RewTerm(func=joint_pos_limits, weight=-5)                  # In WTW paper and code
    _joint_acc_l2 = RewTerm(func=joint_acc_l2, weight=-0.00000005)                     # In WTW paper and code
    _action_rate_l2 = RewTerm(func=action_rate_l2, weight=-0.02)                       # In WTW paper and code

    # Only present in WTW code
    _flat_orientation_l2 = RewTerm(func=flat_orientation_l2, weight=-25)            # Only in WTW code
    _track_joint_positions_l2 = RewTerm(func=track_joint_positions_l2, weight=-0.3) # Only in WTW code
    _penalize_hip_movement = RewTerm(func=penalize_hip_movement, weight=-1.5, params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            body_names=[".*_hip"],
            preserve_order=True
        ),
    })

    # Only present in laboratory code
    joint_torque_limit = RewTerm(
        func=applied_torque_limits,
        weight=-250
    )

    # I chose to use it
    is_terminated = RewTerm(
        func=is_terminated_term,
        weight=-1000,
        params={"term_keys": "base_contact"}
    )

    _penalize_foot_deviation_from_default = RewTerm(
        func=FootDeviationPenalty,
        weight=-15,
        params={
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=GO1_FOOT_NAMES,
                preserve_order=True
            ),
        }
    )
    _track_base_height_l2_cmd_no_sensor = RewTerm(
        func=track_base_height_l2_cmd_no_sensor,
        weight=-19,
        params={
            "command_name": "body_height_orientation_cmd",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names="trunk",
                preserve_order=True
            ),
        }
    )

    # Positive reward term: encourages proper trot
    _trot_gait_positive = RewTerm(
        func=DiagonalFootTrotPositive,
        weight=0.4,  # positive reward
        params={
            "robot_cfg": SceneEntityCfg(name="robot", body_names=GO1_FOOT_NAMES),
            "scale": 8.0,
        }
    )

    # Negative reward term: penalizes undesired gaits (pacing, bounding, pronking)
    _trot_gait_negative = RewTerm(
        func=DiagonalFootTrotNegative,
        weight=-0.4,
        params={
            "robot_cfg": SceneEntityCfg(name="robot", body_names=GO1_FOOT_NAMES),
            "scale": 8.0,
        }
    )

    _trot_diag_joint_symmetry = RewTerm(
        func=DiagonalJointSymmetryReward,
        weight=0.3,  # scale applied here
        params={
            "robot_cfg": SceneEntityCfg(name="robot", body_names=[joint.replace("_joint", "") for joint in GO1_JOINT_NAMES]),
        },
    )

    _diag_motion_balance = RewTerm(
        func=DiagonalMotionBalanceReward,
        weight=0.1,
        params={
            "robot_cfg": SceneEntityCfg(name="robot", body_names=[joint.replace("_joint", "") for joint in GO1_JOINT_NAMES]),
        }
    )
    
    _staying_still_penalty = RewTerm(
        func=NoMotionWhenStationary,
        weight=-0.3,
        params={
            "robot_cfg": SceneEntityCfg(
                name="robot",
                joint_names=GO1_JOINT_NAMES,  # use joint names instead of body_names
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "torque_weight": 0.3,
            "velocity_threshold": 0.05,
        },
    )

    _all_feet_off_ground_penalty = RewTerm(
        func=AllFeetOffGroundPenalty,
        weight=-4,
        params={
            "sensor_cfg": SceneEntityCfg(
                name="contact_forces",
                body_names=GO1_FOOT_NAMES,
                preserve_order=True,
            ),
            "contact_force_threshold": 1.0,
        },
    )

    # ======= NOT USED =======
    # ======= WTW Paper - Agumented Auxiliary =======
    # _track_feet_contact_schedule_forces = RewTerm(
    #     func=track_feet_contact_schedule_forces,
    #     weight=20,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces", 
    #             body_names=".*foot"
    #         ),
    #         "std": 0.25,
    #     }
    # )
    # from quadrupeds_mdp.rewards.linear_velocity import track_foot_contact_schedule_velocities,
    # _track_foot_contact_schedule_velocities = RewTerm(
    #     func=track_foot_contact_schedule_velocities,
    #     weight=0.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=".*foot"
    #         ),
    #         "std": 0.25,
    #     }
    # )
    # TODO: In WTW code, they use a base height + a jump height. For now, I am assuming the sum is 1.3.
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
    # from quadrupeds_mdp.rewards.position import track_base_orientation
    # _track_base_orientation = RewTerm(
    #     func=track_base_orientation,
    #     weight=-10,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names="trunk",
    #             preserve_order=True
    #         ),
    #         "orientation_cmd": "body_height_orientation_cmd",
    #     }
    # )
    # raibert_footswing = RewTerm(
    #     func=RaibertHeuristic,
    #     weight=-10,
    #     params={
    #         "command_name": "base_velocity",
    #         "gait_stance_distances_cmd": "gait_stance_distances_cmd",
    #         "gait_step_freq_cmd": "gait_freq_cmd",
    #         "robot_cfg": SceneEntityCfg(
    #             "robot", body_names=GO1_FOOT_NAMES, preserve_order=True
    #         )
    #     },
    # )

    # ======= WTW Paper - Fixed Auxiliary =======
    _track_feet_slip = RewTerm(
        func=track_feet_slip,
        weight=-0.25,
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
    _joint_torques_l2 = RewTerm(func=joint_torques_l2, weight=-0.0012)
    _track_joint_vel_l2 = RewTerm(func=track_joint_vel_l2, weight=-0.003)
    # from isaaclab.envs.mdp.rewards import joint_deviation_l1
    # _joint_deviation_l1 = RewTerm(func=joint_deviation_l1, weight=0)

    # ======= WTW - Only in code =======
    # from quadrupeds_mdp.rewards.linear_velocity import track_feet_contact_velocity
    # _track_feet_contact_velocity = RewTerm(
    #     func=track_feet_contact_velocity,
    #     weight=0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=".*foot"
    #         ),
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=".*foot"
    #         ),
    #     }
    # )
    # from quadrupeds_mdp.rewards.force import track_feet_contact_forces
    # _track_feet_contact_forces = RewTerm(
    #     func=track_feet_contact_forces,
    #     weight=-0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=".*foot"
    #         ),
    #         "max_contact_force": 100.0,
    #     }
    # )
    # from quadrupeds_mdp.rewards.force import TrackFeetImpactVel
    # _track_feet_impact_vel = RewTerm(
    #     func=TrackFeetImpactVel,
    #     weight=0.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=".*foot"
    #         ),
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=".*foot"
    #         ),
    #     }
    # )