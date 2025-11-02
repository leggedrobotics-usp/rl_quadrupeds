import math
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from quadrupeds_mdp.assets.go1 import GO1_FOOT_NAMES, GO1_JOINT_NAMES
from quadrupeds_mdp.rewards.action import action_rate_penalty
from quadrupeds_mdp.rewards.angular_velocity import track_ang_vel_z_exp
from quadrupeds_mdp.rewards.linear_velocity import (
    track_lin_vel_xy_exp,
    track_joint_vel_l2,
    lin_vel_z_penalty,
    ang_vel_xy_penalty,
    joint_acc_penalty,
)
from quadrupeds_mdp.rewards.position import (
    DiagonalFootTrotPositive,
    DiagonalFootTrotNegative,
    DiagonalJointSymmetryReward,
    DiagonalMotionBalanceReward,
    FootDeviationPenalty,
    track_base_height_l2_cmd_no_sensor,
    track_footswing_height,
    track_joint_positions_bonus,
    penalize_hip_movement,
    NoMotionWhenStationary,
    AllFeetOffGroundPenalty,
    flat_orientation_bonus,
    joint_pos_limits_penalty,
)
from quadrupeds_mdp.rewards.force import (
    applied_torque_limits_bonus,
    track_feet_slip,
    joint_torques_l2,
)
from quadrupeds_mdp.rewards.termination import is_terminated_term


@configclass
class RewardsCfg:
    """Tuned reward configuration for GO1 locomotion with normalized bounded rewards."""

    # === Primary Locomotion Objectives ===
    track_lin_vel_xy_exp = RewTerm(
        func=track_lin_vel_xy_exp,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=track_ang_vel_z_exp,
        weight=2.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # === Height and Posture ===
    track_base_height_l2_cmd_no_sensor = RewTerm(
        func=track_base_height_l2_cmd_no_sensor,
        weight=1.2,
        params={
            "command_name": "body_height_orientation_cmd",
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk", preserve_order=True),
            "k": 1.2,
        },
    )
    flat_orientation_bonus = RewTerm(
        func=flat_orientation_bonus,
        weight=0.8,
        params={"asset_cfg": SceneEntityCfg("robot"), "k": 100},
    )

    # === Gait bonuses ===
    trot_gait_positive = RewTerm(
        func=DiagonalFootTrotPositive,
        weight=0.5,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=GO1_FOOT_NAMES),
            "k": 1.5,
        },
    )
    trot_gait_penalty = RewTerm(
        func=DiagonalFootTrotNegative,
        weight=0.3,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=GO1_FOOT_NAMES),
            "k": 2.0,
        },
    )
    trot_diag_joint_symmetry = RewTerm(
        func=DiagonalJointSymmetryReward,
        weight=0.25,
        params={
            "robot_cfg": SceneEntityCfg(
                "robot", body_names=[joint.replace("_joint", "") for joint in GO1_JOINT_NAMES]
            ),
            "k": 1.2,
        },
    )
    diag_motion_balance = RewTerm(
        func=DiagonalMotionBalanceReward,
        weight=0.15,
        params={
            "robot_cfg": SceneEntityCfg(
                "robot", body_names=[joint.replace("_joint", "") for joint in GO1_JOINT_NAMES]
            ),
            "k": 1.0,
        },
    )

    # === Physical Stability / Ground Contact ===
    all_feet_off_ground_penalty = RewTerm(
        func=AllFeetOffGroundPenalty,
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=GO1_FOOT_NAMES, preserve_order=True
            ),
            "contact_force_threshold": 1.0,
            "k": 4.0,
        },
    )
    penalize_foot_deviation_from_default = RewTerm(
        func=FootDeviationPenalty,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=GO1_FOOT_NAMES, preserve_order=True),
            "k": 0.5,
        },
    )

    # === Energy, Motion & Effort ===
    joint_torques_penalty = RewTerm(
        func=joint_torques_l2,
        weight=0.4,
        params={"asset_cfg": SceneEntityCfg("robot"), "k": 0.5},
    )
    track_joint_vel_penalty = RewTerm(
        func=track_joint_vel_l2,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot"), "k": 0.5},
    )
    joint_acc_penalty = RewTerm(
        func=joint_acc_penalty,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot"), "k": 0.8},
    )
    action_rate_penalty = RewTerm(
        func=action_rate_penalty,
        weight=0.05,
        params={"k": 0.5},
    )
    applied_torque_limits_bonus = RewTerm(
        func=applied_torque_limits_bonus,
        weight=0.6,
        params={"asset_cfg": SceneEntityCfg("robot"), "k": 1.0},
    )

    # === Orientation and Hip Stability ===
    track_joint_positions_bonus = RewTerm(
        func=track_joint_positions_bonus,
        weight=0.15,
        params={"asset_cfg": SceneEntityCfg("robot"), "k": 0.8},
    )
    penalize_hip_movement = RewTerm(
        func=penalize_hip_movement,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_hip"], preserve_order=True),
            "k": 0.8,
        },
    )

    # === Standing / Stillness Control ===
    staying_still_penalty = RewTerm(
        func=NoMotionWhenStationary,
        weight=0.3,
        params={
            "robot_cfg": SceneEntityCfg(
                "robot", joint_names=GO1_JOINT_NAMES, preserve_order=True
            ),
            "command_name": "base_velocity",
            "torque_weight": 0.3,
            "velocity_threshold": 0.05,
            "k": 1.5,
        },
    )

    # === Termination & Physical Safety ===
    is_terminated = RewTerm(
        func=is_terminated_term,
        weight=3.0,
        params={"term_keys": "base_contact", "k": 6.0},
    )

    # === Velocity / Slip penalties ===
    feet_slip_penalty = RewTerm(
        func=track_feet_slip,
        weight=0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "k": 3.0,
        },
    )
    lin_vel_z_penalty = RewTerm(func=lin_vel_z_penalty, weight=0.8, params={"asset_cfg": SceneEntityCfg("robot"), "k": 1.5})
    ang_vel_xy_penalty = RewTerm(func=ang_vel_xy_penalty, weight=0.2, params={"asset_cfg": SceneEntityCfg("robot"), "k": 1.5})

    joint_pos_limits_penalty = RewTerm(func=joint_pos_limits_penalty, weight=0.3, params={"asset_cfg": SceneEntityCfg("robot"), "k": 10.})