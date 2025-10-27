"""Recorder configurations for Go1 locomotion debugging."""

from isaaclab.managers import RecorderManagerBaseCfg, RecorderTermCfg as RecTerm
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.utils import configclass

from quadrupeds_mdp.recorders.locomotion import (
    DebugActionRateL2Recorder,
    DebugFeetContactForcesRecorder,
    DebugJointVelRecorder,
    DebugTrotSyncRecorder,
    DebugRaibertHeuristicRecorder,
    DebugFootDeviationRecorder,
    DebugBaseHeightRecorder,
    DebugFootSwingRecorder,
    DebugIllegalContactRecorder,
    DebugJointPosRecorder,
    DebugJointPosLimitsRecorder,
    DebugWTWCommandFootStatesRecorder
)

@configclass
class RecordersCfg(RecorderManagerBaseCfg):
    """Recorder Manager configuration for Go1 locomotion debugging."""

    # 🔹 Feet contact force tracking
    debug_feet_contact_forces: RecTerm = RecTerm(class_type=DebugFeetContactForcesRecorder)

    # 🔹 Joint velocity tracking
    debug_joint_vel: RecTerm = RecTerm(class_type=DebugJointVelRecorder)

    # 🔹 Trot synchronization tracking
    debug_trot_sync: RecTerm = RecTerm(class_type=DebugTrotSyncRecorder)

    # 🔹 Foot deviation penalty tracking
    debug_foot_deviation: RecTerm = RecTerm(class_type=DebugFootDeviationRecorder)

    # 🔹 Base height tracking
    debug_base_height: RecTerm = RecTerm(class_type=DebugBaseHeightRecorder)

    # 🔹 Foot swing height tracking
    debug_foot_swing: RecTerm = RecTerm(class_type=DebugFootSwingRecorder)

    # 🔹 Joint position tracking
    debug_joint_pos: RecTerm = RecTerm(class_type=DebugJointPosRecorder)

    debug_raibert_heuristic: RecTerm = RecTerm(class_type=DebugRaibertHeuristicRecorder)
    debug_joint_pos_limits: RecTerm = RecTerm(class_type=DebugJointPosLimitsRecorder)
    debug_action_rate: RecTerm = RecTerm(class_type=DebugActionRateL2Recorder)
    debug_illegal_contact: RecTerm = RecTerm(class_type=DebugIllegalContactRecorder)
    debug_wtw_command_foot_states: RecTerm = RecTerm(class_type=DebugWTWCommandFootStatesRecorder)

    # Dataset export options
    dataset_export_dir_path: str = "output/debug_datasets"
    dataset_filename: str = "go1_debug_rewards"
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL
    export_in_record_pre_reset: bool = True