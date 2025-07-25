from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.utils import configclass

from quadrupeds_mdp.actions.inspection.capture_feat_action import CaptureFeaturesActionCfg
from quadrupeds_mdp.actions.velocity_pretrained_locomotion import RobotVelocityActionTrainedLocomotionCfg

from locomotion_go1.locomotion_go1_env_cfg import Go1LocomotionEnvCfg
locomotion_cfg = Go1LocomotionEnvCfg()
vel_ranges = locomotion_cfg.commands.base_velocity.ranges

@configclass
class ActionsCfg:
    # When trying to control the robot manually, activate
    # the manual_cmd below. 
    hl_vel = RobotVelocityActionTrainedLocomotionCfg(
        asset_name="robot",
        policy_path="/rl_quadrupeds/logs/skrl/go1_locomotion/best_loc_2025-06-10_08-28-10_ppo_torch/policy.jit.pt",
        low_level_decimation=4,
        low_level_actions=locomotion_cfg.actions.joint_pos,
        low_level_observations=locomotion_cfg.observations.policy,
        clip={"lin_vel_x": vel_ranges.lin_vel_x,
              "lin_vel_y": vel_ranges.lin_vel_y,
              "ang_vel_z": vel_ranges.ang_vel_z},
        scale={
            "lin_vel_x": vel_ranges.lin_vel_x[1],
            "lin_vel_y": vel_ranges.lin_vel_y[1],
            "ang_vel_z": vel_ranges.ang_vel_z[1],
        },
        debug_vis=False,
        # manual_cmd="base_velocity"
    )

    capture_feat_action = CaptureFeaturesActionCfg(
        asset_name="robot",
        debug_vis=False,
    )