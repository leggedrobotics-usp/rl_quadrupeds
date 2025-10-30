from isaaclab.utils import configclass
from locomotion_go1.locomotion_go1_env_cfg import Go1LocomotionEnvCfg
from quadrupeds_mdp.actions.planner_pretrained_velocity import RobotPlannerActionTrainedNavigationCfg
from quadrupeds_mdp.actions.inspection.capture_feat_action import CaptureFeaturesActionCfg

locomotion_cfg = Go1LocomotionEnvCfg()
vel_ranges = locomotion_cfg.commands.base_velocity.ranges

@configclass
class ActionsCfg:
    viewpoint_action = RobotPlannerActionTrainedNavigationCfg(
        asset_name="robot",
        # locomotion_policy_path="/rl_quadrupeds/logs/skrl/go1_locomotion/best_loc_2025-06-10_08-28-10_ppo_torch/policy.jit.pt",
        locomotion_policy_path="/rl_quadrupeds/logs/rsl_rl/go1_locomotion/2025-10-28_21-15-38/exported/policy.jit.pt",
        locomotion_decimation=4,
        locomotion_actions=locomotion_cfg.actions.joint_pos,
        locomotion_observations=locomotion_cfg.observations.policy,
        ranges=RobotPlannerActionTrainedNavigationCfg.Ranges(
            pos_x=(-1.5, 1.5),
            pos_y=(-1.5, 1.5),
            heading=(-3.14, 3.14),
            v_linear_x=(vel_ranges.lin_vel_x[0], vel_ranges.lin_vel_x[1]),
            v_linear_y=(vel_ranges.lin_vel_y[0], vel_ranges.lin_vel_y[1]),
            v_angular=(vel_ranges.ang_vel_z[0], vel_ranges.ang_vel_z[1]),
        ),
        debug_vis=True,
    )

    capture_feat_action = CaptureFeaturesActionCfg(
        asset_name="robot",
        debug_vis=True,
    )
