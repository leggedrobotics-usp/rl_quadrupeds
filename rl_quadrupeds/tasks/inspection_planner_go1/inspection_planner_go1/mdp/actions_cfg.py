from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.utils import configclass

from quadrupeds_mdp.actions.inspection.capture_feat_action import CaptureFeaturesActionCfg
from quadrupeds_mdp.actions.planner_pretrained_velocity import RobotPlannerActionTrainedNavigationCfg

from locomotion_go1.locomotion_go1_env_cfg import Go1LocomotionEnvCfg
locomotion_cfg = Go1LocomotionEnvCfg()
vel_ranges = locomotion_cfg.commands.base_velocity.ranges

# from obstacle_avoidance_go1.obstacle_avoidance_go1_env_cfg import Go1ObstacleAvoidanceEnvCfg
# navigation_cfg = Go1ObstacleAvoidanceEnvCfg()
# pos_ranges = navigation_cfg.commands.pose_command.ranges

@configclass
class ActionsCfg:
    # When trying to control the robot manually, activate
    # the manual_cmd below. 
    viewpoint_action = RobotPlannerActionTrainedNavigationCfg(
        asset_name="robot",
        # nav_policy_path="/rl_quadrupeds/logs/skrl/go1_obstacle_avoidance/2025-08-15_15-08-19_ppo_torch/policy.jit.pt",
        locomotion_policy_path="/rl_quadrupeds/logs/skrl/go1_locomotion/best_loc_2025-06-10_08-28-10_ppo_torch/policy.jit.pt",
        locomotion_decimation=4,
        locomotion_actions=locomotion_cfg.actions.joint_pos,
        locomotion_observations=locomotion_cfg.observations.policy,
        # nav_decimation=4,
        # nav_actions=navigation_cfg.actions.hl_vel,
        # nav_observations=navigation_cfg.observations.policy,
        ranges=RobotPlannerActionTrainedNavigationCfg.Ranges(
            pos_x=(-1.5, 1.5),
            pos_y=(-1.5, 1.5),
            heading=(-3.14, 3.14),
            v_linear=(vel_ranges.lin_vel_x[0], vel_ranges.lin_vel_x[1]),
            v_angular=(vel_ranges.ang_vel_z[0], vel_ranges.ang_vel_z[1]),
        ),
        debug_vis=True,
        # manual_cmd="base_velocity"
    )

    capture_feat_action = CaptureFeaturesActionCfg(
        asset_name="robot",
        debug_vis=True,
    )