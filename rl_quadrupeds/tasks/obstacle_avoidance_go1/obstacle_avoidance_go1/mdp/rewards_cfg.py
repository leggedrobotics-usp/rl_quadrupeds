from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.envs.mdp.rewards import (
    action_rate_l2,
    is_terminated,
)
from isaaclab.utils import configclass

from quadrupeds_mdp.rewards.exploration import (
    MinDistanceToObjectsAndWalls
)

from quadrupeds_mdp.rewards.position import (
    position_command_error_tanh,
    heading_command_error_abs
)

from quadrupeds_mdp.rewards.velocity import (
    velocity_aligned_with_goal_reward
)

@configclass
class RewardsCfg:
    action_rate_l2 = RewTerm(func=action_rate_l2, weight=-1.0)

    stay_away_from_obstacles = RewTerm(
        func=MinDistanceToObjectsAndWalls,
        weight=-1.5,
        params={
            "objects": ["block1", "block2"],
            "walls": ["right_wall", "left_wall", "front_wall", "back_wall"],
            "safe_distance": 2.5,
            "near_distance": 1.5,
            "sharpness": 2,
        },
    )

    position_tracking = RewTerm(
        func=position_command_error_tanh,
        weight=10.0,
        params={"std": 4.0, "command_name": "pose_command"},
    )

    position_tracking_fine_grained = RewTerm(
        func=position_command_error_tanh,
        weight=7.0,
        params={"std": 0.4, "command_name": "pose_command"},
    )

    orientation_tracking = RewTerm(
        func=heading_command_error_abs,
        weight=-4.5,
        params={"command_name": "pose_command"},
    )

    termination_penalty = RewTerm(func=is_terminated, weight=-20.0)