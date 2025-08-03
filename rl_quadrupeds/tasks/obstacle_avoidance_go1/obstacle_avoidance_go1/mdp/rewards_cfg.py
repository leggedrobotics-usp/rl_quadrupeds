from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.envs.mdp.rewards import (
    action_rate_l2,
    is_terminated
)
from isaaclab.utils import configclass

from quadrupeds_mdp.rewards.exploration import (
    get_min_distance_from_objects_and_walls
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
    action_rate_l2 = RewTerm(func=action_rate_l2, weight=-0.2)

    stay_away_from_obstacles = RewTerm(
        func=get_min_distance_from_objects_and_walls,
        weight=-10,
        params={
            "objects": ["block1", "block2"],
            "walls": ["right_wall", "left_wall", "front_wall", "back_wall"],
        },
    )

    position_tracking = RewTerm(
        func=position_command_error_tanh,
        weight=8.0,
        params={"std": 4.0, "command_name": "pose_command"},
    )

    position_tracking_fine_grained = RewTerm(
        func=position_command_error_tanh,
        weight=4.0,
        params={"std": 0.4, "command_name": "pose_command"},
    )

    orientation_tracking = RewTerm(
        func=heading_command_error_abs,
        weight=-5,
        params={"command_name": "pose_command"},
    )

    velocity_aligned_with_goal = RewTerm(
        func=velocity_aligned_with_goal_reward,
        weight=2.0,
        params={"normalize": True, "clip_min": -10.0, "clip_max": 10.0}
    )

    termination_penalty = RewTerm(func=is_terminated, weight=-50.0)
