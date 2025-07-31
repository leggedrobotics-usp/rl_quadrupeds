from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.envs.mdp.rewards import (
    action_rate_l2,
    is_alive,
    is_terminated
)
from isaaclab.utils import configclass

from quadrupeds_mdp.rewards.position import (
    position_command_error_tanh,
    heading_command_error_abs
)
from quadrupeds_mdp.rewards.ray_caster import (
    get_sum_distance_from_all_objects
)

@configclass
class RewardsCfg:
    action_rate_l2 = RewTerm(
        func=action_rate_l2,
        weight=-0.1
    )

    stay_away_from_obstacles = RewTerm(
        func=get_sum_distance_from_all_objects,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("ray_caster")
        }
    )

    position_tracking = RewTerm(
        func=position_command_error_tanh,
        weight=10.,
        params={"std": 4.0, "command_name": "pose_command"},
    )

    position_tracking_fine_grained = RewTerm(
        func=position_command_error_tanh,
        weight=10.,
        params={"std": 0.4, "command_name": "pose_command"},
    )

    orientation_tracking = RewTerm(
        func=heading_command_error_abs,
        weight=-2.5,
        params={"command_name": "pose_command"},
    )

    termination_penalty = RewTerm(func=is_terminated, weight=-50.0)