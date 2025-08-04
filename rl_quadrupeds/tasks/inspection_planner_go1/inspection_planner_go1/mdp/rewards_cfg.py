from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.envs.mdp.rewards import (
    is_alive
)
from isaaclab.utils import configclass

from quadrupeds_mdp.rewards.inspection import (
    get_inspection_action,
    get_overall_inspection_coverage,
    get_overall_inspection_coverage_gain,
    get_if_inspection_done,
    get_known_inspection_points,
    get_unknown_inspection_points,
    MilestoneCoverageReward
)
from quadrupeds_mdp.rewards.contact import get_illegal_contact
from quadrupeds_mdp.rewards.exploration import (
    check_if_current_robot_viewpoint_not_visited,
    get_env_exploration_percentage
)
from quadrupeds_mdp.rewards.position import (
    viewpoint_action_rate_l2
)


@configclass
class RewardsCfg:
    viewpoint_action_rate_l2 = RewTerm(
        func=viewpoint_action_rate_l2,
        weight=-5
    )

    # penalize_inspection_action = RewTerm(
    #     func=get_inspection_action,
    #     weight=-100
    # )

    overall_inspection_coverage = RewTerm(
        func=get_overall_inspection_coverage,
        weight=100.
    )

    overall_inspection_coverage_gain = RewTerm(
        func=get_overall_inspection_coverage_gain,
        weight=300.
    )

    known_inspection_points = RewTerm(
        func=get_known_inspection_points,
        weight=200.
    )

    env_exploration = RewTerm(
        func=get_env_exploration_percentage,
        weight=10
    )

    current_robot_viewpoint_not_visited = RewTerm(
        func=check_if_current_robot_viewpoint_not_visited,
        weight=10
    )

    inspection_done = RewTerm(
        func=get_if_inspection_done,
        weight=500
    )

    illegal_contact = RewTerm(
        func=get_illegal_contact,
        weight=-200
    )

    inspection_milestones_reward = RewTerm(
        func=MilestoneCoverageReward,
        weight=500
    )