from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.envs.mdp.rewards import (
    is_alive,
    is_terminated_term
)
from isaaclab.utils import configclass

from quadrupeds_mdp.rewards.action import (
    penalize_raw_action_saturation
)
from quadrupeds_mdp.rewards.inspection import (
    get_inspection_action,
    get_overall_inspection_coverage,
    MaxCoverageGainReward,
    get_if_inspection_done,
    KnownInspectionPointsGainReward,
    MilestoneCoverageReward,
    get_robot_stuck_if_inspection_not_done,
    get_robot_closeness_to_ideal_inspection_pose
)
from quadrupeds_mdp.rewards.contact import get_illegal_contact
from quadrupeds_mdp.rewards.exploration import (
    check_if_current_robot_viewpoint_not_visited,
    get_env_exploration_percentage
)
from quadrupeds_mdp.rewards.position import (
    viewpoint_action_l2,
    viewpoint_action_rate_l2,
    viewpoint_towards_objects
)

@configclass
class RewardsCfg:
    # penalize_raw_action_saturation = RewTerm(
    #     func=penalize_raw_action_saturation,
    #     weight=-1.5,
    #     params={
    #         "action_term": "viewpoint_action",
    #     }
    # )

    viewpoint_action_rate_l2 = RewTerm(
        func=viewpoint_action_rate_l2,
        weight=-0.01
    )

    # viewpoint_towards_objects = RewTerm(
    #     func=viewpoint_towards_objects,
    #     weight=5,
    #     params={
    #         "objects_of_interest": ["block1", "block2"],
    #         "distance_weight": 0,
    #     }
    # )

    penalize_inspection_action = RewTerm(
        func=get_inspection_action,
        weight=-0.1
    )

    # penalize_robot_stuck_if_inspection_not_done = RewTerm(
    #     func=get_robot_stuck_if_inspection_not_done,
    #     weight=-10,
    # )

    # overall_inspection_coverage = RewTerm(
    #     func=get_overall_inspection_coverage,
    #     weight=100.
    # )

    overall_inspection_coverage_gain = RewTerm(
        func=MaxCoverageGainReward,
        weight=1000
    )

    known_inspection_points = RewTerm(
        func=KnownInspectionPointsGainReward,
        weight=50
    )

    # env_exploration = RewTerm(
    #     func=get_env_exploration_percentage,
    #     weight=2
    # )

    current_robot_viewpoint_not_visited = RewTerm(
        func=check_if_current_robot_viewpoint_not_visited,
        weight=50
    )

    # inspection_done = RewTerm(
    #     func=is_terminated_term,
    #     weight=100,
    #     params={"term_keys": "inspection_done"}
    # )

    inspection_done = RewTerm(
        func=get_if_inspection_done,
        weight=10000,
    )

    illegal_contact = RewTerm(
        func=is_terminated_term,
        weight=-10000,
        params={"term_keys": "base_contact"}
    )

    # inspection_milestones_reward = RewTerm(
    #     func=MilestoneCoverageReward,
    #     weight=100
    # )

    # comment back for full RL training
    # get_robot_closeness_to_ideal_inspection_pose = RewTerm(
    #     func=get_robot_closeness_to_ideal_inspection_pose,
    #     weight=5,
    # )

    is_alive = RewTerm(
        func=is_alive,
        weight=-0.01
    )

    # specifics for imitation learning + reinforcement learning
    viewpoint_action_l2 = RewTerm(
        func=viewpoint_action_l2,
        weight=-0.1
    )