from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.envs.mdp.rewards import (
    is_alive
)
from isaaclab.utils import configclass

from quadrupeds_mdp.rewards.inspection import (
    get_inspection_action,
    get_overall_inspection_coverage,
    get_overall_inspection_coverage_gain,
    get_if_inspection_done
)

from quadrupeds_mdp.rewards.contact import get_illegal_contact
from quadrupeds_mdp.rewards.inspection import (
    get_env_exploration_percentage,
    get_unknown_inspection_points
)
from quadrupeds_mdp.rewards.velocity import vel_action_rate_l2

@configclass
class RewardsCfg:
    vel_action_rate_l2 = RewTerm(
        func=vel_action_rate_l2,
        weight=-1.
    )

    penalize_inspection_action = RewTerm(
        func=get_inspection_action,
        weight=-0.1
    )

    # overall_inspection_coverage = RewTerm(
    #     func=get_overall_inspection_coverage,
    #     weight=1.
    # )

    overall_inspection_coverage_gain = RewTerm(
        func=get_overall_inspection_coverage_gain,
        weight=20.
    )

    unknown_inspection_points = RewTerm(
        func=get_unknown_inspection_points,
        weight=-20.
    )

    env_exploration = RewTerm(
        func=get_env_exploration_percentage,
        weight=1.0
    )

    # We penalize the agent for being alive because termination 
    # means it has successfully inspected the objects.
    # is_alive = RewTerm(
    #     func=is_alive,
    #     weight=-0.5
    # )

    inspection_done = RewTerm(
        func=get_if_inspection_done,
        weight=100
    )

    illegal_contact = RewTerm(
        func=get_illegal_contact,
        weight=-100
    )