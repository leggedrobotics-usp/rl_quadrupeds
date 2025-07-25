from __future__ import annotations

from dataclasses import field, MISSING
from typing import List, TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab_extensions.assets.articulation.articulation_pub import PublishableArticulation

if TYPE_CHECKING:
    from rl_quadrupeds.isaaclab_extensions.isaaclab_extensions.ros.node_cfg import ROSPublisherCfg

@configclass
class PublishableArticulationCfg(ArticulationCfg):
    """Configuration parameters for an articulation."""

    class_type: type = PublishableArticulation

    ros_publishers_cfgs: List[ROSPublisherCfg] = field(default=None)
    """The configuration object for the ROS publisher."""