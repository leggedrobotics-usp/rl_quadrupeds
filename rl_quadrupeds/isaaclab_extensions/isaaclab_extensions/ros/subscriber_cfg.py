"""Configuration for the ROS subscriber."""

from isaaclab.utils import configclass
from .node_cfg import ROSNodeCfg
from .subscriber import ROSSubscriber

@configclass
class ROSSubscriberCfg(ROSNodeCfg):
    """Configuration for the ROS subscriber."""

    """The subscriber class type to use for receiving data."""
    node_class: type = ROSSubscriber