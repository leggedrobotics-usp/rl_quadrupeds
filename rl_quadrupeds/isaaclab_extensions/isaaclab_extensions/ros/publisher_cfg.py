"""Configuration for the ROS publisher."""

from isaaclab.utils import configclass

from .node_cfg import ROSNodeCfg
from .publisher import ROSPublisher

@configclass
class ROSPublisherCfg(ROSNodeCfg):
    """Configuration for the ROS publisher."""

    """The publisher class type to use for publishing data."""
    node_class: type = ROSPublisher