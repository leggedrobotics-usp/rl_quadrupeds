"""Configuration for the ROS node."""

from dataclasses import MISSING
from isaaclab.utils import configclass

@configclass
class ROSNodeCfg():
    """Configuration for the ROS node."""

    """The class type to use for the node."""
    node_class: type = MISSING
    
    """The name of the topic the node is related to"""
    topic_name: str = MISSING

    """The type of the topic the node is related to"""
    topic_type: type = MISSING

    """The name of the node"""
    node_name: str = MISSING