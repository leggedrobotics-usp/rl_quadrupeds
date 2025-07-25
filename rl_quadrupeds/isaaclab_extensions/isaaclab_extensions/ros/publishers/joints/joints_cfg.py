"""Configuration for the RayCaster ROS publisher."""

from dataclasses import MISSING
from isaaclab.utils import configclass

from isaaclab_extensions.ros.publisher_cfg import ROSPublisherCfg 
from .joints import JointsROSPublisher
from sensor_msgs.msg import JointState as RosJointStateMsg

@configclass
class JointsROSPublisherCfg(ROSPublisherCfg):
    """Configuration for the joint ROS publisher."""

    """The publisher class type to use for publishing data."""
    node_class: type = JointsROSPublisher

    """The type of the topic to publish to"""
    topic_type: type = RosJointStateMsg