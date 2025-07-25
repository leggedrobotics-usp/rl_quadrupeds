"""Configuration for the RayCaster ROS publisher."""

from dataclasses import MISSING
from isaaclab.utils import configclass

from isaaclab_extensions.ros.publisher_cfg import ROSPublisherCfg 
from .tf import TFROSPublisher
from geometry_msgs.msg import TransformStamped as RosTransformStampedMsg

@configclass
class TFROSPublisherCfg(ROSPublisherCfg):
    """Configuration for the joint ROS publisher."""

    """The publisher class type to use for publishing data."""
    node_class: type = TFROSPublisher

    """The type of the topic to publish to"""
    topic_type: type = RosTransformStampedMsg

    """The name of the reference frame for the transformation"""
    ref_frame_id: str = MISSING

    """The name of the child frame for the transformation"""
    child_frame_id: str = MISSING