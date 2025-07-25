"""Configuration for the RayCaster ROS publisher."""

from dataclasses import MISSING
from isaaclab.utils import configclass

from isaaclab_extensions.ros.publisher_cfg import ROSPublisherCfg 
from sensor_msgs.msg import LaserScan as RosLaserScanMsg

from .ray_caster import RayCasterROSPublisher

@configclass
class RayCasterROSPublisherCfg(ROSPublisherCfg):
    """Configuration for the RayCaster ROS publisher."""

    """The publisher class type to use for publishing data."""
    node_class: type = RayCasterROSPublisher

    """The type of the topic to publish to"""
    topic_type: type = RosLaserScanMsg

    """The frame ID for the lidar data."""
    lidar_frame: str = MISSING