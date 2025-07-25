"""Configuration for the velocity command subscriber."""

from dataclasses import MISSING
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from isaaclab_extensions.ros.subscriber_cfg import ROSSubscriberCfg
from geometry_msgs.msg import TwistStamped as RosTwistStampedMsg
from .vel_cmd import ROSVelCmdSubscriber

@configclass
class ROSVelCmdSubscriberCfg(ROSSubscriberCfg, CommandTermCfg):
    """Configuration for the ROS subscriber."""

    """Name of the asset in the environment for which the commands are generated."""
    asset_name: str = MISSING

    """The subscriber class type to use for receiving data."""
    class_type: type = ROSVelCmdSubscriber

    """The type of the topic the node is related to"""
    topic_type: type = RosTwistStampedMsg

    """The type of the topic the node is related to"""
    resampling_time_range: tuple[float, float] = MISSING