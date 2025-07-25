from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.assets.articulation.articulation import Articulation

if TYPE_CHECKING:
    from .articulation_pub_cfg import PublishableArticulationCfg

class PublishableArticulation(Articulation):

    cfg: PublishableArticulationCfg
    """Configuration instance for the articulations."""

    def __init__(self, cfg: PublishableArticulationCfg):
        """Initialize the articulation.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        self._initialize_ros_publishers()

    def _initialize_ros_publishers(self):
        """Initializes the ROS publisher for the sensor data."""
        if getattr(self.cfg, 'ros_publishers_cfgs', False):
            import rclpy
            self._ros_publishers = [
                ros_publisher_cfg.node_class(self.cfg, idx) for \
                idx, ros_publisher_cfg in enumerate(self.cfg.ros_publishers_cfgs)
            ]
            for ros_publisher in self._ros_publishers:
                rclpy.spin_once(ros_publisher, timeout_sec=0)

    def update(self, dt: float):
        self._data.update(dt)

        if getattr(self.cfg, 'ros_publishers_cfgs', False):
            for ros_publisher in self._ros_publishers:
                ros_publisher.publish(self._data)