from __future__ import annotations

from rclpy.node import Node
from tf2_ros import TransformBroadcaster

class ROSPublisher(Node):
    def __init__(self, asset_cfg, idx_publisher: int = None):
        self.asset_cfg = asset_cfg
        if idx_publisher is not None:
            self.ros_publisher_cfg = self.asset_cfg.ros_publishers_cfgs[idx_publisher]
        else:
            self.ros_publisher_cfg = self.asset_cfg.ros_publisher_cfg

        super().__init__(self.ros_publisher_cfg.node_name)

        if self.ros_publisher_cfg.topic_name == "tf":
            self._publisher = TransformBroadcaster(self)
        else:
            # TODO: externalize the qos_profile from create_publisher
            self._publisher = self.create_publisher(
                self.ros_publisher_cfg.topic_type, self.ros_publisher_cfg.topic_name, 10
            )

    def publish(self, data, env_ids):
        raise NotImplementedError(
            "The publish method must be implemented in the subclass."
        )

    def _time_msg(self):
        """Returns the current time as a ROS Time message."""
        return self.get_clock().now().to_msg()