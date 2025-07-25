from __future__ import annotations

from rclpy.node import Node

class ROSSubscriber(Node):
    def __init__(self, cfg):
        self.ros_subscriber_cfg = cfg
        super().__init__(self.ros_subscriber_cfg.node_name)

        # TODO: externalize the qos_profile from create_subscriber
        self._subscription = self.create_subscription(
            self.ros_subscriber_cfg.topic_type, 
            self.ros_subscriber_cfg.topic_name,
            self.callback,
            10
        )
        self._subscription  # prevent unused variable warning

    def callback(self, msg):
        raise NotImplementedError(
            "The callback method must be implemented in the subclass."
        )