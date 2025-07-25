from isaaclab_extensions.ros.publisher import ROSPublisher

class JointsROSPublisher(ROSPublisher):

    def publish(self, data):
        msg = self.ros_publisher_cfg.topic_type()
        msg.header.stamp = self._time_msg()
        msg.name = data.joint_names

        # TODO: generalize for multiple environments
        msg.position = data.joint_pos[0].tolist()
        self._publisher.publish(msg)