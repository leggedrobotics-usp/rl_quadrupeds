from isaaclab_extensions.ros.publisher import ROSPublisher

class TFROSPublisher(ROSPublisher):

    def publish(self, data):
        msg = self.ros_publisher_cfg.topic_type()
        msg.header.stamp = self._time_msg()
        msg.header.frame_id = self.ros_publisher_cfg.ref_frame_id
        msg.child_frame_id = self.ros_publisher_cfg.child_frame_id

        # TODO: generalize for multiple environments
        msg.transform.translation.x = data.root_pos_w[0, 0].item()
        msg.transform.translation.y = data.root_pos_w[0, 1].item()
        msg.transform.translation.z = data.root_pos_w[0, 2].item()
        msg.transform.rotation.x = data.root_quat_w[0, 1].item()
        msg.transform.rotation.y = data.root_quat_w[0, 2].item()
        msg.transform.rotation.z = data.root_quat_w[0, 3].item()
        msg.transform.rotation.w = data.root_quat_w[0, 0].item()
        self._publisher.sendTransform(msg)