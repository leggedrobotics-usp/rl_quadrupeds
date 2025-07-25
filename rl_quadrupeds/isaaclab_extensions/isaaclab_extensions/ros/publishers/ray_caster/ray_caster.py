import math

from isaaclab_extensions.ros.publisher import ROSPublisher

class RayCasterROSPublisher(ROSPublisher):

    def publish(self, data, env_ids):
        msg = self.ros_publisher_cfg.topic_type()
        msg.header.stamp = self._time_msg()
        msg.header.frame_id = self.ros_publisher_cfg.lidar_frame

        # Dummy laser scan properties
        msg.angle_min = self.asset_cfg.pattern_cfg.horizontal_fov_range[0]*math.pi / 180.0
        msg.angle_max = self.asset_cfg.pattern_cfg.horizontal_fov_range[1]*math.pi / 180.0
        msg.angle_increment = self.asset_cfg.pattern_cfg.horizontal_res*math.pi / 180.0

        # msg.time_increment = self.asset_cfg.update_period / \
        #     (self.asset_cfg.pattern_cfg.channels*(msg.angle_max - msg.angle_min)/msg.angle_increment)
        msg.scan_time = self.asset_cfg.update_period

        msg.range_min = 0.
        msg.range_max = 20.

        # TODO: generalize for multiple environments
        msg.ranges = data.ranges[0, :].tolist()  # Assuming data is a 2D array with ranges in the first column
        
        self._publisher.publish(msg)