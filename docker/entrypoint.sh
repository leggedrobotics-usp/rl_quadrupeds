#!/bin/bash

sudo apt update

cd /ros2_ws
# # Install ROS 2 dependencies
rosdep install --from-paths src --ignore-src -r -y
# # Build the ROS 2 workspace
colcon build

source ${ROS_WS}/install/setup.bash
echo "source ${ROS_WS}/install/setup.bash" >> ~/.bashrc
source ~/.bashrc

source /rl_quadrupeds/install.sh install --isaaclab

cd /
exec bash