#!/bin/bash
cd ~

# Clone this repository
git clone https://github.com/toschilt/rl_quadrupeds.git
cd ${ROS_WS}
# Copy all ROS 2 packages from the cloned repository to the ROS workspace
cp -r ~/rl_quadrupeds/ros2/* ./src/

# Install ROS 2 dependencies
rosdep install --from-paths src --ignore-src -r -y
# Build the ROS 2 workspace
colcon build

echo "source ${ROS_WS}/install/setup.bash" >> ~/.bashrc
source ~/.bashrc

cd /
bash