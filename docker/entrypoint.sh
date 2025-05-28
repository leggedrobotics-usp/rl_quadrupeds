#!/bin/bash
# Clone this repository if it doesn't exist
cd ~
git clone https://github.com/toschilt/rl_quadrupeds.git
cd ${ROS_WS}
cp -r ~/rl_quadrupeds/ros2/* ./src/
rosdep install --from-paths src --ignore-src -r -y
colcon build
echo "source ${ROS_WS}/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
cd /
bash