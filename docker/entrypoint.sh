#!/bin/bash
cd ~

REPO_BRANCH="${REPO_BRANCH:-main}"

# Clone this repository
mkdir rl_quadrupeds
git clone --branch "$REPO_BRANCH" https://github.com/leggedrobotics-usp/rl_quadrupeds.git /rl_quadrupeds
cd ${ROS_WS}
# Copy all ROS 2 packages from the cloned repository to the ROS workspace
cp -r /rl_quadrupeds/ros2/* ./src/

# Install ROS 2 dependencies
rosdep install --from-paths src --ignore-src -r -y
# Build the ROS 2 workspace
colcon build

echo "source ${ROS_WS}/install/setup.bash" >> ~/.bashrc
source ~/.bashrc

cd /
exec bash