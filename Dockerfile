FROM osrf/ros:humble-desktop-full

ADD https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.5.0-rc.36%2Brelease.19112.f59b3005.gl.linux-x86_64.release.zip /isaacsim/isaac-sim.zip
ADD https://github.com/isaac-sim/IsaacLab/archive/refs/tags/v2.1.0.zip /isaclab.zip
COPY ros2/my_subscriber_package/ /ros2_ws/src/my_subscriber_package/

ENV ISAACSIM_PATH="/isaacsim" \
    ISAACSIM_PYTHON_EXE="/isaacsim/python.sh" \
    ISAACLAB_FOLDER="/IsaacLab" \
    ROS_WS=/ros2_ws

RUN \
    # ===== SYSTEM DEPENDENCIES =====
    apt update && apt install -y \
    vulkan-tools \
    unzip \
    cmake \
    build-essential && \
    # ===== ISAACSIM SETUP =====
    unzip /isaacsim/isaac-sim.zip -d /isaacsim && \
    rm /isaacsim/isaac-sim.zip && \
    ./isaacsim/post_install.sh && \
    # ===== ISAACLAB SETUP =====
    unzip /isaclab.zip -d / && \
    rm /isaclab.zip && \
    mv /IsaacLab-2.1.0 /IsaacLab && \
    ln -s $ISAACSIM_PATH /IsaacLab/_isaac_sim && \
    export TERM=xterm && /IsaacLab/isaaclab.sh --install && \
    # ===== ROS =====
    . /opt/ros/humble/setup.sh && \
    cd ${ROS_WS} && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build && \
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc && \
    # ===== ALIAS SETUP =====
    echo "alias isaacsim='/isaacsim/isaac-sim.sh'" >> ~/.bashrc && \
    echo "alias isaaclab='/IsaacLab/isaaclab.sh'" >> ~/.bashrc && \
    # ===== CLEANUP =====
    rm -rf /var/lib/apt/lists/*
    
# ===== DEFAULT ENTRYPOINT =====
CMD ["bash"]
