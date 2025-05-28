# RL Quadrupeds (Docker)

This folder contains tools to run a Docker image containing ROS 2 Humble, IsaacSim, and IsaacLab.

## Prerequisites

- Docker Engine installed

## Building the Image

Go to the root folder of this repository and execute:

```bash
docker build -t legro/ros2_humble_isaaclab:latest .
```

## Running the Image

Use the Docker Compose configuration to run the image with the appropriate settings. Run:

```bash
source docker/run_compose.sh
```