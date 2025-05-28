#!/bin/bash

# Default locations for host cache (edit if different)
HOST_OV_CACHE="$HOME/.cache/ov"
HOST_NV_CACHE="$HOME/.nvidia/omniverse"

# If the host cache dirs exist, use them. Otherwise, fall back to Docker volumes.
if [ -d "$HOST_OV_CACHE" ]; then
  export OV_CACHE_PATH="$HOST_OV_CACHE"
else
  export OV_CACHE_PATH="isaac_cache"
fi

if [ -d "$HOST_NV_CACHE" ]; then
  export NV_CACHE_PATH="$HOST_NV_CACHE"
else
  export NV_CACHE_PATH="isaac_nv_cache"
fi

# Launch the container with an interactive shell
docker compose run --rm ros2_humble_isaaclab bash
