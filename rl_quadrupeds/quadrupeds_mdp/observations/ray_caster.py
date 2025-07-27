"""
ray_caster.py

Observation terms for lidar scans from a RayCaster sensor in an environment.

Available functions:
- lidar_scan: Returns a lidar scan from the specified sensor in the environment.
"""

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_extensions.sensors.ray_caster.better.ray_caster import BetterRayCaster

def lidar_scan(
    env: ManagerBasedEnv, 
    sensor_cfg: SceneEntityCfg, 
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    sensor: BetterRayCaster = env.scene.sensors[sensor_cfg.name]
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2]

def lidar_scan_hits_labels(
    env: ManagerBasedEnv, 
    sensor_cfg: SceneEntityCfg, 
    fill_value: float = 20.0,
) -> torch.Tensor:
    """Returns a single tensor with lidar hit positions and object labels.
    Tensor shape: [E, N, 4] where the last dim is (x, y, z, label).
    All NaN, +inf, -inf values are replaced with `fill_value`.
    """
    sensor: BetterRayCaster = env.scene.sensors[sensor_cfg.name]
    ray_hits = sensor.data.ray_hits_w  # [E, N, 3]
    ray_hits = torch.nan_to_num(ray_hits, nan=fill_value, posinf=fill_value, neginf=fill_value)

    hit_labels = sensor.data.hit_labels.unsqueeze(-1).float()  # [E, N, 1]

    return torch.cat([ray_hits, hit_labels], dim=-1)  # [E, N, 4]

def lidar_scan_hits_labels_flattened(
    env: ManagerBasedEnv, 
    sensor_cfg: SceneEntityCfg, 
    fill_value: float = 20.0,
) -> torch.Tensor:
    """Returns a single flattened tensor with lidar hit positions and object labels.
    Output shape: [E, Lx4], where each row contains all rays for that environment flattened.
    Each ray is (x, y, z, label).
    All NaN, +inf, -inf values are replaced with `fill_value`.
    """
    sensor: BetterRayCaster = env.scene.sensors[sensor_cfg.name]

    ray_hits = sensor.data.ray_hits_w  # [E, N, 3]
    hit_labels = sensor.data.hit_labels  # [E, N]

    # Replace NaN and infinities in place (if allowed)
    torch.nan_to_num(ray_hits, nan=fill_value, posinf=fill_value, neginf=fill_value, out=ray_hits)

    # Preallocate the final tensor
    E, N, _ = ray_hits.shape
    lidar_data = torch.empty((E, N, 4), device=ray_hits.device, dtype=ray_hits.dtype)

    # Copy data without using cat (avoids reallocation)
    lidar_data[..., :3] = ray_hits
    lidar_data[..., 3] = hit_labels

    # Flatten (should be cheap if contiguous)
    return lidar_data.reshape(E, N * 4)