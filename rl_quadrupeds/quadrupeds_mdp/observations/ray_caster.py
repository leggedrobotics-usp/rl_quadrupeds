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
    fill_value: float = 20.0,
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    sensor: BetterRayCaster = env.scene.sensors[sensor_cfg.name]
    return torch.nan_to_num(
        sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2],
        nan=fill_value,
        posinf=fill_value,
        neginf=fill_value
    )

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
    ray_hits = sensor.data.ray_hits_w[:, :, :2]  # [E, N, 2]
    half_n = ray_hits.shape[1] // 2
    ray_hits = torch.nan_to_num(ray_hits[:, :half_n, :], nan=fill_value, posinf=fill_value, neginf=fill_value)
    hit_labels = sensor.data.hit_labels[:, :half_n].unsqueeze(-1).float()  # [E, N, 1]
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
    ray_hits = sensor.data.ray_hits_w[:, :, :2]  # [E, N, 2]
    half_n = ray_hits.shape[1] // 2
    ray_hits = torch.nan_to_num(ray_hits[:, :half_n, :], nan=fill_value, posinf=fill_value, neginf=fill_value) # [E, N/2, 2]
    hit_labels = sensor.data.hit_labels[:, :half_n].float()  # [E, N/2]
    E, _, _ = ray_hits.shape
    lidar_data = torch.empty((E, half_n, 3), device=ray_hits.device, dtype=ray_hits.dtype) # [E, N/2, 3]
    lidar_data[..., :2] = ray_hits
    lidar_data[..., 2] = hit_labels
    return lidar_data.reshape(E, half_n * 3)