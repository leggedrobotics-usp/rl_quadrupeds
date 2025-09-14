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
    fill_value: float = 5.,
    num_rays: int | None = None,
    flatten: bool = True,
    return_hits: bool = False,
    return_labels: bool = True,
    normalize: bool = False,
    hits_norm: float = 1.0,
    dist_norm: float = 1.0,
    label_norm: float = 1.0,
) -> torch.Tensor:
    """
    Returns lidar data with configurable options.

    Args:
        env: Simulation environment.
        sensor_cfg: Sensor configuration.
        fill_value: Value to replace NaN/inf.
        num_rays: Number of central rays to slice. If None, use all.
        flatten: If True, flatten per-environment rays.
        return_hits: If True, return hit positions instead of distances.
        return_labels: If True, append labels as extra channel(s).
        normalize: If True, normalize output values.
        hits_norm: Normalization value for hit positions.
        dist_norm: Normalization value for distances.
        label_norm: Normalization value for labels.

    Returns:
        Tensor of shape:
          - If flatten: [E, R*C], else [E, R, C]
          where R = number of rays, C = channels (2 for hits, 1 for distance, +1 if labels).
    """
    sensor: BetterRayCaster = env.scene.sensors[sensor_cfg.name]

    # Raw data
    ray_hits = sensor.data.ray_hits_w.clone()  # [E, N, 3]
    hit_labels = sensor.data.hit_labels.clone()  # [E, N]
    sensor_pos = sensor.data.pos_w[:, :2]  # [E, 2]

    N = ray_hits.shape[1]
    if num_rays is not None:
        half = N // 2
        start = half - num_rays // 2
        end = start + num_rays
        ray_hits = ray_hits[:, start:end, :]
        hit_labels = hit_labels[:, start:end]

    nan_mask = torch.isnan(ray_hits[:, :, 0]) | torch.isinf(ray_hits[:, :, 0])

    if return_hits:
        # Extract XY hit positions
        data = ray_hits[:, :, :2]
        # Replace only invalid values
        data[nan_mask] = fill_value
        # Clamp extreme values
        data = torch.where(torch.isfinite(data), data, torch.full_like(data, fill_value))
        if normalize:
            data = data / hits_norm
    else:
        # Compute distances from sensor to hit
        delta = ray_hits[:, :, :2] - sensor_pos.unsqueeze(1)
        delta[nan_mask] = 0.0  # avoid propagating NaNs into norm
        dists = torch.norm(delta, dim=-1)
        dists[~torch.isfinite(dists)] = fill_value
        if normalize:
            dists = dists / dist_norm
        data = dists.unsqueeze(-1)

    if return_labels:
        labels = hit_labels.float().unsqueeze(-1)
        # Set label to -1 where hits are invalid
        labels[nan_mask] = -1.0
        labels[~torch.isfinite(labels)] = -1.0
        if normalize:
            labels = labels / label_norm
        data = torch.cat([data, labels], dim=-1)

    if flatten:
        E = data.shape[0]
        data = data.reshape(E, -1)

    # Final cleanup - ensure no NaNs or infs remain
    invalid_mask = ~torch.isfinite(data)
    if invalid_mask.any():
        data[invalid_mask] = fill_value

    return data