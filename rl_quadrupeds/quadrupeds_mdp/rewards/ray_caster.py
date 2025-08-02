import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_extensions.sensors.ray_caster.better.ray_caster import BetterRayCaster

def get_sum_distance_from_all_objects(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    sensor: BetterRayCaster = env.scene.sensors[sensor_cfg.name]

    return torch.sum(
        torch.nan_to_num(
            torch.abs(sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2]),
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ), dim=1
    )