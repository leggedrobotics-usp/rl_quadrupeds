import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

def local_pos_rot(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg, 
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    pos = asset.data.root_link_state_w[:, :3] - env.scene.env_origins
    rot = asset.data.root_link_state_w[:, 3:7]  # [batch_size, 4] quaternion
    return torch.cat([
        pos, 
        rot
    ], dim=-1)  # [batch_size, 7] where last dim is (x, y, z, qx, qy, qz, qw)