import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

def local_viewpoint(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg, 
) -> torch.Tensor:
    """Position and orientation of the asset in the local environment frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_viewpoint = asset.data.root_link_state_w[:, :7]
    robot_viewpoint[:, :3] -= env.scene.env_origins
    return robot_viewpoint