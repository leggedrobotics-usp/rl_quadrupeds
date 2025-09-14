import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

def local_viewpoint(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    xy_norm: float = 1.0,
    heading_norm: float = 1.0,
) -> torch.Tensor:
    """Return normalized local (x, y, heading) of the asset in the environment frame.

    Args:
        env: The environment.
        asset_cfg: Configuration for the asset.
        xy_norm: Normalization value for x and y coordinates.
        heading_norm: Normalization value for heading (yaw) angle.

    Returns:
        Tensor of shape [E, 3] with normalized (x, y, heading).
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # World pose: [E, 13] → position (0:3), quaternion (3:7) as (w, x, y, z)
    state_w = asset.data.root_link_state_w

    # Local x, y relative to environment origins
    x = state_w[:, 0] - env.scene.env_origins[:, 0]
    y = state_w[:, 1] - env.scene.env_origins[:, 1]

    # Normalize x, y
    x = x / xy_norm
    y = y / xy_norm

    # Quaternion components (w, x, y, z)
    qw, qx, qy, qz = state_w[:, 3], state_w[:, 4], state_w[:, 5], state_w[:, 6]

    # Normalize quaternion safely to avoid division by zero or NaNs
    norm = torch.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    norm = torch.clamp(norm, min=1e-8)  # prevent zero division
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    # Heading (yaw) from quaternion
    heading = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )

    # Normalize heading
    heading = heading / heading_norm

    # Stack results
    result = torch.stack((x, y, heading), dim=1)

    # Replace any NaN or inf with zero (or safe value)
    result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result