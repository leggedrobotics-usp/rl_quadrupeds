from typing import List, Sequence
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg

from quadrupeds_mdp.utils import quat_to_yaw

class ComputeObjectRelativePose(ManagerTermBase):
    """
    Computes the relative (x, y, z, yaw) of each object w.r.t. robot in a vectorized way.
    Output: [B, num_objects * 4]
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.env = env
        self.objects: List[str] = cfg.params.get("objects", None)
        if self.objects is None:
            raise ValueError("You must specify a list of objects for ComputeObjectRelativePose.")
        self.num_objects = len(self.objects)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # No state to reset
        pass

    def __call__(self, env: ManagerBasedRLEnv, objects: List[str]) -> torch.Tensor:
        B = env.num_envs

        # Robot pose
        robot_pos = env.scene["robot"].data.root_link_state_w[:, :3]   # [B, 3]
        robot_rot = env.scene["robot"].data.root_link_state_w[:, 3:7]  # [B, 4]
        robot_yaw = quat_to_yaw(robot_rot)                             # [B]
        cos_yaw = torch.cos(-robot_yaw)
        sin_yaw = torch.sin(-robot_yaw)

        # Stack all object positions and orientations
        obj_pos_list = []
        obj_rot_list = []
        for obj in self.objects:
            obj_state = env.scene[obj].data.root_link_state_w  # [B, 7 or 13]
            obj_pos_list.append(obj_state[:, :3])
            obj_rot_list.append(obj_state[:, 3:7])

        obj_pos = torch.stack(obj_pos_list, dim=1)  # [B, N, 3]
        obj_rot = torch.stack(obj_rot_list, dim=1)  # [B, N, 4]

        # Relative positions in world frame
        rel_pos = obj_pos - robot_pos.unsqueeze(1)  # [B, N, 3]

        # Rotate into robot frame using 2D rotation around Z
        rel_x = rel_pos[..., 0] * cos_yaw.unsqueeze(-1) - rel_pos[..., 1] * sin_yaw.unsqueeze(-1)
        rel_y = rel_pos[..., 0] * sin_yaw.unsqueeze(-1) + rel_pos[..., 1] * cos_yaw.unsqueeze(-1)
        rel_z = rel_pos[..., 2]  # Keep Z in world frame

        # Relative yaw: object_yaw - robot_yaw
        obj_yaw = quat_to_yaw(obj_rot.view(B * self.num_objects, 4)).view(B, self.num_objects)
        rel_yaw = (obj_yaw - robot_yaw.unsqueeze(-1) + torch.pi) % (2 * torch.pi) - torch.pi

        # Concatenate per object: [B, N, 4] -> [B, N*4]
        rel_pose = torch.stack([rel_x, rel_y, rel_z, rel_yaw], dim=-1).reshape(B, -1)

        return rel_pose