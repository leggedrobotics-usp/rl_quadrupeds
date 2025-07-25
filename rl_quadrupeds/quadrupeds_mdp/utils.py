"""
utils.py

Utility functions for quaternion operations and coordinate transformations in quadruped locomotion tasks.

Available functions:
- batch_quaternion_inverse: Computes the inverse of a batch of quaternions.
- batch_quaternion_rotate: Rotates a batch of vectors by a batch of quaternions.
- convert_batch_foot_to_local_frame: Converts positions from world frame to local frame using batched quaternions.
- quat_from_angle_axis: Converts angle-axis representation to quaternion.
- euler_to_quat: Converts Euler angles (roll, pitch, yaw) to quaternion.
- quat_to_yaw: Extracts yaw angle from a quaternion.
- quat_mul: Multiplies two quaternions.
"""

import torch

def batch_quaternion_inverse(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a batch of quaternions.

    Args:
        q: Tensor of shape [B, 4] (wxyF format)

    Returns:
        Tensor of shape [B, 4], inverse of input quaternions.
    """
    q_conj = torch.cat([q[:, :1], -q[:, 1:]], dim=1)  # [B, 4]
    return q_conj / q.norm(p=2, dim=1, keepdim=True) ** 2

def batch_quaternion_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate a batch of vectors v by a batch of quaternions q.
    B is the batch size, F is the number of vectors per batch item.

    Args:
        q: Tensor of shape [B, 4] (quaternions in wxyF format)
        v: Tensor of shape [B, F, 3]

    Returns:
        Rotated vectors, shape [B, F, 3]
    """
    B, F, _ = v.shape
    q = q.unsqueeze(1).expand(-1, F, -1) # Expand q to [B, F, 4]
    q_w = q[..., :1]         # [B, F, 1]
    q_vec = q[..., 1:]       # [B, F, 3]

    uv = torch.cross(q_vec, v, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return v + 2 * (q_w * uv + uuv)

def convert_batch_foot_to_local_frame(
    pos_w: torch.Tensor,                 # [B, F, 3]
    pos_local_frame_w: torch.Tensor,     # [B, 3]
    quat_local_frame_w: torch.Tensor     # [B, 4]
) -> torch.Tensor:
    """
    Convert positions from world frame to local frame using batched quaternions,
    supporting multiple positions per batch item.

    B is the batch size, F is the number of foots per batch item.

    Args:
        pos_w: Positions in world frame, shape [B, F, 3]
        pos_local_frame_w: Origins of local frames, shape [B, 3]
        quat_local_frame_w: Orientations of local frames, shape [B, 4]

    Returns:
        Positions in the local frame, shape [B, F, 3]
    """
    pos_rel = pos_w - pos_local_frame_w.unsqueeze(1)
    quat_inv = batch_quaternion_inverse(quat_local_frame_w)
    pos_l = batch_quaternion_rotate(quat_inv, pos_rel)
    return pos_l

def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    # angle: (N,)
    # axis: (N, 3)
    half_angle = 0.5 * angle
    sin_half_angle = torch.sin(half_angle).unsqueeze(-1)  # (N, 1)
    cos_half_angle = torch.cos(half_angle)  # (N,)

    q = torch.zeros((angle.shape[0], 4), device=angle.device)
    q[:, 0] = cos_half_angle               # w
    q[:, 1:] = axis * sin_half_angle       # (N, 3) * (N, 1) => (N, 3)
    return q


def euler_to_quat(roll, pitch, yaw):
    # Inputs: [B]
    cr = torch.cos(roll / 2)
    sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)  # [B, 4]

def quat_to_yaw(q):
    # Input: [B, 4] - [w, x, y, z]
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return yaw  # [B]

def quat_mul(q, r):
    """Multiplies two quaternions.

    Args:
        q (Tensor): Quaternions of shape (N, 4) in format [w, x, y, z]
        r (Tensor): Quaternions of shape (N, 4) in format [w, x, y, z]

    Returns:
        Tensor: Result of quaternion multiplication, shape (N, 4)
    """
    w1, x1, y1, z1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    w2, x2, y2, z2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack([w, x, y, z], dim=1)
