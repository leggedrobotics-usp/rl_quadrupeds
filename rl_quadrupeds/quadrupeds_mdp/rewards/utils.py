import torch

def point_to_segment_distance(points: torch.Tensor, seg_start: torch.Tensor, seg_end: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum distance from points to line segments in a fully vectorized way.

    Args:
        points:    (num_envs, N, 2) points
        seg_start: (num_envs, N, 2) start points of segments
        seg_end:   (num_envs, N, 2) end points of segments

    Returns:
        distances: (num_envs, N) distances from each point to each segment
    """
    seg_vec = seg_end - seg_start                           # (num_envs, N, 2)
    seg_len_sq = torch.sum(seg_vec ** 2, dim=-1, keepdim=True)  # (num_envs, N, 1)

    pt_vec = points - seg_start                             # (num_envs, N, 2)
    t = torch.sum(pt_vec * seg_vec, dim=-1, keepdim=True) / (seg_len_sq + 1e-8)
    t = torch.clamp(t, 0.0, 1.0)                            # (num_envs, N, 1)

    closest = seg_start + t * seg_vec                       # (num_envs, N, 2)
    return torch.norm(points - closest, dim=-1)             # (num_envs, N)