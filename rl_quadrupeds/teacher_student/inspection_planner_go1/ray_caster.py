import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

class RayCasterBuilderCfg:
    def __init__(self):
        # degrees
        self.horizontal_fov_range = (-180, 180)
        self.horizontal_res = 5.0  # degrees step
        self.max_distance = 5.0
        self.noise_std = 0.01  # noise stddev added to distance measurements (meters)

        # position in robot frame (x,y)
        self.position_robot_frame = (0.0, 0.0)
        self.ignore_workspace = True  # if True, rays ignore workspace boundaries

class RayCasterBuilder:
    """
    Vectorized Ray Caster

    Usage:
        rc = RayCasterBuilder(cfg, device, num_envs)
        rc.build(workspace_builder, robot_builder, object_interest_builder)
        data = rc.cast_rays(num_rays=60, return_hits=False, return_labels=True, normalize=True, ...)
    """
    def __init__(self, cfg: RayCasterBuilderCfg, device: torch.device, num_envs: int = 1):
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = int(num_envs)

        # placeholders
        self.angles = None        # (R,)
        self.dirs = None          # (R,2) unit vectors in sensor frame (robot heading = 0)
        self.max_distance = float(cfg.max_distance)
        self.noise_std = float(cfg.noise_std)

    def build(self,
              workspace_builder,   # provides .workspace tensor (4,2)
              robot_builder,       # provides .viewpoint (E,3) and maybe .contour (E,4,2)
              object_interest_builder):  # provides .obj_prop['contour'] (E,M,P,2)
        """
        Setup internal geometry references.
        """
        device = self.device
        self.workspace = workspace_builder.workspace.to(device)  # (4,2)
        self.robot_builder = robot_builder
        self.object_interest_builder = object_interest_builder

        # Build ray angles and directions (in robot frame where heading=0)
        fov_lo, fov_hi = self.cfg.horizontal_fov_range
        step = self.cfg.horizontal_res
        # create inclusive range using step (in degrees)
        angles_deg = torch.arange(fov_lo, fov_hi + 1e-6, step, device=device, dtype=torch.float32)
        angles_rad = angles_deg * (math.pi / 180.0)
        self.angles = angles_rad  # (R,)
        cos = torch.cos(angles_rad)
        sin = torch.sin(angles_rad)
        self.dirs = torch.stack([cos, sin], dim=-1)  # (R,2), unit vectors

        # Precompute workspace segments (global same for all envs)
        ws = self.workspace  # (4,2)
        ws_closed = torch.cat([ws, ws[:1]], dim=0)  # (5,2)
        ws_starts = ws_closed[:-1]  # (4,2)
        ws_ends = ws_closed[1:]     # (4,2)
        # store as (S_ws, 2, 2)
        self.ws_segments = torch.stack([ws_starts, ws_ends], dim=1).to(device)  # (4,2,2)

        # Build object segments per env by flattening objects and their polygon edges
        obj_contour = object_interest_builder.obj_prop['contour'].to(device)  # (E,M,P,2)
        E, M, P, _ = obj_contour.shape
        # make closed (wrap around)
        obj_closed = torch.cat([obj_contour, obj_contour[..., 0:1, :]], dim=2)  # (E,M,P+1,2)
        starts = obj_closed[..., :-1, :]  # (E,M,P,2)
        ends = obj_closed[..., 1:, :]     # (E,M,P,2)
        # reshape to (E, S_obj, 2, 2) where S_obj = M*P
        S_obj = M * P
        starts = starts.reshape(E, S_obj, 2)
        ends = ends.reshape(E, S_obj, 2)
        self.obj_segments = torch.stack([starts, ends], dim=2).to(device)  # (E, S_obj, 2, 2)

        # Prepare segment labels for objects: object index per segment
        # each object has P segments; labels shape (E, S_obj)
        obj_idx = torch.arange(M, device=device).unsqueeze(1).expand(M, P).reshape(-1)  # (S_obj,)
        self.obj_segment_labels = obj_idx.unsqueeze(0).expand(E, -1).to(device)  # (E, S_obj)

        # Workspace segments label as -2 for identification
        # store workspace segments expanded per env: (E, S_ws, 2, 2)
        S_ws = self.ws_segments.shape[0]
        self.ws_segments_exp = self.ws_segments.unsqueeze(0).expand(E, -1, -1, -1).to(device)  # (E, S_ws, 2, 2)

    def plot(
        self,
        env_id: int = 0,
        show_hits: bool = True,
        max_rays: Optional[int] = None,
        ax: Optional[plt.Axes] = None
    ):
        """
        Plot workspace, objects, robot, rays and hits for one environment.
        Works with data stored in self._ray_tracing_results.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        else:
            fig = None

        env_id = int(env_id)

        # Ensure ray tracing has been run
        if not hasattr(self, "_ray_tracing_results"):
            ax.set_aspect("equal", adjustable="box")
            ax.legend()
            if fig is not None:
                plt.show()
            return fig, ax

        results = self._ray_tracing_results

        hits = results["hits"][env_id]       # (R,2)
        dists = results["dists"][env_id]     # (R,)
        sensor_pos = results["sensor_pos"][env_id].cpu().numpy()  # (2,)
        dirs = results["dirs"][env_id].cpu().numpy()              # (R,2)

        R_total = hits.shape[0]
        upto = R_total if max_rays is None else min(R_total, int(max_rays))

        hits_upto = hits[:upto].cpu().numpy()
        dists_upto = dists[:upto].cpu().numpy()
        dirs_upto = dirs[:upto]

        # Plot rays up to hit distance or max_distance
        for r in range(upto):
            d = dists_upto[r]
            d_plot = self.max_distance if not np.isfinite(d) or d > self.max_distance else d
            end = sensor_pos + dirs_upto[r] * d_plot
            ax.plot([sensor_pos[0], end[0]], [sensor_pos[1], end[1]], ":", linewidth=0.6, c="gray")

        # Plot hits
        if show_hits:
            valid_mask = np.isfinite(dists_upto) & (dists_upto <= self.max_distance)
            ax.scatter(hits_upto[valid_mask, 0], hits_upto[valid_mask, 1],
                    s=15, c="b", label="hits")

        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        if fig is not None:
            plt.show()
        return fig, ax

    @staticmethod
    def _cross2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        2D cross product returning scalar: a_x*b_y - a_y*b_x
        a, b can be broadcastable with final dim=2.
        """
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def cast_rays(self, num_rays: Optional[int] = None, store_results: bool = True):
        """
        Perform ray tracing for all environments and store raw results internally.
        Does NOT apply normalization, noise, or output formatting.
        """
        device = self.device
        E = self.num_envs

        R_all = self.dirs.shape[0]
        if num_rays is None or num_rays >= R_all:
            dirs = self.dirs.to(device)  # (R_all,2)
            angles = self.angles.to(device)
        else:
            half = R_all // 2
            start = int(half - num_rays // 2)
            end = start + int(num_rays)
            dirs = self.dirs[start:end].to(device)
            angles = self.angles[start:end].to(device)
        R = dirs.shape[0]

        robot_pos = self.robot_builder.viewpoint[:, :2].to(device)   # (E,2)
        robot_head = self.robot_builder.viewpoint[:, 2].to(device)   # (E,)
        if torch.max(torch.abs(robot_head)) > (2.0 * math.pi + 1e-6):
            robot_head = robot_head * (math.pi / 180.0)

        sensor_pos_robot = torch.tensor(self.cfg.position_robot_frame, device=device, dtype=torch.float32)  # (2,)
        cos_h = torch.cos(robot_head)
        sin_h = torch.sin(robot_head)
        rot = torch.stack([torch.stack([cos_h, -sin_h], dim=-1),
                        torch.stack([sin_h,  cos_h], dim=-1)], dim=-2)  # (E,2,2)
        sensor_pos_world = torch.einsum('eij,j->ei', rot, sensor_pos_robot) + robot_pos  # (E,2)

        dirs_expand = dirs.unsqueeze(0).expand(E, -1, -1)  # (E,R,2)
        dirs_world = torch.einsum('eij,erj->eri', rot, dirs_expand)  # (E,R,2)
        dirs_world = dirs_world / (torch.norm(dirs_world, dim=-1, keepdim=True) + 1e-12)

        obj_seg = self.obj_segments  # (E,S_obj,2,2)
        labels_obj = self.obj_segment_labels  # (E,S_obj)

        if self.cfg.ignore_workspace:
            segs = obj_seg
            seg_labels = labels_obj
        else:
            ws_seg = self.ws_segments_exp  # (E,S_ws,2,2)
            S_ws = ws_seg.shape[1]
            segs = torch.cat([obj_seg, ws_seg], dim=1)  # (E, S_total, 2,2)
            labels_ws = -2 * torch.ones((E, S_ws), device=device, dtype=torch.long)  # (E,S_ws)
            seg_labels = torch.cat([labels_obj, labels_ws], dim=1)  # (E, S_total)
        S_total = segs.shape[1]

        O = sensor_pos_world.unsqueeze(1)       # (E,1,2)
        D = dirs_world.unsqueeze(2)             # (E,R,1,2)
        P0 = segs[...,0,:].unsqueeze(1)         # (E,1,S_total,2)
        P1 = segs[...,1,:].unsqueeze(1)         # (E,1,S_total,2)
        V = P1 - P0                             # (E,1,S_total,2)

        O_ = O.unsqueeze(2)
        D_ = D
        V_ = V
        p0_o = P0 - O_
        denom = self._cross2(D_, V_)
        numer_t = self._cross2(p0_o, V_)
        numer_u = self._cross2(p0_o, D_)

        parallel_mask = denom.abs() < 1e-8
        denom_safe = denom.clone()
        denom_safe[parallel_mask] = 1.0

        t = numer_t / denom_safe
        u = numer_u / denom_safe
        valid = (~parallel_mask) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
        t_masked = torch.where(valid, t, torch.full_like(t, float('inf')))

        t_min, idx_min = torch.min(t_masked, dim=-1)  # (E,R)
        D_r = dirs_world
        no_hit_mask = ~torch.isfinite(t_min) | (t_min > self.max_distance)
        t_min_clamped = t_min.clone()
        t_min_clamped[no_hit_mask] = float('inf')

        hits = O + D_r * t_min_clamped[..., None]  # (E,R,2)
        dists_filled = t_min_clamped.clone()
        dists_filled[~torch.isfinite(dists_filled)] = self.max_distance
        dists_filled = torch.clamp(dists_filled, max=self.max_distance)

        idx_min_long = idx_min.long().to(device)
        labels = torch.gather(seg_labels, dim=1, index=idx_min_long)  # (E,R)
        labels = labels.clone().long()
        labels[no_hit_mask] = -1

        results = {
            "hits": hits,         # (E,R,2)
            "dists": dists_filled,  # (E,R)
            "labels": labels,     # (E,R)
            "sensor_pos": sensor_pos_world,  # (E,2)
            "dirs": dirs_world,   # (E,R,2)
            "angles": angles,     # (R,)
            "no_hit_mask": no_hit_mask,  # (E,R)
        }

        # Store raw results internally
        if store_results:
            self._ray_tracing_results = results
        return results

    def get_ray_data(
        self,
        data: dict = None,
        return_hits: bool = False,
        return_labels: bool = True,
        flatten: bool = True,
        normalize: bool = False,
        hits_norm: float = 1.0,
        dist_norm: float = 1.0,
        label_norm: float = 1.0,
        fill_value: float = None,
        add_noise: bool = True,
        num_rays: Optional[int] = None
    ):
        """
        Return formatted ray data based on the last cast_rays computation.
        Can be called multiple times without recomputing ray tracing.
        
        Args:
            num_rays: If specified, returns only the central num_rays rays.
        """
        results = None
        if data is None:
            if not hasattr(self, "_ray_tracing_results"):
                self.cast_rays()
            results = self._ray_tracing_results
        else:
            results = data
            
        device = results["hits"].device
        fill_value = fill_value if fill_value is not None else float(self.max_distance)

        hits = results["hits"].clone()      # (E,R,2)
        dists = results["dists"].clone()    # (E,R)
        labels = results["labels"].clone()  # (E,R)
        no_hit_mask = results["no_hit_mask"]  # (E,R)

        # Select central subset of rays if num_rays is specified
        R_total = hits.shape[1]
        if num_rays is not None and num_rays < R_total:
            half = R_total // 2
            start = half - num_rays // 2
            end = start + num_rays
            hits = hits[:, start:end, :]
            dists = dists[:, start:end]
            labels = labels[:, start:end]
            no_hit_mask = no_hit_mask[:, start:end]

        # Add noise to distances if requested
        if add_noise:
            noise = torch.randn_like(dists) * float(self.noise_std)
            dists = dists + noise
            dists = torch.clamp(dists, min=0.0, max=self.max_distance)

        # Prepare main data
        if return_hits:
            data = hits.clone()
            invalid_hits_mask = no_hit_mask.unsqueeze(-1).expand(-1, -1, 2)
            data[invalid_hits_mask] = float(fill_value)
            if normalize:
                data = data / float(hits_norm)
        else:
            dists = dists.clone()
            dists[no_hit_mask] = float(fill_value)
            data = dists.unsqueeze(-1)
            if normalize:
                data = data / float(dist_norm)


        # Append labels if requested
        if return_labels:
            lab = labels.float().unsqueeze(-1)
            if normalize:
                lab = lab / float(label_norm)
            data = torch.cat([data, lab], dim=-1)

        # Flatten if requested
        if flatten:
            E_out = data.shape[0]
            data = data.reshape(E_out, -1)

        # Final numeric cleanup
        data[~torch.isfinite(data)] = float(fill_value)
        return data

    def get_observation(self, normalized: bool = True, noise: bool = True):
        return self.get_ray_data(
            return_hits=False,
            return_labels=True,
            flatten=True,
            normalize=normalized,
            dist_norm=self.cfg.max_distance,
            label_norm=self.object_interest_builder.cfg.num_objects_per_env,
            fill_value=0.0,
            add_noise=noise,
        )