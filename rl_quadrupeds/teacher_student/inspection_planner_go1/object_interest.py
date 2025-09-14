from typing import Optional

import matplotlib.pyplot as plt
import torch

from workspace import WorkspaceBuilder, WorkspaceBuilderCfg

class ObjectInterestBuilderCfg:
    def __init__(self):
        self.size_x = (0.5, 0.5)      # min,max of object x size (m)
        self.size_y = (0.5, 0.5)      # min,max of object y size (m)
        self.x = (0, 0)           # x position sampling range (m)
        self.y = (0, 0)           # y position sampling range (m)
        self.heading = (-torch.pi, torch.pi)  # heading sampling range (rad)

        self.num_objects_per_env = 1 # number of objects per environment
        self.contour_points_per_edge = 7  # points along each of 4 edges

class ObjectInterestBuilder:
    """
    Vectorized over environments (num_envs) and objects per env (M).
    Results are stored in self.obj_prop dictionary with documented shapes.
    """
    def __init__(
        self,
        cfg: ObjectInterestBuilderCfg,
        workspace_builder_cfg: WorkspaceBuilderCfg,
        device: torch.device,
        num_envs: int = 1,
    ):
        self.cfg = cfg
        self.device = device
        self.num_envs = int(num_envs)
        self.obj_prop = {}
        self._check_cfg_validity(workspace_builder_cfg)

    def build(self):
        self._resample_properties()
        self._create_contour()

    def plot(self, env_id: int = 0, ax: Optional[plt.Axes] = None, **kwargs):
        """Plot all objects of interest in one environment."""
        if ax is None:
            ax = plt.gca()
        contours = self.obj_prop['contour'][env_id]  # (M,P,2)
        M = contours.shape[0]
        for m in range(M):
            c = contours[m].detach().cpu().numpy()
            ax.plot(c[:, 0], c[:, 1], '-', label=f"Obj {m}", **kwargs)
        ax.set_aspect("equal", adjustable="box")
        return ax

    def _check_cfg_validity(self, workspace_builder_cfg: WorkspaceBuilderCfg):
        assert self.cfg.size_x[0] > 0 and self.cfg.size_y[0] > 0, "Object sizes must be positive."
        assert self.cfg.size_x[0] <= self.cfg.size_x[1], "Invalid size_x range."
        assert self.cfg.size_y[0] <= self.cfg.size_y[1], "Invalid size_y range."
        assert self.cfg.x[0] <= self.cfg.x[1], "Invalid x range."
        assert self.cfg.y[0] <= self.cfg.y[1], "Invalid y range."
        assert self.cfg.heading[0] <= self.cfg.heading[1], "Invalid heading range."
        assert self.cfg.num_objects_per_env > 0, "Number of objects must be positive."
        assert self.cfg.contour_points_per_edge >= 2, "At least two contour points per edge are required."

        # rough workspace consistency
        #TODO: we do not need to assert this if we push objects back into the workspace when sampling
        half_x = workspace_builder_cfg.size_x / 2.0
        half_y = workspace_builder_cfg.size_y / 2.0
        assert self.cfg.x[0] >= -half_x and self.cfg.x[1] <= half_x, "Object x position range exceeds workspace bounds."
        assert self.cfg.y[0] >= -half_y and self.cfg.y[1] <= half_y, "Object y position range exceeds workspace bounds."
        assert self.cfg.x[0] - self.cfg.size_x[1] / 2 >= -half_x, "Object x position and size exceed workspace bounds."
        assert self.cfg.x[1] + self.cfg.size_x[1] / 2 <= half_x, "Object x position and size exceed workspace bounds."
        assert self.cfg.y[0] - self.cfg.size_y[1] / 2 >= -half_y, "Object y position and size exceed workspace bounds."
        assert self.cfg.y[1] + self.cfg.size_y[1] / 2 <= half_y, "Object y position and size exceed workspace bounds."

    def _resample_properties(self):
        E = self.num_envs
        M = self.cfg.num_objects_per_env
        device = self.device

        # Sample per environment and per object: shapes (E, M)
        self.obj_prop['size_x'] = torch.empty(E, M, device=device).uniform_(*self.cfg.size_x)
        self.obj_prop['size_y'] = torch.empty(E, M, device=device).uniform_(*self.cfg.size_y)
        self.obj_prop['x'] = torch.empty(E, M, device=device).uniform_(*self.cfg.x)
        self.obj_prop['y'] = torch.empty(E, M, device=device).uniform_(*self.cfg.y)
        # heading range stored as Python floats/torch scalars; make sure to cast to float if torch.pi used
        heading_lo = float(self.cfg.heading[0]) if isinstance(self.cfg.heading[0], torch.Tensor) else self.cfg.heading[0]
        heading_hi = float(self.cfg.heading[1]) if isinstance(self.cfg.heading[1], torch.Tensor) else self.cfg.heading[1]
        self.obj_prop['heading'] = torch.empty(E, M, device=device).uniform_(heading_lo, heading_hi)

    def _create_contour(self) -> None:
        """
        Creates contour points for every object in every env.

        Result shapes:
        - 'contour' : (E, M, P, 2) where P = 4 * points_per_edge
        - 'num_contour_points' : int P
        - 'face_indices' : (E, M, 4, pts_per_edge) indices into the P axis
        """
        E = self.num_envs
        M = self.cfg.num_objects_per_env
        points_per_edge = self.cfg.contour_points_per_edge
        device = self.device

        # parameter t along each edge: (points_per_edge, 1)
        t = torch.linspace(0.0, 1.0, steps=points_per_edge, device=device).unsqueeze(-1)

        # unit square corners (4,2)
        base_corners = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            device=device,
        )  # (4,2)

        # interpolate edges -> (4, points_per_edge, 2)
        corners_next = base_corners.roll(-1, dims=0)  # (4,2)
        edges = (1.0 - t) * base_corners.unsqueeze(1) + t * corners_next.unsqueeze(1)  # (4, P_edge, 2)
        edges = edges.reshape(-1, 2)  # (P, 2) where P = 4 * points_per_edge
        P = edges.shape[0]

        # Broadcast scale to (E, M, 2)
        scales = torch.stack([self.obj_prop['size_x'], self.obj_prop['size_y']], dim=-1)  # (E, M, 2)

        # Expand edges to (E, M, P, 2) and scale
        edges_exp = edges.unsqueeze(0).unsqueeze(0)  # (1,1,P,2)
        scaled_edges = edges_exp * scales.unsqueeze(2)  # (E, M, P, 2)

        # Rotation per object: build rotation matrices (E, M, 2, 2)
        cos_h = torch.cos(self.obj_prop['heading'])  # (E, M)
        sin_h = torch.sin(self.obj_prop['heading'])  # (E, M)
        rot = torch.stack([
            torch.stack([cos_h, -sin_h], dim=-1),
            torch.stack([sin_h,  cos_h], dim=-1)
        ], dim=-2)  # shape (E, M, 2, 2)
        # Multiply: rotated = (E,M,P,2) x (E,M,2,2) -> (E,M,P,2)
        # einsum indices: scaled_edges em p k ; rot em k j -> result em p j
        rotated_edges = torch.einsum('empk,emkj->empj', scaled_edges, rot)

        # Translation: (E, M, 2)
        translations = torch.stack([self.obj_prop['x'], self.obj_prop['y']], dim=-1)  # (E, M, 2)
        translated_edges = rotated_edges + translations.unsqueeze(2)  # (E, M, P, 2)

        # Build face indices: P = 4 * pts_per_edge
        pts_per_edge = P // 4
        idx = torch.arange(P, device=device)  # (P,)
        face_indices = idx.view(4, pts_per_edge)  # (4, pts_per_edge)
        # expand to (E, M, 4, pts_per_edge)
        face_indices = face_indices.unsqueeze(0).unsqueeze(0).expand(E, M, -1, -1).contiguous()

        # Store results
        self.obj_prop['contour'] = translated_edges          # (E, M, P, 2)
        self.obj_prop['num_contour_points'] = P
        self.obj_prop['face_indices'] = face_indices         # (E, M, 4, pts_per_edge)