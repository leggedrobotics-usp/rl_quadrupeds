import matplotlib.pyplot as plt
import torch

from cfg import Cfg
from object_interest import ObjectInterestBuilder
from ray_caster import RayCasterBuilder, RayCasterBuilderCfg
from workspace import WorkspaceBuilder

class RobotBuilderCfg(Cfg):
    def _define_defaults(self):
        self.size_x = 0.6  # robot length (m)
        self.size_y = 0.4  # robot width (m)

        self.x = (-1.5, 1.5)  # sampling range
        self.y = (-1.5, 1.5)
        self.heading = (-torch.pi, torch.pi)

        self.noise_std = {
            'x': 0.05,       # meters
            'y': 0.05,       # meters
            'heading': 0.1   # radians
        }

class RobotBuilder:
    def __init__(
        self, 
        cfg: RobotBuilderCfg,
        workspace_builder: WorkspaceBuilder,
        ray_caster_builder_cfg: RayCasterBuilderCfg,
        object_interest_builder: ObjectInterestBuilder,
        device: torch.device,
        num_envs: int = 1
    ):
        self.cfg = cfg
        self.device = device
        self.num_envs = int(num_envs)

        self.ray_caster_builder_cfg = ray_caster_builder_cfg
        self.workspace_builder = workspace_builder
        self.object_interest_builder = object_interest_builder

        self.ranges = {
            'x': max(abs(self.cfg.x[0]), abs(self.cfg.x[1])),  # for normalization
            'y': max(abs(self.cfg.y[0]), abs(self.cfg.y[1])),  # for normalization
            'heading': max(abs(self.cfg.heading[0]), abs(self.cfg.heading[1]))  # for normalization
        }

        self.build()

    def build(self, fixed_viewpoint: torch.Tensor = None):
        if fixed_viewpoint is not None:
            self.viewpoint = fixed_viewpoint
        else:
            self._resample_properties()
        self._create_contour()
        self._build_ray_caster()

    def enforce_viewpoint_limits(self, viewpoint: torch.Tensor) -> torch.Tensor:
        viewpoint = viewpoint.clone()
        viewpoint[:, 0] = torch.clamp(viewpoint[:, 0], self.cfg.x[0], self.cfg.x[1])
        viewpoint[:, 1] = torch.clamp(viewpoint[:, 1], self.cfg.y[0], self.cfg.y[1])
        viewpoint[:, 2] = torch.remainder(viewpoint[:, 2] + torch.pi, 2*torch.pi) - torch.pi
        return viewpoint

    def normalize(self, viewpoint: torch.Tensor) -> torch.Tensor:
        normalized_viewpoint = viewpoint.clone()
        normalized_viewpoint[:, 0] /= self.ranges['x']
        normalized_viewpoint[:, 1] /= self.ranges['y']
        normalized_viewpoint[:, 2] /= self.ranges['heading']
        return normalized_viewpoint

    def denormalize(self, normalized_viewpoint: torch.Tensor) -> torch.Tensor:
        viewpoint = normalized_viewpoint.clone()
        viewpoint[:, 0] *= self.ranges['x']
        viewpoint[:, 1] *= self.ranges['y']
        viewpoint[:, 2] *= self.ranges['heading']
        return viewpoint

    def get_observation(self, normalized: bool = True, noise: bool = False):
        return {
            'viewpoint': self._get_robot_viewpoint(normalized=normalized, noise=noise),  # (E,3)
            'lidar_scan_w_labels': self.ray_caster_builder.get_observation(normalized=normalized, noise=noise),  # (E,N,4)
        }

    def plot(self, env_id: int = 0, ax: plt.Axes = None, **kwargs):
        """Plot robot rectangle and heading arrow."""
        if ax is None:
            ax = plt.gca()

        c = self.contour[env_id].detach().cpu().numpy()
        # close polygon
        c_closed = torch.cat([self.contour[env_id], self.contour[env_id, 0:1]], dim=0).cpu().numpy()
        ax.plot(c_closed[:, 0], c_closed[:, 1], 'r-', linewidth=2, label="Robot", **kwargs)

        # plot heading arrow
        pos = self.viewpoint[env_id, :2].cpu().numpy()
        heading = self.viewpoint[env_id, 2].item()
        arrow_len = max(self.cfg.size_x/3, self.cfg.size_y/3)
        ax.arrow(
            pos[0], pos[1],
            arrow_len * torch.cos(torch.tensor(heading)).item(),
            arrow_len * torch.sin(torch.tensor(heading)).item(),
            head_width=0.1, head_length=0.1, fc='r', ec='r'
        )
        ax.set_aspect("equal", adjustable="box")

        _, ax = self.ray_caster_builder.plot(ax=ax, env_id=env_id)
        return ax

    def _transform_axes(self, viewpoint: torch.Tensor, apply_to_heading: bool = True) -> torch.Tensor:
        """
        Adapt axes from this simulator (x is right, y is up, heading=0 is x-axis)
        to the IsaacSim pattern (x is down, y is right, heading=0 is x-axis).

        apply_to_heading: if True, rotate heading by +90 degrees. It should not be applied
        if the viewpoint is an action delta (dx, dy, dheading).
        """
        x_old, y_old, h_old = viewpoint[:, 0], viewpoint[:, 1], viewpoint[:, 2]
        x_new = -y_old
        y_new = x_old

        h_new = h_old
        if apply_to_heading:
            h_new = h_old + (torch.pi/2)/self.ranges['heading']
        return torch.stack([x_new, y_new, h_new], dim=-1)

    def _get_robot_viewpoint(self, normalized: bool = False, noise: bool = False) -> torch.Tensor:
        viewpoint = self.viewpoint.clone()
        if normalized:
            viewpoint = self.normalize(viewpoint)

        if noise:
            viewpoint[:, 0] += torch.randn(self.num_envs, device=self.device) * (self.cfg.noise_std['x'] / self.ranges['x'])
            viewpoint[:, 1] += torch.randn(self.num_envs, device=self.device) * (self.cfg.noise_std['y'] / self.ranges['y'])
            viewpoint[:, 2] += torch.randn(self.num_envs, device=self.device) * (self.cfg.noise_std['heading'] / self.ranges['heading'])

        return viewpoint  # (E,3)

    def _resample_properties(self, max_tries: int = 100):
        """
        Sample robot pose (x,y,heading) so that the *full rotated footprint* (self.contour)
        does not intersect with any object polygon. The robot is allowed to extend
        outside the workspace boundaries.
        """
        E = self.num_envs
        device = self.device

        # object polygons (E,M,P,2)
        obj_contour = self.object_interest_builder.obj_prop["contour"].to(device)

        valid = torch.zeros(E, dtype=torch.bool, device=device)
        viewpoint = torch.zeros(E, 3, device=device)

        for _ in range(max_tries):
            to_sample = ~valid
            if not to_sample.any():
                break
            n_sample = int(to_sample.sum().item())

            # candidate viewpoints
            cand = viewpoint.clone()
            cand[to_sample, 0] = torch.empty(n_sample, device=device).uniform_(
                self.cfg.x[0], self.cfg.x[1]
            )
            cand[to_sample, 1] = torch.empty(n_sample, device=device).uniform_(
                self.cfg.y[0], self.cfg.y[1]
            )
            cand[to_sample, 2] = torch.empty(n_sample, device=device).uniform_(
                self.cfg.heading[0], self.cfg.heading[1]
            )

            # compute contour for candidates
            cos_h = torch.cos(cand[:, 2])
            sin_h = torch.sin(cand[:, 2])
            rot = torch.stack(
                [torch.stack([cos_h, -sin_h], dim=-1),
                 torch.stack([sin_h,  cos_h], dim=-1)],
                dim=-2
            )  # (E,2,2)

            base_corners = torch.tensor(
                [[-0.5, -0.5],
                 [ 0.5, -0.5],
                 [ 0.5,  0.5],
                 [-0.5,  0.5]], device=device
            ) * torch.tensor([self.cfg.size_x, self.cfg.size_y], device=device)

            contour = torch.einsum("eij,pj->epi", rot, base_corners) + cand[:, None, :2]  # (E,4,2)

            # --- Object collision check (SAT polygon intersection) ---
            def polygons_intersect(poly_a, poly_b):
                """Vectorized SAT intersection test for convex polygons (E,M,P,2)."""
                def axes(poly):
                    edges = poly[..., 1:, :] - poly[..., :-1, :]
                    edges = torch.cat([edges, poly[..., :1, :] - poly[..., -1:, :]], dim=-2)
                    normals = torch.stack([-edges[..., 1], edges[..., 0]], dim=-1)
                    return normals / (normals.norm(dim=-1, keepdim=True).clamp(min=1e-8))

                axes_a = axes(poly_a)  # (E,M,Ka,2)
                axes_b = axes(poly_b)  # (E,M,Kb,2)
                axes_all = torch.cat([axes_a, axes_b], dim=-2)  # (E,M,Ka+Kb,2)

                proj_a = (poly_a[..., :, None, :] * axes_all[..., None, :, :]).sum(-1)
                proj_b = (poly_b[..., :, None, :] * axes_all[..., None, :, :]).sum(-1)

                min_a, max_a = proj_a.min(dim=2).values, proj_a.max(dim=2).values
                min_b, max_b = proj_b.min(dim=2).values, proj_b.max(dim=2).values

                overlap = (max_a >= min_b) & (max_b >= min_a)
                return overlap.all(dim=-1)  # (E,M)

            robot_poly = contour[:, None, :, :].expand(-1, obj_contour.size(1), -1, -1)  # (E,M,4,2)
            overlaps = polygons_intersect(robot_poly, obj_contour)  # (E,M)
            no_overlap = ~overlaps.any(dim=1)

            # accept valid candidates
            new_valid = (~valid) & no_overlap
            viewpoint[new_valid] = cand[new_valid]
            valid |= new_valid

        if not valid.all():
            raise RuntimeError("Could not sample valid robot positions for all envs")

        self.viewpoint = viewpoint
        self.contour = contour  # store final contour

    def _create_contour(self):
        """Create rectangle contour for robot (with heading)."""
        E = self.num_envs
        device = self.device
        size_x, size_y = self.cfg.size_x, self.cfg.size_y

        # base rectangle corners (centered at origin)
        corners = torch.tensor([
            [-0.5, -0.5],
            [ 0.5, -0.5],
            [ 0.5,  0.5],
            [-0.5,  0.5]
        ], device=device)  # (4,2)

        # scale
        scaled = corners * torch.tensor([size_x, size_y], device=device)

        # rotation matrices
        cos_h = torch.cos(self.viewpoint[:, 2])
        sin_h = torch.sin(self.viewpoint[:, 2])
        rot = torch.stack([
            torch.stack([cos_h, -sin_h], dim=-1),
            torch.stack([sin_h,  cos_h], dim=-1)
        ], dim=-2)  # (E,2,2)

        # apply rotation and translation
        scaled = scaled.unsqueeze(0).expand(E, -1, -1)  # (E,4,2)
        contour = torch.einsum("eij,epj->epi", rot, scaled)  # (E,4,2)
        contour = contour + self.viewpoint[:, None, :2]      # (E,4,2)

        self.contour = contour

    def _build_ray_caster(self):
        self.ray_caster_builder = RayCasterBuilder(
            cfg=self.ray_caster_builder_cfg,
            device=self.device,
            num_envs=self.num_envs
        )
        self.ray_caster_builder.build(
            workspace_builder=self.workspace_builder,
            robot_builder=self,
            object_interest_builder=self.object_interest_builder,
        )