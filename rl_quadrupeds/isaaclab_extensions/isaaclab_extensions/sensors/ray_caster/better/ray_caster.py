# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from collections import defaultdict
import numpy as np
import re
import torch
from collections.abc import Sequence
import omni.log
import omni.physics.tensors.impl.api as physx
import warp as wp
from isaacsim.core.prims import XFormPrim
from pxr import UsdGeom, UsdPhysics
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from isaaclab.utils.warp import convert_to_warp_mesh, kernels
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

from isaaclab_extensions.sensors.ray_caster.better.ray_caster_data import RayCasterData
from isaaclab_extensions.sensors.sensor_base import SensorBase

RAY_CASTER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
).replace(prim_path="/Visuals/RayCaster")


@wp.kernel(enable_backward=False)
def update_pose(positions: wp.array(dtype=wp.vec3), new_positions: wp.array(dtype=wp.vec3)):    
    tid = wp.tid()
    positions[tid] = new_positions[tid]


def spawn_cylinder(
    prim_path: str,
    cfg: "CylinderCfg",
):
    return sim_utils.shapes.spawn_cylinder(
        prim_path=prim_path,
        cfg=cfg,
        translation=cfg.translation,
    )


@configclass
class CylinderCfg(sim_utils.ShapeCfg):
    """Configuration parameters for a cylinder prim.

    See :meth:`spawn_cylinder` for more information.
    """

    func: sim_utils.Callable = spawn_cylinder

    radius: float = sim_utils.MISSING
    """Radius of the cylinder (in m)."""
    height: float = sim_utils.MISSING
    """Height of the cylinder (in m)."""
    axis: sim_utils.Literal["X", "Y", "Z"] = "Z"
    """Axis of the cylinder. Defaults to "Z"."""
    translation: tuple[float, float, float] = sim_utils.MISSING
    """Offset of the cylinder (in m)."""


RAY_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "rays": CylinderCfg(
            radius=0.01,
            height=1.0,
            translation=(0.0, 0.0, -0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
).replace(prim_path="/Visuals/Rays")


class BetterRayCaster(SensorBase):
    """A ray-casting sensor.

    The ray-caster uses a set of rays to detect collisions with meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor can be configured to ray-cast against
    a set of meshes with a given ray pattern.

    The meshes are parsed from the list of primitive paths provided in the configuration. These are then
    converted to warp meshes and stored in the `warp_meshes` list. The ray-caster then ray-casts against
    these warp meshes using the ray pattern provided in the configuration.

    .. note::
        Currently, only static meshes are supported. Extending the warp mesh to support dynamic meshes
        is a work in progress.
    """

    cfg: "BetterRayCasterCfg"
    """The configuration parameters."""

    def __init__(self, cfg: "BetterRayCasterCfg"):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
        # check if sensor path is valid
        # note: currently we do not handle environment indices if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Sensor_[1,2]".
        sensor_path = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_path) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the ray-caster sensor: {self.cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )
        # Initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data = RayCasterData()
        # the warp meshes used for raycasting.
        self.points: dict[str, UsdGeom.Mesh] = {}
        self._views = {}
        self.wp_meshes: dict[str, list[wp.Mesh]] = defaultdict(list)

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(self.points)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> RayCasterData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # resample the drift
        self.drift[env_ids].uniform_(*self.cfg.drift_range)

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        # check if the prim at path is an articulated or rigid prim
        # we do this since for physics-based view classes we can access their data directly
        # otherwise we need to use the xform view class which is slower
        found_supported_prim_class = False
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(
                f"Failed to find a prim at path expression: {self.cfg.prim_path}"
            )
        # create view based on the type of prim
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            self._view = self._physics_sim_view.create_articulation_view(
                self.cfg.prim_path.replace(".*", "*")
            )
            found_supported_prim_class = True
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(
                self.cfg.prim_path.replace(".*", "*")
            )
            found_supported_prim_class = True
        else:
            self._view = XFormPrim(self.cfg.prim_path, reset_xform_properties=False)
            found_supported_prim_class = True
            omni.log.warn(
                f"The prim at path {prim.GetPath().pathString} is not a physics prim! Using XFormPrim."
            )
        # check if prim view class is found
        if not found_supported_prim_class:
            raise RuntimeError(
                f"Failed to find a valid prim view class for the prim paths: {self.cfg.prim_path}"
            )

        # load the meshes by parsing the stage
        self._initialize_warp_meshes()
        # initialize the ray start and directions
        self._initialize_rays_impl()

    def _initialize_warp_meshes(self):
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            self._views[mesh_prim_path] = self._physics_sim_view.create_rigid_body_view(
                mesh_prim_path.replace(".*", "*")
            )

            assert "*" in mesh_prim_path, "Only meshs for every env is supported now"
            prim_paths = sim_utils.find_matching_prim_paths(mesh_prim_path)

            indices = []
            points = []
            n_indices = 0
            for n, path in enumerate(prim_paths):
                # obtain the mesh prim
                mesh_prim = sim_utils.get_first_matching_child_prim(
                    path, lambda prim: prim.GetTypeName() == "Mesh"
                )
                # check if valid
                if mesh_prim is None or not mesh_prim.IsValid():
                    raise RuntimeError(f"Invalid mesh prim path: {path}")
                # cast into UsdGeomMesh
                mesh = UsdGeom.Mesh(mesh_prim)
                # read the vertices and faces
                ptx = np.asarray(mesh.GetPointsAttr().Get())
                idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()) + n_indices
                n_indices += len(ptx)

                points.append(torch.tensor(ptx))
                indices.append(idx)
                # print info
                # print(
                #     f"Read mesh prim: {mesh.GetPath()} with {len(ptx)} vertices and {len(idx)} faces."
                # )

            # add the warp mesh
            points_ = np.vstack(points)
            self.points[mesh_prim_path] = (
                torch.tensor(np.stack(points)).float().contiguous().to(self._device)
            )
            self.wp_meshes[mesh_prim_path] = convert_to_warp_mesh(
                points_, np.hstack(indices), device=self._device
            )

    def _initialize_rays_impl(self):
        # compute ray stars and directions
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(
            self.cfg.pattern_cfg, self._device
        )
        self.num_rays = len(self.ray_directions)
        # apply offset transformation to the rays
        offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
        self.ray_directions = quat_apply(
            offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions
        )
        self.ray_starts += offset_pos
        # repeat the rays for each sensor
        self.ray_starts = self.ray_starts.repeat(self._view.count, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self._view.count, 1, 1)
        # prepare drift
        self.drift = torch.zeros(self._view.count, 3, device=self._device)
        # fill the data buffer
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.ray_hits_w = torch.zeros(
            self._view.count, self.num_rays, 3, device=self._device
        )
        self._data.ranges = torch.zeros(
            self._view.count, self.num_rays, device=self._device
        )
        self.ray_starts_w = torch.zeros(
            self._view.count, self.num_rays, 3, device=self._device
        )
        self.ray_directions_w = torch.zeros(
            self._view.count, self.num_rays, 3, device=self._device
        )
        self.hits = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        self._data.hit_labels = torch.zeros(
            self._view.count, self.num_rays, device=self._device
        )

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # obtain the poses of the sensors
        if isinstance(self._view, XFormPrim):
            sensor_pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            sensor_pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            sensor_pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")

        # Apply drift in-place
        sensor_pos_w.add_(self.drift[env_ids])

        # store poses
        self._data.pos_w[env_ids] = sensor_pos_w
        self._data.quat_w[env_ids] = quat_w

        # ray transform to world
        batch_size = sensor_pos_w.shape[0]
        num_rays = self.num_rays
        expanded_quat = quat_w.unsqueeze(1).expand(-1, num_rays, -1).reshape(-1, 4)
        ray_starts_local = self.ray_starts[env_ids].reshape(-1, 3)
        ray_dirs_local = self.ray_directions[env_ids].reshape(-1, 3)

        ray_starts_w = quat_apply(expanded_quat, ray_starts_local).reshape(batch_size, num_rays, 3)
        ray_directions_w = quat_apply(expanded_quat, ray_dirs_local).reshape(batch_size, num_rays, 3)
        ray_starts_w += sensor_pos_w.unsqueeze(1)

        self.ray_starts_w = ray_starts_w
        self.ray_directions_w = ray_directions_w

        # warp transform mesh poses once, then update all meshes
        for path, view in self._views.items():
            if not isinstance(view, physx.RigidBodyView):
                raise NotImplementedError("TODO")

            pos_w, quat_z = view.get_transforms().split([3, 4], dim=-1)
            pos_w = pos_w.to(self._device)
            quat_w = convert_quat(quat_z.to(self._device), to="wxyz")

            meshes = self.points[path]
            num_envs, num_vertices, _ = meshes.shape
            meshes_flat = meshes.view(num_envs * num_vertices, 3).to(self._device)
            repeated_quat = quat_w.repeat_interleave(num_vertices, dim=0).to(self._device)
            repeated_pos = pos_w.repeat_interleave(num_vertices, dim=0).to(self._device)
            transformed = quat_apply(repeated_quat, meshes_flat) + repeated_pos
            wp.launch(
                kernel=update_pose,
                dim=transformed.shape[0],
                inputs=[
                    self.wp_meshes[path].points,
                    wp.from_torch(transformed.contiguous(), dtype=wp.vec3),
                ],
            )
            self.wp_meshes[path].refit()

        # raycast against all meshes
        all_hits = []
        for mesh in self.wp_meshes.values():
            hit_positions, _, _, _ = raycast_mesh(
                ray_starts_w,
                ray_directions_w,
                max_dist=self.cfg.max_distance,
                mesh=mesh
            )
            all_hits.append(hit_positions)

        hits = torch.stack(all_hits, dim=1)  # (env, mesh, ray, 3)
        self.hits = hits

        dists = torch.norm(hits - sensor_pos_w[:, None, None, :], dim=-1)  # (env, mesh, ray)
        min_indices = torch.argmin(dists, dim=1)  # (env, ray)
        self._data.hit_labels[env_ids] = min_indices.clone().float()
        min_indices_exp = min_indices.unsqueeze(-1).expand(-1, -1, 3)

        selected_hits = torch.gather(
            hits, dim=1, index=min_indices_exp.unsqueeze(1)
        ).squeeze(1)

        self._data.ray_hits_w[env_ids] = selected_hits

        if self.cfg.range_mode:
            self._data.ranges[env_ids] = torch.norm(
                selected_hits - ray_starts_w, dim=-1
            )

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "ray_visualizer"):
                self.ray_visualizer = VisualizationMarkers(RAY_CASTER_MARKER_CFG)
                self.rays = VisualizationMarkers(RAY_MARKER_CFG)
            # set their visibility to true
            self.ray_visualizer.set_visibility(True)
            self.rays.set_visibility(True)
        else:
            if hasattr(self, "ray_visualizer"):
                self.ray_visualizer.set_visibility(False)
                self.rays.set_visibility(False)

    def _debug_vis_callback(self, event):
        # show ray hit positions
        self.ray_visualizer.visualize(self._data.ray_hits_w.view(-1, 3))

        v1 = self.ray_directions_w.view(-1, 3)
        v2 = torch.tensor([0, 0, 1.0], device=self._device).repeat(v1.shape[0], 1)
        axis = torch.cross(v1, v2, dim=1)
        axis = axis / torch.norm(axis, dim=1, keepdim=True)  # Normalize
        cos_theta = torch.sum(v1 * v2, dim=1) / (
            torch.norm(v1, dim=1) * torch.norm(v2, dim=1)
        )
        theta = torch.acos(cos_theta)
        qw = torch.cos(theta / 2).unsqueeze(1)  # Shape (N, 1)
        q_vec = axis * torch.sin(theta / 2).unsqueeze(1)  # Shape (N, 3)
        quat = torch.cat((qw, q_vec), dim=1)  # Shape (N, 4)

        scales = torch.norm(
                torch.clip(self._data.ray_hits_w.view(-1, 3), max=self.cfg.max_distance)
                - self.ray_starts_w.view(-1, 3),
                dim=-1,
            )
        
        self.rays.visualize(
            translations=self.ray_starts_w.view(-1, 3),
            orientations=quat,
            scales=torch.stack([torch.ones_like(scales), torch.ones_like(scales), scales]).T,
        )

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._physics_sim_view = None
        self._view = None