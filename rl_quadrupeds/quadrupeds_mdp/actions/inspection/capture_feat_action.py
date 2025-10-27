from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import DEFORMABLE_TARGET_MARKER_CFG
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CaptureFeaturesAction(ActionTerm):
    """
    Action term responsible for determining whether to capture inspection 
    features from the environment using a sensor.
    It exports a binary action to the environment in the variable 
    `capture_feat_action`.
    """

    cfg: CaptureFeaturesActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: CaptureFeaturesActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]

        self._raw_actions = torch.zeros(
            (env.num_envs, 1),
            dtype=torch.float32,
            device=env.device
        )
        self._processed_actions = torch.zeros(
            (env.num_envs, 1),
            dtype=torch.float32,
            device=env.device
        )

        env.capture_feat_action = torch.zeros(
            (env.num_envs, 1),
            dtype=torch.float32,
            device=env.device
        )

    @property
    def action_dim(self) -> int:
        # the action is a binary action, so the dimension is 1
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        # Ensure 2D shape [num_envs, 1]
        actions = actions.view(-1, 1)

        # Store raw actions
        self._raw_actions[:] = actions

        # Convert ELU outputs to binary: > 0 → 1, else 0
        self._processed_actions = (actions > 0.0).float()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is not None:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0

    def apply_actions(self):
        # Set the capture feature action to the env
        self.env.capture_feat_action[:] = self._processed_actions[:]

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "capture_feat_visualizer"):
                marker_cfg = DEFORMABLE_TARGET_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/capture_feat"
                marker_cfg.markers["target"].radius = 0.1
                marker_cfg.markers["target"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.2, 0.2)
                )
                self.capture_feat_visualizer = VisualizationMarkers(marker_cfg)
            self.capture_feat_visualizer.set_visibility(True)
        else:
            if hasattr(self, "capture_feat_visualizer"):
                self.capture_feat_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.7

        scale = self._processed_actions.clone()
        scale = torch.cat([scale, scale, scale], dim=-1)

        self.capture_feat_visualizer.visualize(translations=base_pos_w, scales=scale)


@configclass
class CaptureFeaturesActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = CaptureFeaturesAction