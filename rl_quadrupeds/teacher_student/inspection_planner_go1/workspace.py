from typing import Optional

import matplotlib.pyplot as plt
import torch

class WorkspaceBuilderCfg():
    def __init__(self):
        self.size_x = 3.0  # size of the workspace in x direction (m)
        self.size_y = 3.0  # size of the workspace in y direction (m)

class WorkspaceBuilder:
    def __init__(self, cfg: WorkspaceBuilderCfg, device):
        self.cfg = cfg
        self.device = device

        self.build()

    def build(self):
        self.workspace = torch.tensor([
            [-self.cfg.size_x / 2, -self.cfg.size_y / 2],
            [ self.cfg.size_x / 2, -self.cfg.size_y / 2],
            [ self.cfg.size_x / 2,  self.cfg.size_y / 2],
            [-self.cfg.size_x / 2,  self.cfg.size_y / 2]
        ], device=self.device)

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs):
        """Plot the workspace polygon."""
        if ax is None:
            ax = plt.gca()
        ws = self.workspace.detach().cpu().numpy()
        ws_closed = torch.cat([self.workspace, self.workspace[0:1]], dim=0).cpu().numpy()
        ax.plot(ws_closed[:, 0], ws_closed[:, 1], 'k-', linewidth=2, label="Workspace", **kwargs)
        ax.set_aspect("equal", adjustable="box")
        return ax