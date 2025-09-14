import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import torch

from coverage import ObjectCoverage, ObjectCoverageCfg
from expert_policy import ExpertPolicy, ExpertPolicyCfg
from object_interest import ObjectInterestBuilder, ObjectInterestBuilderCfg
from ray_caster import RayCasterBuilder, RayCasterBuilderCfg
from robot import RobotBuilder, RobotBuilderCfg
from workspace import WorkspaceBuilder, WorkspaceBuilderCfg

class Simulator:
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cuda:0",
        object_coverage_cfg: Optional[ObjectCoverageCfg] = ObjectCoverageCfg(),
        object_interest_builder_cfg: Optional[ObjectInterestBuilderCfg] = ObjectInterestBuilderCfg(),
        ray_caster_builder_cfg: Optional[RayCasterBuilderCfg] = RayCasterBuilderCfg(),
        robot_builder_cfg: Optional[RobotBuilderCfg] = RobotBuilderCfg(),
        workspace_builder_cfg: Optional[WorkspaceBuilderCfg] = WorkspaceBuilderCfg(),
    ):
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.object_coverage_cfg = object_coverage_cfg
        self.object_interest_builder_cfg = object_interest_builder_cfg
        self.ray_caster_builder_cfg = ray_caster_builder_cfg
        self.robot_builder_cfg = robot_builder_cfg
        self.workspace_builder_cfg = workspace_builder_cfg

        self.build()

    def build(self):
        # robot_viewpoint per env: (E, 3) -> x, y, heading
        self.robot_viewpoint = torch.zeros(self.num_envs, 3, device=self.device)

        self._build_workspace()
        self._build_object_interest()
        self._build_robot()
        self._build_object_coverage()
        self._build_expert_policy()

    def get_observation(
        self, 
        debug: bool = False,
        normalized: bool = True,
        noise: bool = True
    ):
        robot_obs = self.robot_builder.get_observation(normalized=normalized, noise=noise)
        object_coverage_obs = self.object_coverage.get_observation(noise=noise)

        if debug:
            print(f"Robot Viewpoint ({robot_obs['viewpoint'].shape}):", robot_obs['viewpoint'][0])
            print(f"Robot Lidar Scan ({robot_obs['lidar_scan_w_labels'].shape}):", robot_obs['lidar_scan_w_labels'][0])
            print(f"Object Coverage Confidence ({object_coverage_obs['confidence'].shape}):", object_coverage_obs['confidence'][0])

        return torch.cat([
            robot_obs['viewpoint'],  # (E,3)
            robot_obs['lidar_scan_w_labels'].view(self.num_envs, -1),  # (E,N*4)
            object_coverage_obs['confidence'].view(self.num_envs, -1),  # (E,M*P)
        ], dim=-1)

    def get_expert_action(
        self, 
        debug: bool = False,
        normalized: bool = True
    ):
        expert_action = self.expert_policy.get_action(normalized=normalized)
        if debug:
            print(f"Best Robot Position ({expert_action['robot_pose'].shape}):", expert_action['robot_pose'][0])
            print(f"Inspection Action ({expert_action['inspection_action'].shape}):", expert_action['inspection_action'][0])

        return torch.cat([
            expert_action['robot_pose'],  # (E,3)
            expert_action['inspection_action'],  # (E,1)
        ], dim=-1)

    def step(self, debug: bool = False, noise: bool = True):
        self.robot_builder.ray_caster_builder.cast_rays()
        self.object_coverage.update()
        self.object_coverage.randomize_confidence()
        self.expert_policy.update()
        return (
            self.get_observation(debug=debug, noise=noise),
            self.get_expert_action(debug=debug, normalized=True)
        )

    def plot(self, env_id: int = 0):
        """Plot workspace and objects for one environment, keeping focus on the workspace."""
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot all components
        self.workspace_builder.plot(ax=ax)
        self.object_interest_builder.plot(env_id=env_id, ax=ax)
        self.robot_builder.plot(env_id=env_id, ax=ax)
        self.object_coverage.plot(env_id=env_id, ax=ax)
        self.expert_policy.plot(env_id=env_id, ax=ax)

        # Fix axis to workspace extent
        ws_limits = self.workspace_builder.workspace
        if ws_limits is not None:
            xmin, xmax, ymin, ymax = ws_limits[0][0].item(), ws_limits[2][0].item(), ws_limits[0][1].item(), ws_limits[2][1].item()
            margin = 0.1 * max(xmax - xmin, ymax - ymin)
            ax.set_xlim(xmin - margin, xmax + margin)
            ax.set_ylim(ymin - margin, ymax + margin)
            ax.set_aspect("equal")  # avoid distortion
        else:
            ax.set_aspect("equal", "box")  # fallback

        ax.set_title(f"Environment {env_id}")
        ax.legend()
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/simulator_env_{env_id}.png")
        plt.close(fig)

    def _build_workspace(self):
        self.workspace_builder = WorkspaceBuilder(
            cfg=self.workspace_builder_cfg,
            device=self.device
        )
        self.workspace_builder.build()

    def _build_object_interest(self):
        self.object_interest_builder = ObjectInterestBuilder(
            cfg=self.object_interest_builder_cfg,
            workspace_builder_cfg=self.workspace_builder_cfg,
            device=self.device,
            num_envs=self.num_envs
        )
        self.object_interest_builder.build()

    def _build_robot(self):
        self.robot_builder = RobotBuilder(
            cfg=self.robot_builder_cfg,
            workspace_builder=self.workspace_builder,
            ray_caster_builder_cfg=self.ray_caster_builder_cfg,
            object_interest_builder=self.object_interest_builder,
            device=self.device,
            num_envs=self.num_envs
        )
        self.robot_builder.build()

    def _build_object_coverage(self):
        self.object_coverage = ObjectCoverage(
            cfg=self.object_coverage_cfg,
            ray_caster_builder=self.robot_builder.ray_caster_builder,
            robot_builder=self.robot_builder,
            object_interest_builder=self.object_interest_builder,
            device=self.device,
            num_envs=self.num_envs
        )
        self.object_coverage.build()

    def _build_expert_policy(self):
        self.expert_policy = ExpertPolicy(
            cfg=ExpertPolicyCfg(),
            object_coverage=self.object_coverage,
            object_interest_builder=self.object_interest_builder,
            robot_builder=self.robot_builder,
            device=self.device,
            num_envs=self.num_envs
        )
        self.expert_policy.build()

if __name__ == "__main__":
    start = time.time()
    sim = Simulator(num_envs=2048, device="cuda:0")
    sim.step(debug=True, noise=True)
    print(f"{time.time() - start:.2f} sec")
    
    for i in range(20):
        sim.plot(env_id=i)