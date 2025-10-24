import json
import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
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
        cfgs: Optional[dict] = None,
        cfg_json: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.num_envs = int(num_envs)

        # Load configuration
        self.cfgs = self._load_cfgs(cfgs, cfg_json)

        # Initialize configs (falling back to defaults)
        self._init_cfg_objects(self.cfgs)

        # Build simulation components
        self.build()

    # ------------------------------------------------------------------
    # CONFIG HANDLING
    # ------------------------------------------------------------------
    def _load_cfgs(self, cfgs: Optional[dict], cfg_json: Optional[str]) -> dict:
        """Load configs from JSON and merge with user overrides."""
        if cfg_json is not None:
            with open(cfg_json, "r") as f:
                cfgs_from_file = json.load(f)
            cfgs = {**(cfgs or {}), **cfgs_from_file}  # dict > JSON priority
        return self._replace_constants(cfgs or {})

    def _replace_constants(self, obj):
        """Convert strings like 'pi' and '-pi' to float values."""
        if isinstance(obj, str):
            if obj.lower() == "pi":
                return float(torch.pi)
            if obj.lower() == "-pi":
                return float(-torch.pi)
            return obj
        if isinstance(obj, list):
            return [self._replace_constants(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._replace_constants(v) for k, v in obj.items()}
        return obj

    def _init_cfg_objects(self, cfgs: dict):
        """Merge configs with defaults and initialize cfg objects."""
        defaults = {
            "object_coverage_cfg": ObjectCoverageCfg(),
            "object_interest_builder_cfg": ObjectInterestBuilderCfg(),
            "ray_caster_builder_cfg": RayCasterBuilderCfg(),
            "robot_builder_cfg": RobotBuilderCfg(),
            "workspace_builder_cfg": WorkspaceBuilderCfg(),
            "expert_policy_cfg": ExpertPolicyCfg(),
        }

        merged_cfgs = {
            k: type(defaults[k])(**cfgs[k]) if k in cfgs else defaults[k]
            for k in defaults
        }

        # Unpack to attributes
        self.object_coverage_cfg = merged_cfgs["object_coverage_cfg"]
        self.object_interest_builder_cfg = merged_cfgs["object_interest_builder_cfg"]
        self.ray_caster_builder_cfg = merged_cfgs["ray_caster_builder_cfg"]
        self.robot_builder_cfg = merged_cfgs["robot_builder_cfg"]
        self.workspace_builder_cfg = merged_cfgs["workspace_builder_cfg"]
        self.expert_policy_cfg = merged_cfgs["expert_policy_cfg"]

    # ------------------------------------------------------------------
    # SIM BUILDING
    # ------------------------------------------------------------------
    def build(
        self,
        new_simulation: bool = False
    ):
        """
        Build all simulator components.
        Args:
            new_simulation (bool): If True, update just the components for a new 
            entire simulation, maintaining the same workspace and object interest.
        """
        if new_simulation:
            self._build_robot()
            self._build_object_coverage()
            self._build_expert_policy()
            return

        self._build_workspace()
        self._build_object_interest()

    def get_observation(
        self, 
        debug: bool = False,
        normalized: bool = True,
        noise: bool = True,
        correct_axes: bool = True
    ):
        robot_obs = self.robot_builder.get_observation(normalized=normalized, noise=noise)
        object_coverage_obs = self.object_coverage.get_observation(noise=False)

        if correct_axes:
            robot_obs['viewpoint'] = self.robot_builder._transform_axes(
                robot_obs['viewpoint'],
                apply_to_heading=True
            )

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
        normalized: bool = True,
        correct_axes: bool = True
    ):
        expert_action = self.expert_policy.get_action(normalized=normalized)

        if correct_axes:
            expert_action['robot_pose'] = self.expert_policy._transform_axes(
                expert_action['robot_pose'], 
                apply_to_heading=False
            )

        if debug:
            print(f"Best Robot Position ({expert_action['robot_pose'].shape}):", expert_action['robot_pose'][0])
            print(f"Inspection Action ({expert_action['inspection_action'].shape}):", expert_action['inspection_action'][0])

        return torch.cat([
            expert_action['robot_pose'],  # (E,3)
            expert_action['inspection_action'],  # (E,1)
        ], dim=-1)

    def step(
        self, 
        debug: bool = False, 
        noise: bool = True, 
        correct_axes: bool = True, 
        normalized: bool = True,
        randomize_confidence: bool = True,
        rebuild_simulation: bool = True
    ):
        if rebuild_simulation:
            self.build(new_simulation=True)
            
        self.robot_builder.ray_caster_builder.cast_rays()
        self.object_coverage.update()
        if randomize_confidence:
            self.object_coverage.randomize_confidence()
        self.expert_policy.update()
        return (
            self.get_observation(debug=debug, noise=noise, correct_axes=correct_axes, normalized=normalized),
            self.get_expert_action(debug=debug, normalized=normalized, correct_axes=correct_axes)
        )

    def simulate_step(
        self,
        num_steps: int,
        debug: bool = False,
        noise: bool = True,
        correct_axes: bool = True,
        normalized: bool = True,
        randomize_confidence: bool = True,
        rebuild_simulation: bool = True,
        plot: bool = False,
        plot_env_id: Optional[int] = None,
        history_len: int = 1,   # NEW ARGUMENT
        _step_count: int = 0,
        _obs_list: Optional[list] = None,
        _act_list: Optional[list] = None,
        _obs_buffer: Optional[list] = None  # NEW: buffer for rolling history
    ):
        """
        Recursively simulate multiple timesteps by following the expert policy.

        Args:
            num_steps (int): Number of timesteps to simulate.
            history_len (int): Number of past observations to stack together.
        """
        if _obs_list is None:
            _obs_list, _act_list, _obs_buffer = [], [], []

        # First update the world and get obs + action
        obs, act = self.step(
            debug=debug,
            noise=noise,
            correct_axes=correct_axes,
            normalized=normalized,
            randomize_confidence=randomize_confidence,
            rebuild_simulation=rebuild_simulation
        )

        # Maintain rolling buffer of last `history_len` observations
        _obs_buffer.append(obs)
        if len(_obs_buffer) > history_len:
            _obs_buffer.pop(0)

        # Pad if not enough history yet (repeat first obs)
        if len(_obs_buffer) < history_len:
            padded = [_obs_buffer[0]] * (history_len - len(_obs_buffer)) + _obs_buffer
        else:
            padded = _obs_buffer

        # Stack history along a new dimension and flatten for agent
        obs_with_history = torch.cat(padded, dim=-1)

        _obs_list.append(obs_with_history)
        _act_list.append(act)

        # Optional plotting
        if plot:
            if plot_env_id is None:
                for i in range(min(5, self.num_envs)):
                    self.plot(env_id=i, step=_step_count)
            else:
                self.plot(env_id=plot_env_id, step=_step_count)

        # Base case: stop recursion
        if _step_count + 1 >= num_steps or self.object_coverage.is_fully_inspected(any_env=True):
            return torch.stack(_obs_list, dim=0), torch.stack(_act_list, dim=0)
        else:
            # Move the robot to the expert's suggested position
            with torch.no_grad():
                action = act[:, :3].clone()
                if correct_axes:
                    action = self.expert_policy._detransform_axes(action)
                self.robot_builder.viewpoint += self.expert_policy.denormalize(action)
                self.robot_builder.viewpoint = self.robot_builder.enforce_viewpoint_limits(
                    self.robot_builder.viewpoint
                )
                self.robot_builder._create_contour()
                self.robot_builder.ray_caster_builder._ray_tracing_results = None

        # Recursive call
        return self.simulate_step(
            num_steps=num_steps,
            debug=debug,
            noise=noise,
            correct_axes=correct_axes,
            normalized=normalized,
            randomize_confidence=False,
            rebuild_simulation=False,
            plot=plot,
            plot_env_id=plot_env_id,
            history_len=history_len,
            _step_count=_step_count + 1,
            _obs_list=_obs_list,
            _act_list=_act_list,
            _obs_buffer=_obs_buffer
        )

    def plot(self, env_id: int = 0, step: Optional[int] = None):
        """Plot workspace and objects for one environment, keeping focus on the workspace."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot all components
        self.workspace_builder.plot(ax=ax)
        self.object_interest_builder.plot(env_id=env_id, ax=ax)
        self.robot_builder.plot(env_id=env_id, ax=ax)
        self.object_coverage.plot(env_id=env_id, ax=ax)
        self.expert_policy.plot(env_id=env_id, ax=ax)

        # Fix axis to workspace extent
        ws_limits = self.workspace_builder.workspace
        if ws_limits is not None:
            xmin, xmax = ws_limits[0][0].item(), ws_limits[2][0].item()
            ymin, ymax = ws_limits[0][1].item(), ws_limits[2][1].item()
            margin = 0.1 * max(xmax - xmin, ymax - ymin)
            ax.set_xlim(xmin - margin, xmax + margin)
            ax.set_ylim(ymin - margin, ymax + margin)
            ax.set_aspect("equal")
        else:
            ax.set_aspect("equal", "box")

        ax.set_title(f"Environment {env_id} | Step {step if step is not None else 'final'}")

        # Place legend outside the plot area
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        fig.tight_layout()
        os.makedirs("figures", exist_ok=True)
        if step is None:
            fname = f"figures/simulator_env_{env_id}.png"
        else:
            fname = f"figures/simulator_env_{env_id}_step_{step}.png"
        plt.savefig(fname, bbox_inches="tight")
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
            cfg=self.expert_policy_cfg,
            object_coverage=self.object_coverage,
            object_interest_builder=self.object_interest_builder,
            robot_builder=self.robot_builder,
            device=self.device,
            num_envs=self.num_envs
        )
        self.expert_policy.build()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=None, help="Number of simulation steps")
    parser.add_argument("--plot", action="store_true", help="Plot each simulation step")
    parser.add_argument("--plot-env-id", type=int, default=None, help="Environment ID to plot. If omitted, plots first 20 envs.")
    args = parser.parse_args()

    start = time.time()
    sim = Simulator(
        num_envs=2048, 
        device="cuda:0",
        cfg_json="cfg.json"
    )

    if args.num_steps is None:
        # Just run a single step
        sim.step(debug=True, noise=False, correct_axes=True, normalized=True)
        if args.plot:
            if args.plot_env_id is None:
                for i in range(min(5, sim.num_envs)):
                    sim.plot(env_id=i, step=0)
            else:
                sim.plot(env_id=args.plot_env_id, step=0)
    else:
        # Run recursive simulation
        obs_batch, act_batch = sim.simulate_step(
            num_steps=args.num_steps,
            debug=True,
            noise=False,
            correct_axes=True,
            normalized=True,
            plot=args.plot,
            plot_env_id=args.plot_env_id
        )
        print(f"Simulated {args.num_steps} steps with shapes:")
        print(f"  obs_batch: {obs_batch.shape}")
        print(f"  act_batch: {act_batch.shape}")

    print(f"Time taken to simulate: {time.time() - start:.2f} sec")