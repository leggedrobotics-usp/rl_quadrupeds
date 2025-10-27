# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Convert an RSL-RL checkpoint (.pt) to a TorchScript model (.jit.pt) with output normalization [-1, 1].

Usage:
    isaaclab -p pt2jit_rsl_rl.py --task Go1-Locomotion --pt logs/rsl_rl/go1_locomotion/.../model_3490.pt --compare
"""

import argparse
import os
import torch

from isaaclab.app import AppLauncher  # Must be imported before launching app

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Convert an RSL-RL checkpoint (.pt) into TorchScript (.jit.pt).")
parser.add_argument("--task", type=str, required=True, help="Name of the task (e.g., Go1-Locomotion)")
parser.add_argument("--pt", type=str, required=True, help="Path to RSL-RL checkpoint (.pt)")
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments")
parser.add_argument("--compare", action="store_true", help="Compare outputs before/after export")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# -----------------------------------------------------------------------------
# LAUNCH ISAAC SIM
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# IMPORTS THAT REQUIRE ISAAC SIM
# -----------------------------------------------------------------------------
import gymnasium as gym
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_extensions.isaaclab_rl.vecenv_wrapper import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa
import tasks  # noqa

# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------
def main():
    # Load configs manually (no Hydra)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device

    checkpoint_path = retrieve_file_path(args_cli.pt)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load RSL-RL runner and policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path, load_optimizer=False)
    print("[INFO] Checkpoint loaded successfully.")

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # -----------------------------------------------------------------------------
    # WRAP POLICY FOR TORCHSCRIPT EXPORT
    # -----------------------------------------------------------------------------
    class ClippedPolicy(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            """
            Forward pass for TorchScript export.
            Accepts obs of shape [N, obs_dim] and returns actions in [-1, 1].
            """
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)

            with torch.no_grad():
                # Call act() and always return a tensor for tracing
                out = self.model.act(obs, deterministic=True)
                # If act() returns a tuple, take the first element (actions)
                if isinstance(out, tuple):
                    actions = out[0]
                else:
                    actions = out
            return torch.clamp(actions, -1.0, 1.0)

    clipped_policy = ClippedPolicy(policy_nn).eval().to(agent_cfg.device)

    # -----------------------------------------------------------------------------
    # TRACE AND EXPORT
    # -----------------------------------------------------------------------------
    obs, _ = env.get_observations()
    obs = obs.to(agent_cfg.device)

    # Trace with a single batch for TorchScript (works for any batch size later)
    example_input = obs[:1]
    print(f"[INFO] Tracing with input shape {tuple(example_input.shape)} ...")

    traced = torch.jit.trace(clipped_policy, example_input, strict=False, check_trace=False)
    traced.eval()
    print("[INFO] Traced TorchScript schema:", traced.forward.schema)

    export_dir = os.path.join(os.path.dirname(checkpoint_path), "exported")
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, "policy.jit.pt")
    traced.save(export_path)
    print(f"[INFO] Exported TorchScript model with [-1, 1] clipping: {export_path}")

    # -----------------------------------------------------------------------------
    # OPTIONAL COMPARISON
    # -----------------------------------------------------------------------------
    if args_cli.compare:
        jit_policy = torch.jit.load(export_path).to(agent_cfg.device).eval()
        print("[INFO] Comparing PT vs JIT outputs...")
        for i in range(3):
            dummy_obs = torch.randn_like(obs)
            with torch.inference_mode():
                pt_out = clipped_policy(dummy_obs)
                jit_out = jit_policy(dummy_obs)
            print(f"\n[{i}] PT output:  {pt_out}")
            print(f"[{i}] JIT output (clipped): {jit_out}")

    env.close()


# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    simulation_app.close()