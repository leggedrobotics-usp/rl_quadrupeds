# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--pt", type=str, default=False, help="File for conversion.")
parser.add_argument("--task", type=str, default="Go1-Locomotion", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import os
import torch

import skrl
from skrl.agents.torch.ppo import PPO
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.3.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from isaaclab_extensions.isaaclab_rl.skrl import SkrlVecEnvWrapper

import tasks

# config shortcuts
algorithm = args_cli.algorithm.lower()

import torch
from skrl.agents.torch.ppo import PPO
def convert_pt2jit(env,experiment_cfg,path_pt):
    obs, _ = env.reset()
    print(obs)
    runner = Runner(env, experiment_cfg)
    print(f"[INFO] Loading model checkpoint from: {path_pt}")
    runner.agent.load(path_pt)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            torch.nn.Module.__init__(self)
            self._model = model
        def forward(self, inputs):
            actions, log_prob, outputs = self._model.act({"states": inputs}, 'policy')
            return outputs["mean_actions"]

    # Export policy
    obs, _ = env.reset()

    assert type(runner.agent) is PPO 

    agent: PPO = runner.agent
    model = agent.policy
    adapter = ModelWrapper(model)
    traced = torch.jit.trace(adapter, obs, strict=False, check_trace=False)
    traced.eval()
    traced.save('policy.jit.pt')

    policy = torch.jit.load('policy.jit.pt').to(env.device).eval()
    print('Comparing original .pt output with the new .jit.pt output.')
    for i in range(3):
        print(i,'=======================================================')
        dummy_obs = torch.randn(3,obs.shape[-1]).to(env.device)
        with torch.inference_mode():
                # agent stepping
                outputs = runner.agent.act(dummy_obs, timestep=0, timesteps=0)
                # - multi-agent (deterministic) actions
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                # - single-agent (deterministic) actions
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                # env stepping
                print('pt output:',actions)
        output = policy(dummy_obs)
        print('jit output:',output)


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )
    
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    checkpoint_path = os.path.abspath(args_cli.pt)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    
    convert_pt2jit(env,experiment_cfg,checkpoint_path)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
