# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Go1-Inspection-Planner",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point="isaaclab_extensions.envs.custom:CustomizableManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inspection_planner_go1_env_cfg:Go1InspectionPlannerEnvCfg",
        "skrl_cfg_entry_point": f"{__name__}:skrl_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{__name__}:skrl_sac_cfg.yaml",
        "rl_games_cfg_entry_point": f"{__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:Go1InspectionPlannerPPORunnerCfg",
    },
)
print(gym.spec("Go1-Inspection-Planner").kwargs)