# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

@configclass
class Go1InspectionPlannerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128
    # num_steps_per_env = 16
    max_iterations = 100000
    save_interval = 10
    experiment_name = "go1_inspection_planner"
    empirical_normalization = False
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    load_optimizer = False

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.1,
        num_learning_epochs=16,
        num_mini_batches=512,
        # num_learning_epochs=1,
        # num_mini_batches=5,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # symmetry_cfg=RslRlSymmetryCfg(
        #     use_data_augmentation=True,
        #     data_augmentation_func=,
        # )
    )