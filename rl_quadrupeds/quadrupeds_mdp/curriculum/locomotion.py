import torch
from typing import Sequence
from collections import deque
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


@configclass
class Go1LocomotionCurriculumCfg(ManagerTermBase):
    """Synchronized curriculum for Go1 locomotion.
    Progresses when the mean episode reward stops improving
    and the success rate is high enough.
    Handles periodic reward oscillations gracefully.
    """

    def __init__(self, cfg: SceneEntityCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        # ============================================================
        # Very Conservative Curriculum (with is_terminated always active)
        # ============================================================
        self.STAGES = [
            # -------------------------------------------------------------
            # Stage 0: Standing and balancing (static stability)
            # -------------------------------------------------------------
            dict(
                name="stage_00_flat_orientation",
                command_ranges=dict(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
                active_rewards=[
                    "_flat_orientation_l2",
                    "_lin_vel_z_l2",
                    "_joint_pos_limits",
                    "is_terminated",
                ],
            ),

            # -------------------------------------------------------------
            # Stage 1: Stable posture with torque and joint control
            # -------------------------------------------------------------
            dict(
                name="stage_01_posture_and_smoothness",
                command_ranges=dict(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
                active_rewards=[
                    "_flat_orientation_l2",
                    "_joint_pos_limits",
                    "_joint_torques_l2",
                    "_joint_acc_l2",
                    "_action_rate_l2",
                    "joint_torque_limit",
                    "is_terminated",
                ],
            ),

            # -------------------------------------------------------------
            # Stage 2: Controlled base height and no drift (static locomotion prep)
            # -------------------------------------------------------------
            dict(
                name="stage_02_base_height_alignment",
                command_ranges=dict(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
                active_rewards=[
                    "_flat_orientation_l2",
                    "_track_base_height_l2_cmd_no_sensor",
                    "_penalize_foot_deviation_from_default",
                    "_joint_torques_l2",
                    "_joint_acc_l2",
                    "_action_rate_l2",
                    "is_terminated",
                ],
            ),

            # -------------------------------------------------------------
            # Stage 3: Small linear and angular motions (gentle walking commands)
            # -------------------------------------------------------------
            dict(
                name="stage_03_small_motion",
                command_ranges=dict(lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.2, 0.2)),
                active_rewards=[
                    "_track_lin_vel_xy_exp",
                    "_track_ang_vel_z_exp",
                    "_flat_orientation_l2",
                    "_track_base_height_l2_cmd_no_sensor",
                    "_joint_torques_l2",
                    "_joint_acc_l2",
                    "_action_rate_l2",
                    "_lin_vel_z_l2",
                    "_ang_vel_xy_l2",
                    "is_terminated",
                ],
            ),

            # -------------------------------------------------------------
            # Stage 4: Introducing gait structure (trot pattern learning)
            # -------------------------------------------------------------
            dict(
                name="stage_04_trot_gait_learning",
                command_ranges=dict(lin_vel_x=(-0.4, 0.4), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.3, 0.3)),
                active_rewards=[
                    "_track_lin_vel_xy_exp",
                    "_track_ang_vel_z_exp",
                    "_trot_gait_positive",
                    "_trot_gait_negative",
                    "_trot_diag_joint_symmetry",
                    "_diag_motion_balance",
                    "_track_footswing_height",
                    "_flat_orientation_l2",
                    "_joint_torques_l2",
                    "_joint_acc_l2",
                    "_action_rate_l2",
                    "is_terminated",
                ],
            ),

            # -------------------------------------------------------------
            # Stage 5: Full dynamic locomotion (final policy refinement)
            # -------------------------------------------------------------
            dict(
                name="stage_05_full_locomotion",
                command_ranges=dict(lin_vel_x=(-1.5, 1.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)),
                active_rewards=[
                    "_track_lin_vel_xy_exp",
                    "_track_ang_vel_z_exp",
                    "_trot_gait_positive",
                    "_trot_gait_negative",
                    "_trot_diag_joint_symmetry",
                    "_diag_motion_balance",
                    "_track_footswing_height",
                    "_track_joint_positions_l2",
                    "_penalize_hip_movement",
                    "_track_feet_slip",
                    "_staying_still_penalty",
                    "_flat_orientation_l2",
                    "_track_base_height_l2_cmd_no_sensor",
                    "_joint_torques_l2",
                    "_joint_acc_l2",
                    "_action_rate_l2",
                    "_lin_vel_z_l2",
                    "_ang_vel_xy_l2",
                    "is_terminated",
                ],
            ),
        ]
        self.NUM_STAGES = len(self.STAGES)

        # ============================================================
        # Scheduler / Stage Logic
        # ============================================================\
        self.num_steps_per_env = 16
        self.min_stage_time = self.num_steps_per_env * 300
        self.patience_steps = self.num_steps_per_env * 100
        self.improvement_threshold = 0.001
        self.success_threshold = 0.97
        self.regression_threshold = 0.9
        self.warmup_steps = self.num_steps_per_env * 200

        # ============================================================
        # Tracking
        # ============================================================
        self.current_stage = 0
        self.stage_step_counter = 0
        self.default_reward_weights = {}

        if env is not None:
            for name in env.reward_manager._term_names:
                term_cfg = env.reward_manager.get_term_cfg(name)
                self.default_reward_weights[name] = float(getattr(term_cfg, "weight", 1.0))
            self._apply_stage(env, 0)

        self.last_max_improvement_step = 0
        self.max_avg_reward = float("-inf")
        self.success_flags = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.reward_window = deque(maxlen=200)
        self.rel_var = 1.0

    # ============================================================
    # Main update per step
    # ============================================================
    def __call__(self, env: "ManagerBasedRLEnv", env_ids: Sequence[int]):
        stage_step = self.stage_step_counter

        episodic_reward_buf = env.reward_manager._episode_sums
        if isinstance(episodic_reward_buf, dict):
            total_rewards = sum(episodic_reward_buf.values())
            avg_reward = total_rewards.mean().item()
        else:
            avg_reward = episodic_reward_buf.mean().item()

        terminated = env.termination_manager._terminated_buf
        truncated = env.termination_manager._truncated_buf

        # ============================================================
        # Success Rate
        # ============================================================
        self.success_flags |= truncated
        self.success_flags &= ~terminated
        success_rate = self.success_flags.float().mean().item()

        # Initialize defaults to avoid UnboundLocalError
        reward_improvement = 0.0
        rel_improvement = 0.0

        if (stage_step - self.warmup_steps) % (self.num_steps_per_env - 1) == 0:
            # At the end of each episode, store the average reward
            self.reward_window.append(avg_reward)

            if stage_step > self.warmup_steps:
                # ============================================================
                # Reward Tracking & EMA
                # ============================================================
                reward_improvement = avg_reward - self.max_avg_reward
                effective_threshold = max(
                    self.improvement_threshold * (0.5 ** (stage_step / (5 * self.patience_steps))),
                    0.001,
                )
                rel_improvement = reward_improvement / max(abs(self.max_avg_reward), 1e-6)

                if avg_reward > self.max_avg_reward * (1 + effective_threshold):
                    self.max_avg_reward = avg_reward
                    self.last_max_improvement_step = stage_step

                # ============================================================
                # Periodicity-aware variance
                # ============================================================
                if len(self.reward_window) >= self.reward_window.maxlen // 2:
                    rw_tensor = torch.tensor(list(self.reward_window))
                    mean_val = rw_tensor.mean().item()
                    std_val = rw_tensor.std().item()
                    self.rel_var = std_val / max(abs(mean_val), 1e-6)

                    # --- Detect periodicity via autocorrelation
                    rw_centered = rw_tensor - rw_tensor.mean()
                    ac = torch.nn.functional.conv1d(
                        rw_centered.view(1, 1, -1),
                        rw_centered.flip(0).view(1, 1, -1),
                        padding=rw_centered.numel() - 1,
                    ).squeeze()
                    if ac.max() > 0:
                        ac = ac / ac.max()
                    ac_middle = ac[len(ac)//2:]
                    if len(ac_middle) > 50:
                        max_ac = ac_middle[10:50].max().item()  # correlation for short lags
                        if max_ac > 0.8:
                            # strong periodicity detected → suppress variance
                            self.rel_var *= 0.05

                # ============================================================
                # Stage Progression
                # ============================================================
                no_recent_improvement = (
                    stage_step - self.last_max_improvement_step > self.patience_steps
                    and self.rel_var < 0.02
                )

                if (
                    stage_step > self.min_stage_time
                    and no_recent_improvement
                    and success_rate > self.success_threshold
                ):
                    if self.current_stage < self.NUM_STAGES - 1:
                        self._advance_stage(env)

                    # if self.current_stage == self.NUM_STAGES - 1:
                    #     self.current_stage = 1

                # ============================================================
                # Stage Regression
                # ============================================================
                # elif success_rate < self.regression_threshold and self.current_stage > 0:
                #     self._regress_stage(env)

        self.stage_step_counter += 1
        return {
            "stage": self.current_stage,
            "stage_steps": stage_step,
            "avg_reward": avg_reward,
            "max_avg_reward": self.max_avg_reward,
            "reward_improvement": reward_improvement,
            "rel_improvement": rel_improvement,
            "rel_variance": self.rel_var,
            "last_max_improvement_step": self.last_max_improvement_step,
            "time_since_last_improvement": stage_step - self.last_max_improvement_step,
            "success_rate": success_rate,
        }

    # ============================================================
    # Stage transition helpers
    # ============================================================
    def _advance_stage(self, env: "ManagerBasedRLEnv"):
        self.current_stage += 1
        self._reset_tracking()
        self._apply_stage(env, self.current_stage)

    def _regress_stage(self, env: "ManagerBasedRLEnv"):
        self.current_stage -= 1
        self._reset_tracking()
        self._apply_stage(env, self.current_stage)

    def _reset_tracking(self):
        self.stage_step_counter = 0
        self.last_max_improvement_step = 0
        self.max_avg_reward = float("-inf")
        self.success_flags[:] = False
        self.reward_window.clear()
        self.rel_var = 1.0

    def _apply_stage(self, env: "ManagerBasedRLEnv", stage_idx: int):
        stage_cfg = self.STAGES[stage_idx]
        cmd_term = env.command_manager.get_term("base_velocity")

        try:
            cmd_term.set_command_range_for_envs(None, stage_cfg["command_ranges"])
        except Exception:
            pass

        active_set = set(stage_cfg["active_rewards"])
        for name in env.reward_manager._term_names:
            term_cfg = env.reward_manager.get_term_cfg(name)
            term_cfg.weight = self.default_reward_weights.get(name, 1.0) if name in active_set else 0.0
            env.reward_manager.set_term_cfg(name, term_cfg)