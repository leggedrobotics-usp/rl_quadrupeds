import torch
from typing import Sequence
from collections import deque
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

@configclass
class Go1LocomotionCurriculumCfg(ManagerTermBase):
    """Adaptive curriculum for GO1 trot learning with term-based progress tracking and dynamic weight nudging."""

    def __init__(self, cfg: SceneEntityCfg, env: "ManagerBasedRLEnv", start_stage: int = 0):
        super().__init__(cfg, env)

        # ============================================================
        # Curriculum stages
        # ============================================================
        self.STAGES = [
            # ---------------------------------------------------------
            # Stage 0: Standing and balancing
            # ---------------------------------------------------------
            dict(
                name="stage_00_flat_orientation",
                command_ranges=dict(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
                reward_weights={
                    "flat_orientation_bonus": 1.0,
                    "joint_pos_limits_penalty": 0.8,
                    "penalize_hip_movement": 1.0,
                    "applied_torque_limits_bonus": 0.5,
                    "is_terminated": 1.0,
                },
            ),

            # ---------------------------------------------------------
            # Stage 1: Posture and joint control refinement
            # ---------------------------------------------------------
            dict(
                name="stage_01_posture_control",
                command_ranges=dict(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
                reward_weights={
                    "flat_orientation_bonus": 0.9,
                    "joint_pos_limits_penalty": 0.8,
                    "track_joint_positions_bonus": 0.6,
                    "penalize_hip_movement": 0.8,
                    "joint_torques_penalty": 0.4,
                    "action_rate_penalty": 0.3,
                    "staying_still_penalty": 0.5,
                    "is_terminated": 1.0,
                },
            ),

            # ---------------------------------------------------------
            # Stage 2: Controlled height & foot placement alignment
            # ---------------------------------------------------------
            dict(
                name="stage_02_height_alignment",
                command_ranges=dict(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
                reward_weights={
                    "flat_orientation_bonus": 0.8,
                    "track_base_height_l2_cmd_no_sensor": 1.0,
                    "penalize_foot_deviation_from_default": 1.0,
                    "penalize_hip_movement": 0.8,
                    "joint_torques_penalty": 0.4,
                    "applied_torque_limits_bonus": 0.5,
                    "all_feet_off_ground_penalty": 1.0,
                    "is_terminated": 1.0,
                },
            ),

            # ---------------------------------------------------------
            # Stage 3: Gentle motion, prepare for gait learning
            # ---------------------------------------------------------
            dict(
                name="stage_03_small_motion",
                command_ranges=dict(lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.2, 0.2)),
                reward_weights={
                    "track_lin_vel_xy_exp": 1.0,
                    "track_ang_vel_z_exp": 0.8,
                    "flat_orientation_bonus": 0.8,
                    "track_base_height_l2_cmd_no_sensor": 0.8,
                    "penalize_foot_deviation_from_default": 0.8,
                    "penalize_hip_movement": 0.8,
                    "joint_torques_penalty": 0.4,
                    "action_rate_penalty": 0.3,
                    "lin_vel_z_penalty": 0.6,
                    "is_terminated": 1.0,
                },
            ),

            # ---------------------------------------------------------
            # Stage 4: Initial trot pattern (low speed)
            # ---------------------------------------------------------
            dict(
                name="stage_04_trot_init",
                command_ranges=dict(lin_vel_x=(-0.4, 0.4), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.3, 0.3)),
                reward_weights={
                    "track_lin_vel_xy_exp": 1.0,
                    "track_ang_vel_z_exp": 0.8,
                    "trot_gait_positive": 1.0,
                    "trot_gait_negative": 0.2,
                    "trot_diag_joint_symmetry": 0.6,
                    "diag_motion_balance": 0.4,
                    "track_footswing_height": 0.6,
                    "penalize_foot_deviation_from_default": 0.8,
                    "penalize_hip_movement": 1.0,
                    "flat_orientation_bonus": 0.6,
                    "joint_torques_penalty": 0.4,
                    "action_rate_penalty": 0.3,
                    "lin_vel_z_penalty": 0.6,
                    "all_feet_off_ground_penalty": 0.8,
                    "is_terminated": 1.0,
                },
            ),

            # ---------------------------------------------------------
            # Stage 5: Stable trot, symmetry emphasized
            # ---------------------------------------------------------
            dict(
                name="stage_05_stable_trot",
                command_ranges=dict(lin_vel_x=(-0.8, 0.8), lin_vel_y=(-0.25, 0.25), ang_vel_z=(-0.5, 0.5)),
                reward_weights={
                    "track_lin_vel_xy_exp": 1.0,
                    "track_ang_vel_z_exp": 1.0,
                    "trot_gait_positive": 1.2,
                    "trot_diag_joint_symmetry": 0.8,
                    "diag_motion_balance": 0.6,
                    "track_footswing_height": 0.6,
                    "track_feet_slip": 0.4,
                    "penalize_foot_deviation_from_default": 0.8,
                    "penalize_hip_movement": 1.0,
                    "flat_orientation_bonus": 0.6,
                    "joint_torques_penalty": 0.4,
                    "joint_acc_penalty": 0.2,
                    "action_rate_penalty": 0.3,
                    "lin_vel_z_penalty": 0.6,
                    "is_terminated": 1.0,
                },
            ),

            # ---------------------------------------------------------
            # Stage 6: Faster trot with controlled yaw
            # ---------------------------------------------------------
            dict(
                name="stage_06_fast_trot",
                command_ranges=dict(lin_vel_x=(-1.2, 1.2), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.8, 0.8)),
                reward_weights={
                    "track_lin_vel_xy_exp": 1.0,
                    "track_ang_vel_z_exp": 1.0,
                    "trot_gait_positive": 1.2,
                    "trot_gait_negative": 0.2,
                    "trot_diag_joint_symmetry": 0.8,
                    "diag_motion_balance": 0.6,
                    "track_footswing_height": 0.6,
                    "track_feet_slip": 0.4,
                    "penalize_hip_movement": 1.0,
                    "flat_orientation_bonus": 0.6,
                    "track_base_height_l2_cmd_no_sensor": 0.8,
                    "joint_torques_penalty": 0.3,
                    "action_rate_penalty": 0.3,
                    "is_terminated": 1.0,
                },
            ),

            # ---------------------------------------------------------
            # Stage 7: Dynamic trot with lateral stability
            # ---------------------------------------------------------
            dict(
                name="stage_07_dynamic_trot",
                command_ranges=dict(lin_vel_x=(-1.5, 1.5), lin_vel_y=(-0.35, 0.35), ang_vel_z=(-1.0, 1.0)),
                reward_weights={
                    "track_lin_vel_xy_exp": 1.0,
                    "track_ang_vel_z_exp": 1.0,
                    "trot_gait_positive": 1.2,
                    "trot_gait_negative": 0.3,
                    "trot_diag_joint_symmetry": 0.8,
                    "diag_motion_balance": 0.6,
                    "track_footswing_height": 0.5,
                    "track_feet_slip": 0.4,
                    "track_base_height_l2_cmd_no_sensor": 0.8,
                    "penalize_hip_movement": 1.0,
                    "penalize_foot_deviation_from_default": 0.8,
                    "flat_orientation_bonus": 0.6,
                    "joint_torques_penalty": 0.3,
                    "action_rate_penalty": 0.3,
                    "lin_vel_z_penalty": 0.6,
                    "ang_vel_xy_penalty": 0.3,
                    "is_terminated": 1.0,
                },
            ),

            # ---------------------------------------------------------
            # Stage 8: Full-speed stable trot locomotion
            # ---------------------------------------------------------
            dict(
                name="stage_08_full_trot_locomotion",
                command_ranges=dict(lin_vel_x=(-2.0, 2.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.2, 1.2)),
                reward_weights={
                    "track_lin_vel_xy_exp": 1.0,
                    "track_ang_vel_z_exp": 1.0,
                    "trot_gait_positive": 1.3,
                    "trot_gait_negative": 0.2,
                    "trot_diag_joint_symmetry": 0.9,
                    "diag_motion_balance": 0.7,
                    "track_footswing_height": 0.5,
                    "track_feet_slip": 0.4,
                    "penalize_hip_movement": 1.0,
                    "penalize_foot_deviation_from_default": 0.8,
                    "track_base_height_l2_cmd_no_sensor": 0.8,
                    "flat_orientation_bonus": 0.5,
                    "joint_torques_penalty": 0.3,
                    "joint_acc_penalty": 0.2,
                    "action_rate_penalty": 0.3,
                    "is_terminated": 1.0,
                },
            ),
        ]
        self.NUM_STAGES = len(self.STAGES)

        # ============================================================
        # Scheduler parameters
        # ============================================================
        self.num_steps_per_env = 32
        self.min_stage_time = self.num_steps_per_env * 60
        self.patience_steps = self.num_steps_per_env * 8
        self.improvement_threshold = 0.001
        self.success_threshold = 0.96
        self.regression_threshold = 0.9
        self.warmup_steps = self.num_steps_per_env * 0

        # Per-term progression thresholds
        self.term_success_threshold = 0.9  # how close to max per-term reward must be

        # Nudging parameters
        self.weight_lr = 0.1  # smoothing factor for weight updates
        self.max_weight_multiplier = 2.0  # prevent runaway scaling

        # ============================================================
        # Tracking initialization
        # ============================================================
        self.current_stage = max(0, min(start_stage, self.NUM_STAGES - 1))
        self.stage_step_counter = 0
        self.default_reward_weights = {}

        if env is not None:
            for name in env.reward_manager._term_names:
                term_cfg = env.reward_manager.get_term_cfg(name)
                self.default_reward_weights[name] = float(getattr(term_cfg, "weight", 1.0))
            self._apply_stage(env, self.current_stage)

        # Reward and progress tracking
        self.last_max_improvement_step = 0
        self.max_avg_reward = float("-inf")
        self.success_flags = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.reward_window = deque(maxlen=200)
        self.rel_var = 1.0
        self.term_progress_smooth = {}  # smoothed per-term progress

    # ============================================================
    # Main per-step update
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

        self.success_flags |= truncated
        self.success_flags &= ~terminated
        success_rate = self.success_flags.float().mean().item()

        reward_improvement = 0.0
        rel_improvement = 0.0

        # Reset snapshots
        term_progress_snapshot = {}
        term_weights_snapshot = {}
        smoothed_progress_snapshot = {}

        # === Periodic curriculum update ===
        if (stage_step - self.warmup_steps) % (self.num_steps_per_env - 1) == 0:
            self.reward_window.append(avg_reward)
            if stage_step > self.warmup_steps:
                reward_improvement = avg_reward - self.max_avg_reward
                effective_threshold = max(
                    self.improvement_threshold * (0.5 ** (stage_step / (5 * self.patience_steps))),
                    0.001,
                )
                rel_improvement = reward_improvement / max(abs(self.max_avg_reward), 1e-6)
                if avg_reward > self.max_avg_reward * (1 + effective_threshold):
                    self.max_avg_reward = avg_reward
                    self.last_max_improvement_step = stage_step

                # Compute relative variance of total reward
                if len(self.reward_window) >= self.reward_window.maxlen // 2:
                    rw_tensor = torch.tensor(list(self.reward_window))
                    mean_val = rw_tensor.mean().item()
                    std_val = rw_tensor.std().item()
                    self.rel_var = std_val / max(abs(mean_val), 1e-6)

                # === Adaptive per-term progress and weight nudging ===
                term_progress_snapshot = self._compute_term_progress(env)
                self._evaluate_and_adapt_terms(env)

                # Collect per-term diagnostics
                for name in env.reward_manager._term_names:
                    term_cfg = env.reward_manager.get_term_cfg(name)
                    term_weights_snapshot[name] = float(term_cfg.weight)
                    smoothed_progress_snapshot[name] = self.term_progress_smooth.get(name, 0.0)

                # === Stage progression decision ===
                if (
                    stage_step > self.min_stage_time
                    and self._all_terms_good(env)
                    and success_rate > self.success_threshold
                ):
                    if self.current_stage < self.NUM_STAGES - 1:
                        self._advance_stage(env)

        self.stage_step_counter += 1

        # Compute safe numeric summaries
        mean_term_progress = (
            float(torch.tensor(list(term_progress_snapshot.values())).mean().item())
            if term_progress_snapshot else 0.0
        )
        mean_term_weight = (
            float(torch.tensor(list(term_weights_snapshot.values())).mean().item())
            if term_weights_snapshot else 0.0
        )
        mean_smooth_progress = (
            float(torch.tensor(list(smoothed_progress_snapshot.values())).mean().item())
            if smoothed_progress_snapshot else 0.0
        )

        # === Construct return info (numbers only) ===
        return {
            "stage": float(self.current_stage),
            "stage_steps": float(stage_step),
            "avg_reward": float(avg_reward),
            "max_avg_reward": float(self.max_avg_reward),
            "reward_improvement": float(reward_improvement),
            "rel_improvement": float(rel_improvement),
            "rel_variance": float(self.rel_var),
            "last_max_improvement_step": float(self.last_max_improvement_step),
            "time_since_last_improvement": float(stage_step - self.last_max_improvement_step),
            "success_rate": float(success_rate),
            "num_success_envs": float(self.success_flags.sum().item()),
            "num_envs": float(len(self.success_flags)),
            "mean_term_progress": mean_term_progress,
            "mean_smooth_progress": mean_smooth_progress,
            "mean_term_weight": mean_term_weight,
        }

    # ============================================================
    # Per-term evaluation and adaptive nudging
    # ============================================================
    def _evaluate_and_adapt_terms(self, env: "ManagerBasedRLEnv"):
        """Compute per-term normalized progress and adapt reward weights."""
        term_progress = self._compute_term_progress(env)
        stage_cfg = self.STAGES[self.current_stage]
        active_weights = stage_cfg["reward_weights"]

        for name, base_weight in active_weights.items():
            term_cfg = env.reward_manager.get_term_cfg(name)
            progress = term_progress.get(name, 0.0)

            # Smooth progress using EMA for stability
            old_val = self.term_progress_smooth.get(name, progress)
            smoothed = 0.8 * old_val + 0.2 * progress
            self.term_progress_smooth[name] = smoothed

            # Compute adaptive scale — emphasize low-progress terms
            adapt_scale = (1.0 - smoothed)
            lr = self.weight_lr

            new_weight = (1 - lr) * term_cfg.weight + lr * (base_weight * (1.0 + adapt_scale))
            new_weight = float(torch.clamp(torch.tensor(new_weight), 0.0, self.max_weight_multiplier))

            term_cfg.weight = new_weight
            env.reward_manager.set_term_cfg(name, term_cfg)

    def _compute_term_progress(self, env: "ManagerBasedRLEnv") -> dict:
        """Compute normalized progress [0, 1] for each reward term using baseline-normalized raw values."""
        term_progress = {}
        rm = env.reward_manager

        if not hasattr(rm, "raw_term_values") or rm.raw_term_values is None:
            return term_progress

        raw_values = rm.raw_term_values  # shape: (num_envs, num_terms)
        term_names = rm._term_names
        mean_per_term = raw_values.mean(dim=0)

        # Initialize baselines dict if not present
        if not hasattr(self, "_term_baseline"):
            self._term_baseline = {name: 1.0 for name in term_names}

        for name, mean_val in zip(term_names, mean_per_term):
            mean_val = float(mean_val.item())

            if "penalty" in name or "penalize" in name or "is_terminated" in name:
                # --- Penalty terms: lower is better ---
                base = self._term_baseline.get(name, 1.0)
                # Update baseline slowly (EMA)
                self._term_baseline[name] = 0.99 * base + 0.01 * abs(mean_val)

                # Progress = improvement relative to baseline (smaller penalty → higher progress)
                progress = 1.0 - (abs(mean_val) / (base + 1e-6))
            else:
                # --- Bonus/reward terms: higher is better ---
                base = self._term_baseline.get(name, 1.0)
                self._term_baseline[name] = 0.99 * base + 0.01 * max(mean_val, 0.0)

                # Normalize to expected range (assume reward values near 0–1)
                progress = mean_val / (base + 1e-6)

            # Clamp to [0,1] for stability
            progress = float(max(0.0, min(1.0, progress)))
            term_progress[name] = progress

        print(term_progress)

        return term_progress

    def _all_terms_good(self, env: "ManagerBasedRLEnv") -> bool:
        """Check if all active terms have reached the required smoothed progress threshold."""
        stage_cfg = self.STAGES[self.current_stage]
        active_terms = list(stage_cfg["reward_weights"].keys())

        # Use smoothed progress (EMA) so transient spikes don't trigger transitions.
        # If a term has no smoothed value yet, consider it as not ready.
        good_terms = []
        for name in active_terms:
            smoothed = self.term_progress_smooth.get(name, None)
            if smoothed is None:
                # not enough history yet → not good
                good_terms.append(False)
            else:
                good_terms.append(float(smoothed) >= float(self.term_success_threshold))

        return all(good_terms)

    # ============================================================
    # Stage transition helpers
    # ============================================================
    def _advance_stage(self, env: "ManagerBasedRLEnv"):
        self.current_stage += 1
        self._reset_tracking()
        self._apply_stage(env, self.current_stage)

    def _reset_tracking(self):
        self.stage_step_counter = 0
        self.last_max_improvement_step = 0
        self.max_avg_reward = float("-inf")
        self.success_flags[:] = False
        self.reward_window.clear()
        self.rel_var = 1.0
        self.term_progress_smooth.clear()

    def _apply_stage(self, env: "ManagerBasedRLEnv", stage_idx: int):
        """Apply the command and reward settings for a given stage."""
        stage_cfg = self.STAGES[stage_idx]
        cmd_term = env.command_manager.get_term("base_velocity")
        try:
            cmd_term.set_command_range_for_envs(None, stage_cfg["command_ranges"])
        except Exception:
            pass

        stage_weights = stage_cfg["reward_weights"]
        for name in env.reward_manager._term_names:
            term_cfg = env.reward_manager.get_term_cfg(name)
            if name in stage_weights:
                term_cfg.weight = stage_weights[name]
            else:
                term_cfg.weight = 0.0
            env.reward_manager.set_term_cfg(name, term_cfg)