import torch

from isaaclab.managers import RecorderTerm
from isaaclab.managers import SceneEntityCfg

from quadrupeds_mdp.assets.go1 import GO1_FOOT_NAMES
from quadrupeds_mdp.utils import convert_batch_foot_to_local_frame


def _is_invalid_tensor(x):
    """Check if a tensor is None or contains NaNs."""
    return x is None or (torch.is_tensor(x) and torch.isnan(x).any())


def _is_invalid_body_ids(body_ids):
    """Safely check if body_ids are valid (handles slices, lists, tensors)."""
    if body_ids is None:
        return True

    # Treat slice(None, None, None) as invalid
    if isinstance(body_ids, slice):
        if body_ids.start is None and body_ids.stop is None and body_ids.step is None:
            return True
        return False

    if isinstance(body_ids, (list, tuple)) and len(body_ids) == 0:
        return True

    if torch.is_tensor(body_ids) and body_ids.numel() == 0:
        return True

    return False


class DebugRewardRecorderBase(RecorderTerm):
    """Base helper class for reward debugging recorders."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env = env

    def _empty(self, name: str):
        """Return an empty recorder result instead of None."""
        print(f"⚠️  Recorder {name} returning empty data.")
        return name, {}


class DebugFeetContactForcesRecorder(DebugRewardRecorderBase):
    """Records feet contact forces and desired contact states."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.sensor_cfg = SceneEntityCfg("contact_forces", body_names=GO1_FOOT_NAMES, preserve_order=True)
        self.std = 0.25

    def record_post_step(self):
        if self.sensor_cfg is None or not hasattr(self.sensor_cfg, "name"):
            return self._empty("debug/rewards/feet_contact")

        contact_sensor = self.env.scene[self.sensor_cfg.name]
        if contact_sensor is None or not hasattr(self.sensor_cfg, "body_ids"):
            return self._empty("debug/rewards/feet_contact")

        # Match GO1_FOOT_NAMES to indices
        body_names = getattr(contact_sensor, "body_names", None)
        body_ids = [i for i, name in enumerate(body_names) if name in GO1_FOOT_NAMES]
        if not body_ids:
            print(f"⚠️ Could not resolve GO1_FOOT_NAMES: {GO1_FOOT_NAMES}")
            return self._empty("debug/rewards/feet_contact")

        net_forces = getattr(contact_sensor.data, "net_forces_w", None)
        if _is_invalid_tensor(net_forces):
            return self._empty("debug/rewards/feet_contact")

        feet_forces = torch.norm(net_forces[:, body_ids, :], dim=-1)
        if _is_invalid_tensor(feet_forces):
            return self._empty("debug/rewards/feet_contact")

        exp_term = torch.exp(-(feet_forces ** 2) / self.std)

        foot_desired_contact = getattr(self.env, "desired_contact_states", None)
        if _is_invalid_tensor(foot_desired_contact):
            return self._empty("debug/rewards/feet_contact")

        # Align sizes safely
        n_forces = feet_forces.shape[1]
        n_contacts = foot_desired_contact.shape[1]
        if n_forces != n_contacts:
            min_n = min(n_forces, n_contacts)
            feet_forces = feet_forces[:, :min_n]
            exp_term = exp_term[:, :min_n]
            foot_desired_contact = foot_desired_contact[:, :min_n]

        reward_term = torch.mean(-(1 - foot_desired_contact) * (1 - exp_term), dim=1)

        return "debug/rewards/feet_contact", {
            "forces": feet_forces,
            "exp_term": exp_term,
            "desired_contact_states": foot_desired_contact,
            "reward_term": reward_term,
        }


class DebugJointVelRecorder(DebugRewardRecorderBase):
    """Records joint velocity magnitude."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot_cfg = SceneEntityCfg("robot")

    def record_post_step(self):
        if self.robot_cfg is None:
            return self._empty("debug/rewards/joint_vel")

        robot = self.env.scene[self.robot_cfg.name]
        if robot is None or not hasattr(robot.data, "joint_vel"):
            return self._empty("debug/rewards/joint_vel")

        joint_vel = robot.data.joint_vel[:, self.robot_cfg.joint_ids]
        if _is_invalid_tensor(joint_vel):
            return self._empty("debug/rewards/joint_vel")

        l2 = torch.sum(joint_vel ** 2, dim=1)
        return "debug/rewards/joint_vel", {"joint_vel": joint_vel, "l2": l2}


class DebugTrotSyncRecorder(DebugRewardRecorderBase):
    """Records detailed foot phase synchronization metrics consistent with DiagonalFootTrotSynchronization."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.scale = 10.0  # match reward scale

    # -------------------------------
    # Helper functions
    # -------------------------------
    def _circular_diff(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute minimal phase difference on unit circle (range [0, 0.5])."""
        raw = torch.abs(a - b)
        return torch.minimum(raw, 1.0 - raw)

    def _circular_mean(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute circular mean of two phases (wrap-safe)."""
        angles = 2 * torch.pi * torch.stack([a, b], dim=-1)
        sin_mean = torch.mean(torch.sin(angles), dim=-1)
        cos_mean = torch.mean(torch.cos(angles), dim=-1)
        mean_angle = torch.atan2(sin_mean, cos_mean) / (2 * torch.pi)
        return torch.remainder(mean_angle, 1.0)

    def _circular_variance(self, phases: torch.Tensor) -> torch.Tensor:
        """Circular variance of phase distribution (0 → synced, 1 → evenly spaced)."""
        angles = 2 * torch.pi * phases
        sin_mean = torch.mean(torch.sin(angles), dim=-1)
        cos_mean = torch.mean(torch.cos(angles), dim=-1)
        R = torch.sqrt(sin_mean ** 2 + cos_mean ** 2 + 1e-8)
        return 1 - R

    # -------------------------------
    # Main recorder
    # -------------------------------
    def record_post_step(self):
        env = self.env
        phases = getattr(env, "foot_indices", None)
        if _is_invalid_tensor(phases):
            return self._empty("debug/rewards/trot_sync")

        # Expected order: [FL, FR, RL, RR]
        phases = torch.nan_to_num(phases[:, :4], nan=0.0)
        FL, FR, RL, RR = 0, 1, 2, 3

        # --- 1. Diagonal synchronization
        diff_FL_RR = self._circular_diff(phases[:, FL], phases[:, RR])
        diff_FR_RL = self._circular_diff(phases[:, FR], phases[:, RL])
        intra_pair_diff = diff_FL_RR + diff_FR_RL  # want → 0

        # --- 2. Diagonal pairs 180° apart
        mean_FLRR = self._circular_mean(phases[:, FL], phases[:, RR])
        mean_FRRL = self._circular_mean(phases[:, FR], phases[:, RL])
        inter_pair_diff = self._circular_diff(mean_FLRR, mean_FRRL)
        inter_error = torch.abs(inter_pair_diff - 0.5)  # want → 0

        # --- 3. Lateral (pacing) and fore-hind (bounding) errors
        diff_FL_FR = self._circular_diff(phases[:, FL], phases[:, FR])
        diff_RL_RR = self._circular_diff(phases[:, RL], phases[:, RR])
        lateral_error = torch.abs(diff_FL_FR - 0.5) + torch.abs(diff_RL_RR - 0.5)

        diff_FL_RL = self._circular_diff(phases[:, FL], phases[:, RL])
        diff_FR_RR = self._circular_diff(phases[:, FR], phases[:, RR])
        forehind_error = torch.abs(diff_FL_RL - 0.5) + torch.abs(diff_FR_RR - 0.5)

        # --- 4. Pronking (global sync) penalty
        global_variance = self._circular_variance(phases)
        all_sync_penalty = torch.exp(-10.0 * global_variance)

        # --- 5. Combine all errors
        combined_error = (
            0.5 * intra_pair_diff +
            inter_error +
            0.5 * (lateral_error + forehind_error) +
            0.5 * all_sync_penalty
        )
        combined_error = torch.nan_to_num(combined_error, nan=2.0)
        combined_error = torch.clamp(combined_error, 0.0, 2.0)

        # --- 6. Reward mapping
        reward = torch.exp(-self.scale * combined_error)
        reward = torch.clamp(reward, 0.0, 1.0)

        # --- 7. Return structured debug info
        return "debug/rewards/trot_sync", {
            "phases": phases,
            "diff_FL_RR": diff_FL_RR,
            "diff_FR_RL": diff_FR_RL,
            "intra_pair_diff": intra_pair_diff,
            "mean_FLRR": mean_FLRR,
            "mean_FRRL": mean_FRRL,
            "inter_pair_diff": inter_pair_diff,
            "inter_error": inter_error,
            "diff_FL_FR": diff_FL_FR,
            "diff_RL_RR": diff_RL_RR,
            "diff_FL_RL": diff_FL_RL,
            "diff_FR_RR": diff_FR_RR,
            "lateral_error": lateral_error,
            "forehind_error": forehind_error,
            "global_variance": global_variance,
            "all_sync_penalty": all_sync_penalty,
            "combined_error": combined_error,
            "reward": reward,
        }


class DebugFootDeviationRecorder(DebugRewardRecorderBase):
    """Records current vs. default local foot positions and deviation penalty."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot_cfg = SceneEntityCfg("robot", body_names=GO1_FOOT_NAMES, preserve_order=True)
        self.robot = env.scene[self.robot_cfg.name]

    def record_post_step(self):
        if self.robot is None or not hasattr(self.robot.data, "body_pos_w"):
            return self._empty("debug/rewards/foot_dev")

        # --- Retrieve body IDs safely ---
        body_ids = self.robot_cfg.body_ids
        if _is_invalid_body_ids(body_ids):
            # Lazy resolve from body names
            body_names = getattr(self.robot.data, "body_names", None)
            if body_names is None:
                print("⚠️ Could not access robot.data.body_names")
                return self._empty("debug/rewards/foot_dev")

            # Match GO1_FOOT_NAMES to indices
            body_ids = [i for i, name in enumerate(body_names) if name in GO1_FOOT_NAMES]
            if not body_ids:
                print(f"⚠️ Could not resolve GO1_FOOT_NAMES: {GO1_FOOT_NAMES}")
                return self._empty("debug/rewards/foot_dev")

        # --- Compute deviation ---
        foot_pos_w = self.robot.data.body_pos_w[:, body_ids]
        base_pos_w = self.robot.data.root_pos_w[:]
        base_quat_w = self.robot.data.root_quat_w[:]

        if _is_invalid_tensor(foot_pos_w) or _is_invalid_tensor(base_pos_w) or _is_invalid_tensor(base_quat_w):
            return self._empty("debug/rewards/foot_dev")

        foot_pos_l = convert_batch_foot_to_local_frame(foot_pos_w, base_pos_w, base_quat_w)
        default = getattr(self.env, "default_foot_pos_l", None)
        if default is None:
            default = torch.zeros_like(foot_pos_l)

        foot_deviation = torch.sum(torch.square(foot_pos_l - default), dim=(1, 2))
        return "debug/rewards/foot_dev", {
            "foot_pos_l": foot_pos_l,
            "default_foot_pos_l": default,
            "deviation": foot_deviation,
        }


class DebugBaseHeightRecorder(DebugRewardRecorderBase):
    """Records commanded vs actual base height."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.command_name = "body_height_orientation_cmd"
        self.robot_cfg = SceneEntityCfg("robot", body_names="trunk", preserve_order=True)

    def record_post_step(self):
        robot = self.env.scene[self.robot_cfg.name]
        if robot is None or not hasattr(robot.data, "root_pos_w"):
            return self._empty("debug/rewards/base_height")

        target_height = self.env.command_manager.get_command(self.command_name)
        if _is_invalid_tensor(target_height):
            return self._empty("debug/rewards/base_height")
        target_height = target_height[:, 0]

        current_height = robot.data.root_pos_w[:, 2]
        if _is_invalid_tensor(current_height):
            return self._empty("debug/rewards/base_height")

        error = current_height - target_height
        return "debug/rewards/base_height", {
            "target_height": target_height,
            "current_height": current_height,
            "error": error,
            "l2_penalty": error ** 2,
        }


class DebugFootSwingRecorder(DebugRewardRecorderBase):
    """Records foot swing height tracking values."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.foot_height_cmd = "footswing_height_cmd"
        self.command_name = "base_velocity"
        self.robot_cfg = SceneEntityCfg("robot", body_names=GO1_FOOT_NAMES, preserve_order=True)

    def record_post_step(self):
        env = self.env
        robot = env.scene[self.robot_cfg.name]
        if robot is None:
            return self._empty("debug/rewards/foot_swing")

        desired_vel = env.command_manager.get_command(self.command_name)
        desired_foot_height = env.command_manager.get_command(self.foot_height_cmd)
        if _is_invalid_tensor(desired_vel) or _is_invalid_tensor(desired_foot_height):
            return self._empty("debug/rewards/foot_swing")

        foot_indices = getattr(env, "foot_indices", None)
        if _is_invalid_tensor(foot_indices):
            return self._empty("debug/rewards/foot_swing")

        phases = 1 - torch.abs(1.0 - torch.clip((foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        target_height = desired_foot_height * phases + 0.02

        # --- Retrieve body IDs safely ---
        foot_ids = self.robot_cfg.body_ids
        if _is_invalid_body_ids(foot_ids):
            # Lazy resolve from robot.data.body_names
            body_names = getattr(robot.data, "body_names", None)
            if body_names is None:
                print("⚠️ Could not access robot.data.body_names")
                return self._empty("debug/rewards/foot_swing")

            foot_ids = [i for i, name in enumerate(body_names) if name in GO1_FOOT_NAMES]
            if not foot_ids:
                print(f"⚠️ Could not resolve GO1_FOOT_NAMES: {GO1_FOOT_NAMES}")
                return self._empty("debug/rewards/foot_swing")

        actual_height = robot.data.body_pos_w[:, foot_ids, 2]
        if _is_invalid_tensor(actual_height):
            return self._empty("debug/rewards/foot_swing")

        # Align shapes
        if target_height.shape[1] != actual_height.shape[1]:
            min_n = min(target_height.shape[1], actual_height.shape[1])
            target_height = target_height[:, :min_n]
            actual_height = actual_height[:, :min_n]

        deviation = target_height - actual_height

        desired_contact_states = getattr(env, "desired_contact_states", None)
        if _is_invalid_tensor(desired_contact_states):
            return self._empty("debug/rewards/foot_swing")

        # Match contact states shape
        if deviation.shape[1] != desired_contact_states.shape[1]:
            min_n = min(deviation.shape[1], desired_contact_states.shape[1])
            deviation = deviation[:, :min_n]
            desired_contact_states = desired_contact_states[:, :min_n]

        reward_term = torch.square(deviation) * (1 - desired_contact_states) * \
                      torch.linalg.norm(desired_vel, dim=1, keepdim=True)

        return "debug/rewards/foot_swing", {
            "foot_indices": foot_indices,
            "desired_foot_height": desired_foot_height,
            "desired_vel": desired_vel,
            "phases": phases,
            "target_height": target_height,
            "actual_height": actual_height,
            "deviation": deviation,
            "desired_contact_states": desired_contact_states,
            "reward_term": torch.sum(reward_term, dim=1),
        }


class DebugJointPosRecorder(DebugRewardRecorderBase):
    """Records joint positions and deviations from default."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot_cfg = SceneEntityCfg("robot")

    def record_post_step(self):
        robot = self.env.scene[self.robot_cfg.name]
        if robot is None or not hasattr(robot.data, "joint_pos"):
            return self._empty("debug/rewards/joint_pos")

        joint_pos = robot.data.joint_pos[:, self.robot_cfg.joint_ids]
        default_pos = robot.data.default_joint_pos[:, self.robot_cfg.joint_ids]
        if _is_invalid_tensor(joint_pos) or _is_invalid_tensor(default_pos):
            return self._empty("debug/rewards/joint_pos")

        deviation = joint_pos - default_pos
        return "debug/rewards/joint_pos", {
            "joint_pos": joint_pos,
            "default_joint_pos": default_pos,
            "deviation": deviation,
            "l2": torch.sum(deviation ** 2, dim=1),
        }

class DebugRaibertHeuristicRecorder(DebugRewardRecorderBase):
    """Records all intermediate computations of the Raibert heuristic reward."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        # Names of commands to fetch
        self.command_name = "base_velocity"
        self.gait_stance_distances_cmd = "gait_stance_distances_cmd"
        self.gait_step_freq_cmd = "gait_freq_cmd"

        # Robot configuration (feet only)
        self.robot_cfg = SceneEntityCfg("robot", body_names=GO1_FOOT_NAMES, preserve_order=True)

        # Preallocate neutral foot positions (as in reward)
        self.neutral_y_pos_l = torch.zeros((env.num_envs, 4), device=env.device)
        self.neutral_x_pos_l = torch.zeros((env.num_envs, 4), device=env.device)

    def reset(self, env_ids=None):
        """Reset internal reference foot positions."""
        env = self.env
        stance_distances_cmd = env.command_manager.get_command(self.gait_stance_distances_cmd)

        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)

        self.neutral_y_pos_l[env_ids, :] = stance_distances_cmd[env_ids, 0].unsqueeze(1) / 2.
        self.neutral_y_pos_l[env_ids, 1:4:2] *= -1.0
        self.neutral_x_pos_l[env_ids, :] = stance_distances_cmd[env_ids, 1].unsqueeze(1) / 2.
        self.neutral_x_pos_l[env_ids, 2:4] *= -1.0

    def record_post_step(self):
        """Logs all Raibert heuristic components each step."""
        env = self.env
        robot = env.scene[self.robot_cfg.name]
        if robot is None:
            return self._empty("debug/rewards/raibert_heuristic")

        # --- Retrieve commands ---
        cmd = env.command_manager.get_command(self.command_name)
        step_freq_cmd = env.command_manager.get_command(self.gait_step_freq_cmd)
        stance_distances_cmd = env.command_manager.get_command(self.gait_stance_distances_cmd)

        if _is_invalid_tensor(cmd) or _is_invalid_tensor(step_freq_cmd) or _is_invalid_tensor(stance_distances_cmd):
            return self._empty("debug/rewards/raibert_heuristic")

        # --- Decompose command ---
        x_base_vel_cmd = cmd[:, 0].unsqueeze(1)
        z_base_ang_vel_cmd = cmd[:, 2].unsqueeze(1)

        # --- Compute phase and derived commands ---
        foot_indices = getattr(env, "foot_indices", None)
        if _is_invalid_tensor(foot_indices):
            return self._empty("debug/rewards/raibert_heuristic")

        # Convert phase from [-1, 1] → [-0.5, 0.5]
        phases = torch.abs(1.0 - (foot_indices * 2.0)) * 1.0 - 0.5

        # Y velocity command (turning)
        y_base_vel_cmd = z_base_ang_vel_cmd * (stance_distances_cmd[:, 1].unsqueeze(1)) / 2

        # --- Compute commanded foot offsets ---
        xs_foot_offset_cmd = phases * x_base_vel_cmd * (0.5 / step_freq_cmd)
        ys_foot_offset_cmd = phases * y_base_vel_cmd * (0.5 / step_freq_cmd)
        ys_foot_offset_cmd[:, 2:4] *= -1.0

        # --- Commanded neutral + offsets ---
        x_foot_pos_cmd = self.neutral_x_pos_l + xs_foot_offset_cmd
        y_foot_pos_cmd = self.neutral_y_pos_l + ys_foot_offset_cmd
        foot_pos_cmd = torch.stack([x_foot_pos_cmd, y_foot_pos_cmd], dim=2)

        # --- Get actual foot positions ---
        foot_ids = self.robot_cfg.body_ids
        if _is_invalid_body_ids(foot_ids):
            body_names = getattr(robot.data, "body_names", None)
            if body_names is None:
                print("⚠️ Could not access robot.data.body_names")
                return self._empty("debug/rewards/raibert_heuristic")

            foot_ids = [i for i, name in enumerate(body_names) if name in GO1_FOOT_NAMES]
            if not foot_ids:
                print(f"⚠️ Could not resolve GO1_FOOT_NAMES: {GO1_FOOT_NAMES}")
                return self._empty("debug/rewards/raibert_heuristic")

        foot_pos_w = robot.data.body_pos_w[:, foot_ids]
        base_pos_w = robot.data.root_pos_w[:]
        base_quat_w = robot.data.root_quat_w[:]

        foot_pos_l = convert_batch_foot_to_local_frame(
            foot_pos_w,
            base_pos_w,
            base_quat_w,
        )

        # --- Compute error ---
        error = torch.abs(foot_pos_cmd - foot_pos_l[:, :, :2])
        reward_term = torch.sum(torch.square(error), dim=(1, 2))

        # --- Log everything ---
        return "debug/rewards/raibert_heuristic", {
            "cmd": cmd,
            "step_freq_cmd": step_freq_cmd,
            "stance_distances_cmd": stance_distances_cmd,
            "x_base_vel_cmd": x_base_vel_cmd,
            "z_base_ang_vel_cmd": z_base_ang_vel_cmd,
            "y_base_vel_cmd": y_base_vel_cmd,
            "phases": phases,
            "xs_foot_offset_cmd": xs_foot_offset_cmd,
            "ys_foot_offset_cmd": ys_foot_offset_cmd,
            "x_foot_pos_cmd": x_foot_pos_cmd,
            "y_foot_pos_cmd": y_foot_pos_cmd,
            "foot_pos_cmd": foot_pos_cmd,
            "foot_pos_l": foot_pos_l[:, :, :2],
            "error": error,
            "reward_term": reward_term,
        }

class DebugJointPosLimitsRecorder(DebugRewardRecorderBase):
    """Records and logs joint position limit penalties."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Default robot asset configuration
        self.asset_cfg = SceneEntityCfg("robot")

    def record_post_step(self):
        """Logs quantities involved in joint position limit penalties."""
        env = self.env
        asset = env.scene[self.asset_cfg.name]
        if asset is None:
            return self._empty("debug/rewards/joint_pos_limits")

        # --- Retrieve joint data safely ---
        joint_pos = getattr(asset.data, "joint_pos", None)
        soft_limits = getattr(asset.data, "soft_joint_pos_limits", None)
        if _is_invalid_tensor(joint_pos) or _is_invalid_tensor(soft_limits):
            return self._empty("debug/rewards/joint_pos_limits")

        joint_ids = getattr(self.asset_cfg, "joint_ids", None)
        if _is_invalid_body_ids(joint_ids):
            n_joints = joint_pos.shape[1]
            joint_ids = list(range(n_joints))

        # --- Compute limit violations ---
        lower_violation = -(
            joint_pos[:, joint_ids] - soft_limits[:, joint_ids, 0]
        ).clip(max=0.0)
        upper_violation = (
            joint_pos[:, joint_ids] - soft_limits[:, joint_ids, 1]
        ).clip(min=0.0)

        out_of_limits = lower_violation + upper_violation
        reward_term = torch.sum(out_of_limits, dim=1)

        # --- Log everything ---
        return "debug/rewards/joint_pos_limits", {
            "joint_pos": joint_pos,
            "soft_joint_limits_lower": soft_limits[:, :, 0],
            "soft_joint_limits_upper": soft_limits[:, :, 1],
            "joint_ids": torch.tensor(joint_ids, device=env.device),
            "lower_violation": lower_violation,
            "upper_violation": upper_violation,
            "out_of_limits": out_of_limits,
            "reward_term": reward_term,
        }

class DebugActionRateL2Recorder(DebugRewardRecorderBase):
    """Records and logs action rate (L2) penalty values."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def record_post_step(self):
        """Logs the difference between current and previous actions."""
        env = self.env

        # --- Retrieve actions safely ---
        action = getattr(env.action_manager, "action", None)
        prev_action = getattr(env.action_manager, "prev_action", None)

        if _is_invalid_tensor(action) or _is_invalid_tensor(prev_action):
            return self._empty("debug/rewards/action_rate_l2")

        # --- Compute deltas and penalty ---
        delta_action = action - prev_action
        delta_sq = torch.square(delta_action)
        reward_term = torch.sum(delta_sq, dim=1)

        # --- Log everything ---
        return "debug/rewards/action_rate_l2", {
            "action": action,
            "prev_action": prev_action,
            "delta_action": delta_action,
            "delta_action_squared": delta_sq,
            "reward_term": reward_term,
        }


class DebugIllegalContactRecorder(DebugRewardRecorderBase):
    """Records and logs illegal contact detection data."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Default sensor config
        self.sensor_cfg = SceneEntityCfg("contact_forces")
        # Default threshold (can be overridden dynamically)
        params = {
            "body_names": ["trunk", ".*_hip", ".*_thigh", ".*_calf"],
            "threshold": 1.0
        }
        self.body_names = params.get("body_names", [])
        self.threshold = params.get("threshold", 100.0)

    def record_post_step(self):
        env = self.env
        sensor_cfg = self.sensor_cfg
        threshold = self.threshold

        contact_sensor = env.scene[sensor_cfg.name]
        if contact_sensor is None:
            return self._empty("debug/rewards/illegal_contact")

        net_contact_forces = getattr(contact_sensor.data, "net_forces_w_history", None)
        if _is_invalid_tensor(net_contact_forces):
            print("⚠️ net_contact_forces is invalid")
            return self._empty("debug/rewards/illegal_contact")

        body_ids = getattr(sensor_cfg, "body_ids", None)
        if _is_invalid_body_ids(body_ids):
            body_ids = [i for i, name in enumerate(self.body_names)]

        force_norms = torch.norm(net_contact_forces[:, :, body_ids], dim=-1)
        max_force_per_body, _ = torch.max(force_norms, dim=1)
        contact_mask = max_force_per_body > threshold
        collision = torch.any(contact_mask, dim=1)

        num_envs = env.num_envs

        return "debug/rewards/illegal_contact", {
            "threshold": torch.full((num_envs,), threshold, device=env.device),
            "net_contact_forces": net_contact_forces,
            "force_norms": force_norms,
            "max_force_per_body": max_force_per_body,
            "contact_mask": contact_mask,
            "collision": collision,
            "_illegal_contact": env._illegal_contact.clone().expand(num_envs),
        }

class DebugWTWCommandFootStatesRecorder(DebugRewardRecorderBase):
    """
    Recorder for WTWCommandFootStates.
    Logs the environment tensors and the returned observation tensor:
      - foot_indices
      - clock_inputs
      - desired_contact_states
      - wtw_observation (sin/cos of foot phases)
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def record_post_step(self):
        env = self.env

        # --- Retrieve the tensors computed by WTWCommandFootStates
        foot_indices = getattr(env, "foot_indices", None)
        clock_inputs = getattr(env, "clock_inputs", None)
        desired_contacts = getattr(env, "desired_contact_states", None)

        # --- Validate presence and shape
        if (
            _is_invalid_tensor(foot_indices)
            or _is_invalid_tensor(clock_inputs)
            or _is_invalid_tensor(desired_contacts)
        ):
            return self._empty("debug/observations/wtw_command_foot_states")

        # --- Sanitize tensors (avoid NaNs)
        foot_indices = torch.nan_to_num(foot_indices, nan=0.0)
        clock_inputs = torch.nan_to_num(clock_inputs, nan=0.0)
        desired_contacts = torch.nan_to_num(desired_contacts, nan=0.0)

        # --- Reconstruct the returned observation exactly as in WTWCommandFootStates
        wtw_observation = torch.cat(
            [
                torch.sin(2 * torch.pi * foot_indices),
                torch.cos(2 * torch.pi * foot_indices),
            ],
            dim=1,
        )

        # --- Build the debug dictionary
        debug_data = {
            "foot_indices": foot_indices,
            "clock_inputs": clock_inputs,
            "desired_contact_states": desired_contacts,
            "wtw_observation": wtw_observation,
        }

        # --- Return under a consistent key for the recorder manager
        return "debug/observations/wtw_command_foot_states", debug_data