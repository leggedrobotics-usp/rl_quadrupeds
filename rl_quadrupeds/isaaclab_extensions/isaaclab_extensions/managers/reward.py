import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardManager

class ExponentialRewardManager(RewardManager):
    """
    Customizes the standard RewardManager to replicate the paper "Walk These Ways: Tuning Robot 
    Control for Generalization with Multiplicity of Behavior".

    It computes all the rewards terms separately. Combine them by calculating
    $$r_{task} \exp(c_{aux} r_{aux})$$, where
        - $$r_{task}$$ is the sum of positive (task) reward terms.
        - $$r_{aux}$$ is the sum of negative (auxiliary) reward terms.
        - $$c_{aux}$$ is an arbitrary constant.
    """

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._positive_reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._negative_reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)


    def compute(self, dt: float) -> torch.Tensor:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # reset computation
        self._reward_buf[:] = 0.0
        self._positive_reward_buf[:] = 0.0
        self._negative_reward_buf[:] = 0.0
        
        # iterate over all the reward terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt

            if (value >= -1e-16).all():
                # positive reward
                self._positive_reward_buf += value
            else:
                # negative reward
                self._negative_reward_buf += value

            # update episodic sum
            self._episode_sums[name] += value
            # Update current reward for this step.
            self._step_reward[:, term_idx] = value / dt

        # update total reward
        # TODO: c_aux is hardcoded to 0.2, should be a parameter
        self._reward_buf = self._positive_reward_buf * torch.exp(0.2*self._negative_reward_buf)

        return self._reward_buf
    
class NormalizedRewardManager(RewardManager):
    """
    Custom RewardManager using min-max normalization:
    - Each reward term is normalized to [0, 1] using its own min/max.
    - Then the weighted sum across terms is normalized to [-1, 1] per environment.
    """

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # [num_envs, num_terms, 2]: [min, max]
        inf = float('inf')
        self._term_minmax = torch.stack([
            torch.full((self.num_envs, len(self._term_names)), inf, dtype=torch.float, device=self.device),    # min
            torch.full((self.num_envs, len(self._term_names)), -inf, dtype=torch.float, device=self.device)    # max
        ], dim=-1)

        # [num_envs, 2]: [min, max] of total weighted reward
        self._env_minmax = torch.stack([
            torch.full((self.num_envs,), inf, dtype=torch.float, device=self.device),   # min
            torch.full((self.num_envs,), -inf, dtype=torch.float, device=self.device)   # max
        ], dim=-1)

        self._epsilon = 1e-8

    def compute(self, dt: float) -> torch.Tensor:
        self._step_reward[:, :] = 0.0

        for idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            if term_cfg.weight == 0.0:
                continue

            value = term_cfg.func(self._env, **term_cfg.params)

            # === Update min/max ===
            self._term_minmax[:, idx, 0] = torch.minimum(self._term_minmax[:, idx, 0], value)
            self._term_minmax[:, idx, 1] = torch.maximum(self._term_minmax[:, idx, 1], value)

            min_val = self._term_minmax[:, idx, 0]
            max_val = self._term_minmax[:, idx, 1]
            range_val = max_val - min_val

            # Normalize to [0, 1]
            norm_value = (value - min_val) / (range_val + self._epsilon)

            # Apply weight while preserving weight scale
            weighted_reward = norm_value * term_cfg.weight * dt
            self._step_reward[:, idx] = weighted_reward

        # === Combine terms per environment ===
        total_reward = self._step_reward.sum(dim=1)

        # === Update per-env min/max of combined reward ===
        self._env_minmax[:, 0] = torch.minimum(self._env_minmax[:, 0], total_reward)
        self._env_minmax[:, 1] = torch.maximum(self._env_minmax[:, 1], total_reward)

        # === Sign-preserving normalization ===
        max_abs_total = torch.max(
            total_reward.abs(),
            torch.max(self._env_minmax.abs(), dim=1).values
        )
        max_abs_total = torch.clamp(max_abs_total, min=self._epsilon)

        final_normalized_reward = total_reward / max_abs_total

        # === Log per-term normalized contribution ===
        for idx, name in enumerate(self._term_names):
            term_contrib = self._step_reward[:, idx]
            normalized_contrib = term_contrib / max_abs_total
            self._episode_sums[name] += normalized_contrib

        return final_normalized_reward