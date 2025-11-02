import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardManager

class DebugRewardManager(RewardManager):

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
        # iterate over all the reward terms
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # update total reward
            self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value

            # Update current reward for this step.
            self._step_reward[:, self._term_names.index(name)] = value / dt

            print(f"Reward term '{name}': {value}")

        return self._reward_buf

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
    Custom RewardManager using online exponential moving normalization:
    - Each reward term is normalized with a running mean and std.
    - Rewards are standardized to mean 0 and variance 1, but sign is preserved.
    - Final reward is scaled to [-1, 1] per environment using running max abs.
    """

    def __init__(self, cfg: object, env: ManagerBasedRLEnv, alpha: float = 0.01):
        super().__init__(cfg, env)

        self._epsilon = 1e-8
        self._alpha = alpha  # EMA smoothing factor

        num_envs = self.num_envs
        num_terms = len(self._term_names)

        # Running stats for online normalization per reward term
        self._running_mean = torch.zeros((num_envs, num_terms), dtype=torch.float, device=self.device)
        self._running_var = torch.ones((num_envs, num_terms), dtype=torch.float, device=self.device)

        # Running max abs for final total normalization per environment
        self._running_max_abs_total = torch.ones((num_envs,), dtype=torch.float, device=self.device)

    def compute(self, dt: float) -> torch.Tensor:
        self._step_reward[:, :] = 0.0

        # === Compute and normalize each term ===
        for idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            if term_cfg.weight == 0.0:
                continue

            # Compute raw reward term
            value = term_cfg.func(self._env, **term_cfg.params)

            # --- Update running mean and variance (EMA) ---
            delta = value - self._running_mean[:, idx]
            self._running_mean[:, idx] += self._alpha * delta
            self._running_var[:, idx] = (1 - self._alpha) * self._running_var[:, idx] + self._alpha * delta.pow(2)

            # --- Online standardization ---
            std = torch.sqrt(self._running_var[:, idx] + self._epsilon)
            norm_value = value / std  # keeps sign

            # Apply weight and dt
            weighted_reward = norm_value * term_cfg.weight
            self._step_reward[:, idx] = weighted_reward
            # print(f"Term '{name}': \nraw={value}, \nnormed={norm_value}, \nweighted={weighted_reward}")

        # === Combine terms per environment ===
        total_reward = self._step_reward.sum(dim=1)

        # === Update running max abs for sign-preserving normalization ===
        self._running_max_abs_total = torch.maximum(
            self._running_max_abs_total * (1 - self._alpha) + total_reward.abs() * self._alpha,
            torch.full_like(self._running_max_abs_total, self._epsilon)
        )

        final_normalized_reward = total_reward / self._running_max_abs_total

        # === Log per-term normalized contribution ===
        for idx, name in enumerate(self._term_names):
            term_contrib = self._step_reward[:, idx]
            normalized_contrib = term_contrib / self._running_max_abs_total
            self._episode_sums[name] += normalized_contrib
        #     print(f"Term '{name}': normalized contribution={normalized_contrib}")
        # print("Running max abs total:", self._running_max_abs_total)

        return final_normalized_reward

class WeightedRewardTracker(RewardManager):
    """Reward manager that tracks raw (unweighted) reward term values before applying weights."""

    def __init__(self, cfg: object, env):
        """Initialize the reward tracker."""
        super().__init__(cfg, env)
        # Buffer to store raw (unweighted) values for each term per environment
        self._raw_term_values = torch.zeros(
            (self.num_envs, len(self._term_names)), dtype=torch.float, device=self.device
        )

    def compute(self, dt: float) -> torch.Tensor:
        """Computes the weighted reward signal and saves raw term values.

        The raw (unweighted) values of each term are stored in `self._raw_term_values`
        before multiplying by the term's weight.
        """
        # Reset buffers
        self._reward_buf[:] = 0.0
        self._raw_term_values[:] = 0.0

        # Iterate over all reward terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # Skip if weight is zero
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                self._raw_term_values[:, term_idx] = 0.0
                continue

            # Compute raw (unweighted) value
            raw_value = term_cfg.func(self._env, **term_cfg.params)

            # Store raw value before weighting
            self._raw_term_values[:, term_idx] = raw_value

            # Compute weighted and scaled value
            weighted_value = raw_value * term_cfg.weight * dt

            # Update total reward and episodic sums
            self._reward_buf += weighted_value
            self._episode_sums[name] += weighted_value

            # Store current step weighted reward (unscaled by dt for consistency with parent)
            self._step_reward[:, term_idx] = weighted_value / dt

        return self._reward_buf

    @property
    def raw_term_values(self) -> torch.Tensor:
        """Returns the most recent raw (unweighted) reward term values.

        Shape: (num_envs, num_terms)
        """
        return self._raw_term_values