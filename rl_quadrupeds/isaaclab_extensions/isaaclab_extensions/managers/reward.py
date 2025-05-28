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