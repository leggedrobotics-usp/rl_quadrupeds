import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

class is_terminated_term(ManagerTermBase):
    """Termination penalty in [-1, 0]. 0 if not terminated, -1 if terminated."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env, term_keys=".*", k: float = 5.0) -> torch.Tensor:
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            reset_buf += env.termination_manager.get_term(term)

        penalty = (reset_buf * (~env.termination_manager.time_outs)).float()
        penalty_scaled = -torch.exp(-k * (1 - penalty))  # maps to [-1, 0]
        return penalty_scaled