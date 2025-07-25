import torch
from isaaclab.envs import ManagerBasedRLEnv

def get_illegal_contact(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Returns a float tensor indicating whether there is an illegal contact
    for each environment (1 if there is an illegal contact, 0 otherwise).

    It checks the environment's _illegal_contact attribute, which is updated
    based on the contact forces by a Termination Term.
    """
    return env._illegal_contact.float() if hasattr(env, "_illegal_contact") \
        else torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
