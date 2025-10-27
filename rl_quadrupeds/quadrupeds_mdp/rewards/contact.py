import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def get_illegal_contact(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Returns a float tensor indicating whether there is an illegal contact
    for each environment (1 if there is an illegal contact, 0 otherwise).

    It checks the environment's _illegal_contact attribute, which is updated
    based on the contact forces by a Termination Term.
    """
    return env._illegal_contact.item() if hasattr(env, "_illegal_contact") \
        else 0

def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold,
        dim=1
    ).float()