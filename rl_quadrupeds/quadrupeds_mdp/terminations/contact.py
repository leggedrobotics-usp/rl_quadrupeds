import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

def illegal_contact(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Terminate when the contact force on the sensor exceeds the force threshold.

    It exports "_illegal_contact" to the env so termination reason can be evaluated
    when calculating rewards.
    """
    if not hasattr(env, "_illegal_contact"):
        env._illegal_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    collision = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
    env._illegal_contact[:] = collision
    return collision