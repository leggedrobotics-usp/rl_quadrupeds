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

    Additionally:
    - Exports `env._illegal_contact` as an integer scalar tensor containing
      the number of environments that will be terminated at this step.
    - Still returns a boolean tensor mask (one bool per environment) to drive terminations.
    """
    # initialize the tracking attribute if it doesn't exist
    if not hasattr(env, "_illegal_contact"):
        env._illegal_contact = torch.zeros(1, dtype=torch.int32, device=env.device)

    # extract the used quantities
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    # detect collisions (boolean per environment)
    collision = torch.any(
        torch.max(
            torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1),
            dim=1
        )[0] > threshold,
        dim=1
    )

    # store the number of environments that will terminate
    env._illegal_contact[:] = torch.sum(collision).to(torch.int32)

    # return the boolean mask for termination handling
    return collision