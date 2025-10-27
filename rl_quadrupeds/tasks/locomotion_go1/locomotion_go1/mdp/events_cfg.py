from isaaclab.utils import configclass
from isaaclab.managers import (
    EventTermCfg as EventTerm, 
    SceneEntityCfg
)
from isaaclab.envs.mdp.events import (
    randomize_rigid_body_material,
    randomize_rigid_body_mass,
    reset_root_state_uniform,
    reset_joints_by_scale,
)

@configclass
class EventCfg:
    # physics_material = EventTerm(
    #     func=randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.8, 0.8),
    #         "dynamic_friction_range": (0.6, 0.6),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # add_base_mass = EventTerm(
    #     func=randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )

    reset_base = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.4, 0.4), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1., 1.),
            "velocity_range": (0.0, 0.0),
        },
    )