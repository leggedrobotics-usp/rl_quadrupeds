from isaaclab.utils import configclass
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    SceneEntityCfg
)
from isaaclab.envs.mdp.events import (
    randomize_rigid_body_material,
    reset_root_state_uniform,
    reset_joints_by_scale,
)

from quadrupeds_mdp.events.reset import reset_root_state_uniform_if_inspection_done

@configclass
class EventCfg:
    reset_base = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-3, 3), "y": (-3, 3), "z": (0.4, 0.4), "yaw": (-3.14, 3.14)},
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

    reset_block1 = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-2, -2), "y": (-2, -2), "z": (0., 0.), "yaw": (0., 0.)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg(name="block1"),
        },
    )

    reset_block2 = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (2, 2), "y": (2, 2), "z": (0., 0.), "yaw": (0., 0.)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg(name="block2"),
        },
    )