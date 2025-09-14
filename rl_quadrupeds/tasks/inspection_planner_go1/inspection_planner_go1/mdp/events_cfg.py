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
        func=reset_root_state_uniform_if_inspection_done,
        mode="reset",
        params={
            "pose_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5), "z": (0.4, 0.4), "yaw": (-3.14, 3.14)},
            "avoid_pose_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    reset_block1 = EventTerm(
        func=reset_root_state_uniform_if_inspection_done,
        mode="reset",
        params={
            "pose_range": {"x": (0., 0.), "y": (0., 0.), "z": (0., 0.), "yaw": (0., 0.)},
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