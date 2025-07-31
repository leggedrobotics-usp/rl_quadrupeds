from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup, 
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg
)

from isaaclab.envs.mdp.observations import last_action

from quadrupeds_mdp.observations.ray_caster import lidar_scan_hits_labels_flattened
from quadrupeds_mdp.observations.inspection.coverage import ObjectInspectionCoverage
from quadrupeds_mdp.observations.map.scene_map import SceneGroundTruthMap
from quadrupeds_mdp.observations.position import local_pos_rot

@configclass
class NavObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # LiDAR hits with object labels
        # Continuous, used for navigation in the environment
        lidar_scan_w_labels = ObsTerm(
            func=lidar_scan_hits_labels_flattened,
            params={"sensor_cfg": SceneEntityCfg("ray_caster")},
        )

        # Coverage inspection term
        object_coverage = ObsTerm(
            func=ObjectInspectionCoverage,
            params={
                "sensor_cfg": SceneEntityCfg("ray_caster"),
            },
        )

        # The ground truth map of the scene
        gt_map = ObsTerm(
            func=SceneGroundTruthMap,
            params={
                "objects": [
                    "right_wall", "left_wall", "back_wall", "front_wall",
                    "block1", "block2"
                ],
            }
        )

        actions = ObsTerm(func=last_action, noise=Unoise(n_min=-0.01, n_max=0.01))
        pos_and_rot = ObsTerm(
            func=local_pos_rot,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()