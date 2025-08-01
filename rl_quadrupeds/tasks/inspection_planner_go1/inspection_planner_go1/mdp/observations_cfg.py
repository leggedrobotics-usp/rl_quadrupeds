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
from quadrupeds_mdp.observations.inspection.objects_distance import ComputeObjectRelativePose
from quadrupeds_mdp.observations.map.scene_map import SceneGroundTruthMap
from quadrupeds_mdp.observations.position import local_viewpoint

@configclass
class NavObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Last action taken by the robot
        actions = ObsTerm(func=last_action, noise=Unoise(n_min=-0.01, n_max=0.01))

        # Current viewpoint of the robot
        viewpoint = ObsTerm(
            func=local_viewpoint,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # LiDAR hits with object labels
        # Continuous, used for navigation in the environment
        lidar_scan_w_labels = ObsTerm(
            func=lidar_scan_hits_labels_flattened,
            params={"sensor_cfg": SceneEntityCfg("ray_caster")},
        )

        # Coverage inspection score for all objects
        object_coverage_score = ObsTerm(
            func=ObjectInspectionCoverage,
            params={
                "sensor_cfg": SceneEntityCfg("ray_caster"),
            },
        )

        # Shows the proportion of the environment explored
        exploration_proportion = ObsTerm(
            func=SceneGroundTruthMap,
            params={
                "objects": [
                    "right_wall", "left_wall", "back_wall", "front_wall",
                    "block1", "block2"
                ],
            }
        )

        # Computes the relative pose of specified objects w.r.t. the robot
        object_relative_pose = ObsTerm(
            func=ComputeObjectRelativePose,
            params={"objects": ["block1", "block2"]}
        )


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()