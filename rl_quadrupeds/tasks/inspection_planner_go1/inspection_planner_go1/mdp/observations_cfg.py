from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs.mdp import (
    base_lin_vel,
    base_ang_vel
)

from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup, 
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg
)

from quadrupeds_mdp.observations.action import last_action
from quadrupeds_mdp.observations.ray_caster import lidar_scan
from quadrupeds_mdp.observations.inspection.coverage import ObjectInspectionCoverage
from quadrupeds_mdp.observations.inspection.objects_distance import (
    ComputeObjectRelativePose,
    distance_from_robot_to_each_inference_point,
    position_of_each_inference_point
)
from quadrupeds_mdp.observations.map.scene_map import SceneGroundTruthMap
from quadrupeds_mdp.observations.position import local_viewpoint
from quadrupeds_mdp.observations.state import robot_is_ready_for_new_command

@configclass
class NavObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        history_length = 10
        # robot_is_ready = ObsTerm(func=robot_is_ready_for_new_command)

        viewpoint = ObsTerm(
            func=local_viewpoint,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "xy_norm": 2.,
                "heading_norm": 3.14,
            },
        )

        # base_lin_vel = ObsTerm(func=base_lin_vel)
        # base_ang_vel = ObsTerm(func=base_ang_vel)

        # Last action taken by the robot
        viewpoint_action = ObsTerm(
            func=last_action, 
            params={"action_name": "viewpoint_action"}
        )

        capture_feat_action = ObsTerm(
            func=last_action, 
            params={"action_name": "capture_feat_action"}
        )

        # LiDAR hits with object labels
        # Continuous, used for navigation in the environment
        lidar_scan_w_labels = ObsTerm(
            func=lidar_scan,
            params={
                "sensor_cfg": SceneEntityCfg("ray_caster"),
                "num_rays": None,
                "flatten": True,
                "return_hits": False,
                "return_labels": True,
                "normalize": True,
                "dist_norm": 5.0,
                "label_norm": 1, # num walls + num objects - 1
            },
        )

        # Coverage inspection score for all objects
        object_coverage = ObsTerm(
            func=ObjectInspectionCoverage,
            params={
                "sensor_cfg": SceneEntityCfg("ray_caster"),
            },
        )

        # viewpoint_already_visited = ObsTerm(
        #     func=SceneGroundTruthMap,
        #     params={
        #         # "objects": [
        #         #     "right_wall", "left_wall", "back_wall", "front_wall",
        #         #     "block1"
        #         # ],
        #         "objects": [
        #             "block1"
        #         ],
        #     }
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # @configclass
    # class CriticCfg(ObsGroup):
    #     # Current viewpoint of the robot
    #     viewpoint = ObsTerm(
    #         func=local_viewpoint,
    #         params={
    #             "asset_cfg": SceneEntityCfg("robot"),
    #             "xy_norm": 2.5,
    #             "heading_norm": 3.14,
    #         },
    #     )

    #     # Computes the relative pose of specified objects w.r.t. the robot
    #     object_relative_pose = ObsTerm(
    #         func=ComputeObjectRelativePose,
    #         params={"objects": ["block1"]}
    #     )

    #     # Viewpoint already visited
    #     

    #     distance_from_robot_to_each_inference_point = ObsTerm(
    #         func=distance_from_robot_to_each_inference_point
    #     )

    #     position_of_each_inference_point = ObsTerm(
    #         func=position_of_each_inference_point
    #     )

    policy: PolicyCfg = PolicyCfg()
    # critic: CriticCfg = CriticCfg()