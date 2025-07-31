from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import patterns

from isaaclab_extensions.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab_extensions.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab_extensions.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from isaaclab_extensions.sensors.ray_caster.better.ray_caster_cfg import BetterRayCasterCfg

from isaaclab_extensions.ros.publishers.joints.joints_cfg import JointsROSPublisherCfg
from isaaclab_extensions.ros.publishers.ray_caster.ray_caster_cfg import RayCasterROSPublisherCfg
from isaaclab_extensions.ros.publishers.tf.tf_cfg import TFROSPublisherCfg

from quadrupeds_mdp.assets.go1 import UNITREE_GO1_CFG
from isaaclab.assets import RigidObjectCfg

def create_wall(name, size, position):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.MeshCuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.21, 0.5, 0.73), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=(0.7071068, 0., 0., 0.7071068)),
    )

def create_block(name, position):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.5, 0.5, 1.0),  # Block size (x, y, z)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.21, 0.5, 0.73), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=position,  # Position (x, y, z)
            rot=(0.0, 0.0, 0.0, 1.0),  # No rotation
        ),
    )

# def create_contact_sensor_for_object(object_name):
#     return ContactSensorCfg(
#         prim_path=f"{{ENV_REGEX_NS}}/{object_name}",
#         history_length=3,
#         track_air_time=False,
#         debug_vis=True,
#     )

@configclass
class Go1InspectionPlannerSceneCfg(InteractiveSceneCfg):
    ros_publish = True
    
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(1000, 1000))
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    robot = UNITREE_GO1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        # ros_publishers_cfgs=[
        #     JointsROSPublisherCfg(
        #         topic_name="/joint_states",
        #         node_name="isaaclab_go1_joints_publisher",
        #     ),
        #     TFROSPublisherCfg(
        #         topic_name="tf",
        #         node_name="isaaclab_go1_tf_publisher",
        #         ref_frame_id="odom",
        #         child_frame_id="trunk"
        #     )
        # ]
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        track_air_time=True,
        debug_vis=False
    )

    ray_caster = BetterRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.1)),
        mesh_prim_paths=[
            "/World/envs/env_.*/right_wall",
            "/World/envs/env_.*/left_wall",
            "/World/envs/env_.*/front_wall",
            "/World/envs/env_.*/back_wall",
            "/World/envs/env_.*/block_1",
            "/World/envs/env_.*/block_2"
        ],
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, vertical_fov_range=[0, 0], horizontal_fov_range=[-180, 180.1], horizontal_res=5.0
        ),
        max_distance=20.,
        # ros_publisher_cfg=RayCasterROSPublisherCfg(
        #     topic_name="/lidar",
        #     node_name="isaaclab_raycaster_publisher",
        #     lidar_frame="trunk"
        # ),
        debug_vis=False, # Needs to be true to publish /lidar constantly
    )

    # size and position are (x, y, z)
    right_wall = create_wall("right_wall", (8.0, 0.05, 1.0), (4.0, 0.0, 0.5))
    left_wall  = create_wall("left_wall",  (8.0, 0.05, 1.0), (-4.0, 0.0, 0.5))
    front_wall = create_wall("front_wall", (0.05, 8.0, 1.0), (0.0, -4.0, 0.5))
    back_wall  = create_wall("back_wall",  (0.05, 8.0, 1.0), (0.0, 4.0, 0.5))
    block1 = create_block("block_1", (2.0, 1.5, 0.25))
    block2 = create_block("block_2", (-1.0, -2.0, 0.25))