from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

from isaaclab_extensions.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab_extensions.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

from quadrupeds_mdp.assets.go1 import UNITREE_GO1_CFG

@configclass
class Go1LocomotionSceneCfg(InteractiveSceneCfg):   
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(1000, 1000))
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        track_air_time=True,
        debug_vis=False
    )