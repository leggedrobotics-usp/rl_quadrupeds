from isaaclab.utils import configclass

from isaaclab_extensions.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from isaaclab_extensions.sensors.ray_caster.better.ray_caster import BetterRayCaster

@configclass
class BetterRayCasterCfg(RayCasterCfg):
    """Configuration for the ray-cast sensor."""

    class_type: type = BetterRayCaster

    """When range_mode is active, it will calculate the
    distance to the closest hit point for each ray."""
    range_mode: bool = True