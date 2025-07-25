"""
commands_cfg.py

Configurations for command terms in quadruped locomotion tasks.

Available classes:
- QuadrupedGaitFootswingHeightCommandCfg: Configuration for gait footswing height command generator.
- QuadrupedGaitDutyCycleCommandCfg: Configuration for gait duty cycle command generator.
- QuadrupedGaitFrequencyCommandCfg: Configuration for gait frequency command generator.
- QuadrupedGaitStanceDistancesCommandCfg: Configuration for gait stance distances command generator.
- QuadrupedGaitTypeCommandCfg: Configuration for gait type command generator.
- QuadrupedBaseHeightOrientationCommandCfg: Configuration for base height and orientation command generator.
"""

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import CUBOID_MARKER_CFG
from isaaclab.utils import configclass

from .gait import (
    QuadrupedGaitDutyCycleCommand,
    QuadrupedGaitFrequencyCommand,
    QuadrupedGaitStanceDistancesCommand,
    QuadrupedGaitTypeCommand
)
from .position import (
    QuadrupedBaseHeightOrientationCommand,
    QuadrupedGaitFootswingHeightCommand,
)

@configclass
class QuadrupedGaitFootswingHeightCommandCfg(CommandTermCfg):
    """Configuration for gait footswing height command generator."""

    class_type: type = QuadrupedGaitFootswingHeightCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gait footswing height command."""

        footswing_height: tuple[float, float] = MISSING
        """Range for the quadruped robot's footswing height."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

@configclass
class QuadrupedGaitDutyCycleCommandCfg(CommandTermCfg):
    """Configuration for gait type command generator."""

    class_type: type = QuadrupedGaitDutyCycleCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gait duty cycle command."""

        duty_cycle: tuple[float, float] = MISSING
        """Range for the quadruped robot's gait duty cycle [0, 1]."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

@configclass
class QuadrupedGaitFrequencyCommandCfg(CommandTermCfg):
    """Configuration for gait frequency command generator."""

    class_type: type = QuadrupedGaitFrequencyCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the height command."""

        frequency: tuple[float, float] = MISSING
        """Range for the quadruped robot's gait frequency (in Hz)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

@configclass
class QuadrupedGaitStanceDistancesCommandCfg(CommandTermCfg):
    """Configuration for gait stance distances command generator."""

    class_type: type = QuadrupedGaitStanceDistancesCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gait footswing height command."""

        width: tuple[float, float] = MISSING
        """Range for the quadruped robot's gait stance width."""

        length: tuple[float, float] = MISSING
        """Range for the quadruped robot's gait stance length."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

@configclass
class QuadrupedGaitTypeCommandCfg(CommandTermCfg):
    """Configuration for gait type command generator."""

    class_type: type = QuadrupedGaitTypeCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    desired_gait_types: list[tuple] = MISSING
    """List of tuples to be sampled containing the gait phase, offset and bound."""

@configclass
class QuadrupedBaseHeightOrientationCommandCfg(CommandTermCfg):
    """Configuration for base height command generator."""
    class_type: type = QuadrupedBaseHeightOrientationCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the commands."""

        height: tuple[float, float] = MISSING
        """Range for the quadruped robot's base height (in m)."""

        pitch: tuple[float, float] = MISSING
        """Range for the quadruped robot's base pitch (in rad)."""

        roll: tuple[float, float] = MISSING
        """Range for the quadruped robot's base roll (in rad)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    """The configuration for the goal pose visualization marker."""
    height_orientation_visualizer: VisualizationMarkersCfg = CUBOID_MARKER_CFG.replace(prim_path="/Visuals/Command/height_pose")
    # The size is a thin reactangle in the robot's base height plane.
    height_orientation_visualizer.markers["cuboid"].size = (0.5, 0.5, 0.01)