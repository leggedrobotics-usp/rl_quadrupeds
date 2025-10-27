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
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    CUBOID_MARKER_CFG, 
    FRAME_MARKER_CFG, 
    GREEN_ARROW_X_MARKER_CFG
)
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
from .velocity import UniformVelocityCommand

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

@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.

    This parameter is only used if :attr:`heading_command` is True.
    """

    rel_noise_envs: float = 0.0
    """The sampled probability of environments where noise is added to the command. Defaults to 0.0."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    noise_ranges: Ranges = None
    """Distribution ranges for the noise added to the velocity commands. Defaults to None."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)