import math
from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

from quadrupeds_assets.gaits import GAIT_TIMING_OFFSETS
from quadrupeds_commands.commands_cfg import (
    QuadrupedBaseHeightOrientationCommandCfg,
    QuadrupedGaitDutyCycleCommandCfg,
    QuadrupedGaitFootswingHeightCommandCfg,
    QuadrupedGaitFrequencyCommandCfg,
    QuadrupedGaitStanceDistancesCommandCfg,
    QuadrupedGaitTypeCommandCfg,
)

# WTW Commands:
# [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
#     self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
#     self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
#     self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
#     self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
#     self.obs_scales.body_roll_cmd, self.obs_scales.stance_width_cmd,
#     self.obs_scales.stance_length_cmd, self.obs_scales.aux_reward_cmd
# ]

# Big time range sampling to avoid resampling during the episode.
# When the episode restarts, the commands are resampled.
ONCE_PER_EPISODE = (1000, 1000)

@configclass
class CommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(20., 20.),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 0.6), 
            lin_vel_y=(-0.6, 0.6), 
            ang_vel_z=(-0.5, 0.5), 
            heading=(-2*math.pi, 2*math.pi)
        ),
    )
    body_height_orientation_cmd = QuadrupedBaseHeightOrientationCommandCfg(
        asset_name="robot",
        ranges=QuadrupedBaseHeightOrientationCommandCfg.Ranges(
            height=(0.35, 0.35),
            pitch=(0., 0.),
            roll=(0., 0.),
        ),
        resampling_time_range=ONCE_PER_EPISODE,
        debug_vis=False
    )
    gait_freq_cmd = QuadrupedGaitFrequencyCommandCfg(
        asset_name="robot",
        ranges=QuadrupedGaitFrequencyCommandCfg.Ranges(frequency=(3.0, 3.0)),
        resampling_time_range=ONCE_PER_EPISODE,
        debug_vis=False
    )
    gait_type_cmd = QuadrupedGaitTypeCommandCfg(
        asset_name="robot",
        desired_gait_types=[
            GAIT_TIMING_OFFSETS["trotting"],
            # GAIT_TIMING_OFFSETS["bounding"],
            # GAIT_TIMING_OFFSETS["pacing"],
            # GAIT_TIMING_OFFSETS["pronking"]
        ],
        resampling_time_range=ONCE_PER_EPISODE,
        debug_vis=False
    )
    duty_cycle_cmd = QuadrupedGaitDutyCycleCommandCfg(
        asset_name="robot",
        ranges=QuadrupedGaitDutyCycleCommandCfg.Ranges(duty_cycle=(0.5, 0.5)),
        resampling_time_range=ONCE_PER_EPISODE,
        debug_vis=False
    )
    footswing_height_cmd = QuadrupedGaitFootswingHeightCommandCfg(
        asset_name="robot",
        ranges=QuadrupedGaitFootswingHeightCommandCfg.Ranges(footswing_height=(0.1, 0.1)),
        resampling_time_range=ONCE_PER_EPISODE,
        debug_vis=False
    )
    gait_stance_distances_cmd = QuadrupedGaitStanceDistancesCommandCfg(
        asset_name="robot",
        ranges=QuadrupedGaitStanceDistancesCommandCfg.Ranges(
            width=(0.28, 0.28),
            length=(0.64, 0.64)
        ),
        resampling_time_range=ONCE_PER_EPISODE,
        debug_vis=False
    )