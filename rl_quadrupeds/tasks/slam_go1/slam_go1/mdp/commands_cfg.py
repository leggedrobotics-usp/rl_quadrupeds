from isaaclab.utils import configclass
from isaaclab_extensions.ros.subscribers.vel_cmd.vel_cmd_cfg import ROSVelCmdSubscriberCfg

from locomotion_go1.mdp.commands_cfg import CommandsCfg as LocomotionCommandsCfg
LocomotionCommandsCfg = LocomotionCommandsCfg()

# Big time range sampling to avoid resampling during the episode.
# When the episode restarts, the commands are resampled.
ONCE_PER_EPISODE = (1000, 1000)

@configclass
class CommandsCfg():
    """
    Inherited from LocomotionCommandsCfg:
        - body_height_orientation_cmd
        - gait_freq_cmd
        - gait_type_cmd
        - duty_cycle_cmd
        - footswing_height_cmd
        - gait_stance_distances_cmd
    """
    # Removes it from the Locomotion commands because the
    # velocity is already handled by the action cfg.
    base_velocity = None

    body_height_orientation_cmd = LocomotionCommandsCfg.body_height_orientation_cmd
    gait_freq_cmd = LocomotionCommandsCfg.gait_freq_cmd
    gait_type_cmd = LocomotionCommandsCfg.gait_type_cmd
    duty_cycle_cmd = LocomotionCommandsCfg.duty_cycle_cmd
    footswing_height_cmd = LocomotionCommandsCfg.footswing_height_cmd
    gait_stance_distances_cmd = LocomotionCommandsCfg.gait_stance_distances_cmd

    # When trying to control the robot manually through ROS,
    # this should be enabled.
    # base_velocity = ROSVelCmdSubscriberCfg(
    #     asset_name="robot",
    #     topic_name="/cmd_vel",
    #     node_name="isaaclab_go1_vel_cmd_subscriber",
    #     resampling_time_range=(0.001, 0.001),
    # )