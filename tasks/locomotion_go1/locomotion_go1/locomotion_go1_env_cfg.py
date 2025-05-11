import math

# ==== Scene construction ====
import isaaclab.sim as sim_utils                                    # IsaacSim utils
from isaaclab.assets import AssetBaseCfg                            # Import an asset to the simulation
# from isaaclab.assets import ArticulationCfg

# ==== MDP Definition ====
from isaaclab.managers import EventTermCfg as EventTerm             # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup       # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.ObservationGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm         # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.ObservationTermCfg
from isaaclab.managers import RewardTermCfg as RewTerm              # Reward
from isaaclab.managers import TerminationTermCfg as DoneTerm        # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.TerminationTermCfg

# ==== IsaacLab Utils ====
from isaaclab.managers import SceneEntityCfg                        # Recovers a entity from the scene
from isaaclab.scene import InteractiveSceneCfg                      # Configures the scene
from isaaclab.envs import ManagerBasedRLEnvCfg                      # Configures the RL environment (wrapper around everything)
from isaaclab.envs.mdp import (                                     # MDP utils
    base_lin_vel,                                                   # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.observations.base_lin_vel
    base_ang_vel,                                                   # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.observations.base_ang_vel
    generated_commands,                                             # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.observations.generated_commands
    JointPositionActionCfg,                                         # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.actions.actions_cfg.JointPositionActionCfg
)
from isaaclab.envs.mdp.commands.commands_cfg import (               # MDP commands
    UniformVelocityCommandCfg                                       # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.commands.commands_cfg.UniformVelocityCommandCfg
)
from isaaclab.envs.mdp.events import (                              # MDP events
    apply_external_force_torque,                                    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.events.apply_external_force_torque
    push_by_setting_velocity,                                       # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.events.push_by_setting_velocity
    randomize_rigid_body_mass,                                      # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.events.randomize_rigid_body_mass
    randomize_rigid_body_material,                                  # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.events.randomize_rigid_body_material
    reset_root_state_uniform,                                       # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.events.reset_root_state_uniform
    reset_joints_by_scale                                           # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.events.reset_joints_by_scale
)
from isaaclab.envs.mdp.observations import (                        # MDP observations
    height_scan,                                                    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.observations.height_scan
    joint_pos_rel,                                                  # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.observations.joint_pos_rel
    joint_vel_rel,                                                  # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.observations.joint_vel_rel
    last_action,                                                    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.observations.last_action
    projected_gravity                                               # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.observations.projected_gravity
)
from isaaclab.envs.mdp.terminations import (
    illegal_contact,                                                # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.terminations.illegal_contact
    time_out,                                                       # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.terminations.time_out
)
from isaaclab.envs.mdp.rewards import (                             # MDP rewards
    track_lin_vel_xy_exp,                                           # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.track_lin_vel_xy_exp
    track_ang_vel_z_exp,                                            # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.track_ang_vel_z_exp
    lin_vel_z_l2,                                                   # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.lin_vel_z_l2 
    ang_vel_xy_l2,                                                  # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.ang_vel_xy_l2
    joint_torques_l2,                                               # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.joint_torques_l2
    joint_acc_l2,                                                   # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.joint_acc_l2
    # feet_air_time,
    action_rate_l2,                                                 # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.action_rate_l2
    undesired_contacts,                                             # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.undesired_contacts
    flat_orientation_l2,                                            # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.flat_orientation_l2
    joint_pos_limits                                                # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.rewards.joint_pos_limits
)

from isaaclab.sensors import (                                      # Sensor utils
    ContactSensorCfg,                                               # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.ContactSensorCfg
    RayCasterCfg,                                                   # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.RayCasterCfg
    patterns                                                        # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.patterns.html
)

from isaaclab.utils import configclass                              # Decorator to configure the managers
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html#isaaclab.utils.noise.AdditiveUniformNoiseCfg

from isaaclab_tasks.manager_based.navigation.mdp.pre_trained_policy_action import (
    PreTrainedPolicyAction,
    PreTrainedPolicyActionCfg
)

# ==== External assets ====
from quadrupeds_assets.go1 import UNITREE_GO1_CFG

# ==== Paths ====
# from isaaclab.utils.assets import (                                 # Paths to the IssacSim and IsaacLab assets
#     ISAAC_NUCLEUS_DIR, 
#     ISAACLAB_NUCLEUS_DIR
# )

@configclass
class Go1LocomotionSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(1000, 1000))
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        )
    )
    
    robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    

@configclass
class CommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        _base_lin_vel = ObsTerm(func=base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        _base_ang_vel = ObsTerm(func=base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        _projected_gravity = ObsTerm(func=projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=last_action)
        height_scan = ObsTerm(
            func=height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True       # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.ObservationGroupCfg.enable_corruption
            self.concatenate_terms = True       # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.ObservationGroupCfg.concatenate_terms

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:
    joint_pos = JointPositionActionCfg( 
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True
    )

@configclass
class EventCfg:
    physics_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5), 
                "y": (-0.5, 0.5),
                "z": (0.3, 0.3),
                "yaw": (-3.14, 3.14)
            },
            "velocity_range": {
                # "x": (-0.5, 0.5),
                # "y": (-0.5, 0.5),
                # "z": (-0.5, 0.5),
                # "roll": (-0.5, 0.5),
                # "pitch": (-0.5, 0.5),
                # "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    push_robot = EventTerm(
        func=push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class RewardsCfg:
    # -- task
    _track_lin_vel_xy_exp = RewTerm(
        func=track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    _track_ang_vel_z_exp = RewTerm(
        func=track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    _lin_vel_z_l2 = RewTerm(func=lin_vel_z_l2, weight=-2.0)
    _ang_vel_xy_l2 = RewTerm(func=ang_vel_xy_l2, weight=-0.05)
    _dof_torques_l2 = RewTerm(func=joint_torques_l2, weight=-1.0e-5)
    _dof_acc_l2 = RewTerm(func=joint_acc_l2, weight=-2.5e-7)
    _action_rate_l2 = RewTerm(func=action_rate_l2, weight=-0.01)
    # _feet_air_time = RewTerm(
    #     func=feet_air_time,
    #     weight=0.125,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.5,
    #     },
    # )
    _undesired_contacts = RewTerm(
        func=undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )
    # -- optional penalties
    _flat_orientation_l2 = RewTerm(func=flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=joint_pos_limits, weight=0.0)

@configclass
class TerminationsCfg:
    _time_out = DoneTerm(func=time_out, time_out=True)
    base_contact = DoneTerm(
        func=illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
    )

@configclass
class Go1LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    scene: Go1LocomotionSceneCfg = Go1LocomotionSceneCfg(
        num_envs=16, 
        env_spacing=1.0
    )
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation