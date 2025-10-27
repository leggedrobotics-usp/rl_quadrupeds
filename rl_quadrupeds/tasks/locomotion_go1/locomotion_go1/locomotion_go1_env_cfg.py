from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .mdp.scene_cfg import Go1LocomotionSceneCfg
from .mdp.actions_cfg import ActionsCfg
from .mdp.commands_cfg import CommandsCfg
from .mdp.events_cfg import EventCfg
from .mdp.observations_cfg import ObservationsCfg
from .mdp.rewards_cfg import RewardsCfg
from .mdp.terminations_cfg import TerminationsCfg
from .mdp.curriculum_cfg import CurriculumCfg
from .mdp.recorders_cfg import RecordersCfg

@configclass
class Go1LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    scene: Go1LocomotionSceneCfg = Go1LocomotionSceneCfg(
        num_envs=4096, 
        env_spacing=1.0
    )
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()
    # recorders: RecordersCfg = RecordersCfg()

    def __post_init__(self) -> None:
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

@configclass
class Go1LocomotionEnvCfg_PLAY(Go1LocomotionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # disable randomization for play
        self.observations.policy.enable_corruption = False