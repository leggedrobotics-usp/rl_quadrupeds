from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .mdp.scene_cfg import Go1InspectionPlannerSceneCfg
from .mdp.actions_cfg import ActionsCfg
from .mdp.commands_cfg import CommandsCfg
from .mdp.events_cfg import EventCfg
from .mdp.observations_cfg import NavObservationsCfg
from .mdp.rewards_cfg import RewardsCfg
from .mdp.terminations_cfg import TerminationsCfg
from .mdp.curriculum_cfg import CurriculumCfg

@configclass
class Go1InspectionPlannerEnvCfg(ManagerBasedRLEnvCfg):
    scene: Go1InspectionPlannerSceneCfg = Go1InspectionPlannerSceneCfg(
        num_envs=512, 
        env_spacing=10.
    )
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    observations: NavObservationsCfg = NavObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation