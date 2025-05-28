from isaaclab.envs import (
    ManagerBasedEnv,
    ManagerBasedRLEnv
)

from isaaclab.managers import (
    CommandManager,
    CurriculumManager,
    RewardManager,
    TerminationManager
)

from isaaclab_extensions.managers.reward import (
    ExponentialRewardManager
)

class CustomizableManagerBasedRLEnv(ManagerBasedRLEnv):
    """
    Customizes the ManagerBasedRLEnvCfg changing the RewardManager (for now).
    """

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # call the parent class to load the managers for observations and actions.
        ManagerBasedEnv.load_managers(self)

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        # self.reward_manager = RewardManager(self.cfg.rewards, self)
        self.reward_manager = ExponentialRewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")