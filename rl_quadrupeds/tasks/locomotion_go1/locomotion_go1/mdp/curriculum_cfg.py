from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from quadrupeds_mdp.curriculum.locomotion import Go1LocomotionCurriculumCfg

@configclass
class CurriculumCfg:
    locomotion_levels = CurrTerm(func=Go1LocomotionCurriculumCfg)