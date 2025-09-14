from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from quadrupeds_mdp.curriculum.inspection import inspection_levels

@configclass
class CurriculumCfg:
    inspection_levels = CurrTerm(func=inspection_levels)