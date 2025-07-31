from isaaclab.envs.mdp.terminations import (
    time_out
)
from isaaclab.managers import (
    TerminationTermCfg as DoneTerm,                                 # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.TerminationTermCfg
    SceneEntityCfg
)        
from isaaclab.utils import configclass

from quadrupeds_mdp.terminations.contact import illegal_contact

@configclass
class TerminationsCfg:    
    _time_out = DoneTerm(func=time_out, time_out=True)

    base_contact = DoneTerm(
        func=illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["trunk", ".*_hip", ".*_thigh", ".*_calf"]), "threshold": 1.0},
    )