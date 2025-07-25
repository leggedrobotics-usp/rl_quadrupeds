from isaaclab.envs.mdp.terminations import (
    illegal_contact,                                                # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.terminations.illegal_contact
    time_out,                                                       # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.terminations.time_out
)
from isaaclab.managers import (
    TerminationTermCfg as DoneTerm,                                 # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.TerminationTermCfg
    SceneEntityCfg
)        
from isaaclab.utils import configclass

from quadrupeds_mdp.terminations.reset import reset_on_start
@configclass
class TerminationsCfg:
    reset_on_start = DoneTerm(func=reset_on_start)

    _time_out = DoneTerm(func=time_out, time_out=True)

    base_contact = DoneTerm(
        func=illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
    )