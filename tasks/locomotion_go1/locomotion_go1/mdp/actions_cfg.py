from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.utils import configclass

@configclass
class ActionsCfg:
    joint_pos = JointPositionActionCfg( 
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True
    )