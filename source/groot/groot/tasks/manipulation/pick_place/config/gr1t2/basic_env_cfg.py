from isaaclab.utils import configclass

from groot.tasks.manipulation.pick_place.pick_place_env_cfg import GR1T2PickPlaceEnvCfg

##
# Pre-defined configs
##


@configclass
class GR1T2BasicEnvCfg(GR1T2PickPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()


@configclass
class GR1T2BasicEnvCfg_PLAY(GR1T2BasicEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        
        self.viewer.asset_name = "robot"
        self.viewer.origin_type = "asset_root"
        self.viewer.eye = (0.0, 2.5, 0.8)
        self.viewer.lookat = (0.0, -1.2, 0.0)
