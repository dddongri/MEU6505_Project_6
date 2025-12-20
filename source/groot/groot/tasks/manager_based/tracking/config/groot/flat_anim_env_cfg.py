from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from groot.tasks.manager_based.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class GR1T2EnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            "base_link",
            "left_upper_arm_roll_link",
            "left_lower_arm_pitch_link",
            "left_hand_pitch_link",
            "right_upper_arm_roll_link",
            "right_lower_arm_pitch_link",
            "right_hand_pitch_link",
        ]

        # viewer settings
        self.viewer.asset_name = "robot"
        self.viewer.origin_type = "asset_root"
        self.viewer.eye = (0.0, 2.5, 0.8)
        self.viewer.lookat = (0.0, -1.2, 0.0)


@configclass
class GR1T2EnvCfg_PLAY(GR1T2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 1

        self.events.add_joint_default_pos = None
        self.events.push_robot = None

        # viewer settings
        self.viewer.asset_name = "robot"
        self.viewer.origin_type = "asset_root"
        self.viewer.eye = (0.0, 2.5, 0.8)
        self.viewer.lookat = (0.0, -1.2, 0.0)
