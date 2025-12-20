from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import groot.tasks.manager_based.tracking.mdp as mdp
from groot.robots import GR1T2_EXT_DIR

from groot.robots.fourier import GR1T2_HIGH_PD_CFG
# from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

##
# Scene definition
##


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.075, 0.075, 0.075)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            usd_path=f"{GR1T2_EXT_DIR}/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.1, 0.35, 1.08], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{GR1T2_EXT_DIR}/beaker.usd",
            scale=(0.4, 0.4, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    # object = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[-5.45, 0.45, 1.08], rot=[1.0, 0.0, 0.0, 0.0]),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Beaker/beaker_500ml.usd",
    #         scale=(0.4, 0.4, 0.8),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #     ),
    # )
    
    # Humanoid robot configured for pick-place manipulation tasks
    robot: ArticulationCfg = GR1T2_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.93),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.5708,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": -1.5708,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # --
                "head_.*": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                
                # "R_.*": 0.0,
                "L_.*": 0.0,
                
                "R_index_.*": -1.0, "R_middle_.*": -1.0, "R_pinky_.*": -1.0, "R_ring_.*": -1.0, "R_thumb_proximal_yaw_joint": -1.7,
                "R_thumb_proximal_pitch_joint": 0.35, "R_thumb_distal_joint": 1.0
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    
    # Frames
    left_ee_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/left_hand_pitch_link",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/left_hand_ee_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_hand_pitch_link",
                name="left_hand_pitch_link",
                offset=OffsetCfg(
                    pos=(0.0, -0.05, -0.085),    # offset to the center of the gripper
                    rot=(0.5, -0.5, 0.5, -0.5),  # align with end-effector frame
                ),
            ),
        ],
    )
    
    right_ee_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/right_hand_pitch_link",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/right_hand_ee_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/right_hand_pitch_link",
                name="right_hand_pitch_link",
                offset=OffsetCfg(
                    pos=(0.0, 0.05, -0.085),    # offset to the center of the gripper
                    rot=(0.5, 0.5, 0.5, 0.5),  # align with end-effector frame
                ),
            ),
        ],
    )

    # Contact sensors
    gripper_contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/right_hand_pitch_link",
        filter_prim_paths_expr="/World/envs/env_.*/Object",
        track_air_time=False,
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        },
        velocity_range={
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        },
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot",
                                           joint_names=["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", 
                                                        "left_elbow_pitch_joint", 
                                                        "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
                                                        "L_.*", "R_.*",
                                                        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", 
                                                        "right_elbow_pitch_joint", 
                                                        "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint"], use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.1, n_max=0.1),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", 
                                                        "left_elbow_pitch_joint", 
                                                        "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
                                                        "L_.*", "R_.*",
                                                        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", 
                                                        "right_elbow_pitch_joint", 
                                                        "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",])},
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", 
                                                        "left_elbow_pitch_joint", 
                                                        "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
                                                        "L_.*", "R_.*",
                                                        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", 
                                                        "right_elbow_pitch_joint", 
                                                        "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",])},
        )
        hand_state = ObsTerm(func=mdp.get_hand_state)
        object = ObsTerm(func=mdp.object_obs)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        pass

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.0, 0.0],
                "y": [-0.0, 0.0],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.2,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-8)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    success_grasp_task = RewTerm(func=mdp.success_reach_task_reward, weight=0.1)
    abc = RewTerm(func=mdp.push_object_negx_reward, weight=0.3)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.3},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos,
        params={
            "command_name": "motion",
            "threshold": 0.3,
            "body_names": [
                "left_upper_arm_roll_link",
                "left_lower_arm_pitch_link",
                "left_hand_pitch_link",
                "right_upper_arm_roll_link",
                "right_lower_arm_pitch_link",
                "right_hand_pitch_link",
            ],
        },
    )
    
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.95, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 1.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # viewer settings
        self.viewer.asset_name = "robot"
        self.viewer.origin_type = "asset_root"
        self.viewer.eye = (0.0, 2.5, 0.8)
        self.viewer.lookat = (0.0, -1.2, 0.0)
