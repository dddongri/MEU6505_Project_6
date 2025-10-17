from __future__ import annotations

import torch
import isaaclab.envs.mdp as base_mdp
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
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from . import mdp

from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.075, 0.075, 0.075)


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.30, 0.45, 1.05], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/ToyTruck/toy_truck.usd",
            scale=(1.5, 1.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

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
                "R_.*": 0.0,
                "L_.*": 0.0,
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
    EE_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/left_hand_pitch_link",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/left_hand_ee_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_hand_pitch_link",
                name="left_hand_pitch_link",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, -0.085),    # offset to the center of the gripper
                    rot=(1.0, 0.0, 0.0, 0.0),  # align with end-effector frame
                ),
            ),
        ],
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    # Differential IK Action
    gr1_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", 
            "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
        ],
        body_name="left_hand_pitch_link",
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.085),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
        scale=0.25,
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    )

    # TODO: hand-to-hand에서는 오른손 제어도 필요하면 아래를 활성화
    # gr1_right_action = mdp.DifferentialInverseKinematicsActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    #         "right_elbow_pitch_joint","right_wrist_yaw_joint","right_wrist_roll_joint","right_wrist_pitch_joint",
    #     ],
    #     body_name="right_hand_pitch_link",
    #     body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
    #         pos=(0.0, 0.0, -0.085), rot=(1.0, 0.0, 0.0, 0.0)
    #     ),
    #     scale=0.25,
    #     controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    # )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})

        left_eef_pos = ObsTerm(func=mdp.get_left_eef_pos)
        left_eef_quat = ObsTerm(func=mdp.get_left_eef_quat)
        right_eef_pos = ObsTerm(func=mdp.get_right_eef_pos)    # TODO: hand-to-hand용 관측 사용 시 mdp에 구현 필요
        right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat)  # TODO: hand-to-hand용 관측 사용 시 mdp에 구현 필요

        # TODO: hand-to-hand 상대 특성 사용 시 활성화
        # lh_obj_rel = ObsTerm(func=mdp.rel_left_to_object)
        # rh_obj_rel = ObsTerm(func=mdp.rel_right_to_object)
        # hands_rel  = ObsTerm(func=mdp.rel_hands)

        object = ObsTerm(func=mdp.object_obs)

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
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- penalties
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # -- optional penalties
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

    # TODO: hand-to-hand 성공 보상/중간 보상 연결 시 아래와 같이 추가
    # success_transfer = RewTerm(func=mdp.rew_hand_to_hand_success, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=mdp.task_done_hand_to_hand)  # TODO: mdp에 hand-to-hand 성공 조건 구현 필요


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.04, 0.04],
                "y": [-0.04, 0.04],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class GR1T2HandToHandEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 6
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = 2