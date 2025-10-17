# hand_to_hand_env_cfg.py
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

# 로봇 설정 (픽앤플레이스와 동일한 GR1T2 고정관절/PD 설정을 재사용)
from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.075, 0.075, 0.075)


# -----------------------------------------------------------------------------
# Scene definition
# -----------------------------------------------------------------------------
@configclass
class HandToHandSceneCfg(InteractiveSceneCfg):
    """두 손 핸드오버를 위한 씬 구성"""

    # 테이블(선택)
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # 핸드오버 대상 물체
    handover_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HandoverObject",
        # 초기 위치: 왼손 쪽(집기 쉬운 위치) — 필요시 조정
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.30, 0.45, 1.05], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/ToyTruck/toy_truck.usd",
            scale=(1.5, 1.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    # 휴머노이드 로봇 (좌/우 손 사용)
    robot: ArticulationCfg = GR1T2_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.93),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # 오른팔 기본 구부림
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.5708,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # 왼팔 기본 구부림
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": -1.5708,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # 기타
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

    # 바닥
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # 라이트
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # (디버깅) 좌/우 EE 프레임 마커
    left_EE_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/left_hand_pitch_link",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/left_hand_ee_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_hand_pitch_link",
                name="left_hand_pitch_link",
                offset=OffsetCfg(pos=(0.0, 0.0, -0.085), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
        ],
    )
    right_EE_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/right_hand_pitch_link",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/right_hand_ee_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/right_hand_pitch_link",
                name="right_hand_pitch_link",
                offset=OffsetCfg(pos=(0.0, 0.0, -0.085), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
        ],
    )


# -----------------------------------------------------------------------------
# MDP settings
# -----------------------------------------------------------------------------
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 좌손: DIK (엔드이펙터 포즈 제어)
    left_dik = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
        ],
        body_name="left_hand_pitch_link",
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.085), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        scale=0.25,
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    )

    # 우손: DIK (엔드이펙터 포즈 제어) — 핸드오버 정렬/수취용
    right_dik = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
        ],
        body_name="right_hand_pitch_link",
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.085), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        scale=0.25,
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        # 최근 액션
        actions = ObsTerm(func=mdp.last_action)

        # 로봇 루트/관절 상태
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})

        # 물체 상태
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("handover_object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("handover_object")})

        # 좌/우 EE 절대 포즈
        left_eef_pos  = ObsTerm(func=mdp.get_left_eef_pos)
        left_eef_quat = ObsTerm(func=mdp.get_left_eef_quat)
        right_eef_pos  = ObsTerm(func=mdp.get_right_eef_pos)
        right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat)

        # 상대량(핵심): 좌손-물체, 우손-물체, 좌손-우손
        lh_obj_rel = ObsTerm(func=mdp.rel_left_to_object)     # [dx,dy,dz]
        rh_obj_rel = ObsTerm(func=mdp.rel_right_to_object)    # [dx,dy,dz]
        hands_rel  = ObsTerm(func=mdp.rel_hands)              # [dx,dy,dz]

        # (선택) 그립 상태 플래그/거리 기반 의사-그립
        grasp_flags = ObsTerm(func=mdp.get_grasp_flags)       # [lh_flag, rh_flag]

        # (선택) 추가 MDP 상태 묶음
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

    # (핵심 보상들은 mdp.rewards.*에 구현해 연결하는 걸 권장)
    # 여기서는 공통 패널티만 먼저 연결 — 핸드오버 전용 보상은 mdp.rewards에서 추가해 스케일만 여기서 줄 것.
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2     = RewTerm(func=mdp.joint_acc_l2,     weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2,   weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)  # 옵션

    # 예) 핸드오버 전용 보상을 연결하고 싶다면:
    # left_approach   = RewTerm(func=mdp.rew_left_approach,   weight=1.0)
    # left_grasp      = RewTerm(func=mdp.rew_left_grasp,      weight=1.0)
    # handover_align  = RewTerm(func=mdp.rew_handover_align,  weight=1.0)
    # transfer        = RewTerm(func=mdp.rew_transfer,        weight=4.0)
    # stability       = RewTerm(func=mdp.rew_stability,       weight=0.2)
    # energy_penalty  = RewTerm(func=mdp.rew_energy_penalty,  weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("handover_object")},
    )

    # 성공 조건: 오른손만 그립 on & 왼손 off (mdp.task_done_hand_to_hand 내부에서 검증)
    success = DoneTerm(func=mdp.task_done_hand_to_hand)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # 시작 시 왼손 근처에 개체가 오도록 x/y 범위를 좁게 — 필요 시 조정
            "pose_range": {"x": [-0.04, 0.04], "y": [-0.04, 0.04]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("handover_object"),
        },
    )


@configclass
class HandToHandEnvCfg(ManagerBasedRLEnvCfg):
    """두 손 핸드오버(Hand-to-Hand Transfer) 태스크 환경 설정"""

    # Scene
    scene: HandToHandSceneCfg = HandToHandSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # Managers
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # 일반 설정
        self.decimation = 6
        self.episode_length_s = 20.0

        # 시뮬레이션 설정
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = 2
