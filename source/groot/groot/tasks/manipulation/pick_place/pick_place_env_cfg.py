from __future__ import annotations

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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from . import mdp


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.075, 0.075, 0.075)


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Scene for left-hand start grasp -> place on table."""

    # Table (XForm asset; has no .data, so do NOT use as RigidObject in obs)
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Cup object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.45, 0.45, 1.08], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Beaker/beaker_500ml.usd",
            scale=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    # Robot (right arm fixed by not being controlled; left arm used)
    robot: ArticulationCfg = GR1T2_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.93),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # ---------------------------
                # RIGHT ARM: keep comfortable, but NOT controlled
                # ---------------------------
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.2,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,

                # LEFT ARM: 책상 위로 더 올라가게(추천 튜닝)
                "left_shoulder_pitch_joint": 0.65,
                "left_shoulder_roll_joint": 0.22,
                "left_shoulder_yaw_joint": 0.05,
                "left_elbow_pitch_joint": -0.75,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.15,

                # LEFT HAND: 시작부터 CLOSE (쥔 상태)
                "L_index_.*": 0.0,
                "L_middle_.*": 0.0,
                "L_ring_.*": 0.0,
                "L_pinky_.*": 0.0,
                "L_thumb_proximal_yaw_joint": 0.0,   # ✅ 양수 넣으면 limit 터짐
                "L_thumb_proximal_pitch_joint": 0.0,
                "L_thumb_distal_joint": 0.0,

                # Others
                "head_.*": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                "R_.*": 0.0,

            },
            joint_vel={".*": 0.0},
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
    )

    # LEFT EE Frame (IMPORTANT: properly inside SceneCfg)
    ee_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/left_hand_pitch_link",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/left_hand_ee_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_hand_pitch_link",
                name="left_tcp",
                offset=OffsetCfg(
                    # TCP offset (손바닥 쪽으로 약간)
                    pos=(0.0, 0.0, 0.05),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
        ],
    )


@configclass
class ActionsCfg:
    """Left-arm IK (position-only). Hand is fixed open via events (no gripper action)."""

    # Left arm IK (7 joints)
    gr1_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
        ],
        body_name="left_hand_pitch_link",
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        # relative XYZ command scale per env-step (env-step = decimation * sim.dt)
        scale=0.04,
        controller=mdp.DifferentialIKControllerCfg(
            command_type="pose",      # ✅ position only (easier to learn than full pose)
            use_relative_mode=True,       # ✅ delta command -> arm can actually reach the table
            ik_method="dls",
        ),
    )
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)

        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_pitch_joint",
                        "left_wrist_yaw_joint",
                        "left_wrist_roll_joint",
                        "left_wrist_pitch_joint",
                    ],
                )
            },
        )
        robot_joint_vel = ObsTerm(
            func=base_mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_pitch_joint",
                        "left_wrist_yaw_joint",
                        "left_wrist_roll_joint",
                        "left_wrist_pitch_joint",
                    ],
                )
            },
        )

        hand_state = ObsTerm(func=mdp.get_hand_state)   # ee_frame 기반
        object = ObsTerm(func=mdp.object_obs)           # object pose/vel

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        pass

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    reach_xy = RewTerm(func=mdp.reward_reach_target_xy, weight=2.0, params={"target_pos_rel": (0.0, 0.55, 0.86)})
    upright = RewTerm(func=mdp.reward_upright, weight=0.5, params={"upright_cos": 0.92})
    success = RewTerm(func=mdp.reward_place_success_upright, weight=10.0, params={"target_pos_rel": (0.0, 0.55, 0.86)})
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-0.1)

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # object 떨어뜨리면 종료 (너무 낮으면 튕김으로 오탐)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.55, "asset_cfg": SceneEntityCfg("object")},
    )

    success = DoneTerm(func=mdp.task_done_place_upright, params={"target_pos_rel": (0.0, 0.55, 0.86)})

@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # (1) reset: 컵을 왼손 TCP 근처로 순간이동
    place_object = EventTerm(
        func=mdp.place_object_to_left_hand,
        mode="reset",
        params={
            "obj_offset_tcp": (0.0, 0.08, 0.0),
            "align_with_hand": False,
            "force_upright": True,
        },
    )

    # (2) reset: “붙어있음” 플래그 ON
    attach_object = EventTerm(
        func=mdp.set_object_attached,
        mode="reset",
        params={"attached": True},
    )

    # (3) interval: 손은 계속 편 상태로 고정(떨림 제거)
    keep_hand_open = EventTerm(
        func=mdp.keep_left_hand_open,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={"open_value": 0.0},
    )

    # (4) interval: attached=True일 때만 컵을 손에 “붙여서” 따라오게 함
    weld_object = EventTerm(
        func=mdp.keep_object_welded_to_left_hand,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "obj_offset_tcp": (0.0, 0.08, 0.0),
            "align_with_hand": False,
            "force_upright": True,
        },
    )

    # (5) interval: 목표 근처면 attached=False로 바꿔서 “용접 끊기” => 컵이 떨어짐
    auto_release = EventTerm(
        func=mdp.release_object_if_at_target,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "target_pos_rel": (0.0, 0.55, 0.86),
            "xy_thresh": 0.06,
            "z_min": 0.75,
            "upright_cos": 0.92,
            "speed_thresh": 0.35,
        },
    )

@configclass
class GR1T2PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=256, env_spacing=2.5, replicate_physics=True)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # unused
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 6              # 크게 할수록 제어가 부드러워짐
        self.episode_length_s = 8.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = 3
