from __future__ import annotations

import torch
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-5.45, 0.45, 1.08], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Beaker/beaker_500ml.usd",
            scale=(0.4, 0.4, 0.8),
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
                # right hand closed / left hand open by default
                # slightly relaxed to avoid interpenetration
                "R_index_.*": -0.74,
                "R_middle_.*": -0.74,
                "R_pinky_.*": -1.0,
                "R_ring_.*": -1.0,
                "R_thumb_proximal_yaw_joint": -1.73,
                "R_thumb_proximal_pitch_joint": 0.25,
                "R_thumb_distal_joint": 0.7,
                "L_index_.*": 0.0,
                "L_middle_.*": 0.0,
                "L_pinky_.*": 0.0,
                "L_ring_.*": 0.0,
                "L_thumb_.*": 0.0,
                # right-arm
                # lift hand slightly above table to avoid initial impact
                "right_shoulder_pitch_joint": -0.2,
                "right_shoulder_roll_joint": 0.15,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.2,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.25,
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
                    pos=(0.0, 0.0, -0.085),
                    rot=(1.0, 0.0, 0.0, 0.0),
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
    symmetric_hands = mdp.SymmetricDualIKActionCfg(
        asset_name="robot",
        left_joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
        ],
        right_joint_names=[
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
        ],
        left_body_name="left_hand_pitch_link",
        right_body_name="right_hand_pitch_link",
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.085),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        scale=0.25,
        include_grasp=True,
        center_z_floor=1.0,
        mirror_rotation=False,  # mirror only position (plane symmetry), keep rotations independent
    )


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
        right_eef_pos = ObsTerm(func=mdp.get_right_eef_pos)
        right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat)

        object = ObsTerm(func=mdp.object_obs)

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
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.rew_alive, weight=0.2)

    # Smoothness
    # Penalize fast action changes (func returns negative cost)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.001)

    # Handover shaping
    right_hold = RewTerm(func=mdp.rew_right_hold, weight=4.0, params={"dist_th": 0.07})
    # approach terms return negative distance; keep positive weights to encourage proximity
    left_approach = RewTerm(func=mdp.rew_left_approach_simple, weight=3.0, params={"min_height": 0.85})
    hands_proximity = RewTerm(func=mdp.rew_hands_proximity, weight=2.0, params={"min_height": 0.85})
    hand_front_penalty = RewTerm(func=mdp.rew_hand_front_penalty, weight=3.0, params={"min_x": 0.12, "gate_height": 0.86})
    palm_sagittal = RewTerm(
        func=mdp.rew_palm_sagittal,
        weight=0.5,
        params={"palm_offset": 0.06, "warmup_steps": 12, "min_height": 0.86, "hand_sep_max": 0.28},
    )
    palm_inward = RewTerm(
        func=mdp.rew_palm_inward,
        weight=1.5,
        params={"warmup_steps": 12, "min_height": 0.86, "thresh": 0.2, "hand_sep_max": 0.28},
    )
    palm_alignment = RewTerm(
        func=mdp.rew_palm_alignment,
        weight=6.0,
        params={"palm_offset": 0.06, "near_thresh": 0.18, "hand_sep_thresh": 0.25, "warmup_steps": 18},
    )
    left_grasp_ready = RewTerm(func=mdp.rew_left_grasp_ready, weight=4.0, params={"dist_th": 0.07})
    dual_hold = RewTerm(func=mdp.rew_dual_hold_bonus, weight=4.0, params={"dist_th": 0.07})
    guarded_transfer = RewTerm(func=mdp.rew_guarded_transfer, weight=2.0)
    # keep per-step success small; one-shot bonuses carry the big signal
    transfer_success = RewTerm(func=mdp.rew_transfer_success, weight=1.0, params={"vel_thresh": 0.25, "min_height": 0.7})
    left_hold_stable = RewTerm(func=mdp.rew_left_hold_stable, weight=3.0)
    close_bonus = RewTerm(
        func=mdp.rew_close_bonus,
        weight=60.0,
        params={"hand_obj_thresh": 0.11, "hand_sep_thresh": 0.21, "min_height": 0.86, "vel_thresh": 0.22},
    )
    handover_bonus = RewTerm(
        func=mdp.rew_handover_bonus,
        weight=90.0,
        params={"vel_thresh": 0.22, "min_height": 0.82, "palm_align_thresh": 0.10},
    )
    hand_height_band = RewTerm(
        func=mdp.rew_hand_height_band_penalty,
        weight=12.0,
        params={"min_height": 0.92, "max_height": 1.15, "weight_high": 1.0},
    )
    hands_low_guard = RewTerm(func=mdp.rew_hands_low_termination_guard, weight=2.0)
    height_shaping = RewTerm(
        func=mdp.rew_height_shaping_banded,
        weight=2.5,
        params={"min_height": 0.88, "target_height": 1.08, "max_height": 1.18},
    )
    object_height_penalty = RewTerm(func=mdp.rew_object_height_penalty, weight=18.0, params={"min_height": 0.83})
    object_low_guard = RewTerm(func=mdp.rew_object_low_termination_guard, weight=2.0)
    object_upright = RewTerm(func=mdp.rew_object_upright, weight=0.5)
    release_penalty = RewTerm(func=mdp.rew_release_penalty, weight=6.0, params={"dist_th": 0.06})
    exchange_zone = RewTerm(
        func=mdp.rew_exchange_zone,
        weight=20.0,
        params={"target_offset": (0.25, 0.0, 0.1), "sigma": 0.08, "obj_height_min": 0.85, "hand_obj_thresh": 0.12},
    )
    # post-handover cleanup
    post_handover_separation = RewTerm(func=mdp.rew_post_handover_separation, weight=1.5)
    post_handover_arm_home = RewTerm(func=mdp.rew_post_handover_arm_home, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=mdp.task_done_hand_to_hand)
    hands_too_low = DoneTerm(func=mdp.hands_below_min_height, params={"min_height": 0.85})
    object_stuck = DoneTerm(
        func=mdp.object_stuck,
        params={
            "table_height": 0.55,
            "table_margin": 0.05,
            "hand_dist": 0.20,
            "vel_thresh": 0.02,
            "spawn_band": 0.05,
            "settle_steps": 30,
        },
    )
    hands_clamped = DoneTerm(
        func=mdp.object_clamped_between_hands,
        params={
            "hand_dist": 0.12,
            "hand_sep": 0.18,
            "vel_thresh": 0.01,
            "z_progress_tol": 0.003,
            "sep_progress_tol": 0.002,
            "hand_vel_thresh": 0.05,
            "settle_steps": 60,
        },
    )
    object_at_rest = DoneTerm(
        func=mdp.object_at_rest_on_table,
        params={"vel_thresh": 0.05, "height_thresh": 0.9, "settle_steps": 10},
    )
    both_off = DoneTerm(func=mdp.both_hands_released)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_extras = EventTerm(func=mdp.reset_episode_extras, mode="reset")
    jitter_hands = EventTerm(func=mdp.randomize_hand_pose, mode="reset")
    place_object = EventTerm(func=mdp.place_object_to_right_hand, mode="reset")
    tick_counter = EventTerm(func=mdp.inc_step_counter, mode="step")


@configclass
class GR1T2HandToHandEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1000, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    commands = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 6
        self.episode_length_s = 20.0 / 9
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = 2

        # Lock down the waist/trunk to keep torso steady.
        if "trunk" in self.scene.robot.actuators:
            trunk = self.scene.robot.actuators["trunk"]
            trunk.stiffness = 1e9
            trunk.damping = 1e6
            trunk.friction = 10.0
            trunk.velocity_limit = 0.0

        # Strengthen finger actuators for better grip.
        self.scene.robot.actuators["hands"] = ImplicitActuatorCfg(
            joint_names_expr=[
                "L_index_.*", "L_middle_.*", "L_pinky_.*", "L_ring_.*", "L_thumb_.*",
                "R_index_.*", "R_middle_.*", "R_pinky_.*", "R_ring_.*", "R_thumb_.*",
            ],
            stiffness=1500.0,
            damping=50.0,
            effort_limit_sim=200.0,
            velocity_limit_sim=5.0,
        )
