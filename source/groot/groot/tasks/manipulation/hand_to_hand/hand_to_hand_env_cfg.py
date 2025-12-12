from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
from .hand_to_hand_env import GR1T2HandToHandEnv

from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG  # isort: skip


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
        disable_arms=True,   # freeze arms so only references are generated
        disable_grasp=True,  # keep fingers fixed (no random grasp)
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
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_extras = EventTerm(func=mdp.reset_episode_extras, mode="reset")
    jitter_hands = EventTerm(func=mdp.randomize_hand_pose, mode="reset")
    place_object = EventTerm(func=mdp.place_object_to_right_hand, mode="reset")
    build_right_arm_reference = EventTerm(func=mdp.build_right_arm_reference_trajectory, mode="reset")
    log_traj_csv = EventTerm(func=mdp.log_traj_csv_step, mode="reset")
    tick_counter = EventTerm(func=mdp.inc_step_counter, mode="step")


@configclass
class GR1T2HandToHandEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1000, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    debug_ik_follow: bool = True
    # trajectory CSV logging (debug)
    log_traj_csv: bool = True
    log_traj_env_id: int = 0
    log_traj_dir: str = "logs/hand2hand_traj"
    log_traj_write_every: int = 1
    rewards = None  # reward structure removed; reference trajectories only
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

        # debug flag: enable IK follow of reference (disables arm freeze)
        if self.debug_ik_follow:
            self.actions.symmetric_hands.disable_arms = False
        self.class_type = GR1T2HandToHandEnv

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
