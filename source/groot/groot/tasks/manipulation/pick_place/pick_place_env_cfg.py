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
from isaaclab.assets import Articulation, DeformableObject, RigidObject



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
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.45, 0.45, 1.08], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Beaker/beaker_500ml.usd",
            scale=(0.3, 0.3, 0.4),
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
    
    # Joint Position Action
    # gr1_action = mdp.JointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         "left_shoulder_pitch_joint",
    #         "left_shoulder_roll_joint",
    #         "left_shoulder_yaw_joint",
    #         "left_elbow_pitch_joint",
    #         "left_wrist_yaw_joint",
    #         "left_wrist_roll_joint",
    #         "left_wrist_pitch_joint",
    #         "right_shoulder_pitch_joint",
    #         "right_shoulder_roll_joint",
    #         "right_shoulder_yaw_joint",
    #         "right_elbow_pitch_joint",
    #         "right_wrist_yaw_joint",
    #         "right_wrist_roll_joint",
    #         "right_wrist_pitch_joint",
    #     ],
    #     scale=0.5,
    #     use_default_offset=True
    # )

    # Differential IK Action
    gr1_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint", 
            "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint"
        ],
        body_name="right_hand_pitch_link",
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.085),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
        scale=0.25,
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    )

    gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            # 손가락 관절만 포함
            "R_index_proximal_joint",
            "R_index_intermediate_joint",
            "R_middle_proximal_joint",
            "R_middle_intermediate_joint",
            "R_pinky_proximal_joint",
            "R_pinky_intermediate_joint",
            "R_ring_proximal_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_distal_joint"
        ],
        scale=0.25,  # 스케일은 적절히 조정
        use_default_offset=True
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
        # right_eef_pos = ObsTerm(func=mdp.get_right_eef_pos)
        # right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat)

        # hand_joint_state = ObsTerm(func=mdp.get_hand_state)
        # head_joint_state = ObsTerm(func=mdp.get_head_state)

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






def custom_object_target_distance_bonus(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_pos: tuple[float, float, float],
    threshold: float,
) -> torch.Tensor:
    """
    물체가 목표 위치(target_pos)로부터 threshold 거리 이내에 있으면
    L2 거리 기준으로 quadratic 리워드
    """
    object_pos = env.scene[object_cfg.name].data.root_pos_w

    target_pos_tensor = torch.tensor(target_pos, device=env.device, dtype=torch.float32).unsqueeze(0)

    distance = torch.norm(object_pos - target_pos_tensor, p=2, dim=1)

    bonus = (distance < threshold).float() * (threshold - distance)**2

    return bonus

def custom_robot_link_to_object_distance_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    link_name: str,
    threshold: float,
) -> torch.Tensor:
    """
    로봇(robot_cfg)의 특정 링크(link_name)가 물체(object_cfg)로부터
    threshold 거리 이내에 있으면 1.0의 보너스를 반환합니다.
    """

    robot = env.scene[robot_cfg.name]
    object_pos = env.scene[object_cfg.name].data.root_pos_w

    try:
        link_index = robot.body_names.index(link_name)
    except ValueError:

        raise ValueError(f"Link '{link_name}' not found in robot '{robot_cfg.name}'. "
                         f"Available links: {robot.body_names}")


    all_body_pos_w = robot.data.body_pos_w
    eef_pos_w = all_body_pos_w[:, link_index, :]
    distance = torch.norm(eef_pos_w - object_pos, p=2, dim=1)

    bonus = (distance < threshold).float() * (threshold - distance)**2
    return bonus



BOX_POS = (0.6, 0.5, 1.0)
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    #물체 위치를 박스쪽으로
    success_bonus = RewTerm(
        func=custom_object_target_distance_bonus,
        weight=100.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "target_pos": BOX_POS,
            "threshold": 0.5,
        },
    )

    #로봇 손이 object에 붙어있게
    right_hand_link_to_object_bonus = RewTerm(
        func=custom_robot_link_to_object_distance_bonus, 
        weight=100.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),  
            "object_cfg": SceneEntityCfg("object"),
            "link_name": "right_hand_pitch_link",
            "threshold": 0.5,
        },
    )
    # -- penalties

    """
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # -- optional penalties
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    """

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=mdp.task_done_pick_place)




# 사용자가 제공한 로봇의 오른쪽 팔 관절 이름
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
]


def reset_robot_arm_randomly(env, env_ids: torch.Tensor):
    """
    로봇의 오른팔 관절 각도를 랜덤하게 설정합니다.
    (오브젝트 위치는 변경하지 않습니다.)
    
    (내부에 'Lazy Initialization' 로직이 포함되어 있습니다.)
    """
    
    # --- 1. Lazy Initialization (처음 한 번만 실행) ---
    if not hasattr(env, "_reset_arm_setup_done"):
        print("[Info] 'reset_robot_arm_randomly' 1회성 초기 설정 수행 중...")
        
        try:
            robot: Articulation = env.scene["robot"]
        except KeyError:
            print(f"오류: env.scene에 'robot' 키가 없습니다! 사용 가능한 키: {env.scene.keys()}")
            raise

        robot_joint_names = robot.data.joint_names
        try:
            indices = [robot_joint_names.index(name) for name in RIGHT_ARM_JOINT_NAMES]
        except ValueError as e:
            print(f"오류: {e}. 로봇에 해당 관절 이름이 없습니다.")
            print(f"로봇의 관절 이름: {robot_joint_names}")
            raise
            
        env.right_arm_joint_indices = torch.tensor(
            indices, device=env.device, dtype=torch.long
        )
        
        env._reset_arm_setup_done = True
    # --- 초기화 종료 ---

    # --- 2. 리셋 로직 실행 ---
    
    # 2.1. 씬 및 설정된 속성 가져오기
    robot: Articulation = env.scene["robot"]
    arm_indices: torch.Tensor = env.right_arm_joint_indices
    num_resets = len(env_ids)

    # 2.2. 로봇의 *현재* 관절 상태를 가져옵니다.
    # (속성 이름: joint_pos, joint_vel)
    dof_pos_current = robot.data.joint_pos
    dof_vel_current = robot.data.joint_vel

    # 2.3. 오른팔 관절의 랜덤 각도 생성 (관절 한계 내)
    arm_limits = robot.data.soft_joint_pos_limits[env_ids][:, arm_indices]
    lower_limits = arm_limits[..., 0]
    upper_limits = arm_limits[..., 1]

    random_angles = (
        torch.rand(num_resets, len(arm_indices), device=env.device) 
        * (upper_limits - lower_limits) + lower_limits
    )

    # 2.4. 리셋할 환경의 DOF 상태만 복제합니다.
    dof_pos_resets = dof_pos_current[env_ids].clone()
    dof_vel_resets = dof_vel_current[env_ids].clone() 

    # 2.5. 복제된 텐서에서 팔 각도 부분만 덮어씁니다.
    dof_pos_resets[:, arm_indices] = random_angles

    # 2.6. (중요) 리셋할 환경의 모든 관절 속도를 0으로 설정합니다.
    # (dof_vel_resets는 이미 env_ids 환경의 속도를 복제한 상태입니다)
    dof_vel_resets[:] = 0.0 
    # 또는: dof_vel_resets.zero_()

    # 2.7. (핵심) 수정된 로컬 텐서를 시뮬레이션의 *원본* 텐서 버퍼에 다시 덮어씁니다.
    # 'dof_pos_current'는 'robot.data.joint_pos'를 가리키고 있습니다.
    # 이 'Scatter' 작업이 시뮬레이션 상태를 실제로 변경합니다.
    dof_pos_current[env_ids] = dof_pos_resets
    dof_vel_current[env_ids] = dof_vel_resets





@configclass
class EventCfg:

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [0.4, 0.6],
                "y": [0.0, 0.2],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


    reset_arm = EventTerm(
        func=reset_robot_arm_randomly, 
        mode="reset"
    )
    



@configclass
class GR1T2PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
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
