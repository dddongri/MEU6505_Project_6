from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from groot.tasks.manager_based.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def object_obs(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Object observations (in world frame):
        object pos,
        object quat,
        left_eef to object,
        right_eef_to object,
    """

    # body_pos_w = env.scene["robot"].data.body_pos_w
    # left_eef_idx = env.scene["robot"].data.body_names.index("left_hand_pitch_link")
    # right_eef_idx = env.scene["robot"].data.body_names.index("right_hand_pitch_link")
    
    # left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene["robot"].data.root_pos_w
    hand_pos_states = env.scene["left_ee_frame"].data.target_pos_w[:, 0, :]
    hand_ori_states = env.scene["left_ee_frame"].data.target_quat_w[:, 0, :]
    # left_eef_ori = env.scene["robot"].data.body_quat_w[:, left_eef_idx]
    # right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene["robot"].data.root_pos_w

    object_pos = env.scene["object"].data.root_pos_w
    object_quat = env.scene["object"].data.root_quat_w

    left_eef_to_object = object_pos - hand_pos_states
    # right_eef_to_object = object_pos - right_eef_pos
    
    # left_eef_rot_error = env.scene.quat_mul(
    #     quat_conjugate(left_eef_ori),
    #     object_quat,
    # )

    hand_pos_states = env.scene["left_ee_frame"].data.target_pos_w[:, 0, :]
    dist = torch.norm(left_eef_to_object, dim=-1)        # [num_envs]

    close_env_ids = torch.nonzero(dist < 0.05, as_tuple=False).squeeze(-1)  # [K]

    if close_env_ids.numel() > 0:
        hand_close(env, close_env_ids)
        
    # print("distance: ", dist[0])    

    return torch.cat(
        (
            object_pos,
            object_quat,
            left_eef_to_object,
        ),
        dim=1,
    )

def get_left_eef_pos(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_hand_pitch_link")
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins

    return left_eef_pos


def get_left_eef_quat(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_hand_pitch_link")
    left_eef_quat = body_quat_w[:, left_eef_idx]

    return left_eef_quat


def get_right_eef_pos(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = env.scene["robot"].data.body_names.index("right_hand_pitch_link")
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    return right_eef_pos


def get_right_eef_quat(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    right_eef_idx = env.scene["robot"].data.body_names.index("right_hand_pitch_link")
    right_eef_quat = body_quat_w[:, right_eef_idx]

    return right_eef_quat


def get_hand_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    hand_pos_states = env.scene["left_ee_frame"].data.target_pos_w[:, 0, :] - env.scene["robot"].data.root_pos_w
    hand_ori_states = env.scene["left_ee_frame"].data.target_quat_w[:, 0, :]
    
    right_hand_pos_states = env.scene["right_ee_frame"].data.target_pos_w[:, 0, :] - env.scene["robot"].data.root_pos_w
    right_hand_ori_states = env.scene["right_ee_frame"].data.target_quat_w[:, 0, :]
    
    # hand_joint_states = env.scene["robot"].data.joint_pos[:, -22:]  # Hand joints are last 22 entries of joint state

    # print("target_pos_w: ", env.scene["left_ee_frame"].data.target_pos_w[0, 0, :])  # [-0.2238,  0.3406,  1.0976]
    # print("env.scene.env_origins: ", env.scene["robot"].data.root_pos_w[0])
    # print("hand_pos_states: ", hand_pos_states[0])
    # print("hand_ori_states: ", hand_ori_states[0])
    
    return torch.cat((hand_pos_states, hand_ori_states, right_hand_pos_states, right_hand_ori_states), dim=1)


def get_head_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    robot_joint_names = env.scene["robot"].data.joint_names
    head_joint_names = ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]
    indexes = torch.tensor([robot_joint_names.index(name) for name in head_joint_names], dtype=torch.long)
    head_joint_states = env.scene["robot"].data.joint_pos[:, indexes]

    return head_joint_states


def get_all_robot_link_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_link_state_w[:, :, :]
    all_robot_link_pos = body_pos_w

    return all_robot_link_pos


def hand_close(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    N = env.num_envs
    robot = env.scene["robot"]

    hand_joint1_ids = robot.find_joints([
        "L_index_.*",
        "L_middle_.*",
        "L_pinky_.*",
        "L_ring_.*",
    ])[0]

    hand_joint2_ids = robot.find_joints([
        "L_thumb_proximal_yaw_joint",
    ])[0]
    
    hand_joint3_ids = robot.find_joints([
        "L_thumb_proximal_pitch_joint",
    ])[0]
    
    hand_joint4_ids = robot.find_joints([
        "L_thumb_distal_joint",
    ])[0]

    close_hand_joint1 = robot.data.joint_pos_target[:, hand_joint1_ids].clone()
    close_hand_joint2 = robot.data.joint_pos_target[:, hand_joint2_ids].clone()
    close_hand_joint3 = robot.data.joint_pos_target[:, hand_joint3_ids].clone()
    close_hand_joint4 = robot.data.joint_pos_target[:, hand_joint4_ids].clone()

    if env_ids is None:
        close_hand_joint1[:] = 0.0
        close_hand_joint2[:] = 0.0
        close_hand_joint3[:] = 0.0
        close_hand_joint4[:] = 0.0
    else:
        # close_hand_joint1[env_ids] = -1.0
        # close_hand_joint2[env_ids] = -1.7
        # close_hand_joint3[env_ids] = 0.35
        # close_hand_joint4[env_ids] = 1.0
        
        close_hand_joint1[env_ids] = -0.5
        close_hand_joint2[env_ids] = -0.85
        close_hand_joint3[env_ids] = 0.175
        close_hand_joint4[env_ids] = 0.5

    robot.set_joint_position_target(close_hand_joint1, joint_ids=hand_joint1_ids)
    robot.set_joint_position_target(close_hand_joint2, joint_ids=hand_joint2_ids)
    robot.set_joint_position_target(close_hand_joint3, joint_ids=hand_joint3_ids)
    robot.set_joint_position_target(close_hand_joint4, joint_ids=hand_joint4_ids)

def align_object_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    hand_quat = env.scene["left_ee_frame"].data.target_quat_w[:, 0, :]   # [N, 4]
    object_quat = env.scene["object"].data.root_quat_w              # [N, 4]

    q_rel = quat_mul(quat_conjugate(hand_quat), object_quat)        # [N, 4]
    w = torch.clamp(torch.abs(q_rel[..., 0]), 0.0, 1.0)

    angle = 2.0 * torch.acos(w)

    norm_err = angle / torch.pi

    ori_reward = torch.exp(-3.0 * norm_err)

    hand_pos = env.scene["left_ee_frame"].data.target_pos_w[:, 0, :]
    obj_pos  = env.scene["object"].data.root_pos_w
    dist = torch.norm(obj_pos - hand_pos, dim=-1)

    d0 = 0.1
    proximity = torch.exp(-dist / d0)

    reward = proximity * ori_reward
    return reward