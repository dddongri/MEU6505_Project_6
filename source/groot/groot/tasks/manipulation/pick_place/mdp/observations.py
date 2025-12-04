# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_conjugate, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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

    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_hand_pitch_link")
    # right_eef_idx = env.scene["robot"].data.body_names.index("right_hand_pitch_link")
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene["robot"].data.root_pos_w
    left_eef_ori = env.scene["robot"].data.body_quat_w[:, left_eef_idx]
    # right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene["robot"].data.root_pos_w

    object_pos = env.scene["object"].data.root_pos_w - env.scene["robot"].data.root_pos_w
    object_quat = env.scene["object"].data.root_quat_w

    left_eef_to_object = object_pos - left_eef_pos
    # right_eef_to_object = object_pos - right_eef_pos
    
    # left_eef_rot_error = env.scene.quat_mul(
    #     quat_conjugate(left_eef_ori),
    #     object_quat,
    # )
    
    # print("Object Quaternion:", object_quat[0])

    return torch.cat(
        (
            object_pos,
            object_quat,
            left_eef_to_object,
            # left_eef_rot_error
            # right_eef_to_object,
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
    hand_pos_states = env.scene["ee_frame"].data.target_pos_w[:, 0, :] - env.scene["robot"].data.root_pos_w
    hand_ori_states = env.scene["ee_frame"].data.target_quat_w[:, 0, :]
    
    # hand_joint_states = env.scene["robot"].data.joint_pos[:, -22:]  # Hand joints are last 22 entries of joint state

    # print("target_pos_w: ", env.scene["ee_frame"].data.target_pos_w[0, 0, :])  # [-0.2238,  0.3406,  1.0976]
    # print("env.scene.env_origins: ", env.scene["robot"].data.root_pos_w[0])
    # print("hand_pos_states: ", hand_pos_states[0])
    # print("hand_ori_states: ", hand_ori_states[0])
    
    return torch.cat((hand_pos_states, hand_ori_states), dim=1)


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
