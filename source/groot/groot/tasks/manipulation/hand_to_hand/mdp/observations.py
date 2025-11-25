# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

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
    right_eef_idx = env.scene["robot"].data.body_names.index("right_hand_pitch_link")
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
    object_quat = env.scene["object"].data.root_quat_w

    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos

    return torch.cat(
        (
            object_pos,
            object_quat,
            left_eef_to_object,
            right_eef_to_object,
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
    hand_joint_states = env.scene["robot"].data.joint_pos[:, -22:]  # Hand joints are last 22 entries of joint state

    return hand_joint_states


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
    

def rel_left_to_object(env: ManagerBasedRLEnv) -> torch.Tensor:
    left = get_left_eef_pos(env)
    obj = env.scene["object"].data.root_pos_w - env.scene.env_origins
    return obj - left


def rel_right_to_object(env: ManagerBasedRLEnv) -> torch.Tensor:
    right = get_right_eef_pos(env)
    obj = env.scene["object"].data.root_pos_w - env.scene.env_origins
    return obj - right


def rel_hands(env: ManagerBasedRLEnv) -> torch.Tensor:
    left = get_left_eef_pos(env)
    right = get_right_eef_pos(env)
    return right - left


def get_grasp_flags(env: ManagerBasedRLEnv, dist_th: float = 0.2, z_tol: float = 0.2) -> torch.Tensor:
    """[N,2] = [left_flag, right_flag] (손바닥 기준 원통 거리 + 느슨한 임계 + 명령)"""
    # palm positions: hand_pitch_link origin + local -Z offset
    def _palm_pos(link: str):
        idx = env.scene["robot"].data.body_names.index(link)
        pos = env.scene["robot"].data.body_pos_w[:, idx] - env.scene.env_origins
        quat = env.scene["robot"].data.body_quat_w[:, idx]
        offset = torch.tensor([0.0, 0.0, -0.06], device=env.device).expand(quat.shape[0], -1)
        return pos + math_utils.quat_apply(quat, offset)

    left_palm = _palm_pos("left_hand_pitch_link")
    right_palm = _palm_pos("right_hand_pitch_link")
    obj_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins

    diff_left = obj_pos - left_palm
    diff_right = obj_pos - right_palm
    lh = ((torch.norm(diff_left[:, :2], dim=-1) < dist_th) & (torch.abs(diff_left[:, 2]) < z_tol)).float().unsqueeze(-1)
    rh = ((torch.norm(diff_right[:, :2], dim=-1) < dist_th) & (torch.abs(diff_right[:, 2]) < z_tol)).float().unsqueeze(-1)
    flags = torch.cat([lh, rh], dim=-1)
    # if grasp command exists, require both command and proximity
    g_cmd = env.extras.get("grasp_cmd", None)
    if g_cmd is not None:
        cmd = (g_cmd > 0.5).float()
        flags = flags * cmd
    return flags
