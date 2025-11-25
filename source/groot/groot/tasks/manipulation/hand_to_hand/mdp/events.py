# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.math as math_utils
import re
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_object_poses_nut_pour(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    sorting_beaker_cfg: SceneEntityCfg = SceneEntityCfg("sorting_beaker"),
    factory_nut_cfg: SceneEntityCfg = SceneEntityCfg("factory_nut"),
    sorting_bowl_cfg: SceneEntityCfg = SceneEntityCfg("sorting_bowl"),
    sorting_scale_cfg: SceneEntityCfg = SceneEntityCfg("sorting_scale"),
):
    """Reset the asset root states to a random position and orientation uniformly within the given ranges.

    Args:
        env: The RL environment instance.
        env_ids: The environment IDs to reset the object poses for.
        sorting_beaker_cfg: The configuration for the sorting beaker asset.
        factory_nut_cfg: The configuration for the factory nut asset.
        sorting_bowl_cfg: The configuration for the sorting bowl asset.
        sorting_scale_cfg: The configuration for the sorting scale asset.
        pose_range: The dictionary of pose ranges for the objects. Keys are
                    ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.
    """
    # extract the used quantities (to enable type-hinting)
    sorting_beaker = env.scene[sorting_beaker_cfg.name]
    factory_nut = env.scene[factory_nut_cfg.name]
    sorting_bowl = env.scene[sorting_bowl_cfg.name]
    sorting_scale = env.scene[sorting_scale_cfg.name]

    # get default root state
    sorting_beaker_root_states = sorting_beaker.data.default_root_state[env_ids].clone()
    factory_nut_root_states = factory_nut.data.default_root_state[env_ids].clone()
    sorting_bowl_root_states = sorting_bowl.data.default_root_state[env_ids].clone()
    sorting_scale_root_states = sorting_scale.data.default_root_state[env_ids].clone()

    # get pose ranges
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=sorting_beaker.device)

    # randomize sorting beaker and factory nut together
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=sorting_beaker.device
    )


def randomize_hand_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    noise: float = 0.1,
):
    """Jitter arm joints a little while keeping the base pose fixed."""
    robot = env.scene["robot"]
    names = robot.data.joint_names
    arm_names = [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    ]
    idx = [names.index(n) for n in arm_names if n in names]
    if not idx:
        return
    default_q = robot.data.default_joint_pos[env_ids][:, idx]
    # joint limits not available in data; assume symmetric +/-1.5 rad bounds
    lower = torch.full_like(default_q, -1.5)
    upper = torch.full_like(default_q, 1.5)
    noise_q = (torch.rand_like(default_q) * 2 - 1) * noise
    target = (default_q + noise_q).clamp(lower, upper)
    robot.set_joint_position_target(target, joint_ids=idx, env_ids=env_ids)
    zeros = torch.zeros_like(target)
    robot.set_joint_velocity_target(zeros, joint_ids=idx, env_ids=env_ids)


def place_object_to_right_hand(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # absolute offset in hand frame (equivalent to old 0.1,0.1,-0.2 scaled by 0.4 -> 0.04,0.04,-0.08)
    palm_offset: float | tuple[float, float, float] = (0.02, 0.04, -0.08),
    close_right: bool = True,
):
    """Place object at the right hand (pre-grasp pose) at reset.

    Positions the object at the current right hand pose with a small offset along the hand's
    -Z axis (palm forward) and zeroes its velocity, so training can start from a grasped state.
    """
    robot = env.scene["robot"]
    obj = env.scene["object"]

    # current right hand pose
    rh_idx = robot.data.body_names.index("right_hand_pitch_link")
    rh_pos = robot.data.body_pos_w[env_ids, rh_idx]
    rh_quat = robot.data.body_quat_w[env_ids, rh_idx]

    # local offset toward palm (allow xyz tuple), absolute (no scaling with object size)
    if isinstance(palm_offset, tuple):
        offset_local = torch.tensor(palm_offset, device=env.device, dtype=torch.float32)
    else:
        offset_local = torch.tensor([0.0, 0.0, -float(palm_offset)], device=env.device)

    rot_mats = math_utils.matrix_from_quat(rh_quat)
    offset_dir = offset_local[None, :, None].repeat(len(env_ids), 1, 1)
    offset_vec = (rot_mats @ offset_dir).squeeze(-1)
    target_pos = rh_pos + offset_vec

    # use identity orientation to match env default (no extra rotation)
    upright_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32)
    target_quat = upright_quat.expand(len(env_ids), -1)

    # write pose/vel to sim
    obj.write_root_pose_to_sim(
        torch.cat([target_pos, target_quat], dim=-1),
        env_ids=env_ids,
    )
    zeros = torch.zeros_like(obj.data.root_lin_vel_w[env_ids])
    obj.write_root_velocity_to_sim(
        torch.cat([zeros, zeros], dim=-1),
        env_ids=env_ids,
    )
    # optionally close right hand joints to hold the object
    if close_right:
        joint_names = robot.data.joint_names
        close_cmd = {
            "R_index_.*": -1.0,
            "R_middle_.*": -1.0,
            "R_pinky_.*": -1.0,
            "R_ring_.*": -1.0,
            "R_thumb_proximal_yaw_joint": -1.7,
            "R_thumb_proximal_pitch_joint": 0.35,
            "R_thumb_distal_joint": 1.0,
        }
        # build target tensor
        targets = robot.data.joint_pos_target.clone()
        for pattern, val in close_cmd.items():
            regex = re.compile(pattern)
            idx = [i for i, name in enumerate(joint_names) if regex.match(name)]
            if idx:
                idx_tensor = torch.tensor(idx, device=env.device, dtype=torch.long)
                targets[env_ids.unsqueeze(-1), idx_tensor] = val
        robot.set_joint_position_target(targets[env_ids], joint_ids=None, env_ids=env_ids)
