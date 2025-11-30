# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the hand-to-hand task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _handover_active(env: "ManagerBasedRLEnv", grasp_dist: float = 0.14) -> torch.Tensor:
    """Return mask where left is holding and right is released (handover done)."""
    g_cmd = env.extras.get("grasp_cmd", None)
    handover_mask = env.extras.get("handover_mask", None)
    from .observations import get_grasp_flags

    if g_cmd is not None:
        left_on = g_cmd[:, 0] > 0.5
        right_on = g_cmd[:, 1] > 0.5
    else:
        grasp = get_grasp_flags(env, dist_th=grasp_dist)
        left_on = grasp[:, 0] > 0.5
        right_on = grasp[:, 1] > 0.5
    mask = left_on & (~right_on)
    if handover_mask is not None:
        mask = mask | handover_mask
    return mask


def task_done_hand_to_hand(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    max_object_vel: float = 0.5,
    settle_steps: int = 2,
    grasp_dist: float = 0.14,
) -> torch.Tensor:
    # 기준:
    #  - 왼손이 객체를 잡고 있음(거리 기반)
    #  - 오른손은 더 이상 잡지 않음(거리 멀거나 그립 커맨드 off)
    #  - 객체 속도가 충분히 느림
    obj = env.scene[object_cfg.name]
    obj_vel = torch.norm(obj.data.root_vel_w, dim=1)              # [N]

    from .observations import get_grasp_flags, rel_left_to_object, rel_right_to_object

    g_cmd = env.extras.get("grasp_cmd", None)
    handover_mask = env.extras.get("handover_mask", None)
    grasp = get_grasp_flags(env, dist_th=grasp_dist)
    if g_cmd is not None:
        # left: allow proximity or command; right: command only (prevent lingering proximity from blocking success)
        left_on = (g_cmd[:, 0] > 0.5) | (grasp[:, 0] > 0.5)
        right_on = g_cmd[:, 1] > 0.5
    else:
        left_on = grasp[:, 0] > 0.5
        right_on = grasp[:, 1] > 0.5

    # 오른손이 멀리 떨어졌는지/잡지 않았는지
    right_far = torch.norm(rel_right_to_object(env), dim=-1) > grasp_dist
    right_released = (~right_on) | right_far
    left_close = torch.norm(rel_left_to_object(env), dim=-1) < grasp_dist

    # 객체 정지 + 왼손 잡음 + 오른손 놓음
    step_counter = env.extras.get("step_counter", None)
    if step_counter is None:
        warm_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    else:
        warm_mask = step_counter >= 5
    done_now = (obj_vel < max_object_vel) & left_on & right_released & left_close
    if handover_mask is not None:
        done_now = done_now | (handover_mask & (obj_vel < max_object_vel))
    done_now = done_now & warm_mask

    counter = env.extras.get("success_counter", torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
    counter = torch.where(done_now, counter + 1, torch.zeros_like(counter))
    env.extras["success_counter"] = counter
    term_counts = env.extras.get("termination_counts", {"success": 0})
    if "last_term" in env.extras:
        last_term = env.extras["last_term"]
    else:
        last_term = [None] * env.num_envs
    last_term = list(last_term)
    done_envs = torch.where(counter >= settle_steps)[0]
    for idx in done_envs.tolist():
        last_term[idx] = "success"
    env.extras["last_term"] = last_term
    term_counts["success"] = term_counts.get("success", 0) + len(done_envs)
    env.extras["termination_counts"] = term_counts
    return counter >= settle_steps


def object_stuck(
    env: "ManagerBasedRLEnv",
    table_height: float = 0.55,
    table_margin: float = 0.05,
    hand_dist: float = 0.20,
    vel_thresh: float = 0.02,
    z_progress_tol: float = 0.003,
    spawn_band: float = 0.05,
    settle_steps: int = 30,
) -> torch.Tensor:
    """Terminate if the object is clamped between hands/table without progress."""
    obj = env.scene["object"]
    rel_pos = obj.data.root_pos_w - env.scene.env_origins
    obj_vel = torch.norm(obj.data.root_vel_w, dim=1)

    from .observations import rel_left_to_object, rel_right_to_object

    # near-plane: around table height only
    near_table = rel_pos[:, 2] < (table_height + table_margin)
    near_plane = near_table

    left_close = torch.norm(rel_left_to_object(env), dim=-1) < hand_dist
    right_close = torch.norm(rel_right_to_object(env), dim=-1) < hand_dist
    close_any = left_close | right_close

    prev_z = env.extras.get("prev_obj_z", rel_pos[:, 2].clone())
    dz = torch.abs(rel_pos[:, 2] - prev_z)
    env.extras["prev_obj_z"] = rel_pos[:, 2].clone()

    stuck_now = near_plane & close_any & (obj_vel < vel_thresh) & (dz < z_progress_tol)
    # don't flag stuck after handover is active
    handover_active = _handover_active(env, grasp_dist=hand_dist)
    stuck_now = stuck_now & (~handover_active)

    step_counter = env.extras.get("step_counter", None)
    if step_counter is None:
        warm_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    else:
        warm_mask = step_counter >= 10
    stuck_now = stuck_now & warm_mask

    counter = env.extras.get("stuck_counter", torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
    counter = torch.where(stuck_now, counter + 1, torch.zeros_like(counter))
    env.extras["stuck_counter"] = counter
    done_envs = torch.where(counter >= settle_steps)[0]
    if done_envs.numel() > 0:
        term_counts = env.extras.get("termination_counts", {})
        term_counts["object_stuck"] = term_counts.get("object_stuck", 0) + len(done_envs)
        env.extras["termination_counts"] = term_counts
        if "last_term" in env.extras:
            last_term = env.extras["last_term"]
        else:
            last_term = [None] * env.num_envs
        last_term = list(last_term)
        for idx in done_envs.tolist():
            last_term[idx] = "object_stuck"
        env.extras["last_term"] = last_term
    return counter >= settle_steps


def object_clamped_between_hands(
    env: "ManagerBasedRLEnv",
    hand_dist: float = 0.12,
    hand_sep: float = 0.18,
    vel_thresh: float = 0.01,
    z_progress_tol: float = 0.003,
    sep_progress_tol: float = 0.002,
    hand_vel_thresh: float = 0.05,
    settle_steps: int = 40,
) -> torch.Tensor:
    """Terminate if the object is pinched between both hands with no motion (any height)."""
    obj = env.scene["object"]
    obj_vel = torch.norm(obj.data.root_vel_w, dim=1)

    from .observations import rel_left_to_object, rel_right_to_object, rel_hands

    handover_active = _handover_active(env, grasp_dist=hand_dist)
    if not torch.any(handover_active):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    left_vec = rel_left_to_object(env)
    right_vec = rel_right_to_object(env)
    hands_vec = rel_hands(env)
    left_close = torch.norm(left_vec, dim=-1) < hand_dist
    right_close = torch.norm(right_vec, dim=-1) < hand_dist
    hands_sep = torch.norm(hands_vec, dim=-1)
    hands_close = hands_sep < hand_sep

    prev_sep = env.extras.get("prev_hand_sep", hands_sep.clone())
    env.extras["prev_hand_sep"] = hands_sep.clone()
    sep_still = torch.abs(hands_sep - prev_sep) < sep_progress_tol

    prev_z = env.extras.get("prev_obj_z_clamp", obj.data.root_pos_w[:, 2].clone())
    dz = torch.abs((obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]) - prev_z)
    env.extras["prev_obj_z_clamp"] = (obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).clone()

    # hand linear velocity
    robot = env.scene["robot"]
    names = robot.data.body_names
    l_idx = names.index("left_hand_pitch_link")
    r_idx = names.index("right_hand_pitch_link")
    hand_lin_vel = torch.max(
        torch.norm(robot.data.body_lin_vel_w[:, l_idx], dim=-1),
        torch.norm(robot.data.body_lin_vel_w[:, r_idx], dim=-1),
    )
    hands_slow = hand_lin_vel < hand_vel_thresh

    stuck_now = (
        left_close
        & right_close
        & hands_close
        & (obj_vel < vel_thresh)
        & (dz < z_progress_tol)
        & sep_still
        & hands_slow
    )

    step_counter = env.extras.get("step_counter", None)
    warm_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool) if step_counter is None else step_counter >= 10
    stuck_now = stuck_now & warm_mask & handover_active

    counter = env.extras.get("clamp_counter", torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
    counter = torch.where(stuck_now, counter + 1, torch.zeros_like(counter))
    env.extras["clamp_counter"] = counter
    done_envs = torch.where(counter >= settle_steps)[0]
    if done_envs.numel() > 0:
        term_counts = env.extras.get("termination_counts", {})
        term_counts["hands_clamped"] = term_counts.get("hands_clamped", 0) + len(done_envs)
        env.extras["termination_counts"] = term_counts
        if "last_term" in env.extras:
            last_term = env.extras["last_term"]
        else:
            last_term = [None] * env.num_envs
        last_term = list(last_term)
        for idx in done_envs.tolist():
            last_term[idx] = "hands_clamped"
        env.extras["last_term"] = last_term
    return counter >= settle_steps


def both_hands_released(env: "ManagerBasedRLEnv", dist_th: float = 0.06) -> torch.Tensor:
    """Terminate when both hands have let go of the object."""
    from .observations import get_grasp_flags
    g = get_grasp_flags(env, dist_th=dist_th)  # [N,2]
    both_off = (g[:, 0] < 0.5) & (g[:, 1] < 0.5)
    # ignore initial warmup steps to let the system settle
    step_counter = env.extras.get("step_counter", None)
    if step_counter is None:
        warm_mask = torch.ones_like(both_off, dtype=torch.bool)
    else:
        warm_mask = step_counter >= 5
    both_off = both_off & warm_mask
    # require several consecutive frames off to avoid flicker
    counter = env.extras.get("both_off_counter", torch.zeros_like(both_off, dtype=torch.long))
    counter = torch.where(both_off, counter + 1, torch.zeros_like(counter))
    env.extras["both_off_counter"] = counter
    return counter >= 2
