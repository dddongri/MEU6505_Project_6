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


def task_done_hand_to_hand(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    max_object_vel: float = 0.35,
    settle_steps: int = 3,
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
    if g_cmd is not None:
        left_on = g_cmd[:, 0] > 0.5
        right_on = g_cmd[:, 1] > 0.5
    else:
        grasp = get_grasp_flags(env, dist_th=grasp_dist)
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
