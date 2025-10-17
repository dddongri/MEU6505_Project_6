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
    max_object_vel: float = 0.25,
)-> torch.Tensor:
    # 1) 우선 객체 속도 안정 조건 (fallback)
    obj = env.scene[object_cfg.name]
    obj_vel = torch.norm(obj.data.root_vel_w, dim=1)              # [N]
    done = obj_vel < max_object_vel                               # [N] bool

    # 2) (선택) 손 위치 기반 추가 조건을 쓰고 싶다면, 브로드캐스팅을 올바르게:
    robot = env.scene["robot"]
    body_pos_w = robot.data.body_pos_w                             # [N, B, 3]
    env_origins = env.scene.env_origins[:, None, :]                # [N, 1, 3]
    body_pos_rel = body_pos_w - env_origins                        # [N, B, 3]

    # 필요 시 바디 인덱스
    names = robot.data.body_names
    try:
        left_idx = names.index("left_hand_pitch_link")
        right_idx = names.index("right_hand_pitch_link")
        left_x  = body_pos_rel[:, left_idx, 0]
        right_x = body_pos_rel[:, right_idx, 0]
        # 예) 왼손이 특정 x 이내 + 오른손이 뒤로 빠짐 등의 추가 성공 조건이 있다면 논리곱
        # done = done & (left_x < 0.30) & (right_x < 0.26)
    except ValueError:
        # 바디 이름이 없으면 위치 조건은 건너뛰고 속도 조건만 유지
        pass

    return done