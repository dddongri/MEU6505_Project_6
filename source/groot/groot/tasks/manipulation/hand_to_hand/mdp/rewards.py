from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import (
    rel_left_to_object,
    rel_right_to_object,
    rel_hands,
    get_grasp_flags,
)


def joint_torques_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return -torch.sum(env.scene["robot"].data.applied_torque**2, dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return -torch.sum(env.scene["robot"].data.joint_acc**2, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return -torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv) -> torch.Tensor:
    lower = env.scene["robot"].data.joint_lower_limits
    upper = env.scene["robot"].data.joint_upper_limits
    q = env.scene["robot"].data.joint_pos
    below = (lower - q).clamp(max=0.0).abs()
    above = (q - upper).clamp(min=0.0).abs()
    return -(below + above).sum(dim=1)


def rew_left_approach(env: ManagerBasedRLEnv) -> torch.Tensor:
    rel = rel_left_to_object(env)
    return -torch.norm(rel, dim=-1)


def rew_right_stability(env: ManagerBasedRLEnv) -> torch.Tensor:
    v = env.scene["object"].data.root_lin_vel_w  # [N,3]
    return -torch.norm(v, dim=-1)


def rew_hands_proximity(env: ManagerBasedRLEnv) -> torch.Tensor:
    rel = rel_hands(env)
    return -torch.norm(rel, dim=-1)


def rew_transfer(env: ManagerBasedRLEnv) -> torch.Tensor:
    g = get_grasp_flags(env)  # [N,2] = [left,right]
    lh, rh = g[:, 0], g[:, 1]
    return ((lh > 0.5) & (rh < 0.5)).float() * 4.0


def rew_align_to_exchange(env: ManagerBasedRLEnv,
                          center: torch.Tensor | None = None) -> torch.Tensor:
    """
    교환지점(월드) C를 기준으로 점대칭 정렬 유도:
      p_left* = 2C - p_right , p_right* = 2C - p_left
    위치 정렬 비용만 사용(단순/안정). 필요시 회전 정렬을 추가.
    """
    if center is None:
        # 몸 중심축, 테이블 앞, 명치 높이 근처 (env origin 기준 오프셋)
        # (프로젝트에 맞게 수치 조정 가능)
        center = torch.tensor([0.00, 0.45, 1.05], device=env.device)

    left = env.scene["robot"].data.body_pos_w[:, env.scene["robot"].data.body_names.index("left_hand_pitch_link")]
    right = env.scene["robot"].data.body_pos_w[:, env.scene["robot"].data.body_names.index("right_hand_pitch_link")]
    left = left - env.scene.env_origins
    right = right - env.scene.env_origins

    lh_t = 2 * center - right
    rh_t = 2 * center - left
    pos_cost = torch.norm(left - lh_t, dim=-1) + torch.norm(right - rh_t, dim=-1)
    return -pos_cost
