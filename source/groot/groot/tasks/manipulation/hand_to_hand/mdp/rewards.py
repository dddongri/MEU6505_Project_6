from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_height_bonus(env: ManagerBasedRLEnv, min_height: float = 0.8, target_height: float = 1.0) -> torch.Tensor:
    """Bonus when the object is carried above a safe height."""
    obj_z = env.scene["object"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    bonus = (obj_z - min_height) / max(target_height - min_height, 1e-3)
    return torch.clamp(bonus, min=0.0, max=1.0)

def rew_clamp_penalty(env: ManagerBasedRLEnv,
                      hand_dist: float = 0.14,
                      hand_sep: float = 0.22,
                      vel_thresh: float = 0.05) -> torch.Tensor:
    """Penalty when the object is simultaneously close to both hands and hands are close."""
    left_vec = rel_left_to_object(env)
    right_vec = rel_right_to_object(env)
    hands_vec = rel_hands(env)
    left_close = torch.norm(left_vec, dim=-1) < hand_dist
    right_close = torch.norm(right_vec, dim=-1) < hand_dist
    hands_close = torch.norm(hands_vec, dim=-1) < hand_sep
    obj_vel = torch.norm(env.scene["object"].data.root_vel_w, dim=1)
    clamped = left_close & right_close & hands_close & (obj_vel < vel_thresh)
    return -clamped.float()

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


def rew_left_grasp_bonus(env: ManagerBasedRLEnv, dist_th: float = 0.12) -> torch.Tensor:
    """Bonus when 왼손이 물체 근처에서 그립 명령까지 주는 경우를 장려."""
    g = get_grasp_flags(env, dist_th=dist_th)
    lh = g[:, 0] > 0.5
    rel = rel_left_to_object(env)
    near = torch.norm(rel, dim=-1) < dist_th
    return (lh & near).float() * 2.0


def rew_align_to_exchange(env: ManagerBasedRLEnv,
                          center: torch.Tensor | None = None,
                          table_height: float = 0.55,
                          z_margin: float = 0.1) -> torch.Tensor:
    """
    교환지점(월드) C를 기준으로 점대칭 정렬 유도:
      p_left* = 2C - p_right , p_right* = 2C - p_left
    위치 정렬 비용만 사용(단순/안정). 필요시 회전 정렬을 추가.
    """
    left = env.scene["robot"].data.body_pos_w[:, env.scene["robot"].data.body_names.index("left_hand_pitch_link")]
    right = env.scene["robot"].data.body_pos_w[:, env.scene["robot"].data.body_names.index("right_hand_pitch_link")]
    left = left - env.scene.env_origins
    right = right - env.scene.env_origins

    if center is None:
        center = 0.5 * (left + right)
        center[:, 2] = torch.clamp(center[:, 2], min=table_height + z_margin)

    lh_t = 2 * center - right
    rh_t = 2 * center - left
    pos_cost = torch.norm(left - lh_t, dim=-1) + torch.norm(right - rh_t, dim=-1)
    return -pos_cost


def rew_post_handover_arm_home(env: ManagerBasedRLEnv,
                               target: tuple[float, float, float] = (0.2, 0.25, 1.0),
                               palm_offset: float = 0.06,
                               vel_weight: float = 0.05) -> torch.Tensor:
    """
    After handover (left grasped, right released), pull right hand toward a home pose
    in front of the body to avoid flinging backward.
    """
    g_cmd = env.extras.get("grasp_cmd", None)
    handover_mask = env.extras.get("handover_mask", None)
    if g_cmd is not None:
        mask = (g_cmd[:, 0] > 0.5) & (g_cmd[:, 1] < 0.5)
    else:
        g = get_grasp_flags(env)
        mask = (g[:, 0] > 0.5) & (g[:, 1] < 0.5)
    if handover_mask is not None:
        mask = mask | handover_mask
    if not torch.any(mask.bool()):
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    names = robot.data.body_names
    idx = names.index("right_hand_pitch_link")
    pos = robot.data.body_pos_w[:, idx] - env.scene.env_origins
    quat = robot.data.body_quat_w[:, idx]
    offset = torch.tensor([0.0, 0.0, -palm_offset], device=env.device).expand(quat.shape[0], -1)
    palm = pos + math_utils.quat_apply(quat, offset)

    tgt = torch.tensor(target, device=env.device)
    dist = torch.norm(palm - tgt, dim=-1)
    vel_cost = torch.norm(robot.data.body_lin_vel_w[:, idx], dim=-1)
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[mask] = -(dist[mask] + vel_weight * vel_cost[mask])
    return reward


def rew_post_handover_posture(env: ManagerBasedRLEnv, vel_weight: float = 0.02) -> torch.Tensor:
    """After the object is in the left hand, bias arm joints back toward the relaxed default pose."""
    g_cmd = env.extras.get("grasp_cmd", None)
    handover_mask = env.extras.get("handover_mask", None)
    if g_cmd is None and handover_mask is None:
        return torch.zeros(env.num_envs, device=env.device)

    if g_cmd is not None:
        active = (g_cmd[:, 0] > 0.5) & (g_cmd[:, 1] < 0.5)
    else:
        active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if handover_mask is not None:
        active = active | handover_mask
    if not torch.any(active.bool()):
        return torch.zeros(env.num_envs, device=env.device)

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
        return torch.zeros(env.num_envs, device=env.device)

    idx_tensor = torch.tensor(idx, device=env.device, dtype=torch.long)
    q = robot.data.joint_pos[:, idx_tensor]
    q_def = robot.data.default_joint_pos[:, idx_tensor]
    vel = robot.data.joint_vel[:, idx_tensor]

    err = torch.norm(q - q_def, dim=1)
    vel_cost = torch.norm(vel, dim=1)
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[active] = -(err[active] + vel_weight * vel_cost[active])
    return reward


def rew_guarded_transfer(env: ManagerBasedRLEnv,
                         near_tol: float = 0.12,
                         grasp_tol: float = 0.06) -> torch.Tensor:
    """
    Sequence shaping:
      - Penalize opening the source hand before the receiving hand is close.
      - Bonus when both hands grasp (handover overlap).
      - Bonus when source releases after receiver grasped near the object.
    Uses grasp_cmd from action extras; falls back to zeros if unavailable.
    """
    g_cmd = env.extras.get("grasp_cmd", None)
    if g_cmd is None:
        return torch.zeros(env.num_envs, device=env.device)

    # inferred grasps from command bits
    left_cmd = g_cmd[:, 0] > 0.5
    right_cmd = g_cmd[:, 1] > 0.5

    # proximity to object
    left_close = torch.norm(rel_left_to_object(env), dim=-1) < near_tol
    right_close = torch.norm(rel_right_to_object(env), dim=-1) < near_tol

    # penalties/rewards
    reward = torch.zeros(env.num_envs, device=env.device)

    # bad: right opens while left is not yet close
    bad_release = (~right_cmd) & (~left_close)
    reward -= bad_release.float() * 1.0

    # good: both grasping near the object (handover overlap)
    dual_hold = left_cmd & right_cmd & left_close & right_close
    reward += dual_hold.float() * 0.5

    # penalize lingering with the source hand after the receiver is close
    cling = right_cmd & left_cmd & left_close
    reward -= cling.float() * 0.5

    # good: right releases after left grasped near object
    safe_release = (~right_cmd) & left_cmd & left_close
    reward += safe_release.float() * 1.0

    handover_mask = env.extras.get("handover_mask", None)
    if handover_mask is not None:
        reward += ((handover_mask) & (~right_cmd)).float() * 0.2

    return reward


def rew_release_penalty(env: ManagerBasedRLEnv, dist_th: float = 0.06) -> torch.Tensor:
    """Penalty if 양손 모두 객체를 놓은 상태."""
    g = get_grasp_flags(env, dist_th=dist_th)  # [N,2]
    both_off = (g[:, 0] < 0.5) & (g[:, 1] < 0.5)
    # ignore penalty for initial warmup steps
    step_counter = env.extras.get("step_counter", None)
    warm_mask = torch.ones_like(both_off, dtype=torch.bool)
    if step_counter is not None:
        warm_mask = step_counter >= 10
    both_off = both_off & warm_mask
    return -both_off.float() * 5.0
