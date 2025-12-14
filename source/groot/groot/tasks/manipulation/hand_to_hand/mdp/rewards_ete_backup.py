from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _handover_active(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return mask where left is holding and right is released (handover done)."""

    g_cmd = env.extras.get("grasp_cmd", None)
    handover_mask = env.extras.get("handover_mask", None)
    g = get_grasp_flags(env)  # already combines command bits if present

    grasp_mask = (g[:, 0] > 0.5) & (g[:, 1] < 0.5)
    cmd_mask = None
    if g_cmd is not None:
        cmd_mask = (g_cmd[:, 0] > 0.5) & (g_cmd[:, 1] < 0.5)

    mask = grasp_mask if cmd_mask is None else grasp_mask & cmd_mask
    if handover_mask is not None:
        mask = mask | handover_mask
    return mask


def rew_post_handover_separation(
    env: ManagerBasedRLEnv, target_sep: float = 0.25, min_sep: float = 0.18
) -> torch.Tensor:
    """After handover, reward separating hands toward a comfortable distance."""

    mask = _handover_active(env)
    if not torch.any(mask):
        return torch.zeros(env.num_envs, device=env.device)

    sep = torch.norm(rel_hands(env), dim=-1)
    cost = torch.clamp(target_sep - sep, min=0.0)

    reward = torch.zeros(env.num_envs, device=env.device)
    reward[mask] = -cost[mask]

    too_close = sep < min_sep
    reward[mask & too_close] -= (min_sep - sep[mask & too_close]) * 2.0
    return reward


def rew_palm_alignment(
    env: ManagerBasedRLEnv,
    palm_offset: float = 0.06,
    near_thresh: float = 0.18,
    hand_sep_thresh: float = 0.25,
    warmup_steps: int = 18,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    순수 손바닥 얼라인 보상:
    1) 두 손바닥이 서로를 바라보는 정도 (raw_align)
    2) 리셋 시 기준 손바닥 방향과의 일치도 (baseline_score)
    를 합산해서 반환한다.
    """

    # 현재 손바닥 위치/방향
    l_palm, r_palm, n_l, n_r = _palm_pos_and_dir(env, palm_offset)

    # 리셋 시 저장해둔 baseline 방향
    n_l0 = env.extras.get("baseline_palm_dir_left", None)
    n_r0 = env.extras.get("baseline_palm_dir_right", None)

    # baseline이 없으면 보상 0
    if n_l0 is None or n_r0 is None:
        return torch.zeros(env.num_envs, device=env.device)

    hand_sep = torch.norm(rel_hands(env), dim=-1)
    obj_z = env.scene["object"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    warm_ok = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    if "step_counter" in env.extras:
        warm_ok = env.extras["step_counter"] >= warmup_steps

    active = (hand_sep < hand_sep_thresh) & (obj_z > near_thresh) & warm_ok
    if not torch.any(active):
        return torch.zeros(env.num_envs, device=env.device)

    # 1) 서로를 바라보는 정도
    dir_lr = r_palm - l_palm
    dir_lr = dir_lr / torch.clamp(torch.norm(dir_lr, dim=-1, keepdim=True), min=eps)

    align_l = torch.sum(n_l * dir_lr, dim=-1)      # left faces right
    align_r = torch.sum(n_r * (-dir_lr), dim=-1)   # right faces left
    raw_align = 0.5 * (align_l + align_r)

    # 2) 리셋 기준 방향과의 일치도
    bl_l = torch.sum(n_l * n_l0, dim=-1)
    bl_r = torch.sum(n_r * n_r0, dim=-1)
    baseline_score = 0.5 * (bl_l + bl_r)

    reward = torch.zeros(env.num_envs, device=env.device)
    reward[active] = (raw_align + 0.5 * baseline_score)[active]
    return reward


def _palm_pos_and_dir(
    env: ManagerBasedRLEnv, palm_offset: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (left_palm_pos, right_palm_pos, left_palm_dir, right_palm_dir) using offset vector as palm axis."""

    robot = env.scene["robot"]
    names = robot.data.body_names
    li = names.index("left_hand_pitch_link")
    ri = names.index("right_hand_pitch_link")

    offset_local = torch.tensor([0.0, 0.0, -palm_offset], device=env.device).expand(env.num_envs, -1)

    l_pos = robot.data.body_pos_w[:, li] - env.scene.env_origins
    r_pos = robot.data.body_pos_w[:, ri] - env.scene.env_origins
    l_quat = robot.data.body_quat_w[:, li]
    r_quat = robot.data.body_quat_w[:, ri]

    l_off = math_utils.quat_apply(l_quat, offset_local)
    r_off = math_utils.quat_apply(r_quat, offset_local)

    l_palm = l_pos + l_off
    r_palm = r_pos + r_off

    n_l = torch.nn.functional.normalize(l_off, dim=-1)
    n_r = torch.nn.functional.normalize(r_off, dim=-1)
    return l_palm, r_palm, n_l, n_r


def reset_episode_extras(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """Reset all per-episode extras deterministically at env reset."""

    device = env.device
    n = env.num_envs

    # step counter
    env.extras["step_counter"] = torch.zeros(n, device=device, dtype=torch.long)

    # one-shot flags
    env.extras["close_bonus_given"] = torch.zeros(n, device=device, dtype=torch.bool)
    env.extras["handover_bonus_given"] = torch.zeros(n, device=device, dtype=torch.bool)

    # handover bookkeeping
    env.extras["handover_mask"] = torch.zeros(n, device=device, dtype=torch.bool)

    # grasp command cache (if your action writes it)
    env.extras["grasp_cmd"] = torch.zeros((n, 2), device=device, dtype=torch.float)

    # --- baseline palm direction at reset (리셋 직후 손바닥 방향 저장) ---
    robot = env.scene["robot"]
    names = robot.data.body_names
    li = names.index("left_hand_pitch_link")
    ri = names.index("right_hand_pitch_link")

    # 손바닥 방향은 hand_pitch_link 의 -Z 방향으로 정의
    offset_local = torch.tensor([0.0, 0.0, -0.06], device=env.device).expand(env.num_envs, -1)

    l_off0 = math_utils.quat_apply(robot.data.body_quat_w[:, li], offset_local)
    r_off0 = math_utils.quat_apply(robot.data.body_quat_w[:, ri], offset_local)

    env.extras["baseline_palm_dir_left"] = torch.nn.functional.normalize(l_off0, dim=-1)
    env.extras["baseline_palm_dir_right"] = torch.nn.functional.normalize(r_off0, dim=-1)


def inc_step_counter(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """Increment per-env step counter every sim step."""

    if "step_counter" not in env.extras or env.extras["step_counter"].shape[0] != env.num_envs:
        env.extras["step_counter"] = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env.extras["step_counter"] += 1


def rew_hand_front_penalty(env: ManagerBasedRLEnv, min_x: float = 0.12, gate_height: float = 0.86) -> torch.Tensor:
    """
    Penalize if either hand moves behind a forward threshold in the robot root frame.
    Active only when at least one hand grasps or the object is held high enough.
    """

    obj_z = env.scene["object"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    g = get_grasp_flags(env)
    grasp_on = (g[:, 0] > 0.5) | (g[:, 1] > 0.5)
    active = grasp_on | (obj_z > gate_height)
    if not torch.any(active):
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    names = robot.data.body_names
    li = names.index("left_hand_pitch_link")
    ri = names.index("right_hand_pitch_link")

    root_pos = robot.data.root_pos_w - env.scene.env_origins
    left_pos = robot.data.body_pos_w[:, li] - env.scene.env_origins
    right_pos = robot.data.body_pos_w[:, ri] - env.scene.env_origins

    left_x = left_pos[:, 0] - root_pos[:, 0]
    right_x = right_pos[:, 0] - root_pos[:, 0]

    cost_l = torch.clamp(min_x - left_x, min=0.0)
    cost_r = torch.clamp(min_x - right_x, min=0.0)
    cost = cost_l + cost_r

    reward = torch.zeros(env.num_envs, device=env.device)
    reward[active] = -cost[active]
    return reward


def rew_palm_inward(
    env: ManagerBasedRLEnv,
    warmup_steps: int = 12,
    min_height: float = 0.86,
    thresh: float = 0.2,
    hand_sep_max: float = 0.28,
    palm_offset: float = 0.06
) -> torch.Tensor:
    """
    Encourage palms to face inward (right→+Y, left→-Y) to avoid dorsal-out flipping.
    Penalize only when the average inward score drops below a threshold.
    """

    obj_z = env.scene["object"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    warm = env.extras.get("step_counter", None)
    warm_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool) if warm is None else (warm >= warmup_steps)

    hand_sep = torch.norm(rel_hands(env), dim=-1)
    active = (obj_z > min_height) & warm_mask & (hand_sep < hand_sep_max)
    if not torch.any(active):
        return torch.zeros(env.num_envs, device=env.device)

    _, _, n_l, n_r = _palm_pos_and_dir(env, palm_offset)

    root_quat = env.scene["robot"].data.root_quat_w
    y_root = torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(env.num_envs, -1)
    y_root_world = math_utils.quat_apply(root_quat, y_root)

    score = 0.5 * (
        torch.sum(n_r * y_root_world, dim=-1) + torch.sum(n_l * (-y_root_world), dim=-1)
    )

    penalty = torch.clamp(thresh - score, min=0.0)

    reward = torch.zeros(env.num_envs, device=env.device)
    reward[active] = -penalty[active]
    return reward


def rew_hand_height_penalty(env: ManagerBasedRLEnv, min_height: float = 0.92) -> torch.Tensor:
    """Penalty if either hand drops below a minimum height (to keep hands off the table)."""

    robot = env.scene["robot"]
    names = robot.data.body_names
    li = names.index("left_hand_pitch_link")
    ri = names.index("right_hand_pitch_link")

    l_z = robot.data.body_pos_w[:, li, 2] - env.scene.env_origins[:, 2]
    r_z = robot.data.body_pos_w[:, ri, 2] - env.scene.env_origins[:, 2]

    low_l = torch.clamp(min_height - l_z, min=0.0)
    low_r = torch.clamp(min_height - r_z, min=0.0)

    g = get_grasp_flags(env)
    grasp_on = (g[:, 0] > 0.5) | (g[:, 1] > 0.5)

    obj_z = env.scene["object"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    active = grasp_on | (obj_z > min_height - 0.05)

    reward = torch.zeros(env.num_envs, device=env.device)
    reward[active] = -(low_l + low_r)[active]
    return reward


def rew_hand_height_band_penalty(
    env: ManagerBasedRLEnv, min_height: float, max_height: float, weight_high: float = 1.0
) -> torch.Tensor:
    """Soft penalty when hands go below min or above max height (env-origin frame)."""

    warmup_steps = 10
    robot = env.scene["robot"]
    names = robot.data.body_names
    li = names.index("left_hand_pitch_link")
    ri = names.index("right_hand_pitch_link")

    l_z = robot.data.body_pos_w[:, li, 2] - env.scene.env_origins[:, 2]
    r_z = robot.data.body_pos_w[:, ri, 2] - env.scene.env_origins[:, 2]

    step_counter = env.extras.get("step_counter")
    if step_counter is None:
        active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    else:
        active = step_counter >= warmup_steps
    if not torch.any(active):
        return torch.zeros(env.num_envs, device=env.device)

    low_l = torch.clamp(min_height - l_z, min=0.0)
    low_r = torch.clamp(min_height - r_z, min=0.0)
    high_l = torch.clamp(l_z - max_height, min=0.0)
    high_r = torch.clamp(r_z - max_height, min=0.0)

    penalty = (low_l + low_r) + weight_high * (high_l + high_r)

    reward = torch.zeros(env.num_envs, device=env.device)
    reward[active] = -penalty[active]
    return reward


def rew_hands_low_termination_guard(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty that ramps up when either hand drops below the termination height."""

    robot = env.scene["robot"]
    names = robot.data.body_names
    li = names.index("left_hand_pitch_link")
    ri = names.index("right_hand_pitch_link")

    l_z = robot.data.body_pos_w[:, li, 2] - env.scene.env_origins[:, 2]
    r_z = robot.data.body_pos_w[:, ri, 2] - env.scene.env_origins[:, 2]

    low_l = torch.clamp(0.85 - l_z, min=0.0)
    low_r = torch.clamp(0.85 - r_z, min=0.0)
    return -(low_l + low_r)


def rew_object_height_penalty(env: ManagerBasedRLEnv, min_height: float = 0.83) -> torch.Tensor:
    """Penalty when the object drops below a safe working height."""

    obj_z = env.scene["object"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    low = torch.clamp(min_height - obj_z, min=0.0)
    return -low


def rew_object_low_termination_guard(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty that ramps up when the object approaches the dropping threshold."""

    obj_z = env.scene["object"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    low = torch.clamp(0.83 - obj_z, min=0.0)
    return -low


def rew_exchange_zone(
    env: ManagerBasedRLEnv,
    target_offset: tuple[float, float, float] = (0.25, 0.0, 0.1),
    sigma: float = 0.08,
    obj_height_min: float = 0.85,
    hand_obj_thresh: float = 0.12
) -> torch.Tensor:
    """
    Reward placing the object near a chest-height exchange zone in front of the robot.
    Requires both hands near the object and object above a safe height.
    """

    obj_pos = env.scene["object"].data.root_pos_w  # world frame
    obj_z = obj_pos[:, 2] - env.scene.env_origins[:, 2]

    # target position in world: robot root + rotated offset (world frame)
    root_pos = env.scene["robot"].data.root_pos_w
    root_quat = env.scene["robot"].data.root_quat_w
    offset = torch.tensor(target_offset, device=env.device).expand(env.num_envs, -1)
    target = root_pos + math_utils.quat_apply(root_quat, offset)

    # gating: object high enough, right grasp on, left hand near object
    g = get_grasp_flags(env, dist_th=hand_obj_thresh)
    right_on = g[:, 1] > 0.5
    left_close = torch.norm(rel_left_to_object(env), dim=-1) < hand_obj_thresh

    active = (obj_z > obj_height_min) & right_on & left_close
    if not torch.any(active):
        return torch.zeros(env.num_envs, device=env.device)

    dist2 = torch.sum((obj_pos - target) ** 2, dim=-1)

    reward = torch.zeros(env.num_envs, device=env.device)
    reward[active] = torch.exp(-dist2[active] / (2 * sigma * sigma))
    return reward


def _init_flag(env: ManagerBasedRLEnv, key: str) -> torch.Tensor:
    """Ensure a per-env boolean flag exists (reset handled by reset_episode_extras)."""

    if key not in env.extras or env.extras[key].shape[0] != env.num_envs:
        env.extras[key] = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return env.extras[key]


def rew_close_bonus(
    env: ManagerBasedRLEnv,
    hand_obj_thresh: float = 0.12,
    hand_sep_thresh: float = 0.22,
    min_height: float = 0.85,
    vel_thresh: float = 0.25
) -> torch.Tensor:
    """One-time bonus when both hands are near the object at safe height/speed."""

    flag = _init_flag(env, "close_bonus_given")

    obj = env.scene["object"]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    obj_v = torch.norm(obj.data.root_vel_w, dim=-1)

    left_close = torch.norm(rel_left_to_object(env), dim=-1) < hand_obj_thresh
    right_close = torch.norm(rel_right_to_object(env), dim=-1) < hand_obj_thresh
    hand_sep_ok = torch.norm(rel_hands(env), dim=-1) < hand_sep_thresh

    cond = left_close & right_close & hand_sep_ok & (obj_z > min_height) & (obj_v < vel_thresh)
    new_hit = cond & (~flag)

    reward = torch.zeros(env.num_envs, device=env.device)
    reward[new_hit] = 1.0

    if torch.any(new_hit):
        env.extras["close_bonus_given"] = flag | new_hit
    return reward


def rew_handover_bonus(
    env: ManagerBasedRLEnv,
    vel_thresh: float = 0.25,
    min_height: float = 0.8,
    palm_align_thresh: float = 0.0
) -> torch.Tensor:
    """
    One-time bonus for successful handover: left grasped, right released, stable, high enough,
    and palms roughly facing (dot average > threshold).
    """

    flag = _init_flag(env, "handover_bonus_given")

    g = get_grasp_flags(env)
    left_on = g[:, 0] > 0.5
    right_off = g[:, 1] < 0.5

    obj = env.scene["object"]
    vel = torch.norm(obj.data.root_vel_w, dim=1)
    height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    robot = env.scene["robot"]
    names = robot.data.body_names
    li = names.index("left_hand_pitch_link")
    ri = names.index("right_hand_pitch_link")

    n_local = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
    n_l = math_utils.quat_apply(robot.data.body_quat_w[:, li], n_local)
    n_r = math_utils.quat_apply(robot.data.body_quat_w[:, ri], n_local)

    left_pos = robot.data.body_pos_w[:, li] - env.scene.env_origins
    right_pos = robot.data.body_pos_w[:, ri] - env.scene.env_origins
    dir_lr = right_pos - left_pos
    dir_lr = dir_lr / torch.clamp(torch.norm(dir_lr, dim=-1, keepdim=True), min=1e-5)

    align_l = torch.sum(n_l * dir_lr, dim=-1)
    align_r = torch.sum(n_r * (-dir_lr), dim=-1)
    align_score = 0.5 * (align_l + align_r)

    cond = left_on & right_off & (vel < vel_thresh) & (height > min_height) & (align_score > palm_align_thresh)

    new_hit = cond & (~flag)
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[new_hit] = 1.0

    if torch.any(new_hit):
        env.extras["handover_bonus_given"] = flag | new_hit
    return reward


def rew_palm_sagittal(
    env: ManagerBasedRLEnv,
    palm_offset: float = 0.06,
    warmup_steps: int = 12,
    min_height: float = 0.86,
    hand_sep_max: float = 0.28,
    tol: float = 0.2
) -> torch.Tensor:
    """
    Keep palms roughly facing the sagittal plane (avoid flipping palms sideways).
    Penalizes the Y-component of each palm normal (robot frame: +Y is left, +X is forward).
    Only penalize when |dot(n, Y)| exceeds tol; otherwise 0.
    Applies when the object is above a safe height and after a short warmup.
    """

    obj_z = env.scene["object"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    warm = env.extras.get("step_counter", None)
    warm_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool) if warm is None else (warm >= warmup_steps)

    hand_sep = torch.norm(rel_hands(env), dim=-1)
    active = (obj_z > min_height) & warm_mask & (hand_sep < hand_sep_max)
    if not torch.any(active):
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    names = robot.data.body_names
    li = names.index("left_hand_pitch_link")
    ri = names.index("right_hand_pitch_link")

    l_quat = robot.data.body_quat_w[:, li]
    r_quat = robot.data.body_quat_w[:, ri]

    n_local = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
    n_l = math_utils.quat_apply(l_quat, n_local)
    n_r = math_utils.quat_apply(r_quat, n_local)

    root_quat = robot.data.root_quat_w
    y_root = torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(env.num_envs, -1)
    y_root_world = math_utils.quat_apply(root_quat, y_root)

    dev = 0.5 * (
        torch.abs(torch.sum(n_l * y_root_world, dim=-1))
        + torch.abs(torch.sum(n_r * y_root_world, dim=-1))
    )

    penalty = torch.clamp(dev - tol, min=0.0)

    reward = torch.zeros(env.num_envs, device=env.device)
    reward[active] = -penalty[active]
    return reward


def rew_object_upright(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward keeping the object upright based on its local +Z axis in world frame."""

    obj_quat = env.scene["object"].data.root_quat_w
    z_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1)
    z_world = math_utils.quat_apply(obj_quat, z_local)
    z_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1)

    upright_score = torch.sum(z_world * z_up, dim=-1)
    return upright_score


from .observations import (
    rel_left_to_object,
    rel_right_to_object,
    rel_hands,
    get_grasp_flags,
)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return -torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)


def rew_hands_proximity(env: ManagerBasedRLEnv, min_height: float = 0.9) -> torch.Tensor:
    """Bring hands closer, only when the object is sufficiently above the table."""

    obj_z = env.scene["object"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    active = obj_z > min_height

    reward = torch.zeros(env.num_envs, device=env.device)
    if torch.any(active):
        rel = rel_hands(env)[active]
        reward[active] = -torch.norm(rel, dim=-1)

    # below height: no bonus, mild penalty proportional to how low
    if torch.any(~active):
        reward[~active] = -0.5 * torch.clamp(min_height - obj_z[~active], min=0.0)

    mask = _handover_active(env)
    reward = torch.where(mask, torch.zeros_like(reward), reward)
    return reward


def rew_post_handover_arm_home(
    env: ManagerBasedRLEnv,
    target: tuple[float, float, float] = (0.2, 0.25, 1.0),
    palm_offset: float = 0.06,
    vel_weight: float = 0.05
) -> torch.Tensor:
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

    target_world = torch.tensor(target, device=env.device).expand(env.num_envs, -1)
    tgt = target_world - env.scene.env_origins

    dist = torch.norm(palm - tgt, dim=-1)

    vel_cost = torch.norm(robot.data.body_lin_vel_w[:, idx], dim=-1)

    reward = torch.zeros(env.num_envs, device=env.device)
    reward[mask] = -(dist[mask] + vel_weight * vel_cost[mask])
    return reward


def rew_guarded_transfer(env: ManagerBasedRLEnv, near_tol: float = 0.15, grasp_tol: float = 0.06) -> torch.Tensor:
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

    step_counter = env.extras.get("step_counter", None)
    warm_ok = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool) if step_counter is None else (step_counter >= 12)

    # bad: right opens while left is not yet close (after warmup)
    bad_release = (~right_cmd) & (~left_close) & warm_ok
    reward -= bad_release.float() * 1.0

    # good: both grasping near the object (handover overlap)
    dual_hold = left_cmd & right_cmd & left_close & right_close
    reward += dual_hold.float() * 0.5

    # penalize lingering with the source hand after the receiver is close
    cling = right_cmd & left_cmd & left_close
    reward -= cling.float() * 0.2

    # good: right releases after left grasped near object
    safe_release = (~right_cmd) & left_cmd & left_close
    reward += safe_release.float() * 0.6

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


# --------------------------------------------------------------------------
# Simpler handover shaping (right hold -> left approach -> transfer -> left hold)
# --------------------------------------------------------------------------

def rew_right_hold(env: ManagerBasedRLEnv, dist_th: float = 0.07) -> torch.Tensor:
    """Encourage right hand to hold/keep object before transfer."""

    g = get_grasp_flags(env, dist_th=dist_th)
    right_on = g[:, 1] > 0.5
    left_on = g[:, 0] > 0.5
    near = torch.norm(rel_right_to_object(env), dim=-1) < dist_th

    # once left grasps, stop rewarding right_hold
    return ((right_on & near) & (~left_on)).float()


def rew_left_approach_simple(env: ManagerBasedRLEnv, min_height: float = 0.9) -> torch.Tensor:
    """Pull left hand toward object, but only when object is well above the table."""

    obj_z = env.scene["object"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    active = obj_z > min_height

    reward = torch.zeros(env.num_envs, device=env.device)
    if torch.any(active):
        rel = rel_left_to_object(env)[active]
        reward[active] = -torch.norm(rel, dim=-1)

    if torch.any(~active):
        reward[~active] = -0.5 * torch.clamp(min_height - obj_z[~active], min=0.0)  # mild penalty when too low
    return reward


def rew_left_grasp_ready(env: ManagerBasedRLEnv, dist_th: float = 0.07) -> torch.Tensor:
    """Bonus when left hand is close and command/grasp flag is on."""

    g = get_grasp_flags(env, dist_th=dist_th)
    left_on = g[:, 0] > 0.5
    near = torch.norm(rel_left_to_object(env), dim=-1) < dist_th
    return (left_on & near).float()


def rew_transfer_success(env: ManagerBasedRLEnv, vel_thresh: float = 0.3, min_height: float = 0.7) -> torch.Tensor:
    """Handover success: left holds, right released, object stable and above table."""

    g = get_grasp_flags(env)
    left_on = g[:, 0] > 0.5
    right_off = g[:, 1] < 0.5

    obj = env.scene["object"]
    vel = torch.norm(obj.data.root_vel_w, dim=1)
    height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    good = left_on & right_off & (vel < vel_thresh) & (height > min_height)
    return good.float()


def rew_left_hold_stable(env: ManagerBasedRLEnv) -> torch.Tensor:
    """After left holds, reward low object velocity."""

    g = get_grasp_flags(env)
    left_on = g[:, 0] > 0.5
    vel = torch.norm(env.scene["object"].data.root_vel_w, dim=1)

    reward = torch.zeros(env.num_envs, device=env.device)
    reward[left_on] = -vel[left_on]
    return reward


def rew_height_shaping(env: ManagerBasedRLEnv, min_height: float = 0.88, target_height: float = 1.08) -> torch.Tensor:
    """Encourage keeping object off the table; bonus ramps up between min and target, only when grasped."""

    g = get_grasp_flags(env)
    grasp_on = (g[:, 0] > 0.5) | (g[:, 1] > 0.5)

    obj_z = env.scene["object"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    bonus = (obj_z - min_height) / max(target_height - min_height, 1e-3)
    bonus = torch.clamp(bonus, min=0.0, max=1.0)

    reward = torch.zeros(env.num_envs, device=env.device)
    reward[grasp_on] = bonus[grasp_on]
    return reward


def rew_height_shaping_banded(
    env: ManagerBasedRLEnv, min_height: float, target_height: float, max_height: float
) -> torch.Tensor:
    """Banded shaping: maximal near target band, decays toward both min and max limits."""

    obj_z = env.scene["object"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    up = torch.clamp((obj_z - min_height) / max(target_height - min_height, 1e-3), min=0.0, max=1.0)
    down = torch.clamp((max_height - obj_z) / max(max_height - target_height, 1e-3), min=0.0, max=1.0)

    bonus = up * down

    reward = torch.zeros(env.num_envs, device=env.device)
    active = obj_z > 0.6
    reward[active] = bonus[active]
    return reward


def rew_dual_hold_bonus(env: ManagerBasedRLEnv, dist_th: float = 0.07) -> torch.Tensor:
    """Small bonus when 양손 모두 근접+그립 on 상태(전이 겹침)."""

    g = get_grasp_flags(env, dist_th=dist_th)
    lh = g[:, 0] > 0.5
    rh = g[:, 1] > 0.5

    left_close = torch.norm(rel_left_to_object(env), dim=-1) < dist_th
    right_close = torch.norm(rel_right_to_object(env), dim=-1) < dist_th

    dual = lh & rh & left_close & right_close
    return dual.float()


def rew_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant alive reward per step to discourage intentional early termination."""

    return torch.ones(env.num_envs, device=env.device)
