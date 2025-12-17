from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# --------------------------------------------------------------------------------
# Internal helpers / caches (stored on env instance)
# --------------------------------------------------------------------------------

def _ensure_obj_attached(env: "ManagerBasedEnv") -> None:
    """Create per-env boolean flag: object is 'attached/welded' to left hand."""
    if not hasattr(env, "_obj_attached"):
        env._obj_attached = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

def _ensure_left_hand_joint_ids(env: "ManagerBasedEnv", robot_cfg: SceneEntityCfg) -> None:
    """Cache left hand joint indices that match GR1 left hand naming."""
    if hasattr(env, "_left_hand_joint_ids"):
        return
    robot = env.scene[robot_cfg.name]
    names = list(robot.joint_names)
    pats = [
        r"^L_index_.*",
        r"^L_middle_.*",
        r"^L_ring_.*",
        r"^L_pinky_.*",
        r"^L_thumb_.*",
    ]
    ids: list[int] = []
    for i, n in enumerate(names):
        if any(re.match(p, n) for p in pats):
            ids.append(i)
    if len(ids) == 0:
        raise RuntimeError(
            "Could not find left hand joints by regex. "
            "Check your robot joint names. Example expected: L_index_*, L_thumb_* ..."
        )
    env._left_hand_joint_ids = torch.tensor(ids, device=env.device, dtype=torch.long)


# --------------------------------------------------------------------------------
# Reset/interval events
# --------------------------------------------------------------------------------

def set_object_attached(env: "ManagerBasedEnv", env_ids: torch.Tensor, attached: bool = True) -> None:
    """Set attached flag for selected envs."""
    _ensure_obj_attached(env)
    env._obj_attached[env_ids] = attached


def keep_left_hand_open(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    open_value: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Force all left hand joints to stay at open_value (and vel=0) to remove jitter."""
    _ensure_left_hand_joint_ids(env, robot_cfg)
    robot = env.scene[robot_cfg.name]
    ids = env._left_hand_joint_ids

    qpos = robot.data.joint_pos[env_ids].clone()
    qvel = robot.data.joint_vel[env_ids].clone()
    qpos[:, ids] = float(open_value)
    qvel[:, ids] = 0.0

    # NOTE: velocity cannot be None (will crash). Provide qvel.
    robot.write_joint_state_to_sim(qpos, qvel, env_ids=env_ids)


def place_object_to_left_hand(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    obj_offset_tcp: tuple[float, float, float] = (0.0, 0.06, 0.0),
    align_with_hand: bool = True,
    # ✅ 추가: 컵을 월드 기준 "upright"로 강제
    force_upright: bool = False,
    upright_quat_w: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Reset-time placement: put object near LEFT TCP (NOT welded).
    """
    obj = env.scene[object_cfg.name]
    ee = env.scene.sensors[ee_cfg.name]

    tcp_pos_w = ee.data.target_pos_w[env_ids, 0, :]
    tcp_quat_w = ee.data.target_quat_w[env_ids, 0, :]

    off_tcp = torch.tensor(obj_offset_tcp, device=tcp_pos_w.device, dtype=tcp_pos_w.dtype).unsqueeze(0)
    off_tcp = off_tcp.repeat(env_ids.shape[0], 1)
    off_w = math_utils.quat_apply(tcp_quat_w, off_tcp)

    obj_pos_w = tcp_pos_w + off_w

    if force_upright:
        obj_quat_w = torch.tensor(upright_quat_w, device=tcp_pos_w.device, dtype=tcp_pos_w.dtype).unsqueeze(0)
        obj_quat_w = obj_quat_w.repeat(env_ids.shape[0], 1)
    elif align_with_hand:
        obj_quat_w = tcp_quat_w
    else:
        obj_quat_w = obj.data.default_root_state[env_ids, 3:7].clone()

    obj.write_root_pose_to_sim(torch.cat([obj_pos_w, obj_quat_w], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros((env_ids.shape[0], 6), device=env.device), env_ids=env_ids)

def keep_object_welded_to_left_hand(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    obj_offset_tcp: tuple[float, float, float] = (0.0, 0.06, 0.0),
    align_with_hand: bool = True,
    # ✅ 추가: 컵을 월드 기준 "upright"로 강제
    force_upright: bool = False,
    upright_quat_w: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """attached=True인 env에서만 object를 손 TCP로 따라오게 한다."""
    # attached 플래그가 없으면 기본 True로 간주
    if not hasattr(env, "_obj_attached"):
        env._obj_attached = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    mask = env._obj_attached[env_ids]
    if not bool(mask.any()):
        return

    ids = env_ids[mask]

    obj = env.scene[object_cfg.name]
    ee = env.scene.sensors[ee_cfg.name]

    tcp_pos_w = ee.data.target_pos_w[ids, 0, :]
    tcp_quat_w = ee.data.target_quat_w[ids, 0, :]

    off_tcp = torch.tensor(obj_offset_tcp, device=tcp_pos_w.device, dtype=tcp_pos_w.dtype).unsqueeze(0).repeat(ids.shape[0], 1)
    off_w = math_utils.quat_apply(tcp_quat_w, off_tcp)
    obj_pos_w = tcp_pos_w + off_w

    if force_upright:
        obj_quat_w = torch.tensor(upright_quat_w, device=tcp_pos_w.device, dtype=tcp_pos_w.dtype).unsqueeze(0).repeat(ids.shape[0], 1)
    elif align_with_hand:
        obj_quat_w = tcp_quat_w
    else:
        obj_quat_w = obj.data.root_quat_w[ids].clone()

    obj.write_root_pose_to_sim(torch.cat([obj_pos_w, obj_quat_w], dim=-1), env_ids=ids)
    obj.write_root_velocity_to_sim(torch.zeros((ids.shape[0], 6), device=env.device, dtype=tcp_pos_w.dtype), env_ids=ids)

def release_object_if_at_target(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    target_pos_rel: tuple[float, float, float] = (0.0, 0.55, 0.86),
    xy_thresh: float = 0.06,
    z_min: float = 0.75,
    upright_cos: float = 0.92,
    speed_thresh: float = 0.35,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """
    Interval-time: detach (attached=False) when object is near target XY, above z_min, upright-ish and slow.
    IMPORTANT: once attached=False, keep_object_welded_to_left_hand will stop overwriting pose => cup can drop.
    """
    _ensure_obj_attached(env)
    obj = env.scene[object_cfg.name]

    pos_rel = obj.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    lin = obj.data.root_lin_vel_w[env_ids]
    ang = obj.data.root_ang_vel_w[env_ids]

    tgt = torch.tensor(target_pos_rel, device=env.device, dtype=pos_rel.dtype).unsqueeze(0).repeat(env_ids.shape[0], 1)
    dxy = torch.norm(pos_rel[:, :2] - tgt[:, :2], dim=-1)
    xy_ok = dxy < float(xy_thresh)
    z_ok = pos_rel[:, 2] > float(z_min)
    slow = (torch.norm(lin, dim=-1) < float(speed_thresh)) & (torch.norm(ang, dim=-1) < float(speed_thresh))

    quat_w = obj.data.root_quat_w[env_ids]
    z_axis = math_utils.quat_apply(
        quat_w,
        torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=pos_rel.dtype).unsqueeze(0).repeat(env_ids.shape[0], 1),
    )
    upright = z_axis[:, 2] > float(upright_cos)

    attached = env._obj_attached[env_ids]
    do_release = xy_ok & z_ok & upright & slow & attached

    if bool(do_release.any()):
        rel_ids = env_ids[do_release]
        env._obj_attached[rel_ids] = False
        # zero velocities at release to reduce bouncing
        obj.write_root_velocity_to_sim(torch.zeros((rel_ids.shape[0], 6), device=env.device, dtype=pos_rel.dtype), env_ids=rel_ids)
