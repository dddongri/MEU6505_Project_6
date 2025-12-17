from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _ensure_obj_attached(env: "ManagerBasedEnv") -> None:
    if not hasattr(env, "_obj_attached"):
        env._obj_attached = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)


def _target_pos_rel_const(env: "ManagerBasedEnv", target_pos_rel=(0.0, 0.55, 0.86)) -> torch.Tensor:
    return torch.tensor(target_pos_rel, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)


def reward_reach_target_xy(
    env: "ManagerBasedEnv",
    target_pos_rel: tuple[float, float, float] = (0.0, 0.55, 0.86),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Smooth shaping reward using only XY distance to target."""
    obj = env.scene[object_cfg.name]
    pos_rel = obj.data.root_pos_w - env.scene.env_origins
    tgt = _target_pos_rel_const(env, target_pos_rel)
    dxy = torch.norm(pos_rel[:, :2] - tgt[:, :2], dim=-1)
    return 1.0 / (1.0 + dxy * dxy)


def reward_upright(
    env: "ManagerBasedEnv",
    upright_cos: float = 0.92,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward cup being upright (local +Z aligned with world +Z)."""
    obj = env.scene[object_cfg.name]
    quat_w = obj.data.root_quat_w
    z_axis = math_utils.quat_apply(
        quat_w,
        torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=quat_w.dtype).unsqueeze(0).repeat(env.num_envs, 1),
    )
    # map z_cos from [-1,1] to [0,1] (and clamp)
    z_cos = torch.clamp(z_axis[:, 2], -1.0, 1.0)
    return torch.clamp((z_cos - upright_cos) / (1.0 - upright_cos + 1e-6), 0.0, 1.0)


def reward_place_success_upright(
    env: "ManagerBasedEnv",
    target_pos_rel: tuple[float, float, float] = (0.0, 0.55, 0.86),
    xy_thresh: float = 0.06,
    z_band: tuple[float, float] = (0.72, 1.05),
    upright_cos: float = 0.92,
    speed_thresh: float = 0.25,
    require_released: bool = True,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Sparse success reward: near target XY, on table height band, upright, slow, and released."""
    _ensure_obj_attached(env)
    obj = env.scene[object_cfg.name]

    pos_rel = obj.data.root_pos_w - env.scene.env_origins
    lin = obj.data.root_lin_vel_w
    ang = obj.data.root_ang_vel_w
    tgt = _target_pos_rel_const(env, target_pos_rel)

    dxy = torch.norm(pos_rel[:, :2] - tgt[:, :2], dim=-1)
    xy_ok = dxy < float(xy_thresh)
    z_ok = (pos_rel[:, 2] > float(z_band[0])) & (pos_rel[:, 2] < float(z_band[1]))
    slow = (torch.norm(lin, dim=-1) < float(speed_thresh)) & (torch.norm(ang, dim=-1) < float(speed_thresh))

    quat_w = obj.data.root_quat_w
    z_axis = math_utils.quat_apply(
        quat_w,
        torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=quat_w.dtype).unsqueeze(0).repeat(env.num_envs, 1),
    )
    upright = z_axis[:, 2] > float(upright_cos)

    if require_released:
        released = ~env._obj_attached
        cond = xy_ok & z_ok & upright & slow & released
    else:
        cond = xy_ok & z_ok & upright & slow

    return cond.float()
