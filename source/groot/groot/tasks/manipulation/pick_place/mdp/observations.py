from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def get_hand_state(
    env: ManagerBasedEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return left TCP pose (pos_rel(3) + quat(4))."""
    ee = env.scene.sensors[ee_cfg.name]
    tcp_pos_w = ee.data.target_pos_w[:, 0, :]
    tcp_quat_w = ee.data.target_quat_w[:, 0, :]
    tcp_pos_rel = tcp_pos_w - env.scene.env_origins
    return torch.cat([tcp_pos_rel, tcp_quat_w], dim=-1)


def object_obs(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return object state (pos_rel(3) + quat(4) + linvel(3) + angvel(3))."""
    obj = env.scene[object_cfg.name]
    pos_rel = obj.data.root_pos_w - env.scene.env_origins
    quat_w = obj.data.root_quat_w
    lin = obj.data.root_lin_vel_w
    ang = obj.data.root_ang_vel_w
    return torch.cat([pos_rel, quat_w, lin, ang], dim=-1)
