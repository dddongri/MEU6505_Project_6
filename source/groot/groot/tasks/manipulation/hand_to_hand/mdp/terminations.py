# terminations.py
from __future__ import annotations
import torch

def term_timeout(scene, env_ids, cache):
    # 엔브 별 스텝 카운터와 max_step은 보통 env에 있음
    t = scene["timesteps"][env_ids]
    max_t = scene["max_timesteps"]
    return (t >= max_t)

def term_dropped(scene, env_ids, cache):
    # 물체가 바닥보다 아래로 떨어지면 종료
    z = scene["handover_object"].pose_w[env_ids, 2]
    return (z < scene["floor_z"] + 0.02)

def term_success(scene, env_ids, cache):
    # 오른손만 쥐고 있고, 왼손은 놓은 상태
    g = cache.get("grasp_flags")
    if g is None:
        from .observations import obs_grasp_flags
        g = obs_grasp_flags(scene, env_ids, cache)
    lh, rh = g[:, 0], g[:, 1]
    return ((rh > 0.5) & (lh < 0.5))
