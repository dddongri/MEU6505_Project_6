# rewards.py
from __future__ import annotations
import torch

def rew_left_approach(scene, env_ids, cache):
    rel = cache.get("lh_obj_rel")
    if rel is None:
        # 관측 term이 먼저 호출되지 않는 경우 대비
        from .observations import obs_lh_obj_rel
        rel = obs_lh_obj_rel(scene, env_ids, cache)
    # 물체와 왼손 거리 줄이기 (음수 거리)
    return -torch.norm(rel, dim=-1)

def rew_left_grasp(scene, env_ids, cache):
    # 왼손 그립 성공 보너스
    g = cache.get("grasp_flags")
    if g is None:
        from .observations import obs_grasp_flags
        g = obs_grasp_flags(scene, env_ids, cache)
    lh = g[:, 0]
    return lh * 2.0  # 스케일은 env_cfg에서 최종 조정

def rew_handover_align(scene, env_ids, cache):
    # 두 손 사이 정렬(가까울수록 보상 ↑)
    rel = cache.get("hands_rel")
    if rel is None:
        from .observations import obs_hands_rel
        rel = obs_hands_rel(scene, env_ids, cache)
    return -torch.norm(rel, dim=-1)

def rew_transfer(scene, env_ids, cache):
    # 오른손 그립 on & 왼손 그립 off 상태를 보상
    g = cache.get("grasp_flags")
    if g is None:
        from .observations import obs_grasp_flags
        g = obs_grasp_flags(scene, env_ids, cache)
    lh, rh = g[:, 0], g[:, 1]
    # 오른손만 쥔 상태(핸드오버 완료)에 보너스
    return ((rh > 0.5) & (lh < 0.5)).float() * 4.0

def rew_stability(scene, env_ids, cache):
    # 물체 속도 안정성 (속도 낮을수록 보상)
    v = scene["handover_object"].lin_vel_w[env_ids]
    return -torch.norm(v, dim=-1)

def rew_energy_penalty(scene, env_ids, cache):
    # 조인트 토크/액션 크기 패널티 예시 (필요시 수정)
    act = scene["actions"][env_ids]  # 정책이 준 액션 캐싱해뒀다고 가정
    return -0.001 * torch.norm(act, dim=-1)
