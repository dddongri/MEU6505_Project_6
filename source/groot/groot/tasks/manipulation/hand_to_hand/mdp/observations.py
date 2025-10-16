# observations.py
from __future__ import annotations
import torch

# 캐시에 공통 계산을 저장해 여러 term이 재사용할 수 있게 구성하는 걸 권장
def obs_lh_obj_rel(scene, env_ids, cache):
    # 왼손 EE와 물체의 월드 포즈(또는 위치) 취득
    lh_pos = scene["left_arm"].ee_pose_w[env_ids, :3]      # [N,3]
    obj_pos = scene["handover_object"].pose_w[env_ids, :3] # [N,3]
    rel = obj_pos - lh_pos
    cache["lh_obj_rel"] = rel
    return rel  # [N,3]

def obs_rh_obj_rel(scene, env_ids, cache):
    rh_pos = scene["right_arm"].ee_pose_w[env_ids, :3]
    obj_pos = scene["handover_object"].pose_w[env_ids, :3]
    rel = obj_pos - rh_pos
    cache["rh_obj_rel"] = rel
    return rel

def obs_hands_rel(scene, env_ids, cache):
    lh_pos = scene["left_arm"].ee_pose_w[env_ids, :3]
    rh_pos = scene["right_arm"].ee_pose_w[env_ids, :3]
    rel = rh_pos - lh_pos
    cache["hands_rel"] = rel
    return rel

def obs_grasp_flags(scene, env_ids, cache):
    # 그립 센서/접촉 기반 플래그. 없으면 집게-물체 거리 < thresh 로 더미 구성 가능
    lh_g = scene["left_gripper"].grasp_flag[env_ids].float().unsqueeze(-1)   # [N,1]
    rh_g = scene["right_gripper"].grasp_flag[env_ids].float().unsqueeze(-1)  # [N,1]
    out = torch.cat([lh_g, rh_g], dim=-1)  # [N,2]
    cache["grasp_flags"] = out
    return out

def obs_obj_linvel(scene, env_ids, cache):
    v = scene["handover_object"].lin_vel_w[env_ids]  # [N,3]
    return v

def obs_obj_angvel(scene, env_ids, cache):
    w = scene["handover_object"].ang_vel_w[env_ids]  # [N,3]
    return w
