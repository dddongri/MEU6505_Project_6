from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# 1) 물체에 가까이 가는 리워드 (dense, exp 기반)
def approach_object(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene["object"].data.root_pos_w

    distance = torch.norm(object_pos - ee_tcp_pos, dim=-1, p=2)

    # workspace scale에 맞게 조절 (예: 20cm 정도를 기준)
    # d_norm = d / d0, reward = exp(-d_norm)
    d0 = 0.2  # 20cm 기준
    d_norm = distance / d0
    reward = torch.exp(-d_norm)  # 0 ~ 1

    return reward


# 2) 거리가 줄어드는 "progress" 리워드 (선택, 사용하면 reset 처리 필수)
def reach_to_object(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene["object"].data.root_pos_w

    distance = torch.norm(object_pos - ee_tcp_pos, dim=-1, p=2)

    # 에피소드 시작 스텝(혹은 reset된 env)에서는 progress=0으로
    if not hasattr(env, "_prev_reach_distance"):
        env._prev_reach_distance = distance.clone()
        return torch.zeros_like(distance)

    prev_distance = env._prev_reach_distance

    progress = prev_distance - distance  # 가까워지면 +, 멀어지면 -
    env._prev_reach_distance = distance.clone()

    # 너무 큰 값 튀는 거 방지용 (선택)
    reward = torch.clamp(progress, -0.05, 0.05)

    return reward


# 3) "reach 성공" 리워드 – 임계 거리 안에 들어오면 보너스
def success_reach_task_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene["object"].data.root_pos_w

    distance = torch.norm(object_pos - ee_tcp_pos, dim=-1, p=2)
    threshold = 0.05  # 5 cm

    reward = (distance < threshold).float()
    return reward


# 4) grasp 성공 리워드 – contact sensor 기반 (sparse + 큰 보너스)
def success_grasp_task_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene["gripper_contact_sensor"]
    net_forces = contact_sensor.data.net_forces_w
    force_mag = torch.norm(net_forces, dim=-1)
    max_force = torch.max(force_mag, dim=-1)[0]

    # 일정 이상의 접촉이 있으면 grasp 성공으로 간주
    # reward = (max_force > 1.0).float() & (max_force < 5.0).float()
    reward = ((max_force > 0.5) & (max_force < 5.0)).float()
    return reward


# 5) object lift 리워드 – 높이에 따라 shaping + 성공 보너스
def object_lifted_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    object_pos = env.scene["object"].data.root_pos_w
    z = object_pos[..., 2]

    # 테이블 높이 (예시), 환경에 맞게 조정
    table_height = 1.08
    target_lift = 0.10  # 테이블에서 10cm 들어 올리면 full reward

    # 0 ~ 1 사이로 normalize
    lift_amount = torch.clamp((z - table_height) / target_lift, 0.0, 1.0)

    # 좀 더 강하게 주고 싶으면 exp나 제곱 사용도 가능
    # reward = lift_amount ** 2
    reward = lift_amount
    return reward


# 6) 시간 페널티 (그대로, weight만 - 붙여서 사용)
def time_elapsed(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


# 7) grasp를 위한 통합 shaping 리워드 (거리 + contact 보너스)
def approach_grasp_object(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene["object"].data.root_pos_w

    distance = torch.norm(object_pos - ee_tcp_pos, dim=-1, p=2)

    # 1) 거리 기반 (exp)
    d0 = 0.1  # grasp는 더 근접해야 하니 10cm 기준
    d_norm = distance / d0
    dist_reward = torch.exp(-d_norm)  # 0~1

    # 2) contact 기반 보너스
    contact_sensor: ContactSensor = env.scene["gripper_contact_sensor"]
    net_forces = contact_sensor.data.net_forces_w
    force_mag = torch.norm(net_forces, dim=-1)
    max_force = torch.max(force_mag, dim=-1)[0]

    # grasp_flag = (max_force > 0.5).float() & (max_force < 5).float()  # 약간 더 낮은 threshold로 "접촉 시작" 감지
    grasp_flag = ((max_force > 0.5) & (max_force < 5.0)).float()

    # 3) 합치기: 가까이 갈수록 +, 접촉 있으면 추가 보너스
    # dist_reward: [0,1], grasp_flag: {0,1}
    reward = dist_reward + 1.5 * grasp_flag

    return reward
