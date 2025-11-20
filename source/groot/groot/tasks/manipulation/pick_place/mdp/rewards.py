from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =====================================================================================
#  기존 보상 함수 (원본 그대로 유지)
# =====================================================================================
def approach_object(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins

    distance = torch.norm(object_pos - ee_tcp_pos, dim=-1, p=2)

    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)
    return reward * 100


def success_reach_task_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins

    distance = torch.norm(object_pos - ee_tcp_pos, dim=-1, p=2)
    threshold = 0.03

    return torch.where(
        distance < threshold,
        torch.ones_like(distance),
        torch.zeros_like(distance)
    )


def success_grasp_task_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene["gripper_contact_sensor"]
    contact_data = contact_sensor.get_contact_data()

    return torch.where(
        contact_data.sum(dim=-1) > 0,
        torch.ones_like(contact_data.sum(dim=-1)),
        torch.zeros_like(contact_data.sum(dim=-1))
    )


# =====================================================================================
#   ★ 추가된 Pick → Place 기능
# =====================================================================================

# 1) Pick 상태 판정
def is_holding_object(env: ManagerBasedRLEnv) -> torch.Tensor:
    object_pos = env.scene["object"].data.root_pos_w
    eef_pos = env.scene["robot"].data.body_pos_w[
        :, env.scene["robot"].body_names.index("left_hand_pitch_link")
    ]

    dist = torch.norm(object_pos - eef_pos, dim=-1)
    return dist < 0.10     # 10cm 이내면 잡고 있다고 판단


# 2) Place 타겟 위치
PLACE_TARGET = torch.tensor([0.15, 0.55, 1.05])


# 3) Place 이동 보상
def place_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    holding = is_holding_object(env)
    if not torch.any(holding):
        return torch.zeros(env.num_envs, device=env.device)

    target = PLACE_TARGET.to(env.device)
    eef_pos = env.scene["robot"].data.body_pos_w[
        :, env.scene["robot"].body_names.index("left_hand_pitch_link")
    ]

    dist = torch.norm(eef_pos - target, dim=-1)

    # 가까워질수록 reward 증가
    return torch.exp(-6 * dist) * holding.float()


# 4) Place 성공 보상 + 자동 OPEN
def place_success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    holding = is_holding_object(env)
    if not torch.any(holding):
        return torch.zeros(env.num_envs, device=env.device)

    target = PLACE_TARGET.to(env.device)
    eef_pos = env.scene["robot"].data.body_pos_w[
        :, env.scene["robot"].body_names.index("left_hand_pitch_link")
    ]

    dist = torch.norm(eef_pos - target, dim=-1)
    reached = dist < 0.06

    success = reached & holding

    # ★ 도착 시 자동으로 손 OPEN (놓기)
    if torch.any(success):
        env.actions.gripper_action.current_command[:] = 0.0

    return success.float()














def success_reach_task_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    r"""Reward the robot for successfully placing the object at the target location.

    The reward is given when the object is within a certain threshold distance from the target location.
    """
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins

    # Compute the distance of the object to the target location
    distance = torch.norm(object_pos - ee_tcp_pos, dim=-1, p=2)

    # Define a threshold distance for success
    threshold = 0.03  # 3 cm

    # Reward the robot for successfully placing the object
    reward = torch.where(distance < threshold, torch.ones_like(distance), torch.zeros_like(distance))
    return reward

def success_grasp_task_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    r"""Reward the robot for successfully grasping the object.

    The reward is given when the contact sensor detects contact between the gripper and the object.
    """
    contact_sensor: ContactSensor = env.scene["gripper_contact_sensor"]
    contact_data = contact_sensor.get_contact_data()

    # Reward the robot for successfully grasping the object
    reward = torch.where(contact_data.sum(dim=-1) > 0, torch.ones_like(contact_data.sum(dim=-1)), torch.zeros_like(contact_data.sum(dim=-1)))
    return reward