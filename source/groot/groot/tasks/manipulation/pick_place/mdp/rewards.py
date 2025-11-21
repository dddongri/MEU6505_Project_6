from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def approach_object(env: ManagerBasedRLEnv) -> torch.Tensor:
    r"""Reward the robot for reaching the drawer handle using inverse-square law.

    It uses a piecewise function to reward the robot for reaching the handle.

    .. math::

        reward = \begin{cases}
            2 * (1 / (1 + distance^2))^2 & \text{if } distance \leq threshold \\
            (1 / (1 + distance^2))^2 & \text{otherwise}
        \end{cases}

    """
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
    
    # print("EE TCP Position and Object Position:")
    # print(ee_tcp_pos)
    # print(object_pos)

    # Compute the distance of the end-effector to the handle
    distance = torch.norm(object_pos - ee_tcp_pos, dim=-1, p=2)

    # Reward the robot for reaching the handle
    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)
    return reward * 100


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