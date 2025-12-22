from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.assets import Articulation

from groot.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
        Get joint torques applied on the articulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    return torch.sum(asset.data.applied_torque[:, asset_cfg.joint_ids])


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def approach_object(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_tcp_pos = env.scene["left_ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene["object"].data.root_pos_w

    distance = torch.norm(object_pos - ee_tcp_pos, dim=-1, p=2)

    d0 = 0.2
    d_norm = distance / d0
    reward = torch.exp(-d_norm)  # 0 ~ 1

    return reward

def reach_to_object(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_tcp_pos = env.scene["left_ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene["object"].data.root_pos_w

    distance = torch.norm(object_pos - ee_tcp_pos, dim=-1, p=2)

    if not hasattr(env, "_prev_reach_distance"):
        env._prev_reach_distance = distance.clone()
        return torch.zeros_like(distance)

    prev_distance = env._prev_reach_distance

    progress = prev_distance - distance
    env._prev_reach_distance = distance.clone()

    reward = torch.clamp(progress, -0.05, 0.05)

    return reward


def success_reach_task_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_left_tcp_pos = env.scene["right_ee_frame"].data.target_pos_w[..., 0, :]
    ee_right_tcp_pos = env.scene["right_ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene["object"].data.root_pos_w

    # object x < 0 -> left, else -> right
    use_left = object_pos[..., 0] < 0.0                                          # (N,)

    ee_tcp_pos = torch.where(use_left.unsqueeze(-1), ee_left_tcp_pos, ee_right_tcp_pos)  # (N,3)

    distance = torch.norm(object_pos - ee_tcp_pos, dim=-1, p=2)                  # (N,)
    threshold = 0.03  # 3 cm

    reward = (distance < threshold).to(distance.dtype)                           # float tensor

    return reward


def success_grasp_task_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene["gripper_contact_sensor"]
    net_forces = contact_sensor.data.net_forces_w
    force_mag = torch.norm(net_forces, dim=-1)
    max_force = torch.max(force_mag, dim=-1)[0]

    # reward = (max_force > 1.0).float() & (max_force < 5.0).float()
    reward = ((max_force > 3)).float()
    return reward


def object_lifted_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    object_pos = env.scene["object"].data.root_pos_w
    z = object_pos[..., 2]

    table_height = 1.08
    target_lift = 0.10

    lift_amount = torch.clamp((z - table_height) / target_lift, 0.0, 1.0)

    reward = lift_amount
    return reward


def time_elapsed(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def approach_grasp_object(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_tcp_pos = env.scene["left_ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene["object"].data.root_pos_w

    distance = torch.norm(object_pos - ee_tcp_pos, dim=-1, p=2)

    d0 = 0.1
    d_norm = distance / d0
    dist_reward = torch.exp(-d_norm)  # 0~1

    contact_sensor: ContactSensor = env.scene["gripper_contact_sensor"]
    net_forces = contact_sensor.data.net_forces_w
    force_mag = torch.norm(net_forces, dim=-1)
    max_force = torch.max(force_mag, dim=-1)[0]

    grasp_flag = ((max_force > 0.5) & (max_force < 5.0)).float()

    # dist_reward: [0,1], grasp_flag: {0,1}
    reward = dist_reward + 1.5 * grasp_flag

    return reward

def push_object_negx_reward(env: ManagerBasedRLEnv, scale: float = 1.0) -> torch.Tensor:
    # object x position (N,)
    x = env.scene["object"].data.root_pos_w[..., 0]

    if not hasattr(env, "_prev_obj_x"):
        env._prev_obj_x = x.clone()

    progress = (env._prev_obj_x - x)

    env._prev_obj_x = x.clone()

    return scale * progress

def close_left_gripper_when_xneg_reward(
    env: ManagerBasedRLEnv,
    left_grip_action: torch.Tensor,   # (N,) or (N,1)
    scale: float = 0.5,
    x_gate: float = 0.0,
) -> torch.Tensor:
    x = env.scene["object"].data.root_pos_w[..., 0]  # (N,)
    cond = (x < x_gate)

    a = left_grip_action.squeeze(-1)
    close_cmd = torch.clamp(-a, min=0.0, max=1.0)

    return scale * cond.to(a.dtype) * close_cmd
