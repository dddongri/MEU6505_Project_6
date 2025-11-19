# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab.assets import Articulation, DeformableObject, RigidObject


def reset_object_poses_nut_pour(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    sorting_beaker_cfg: SceneEntityCfg = SceneEntityCfg("sorting_beaker"),
    factory_nut_cfg: SceneEntityCfg = SceneEntityCfg("factory_nut"),
    sorting_bowl_cfg: SceneEntityCfg = SceneEntityCfg("sorting_bowl"),
    sorting_scale_cfg: SceneEntityCfg = SceneEntityCfg("sorting_scale"),
):
    """Reset the asset root states to a random position and orientation uniformly within the given ranges.

    Args:
        env: The RL environment instance.
        env_ids: The environment IDs to reset the object poses for.
        sorting_beaker_cfg: The configuration for the sorting beaker asset.
        factory_nut_cfg: The configuration for the factory nut asset.
        sorting_bowl_cfg: The configuration for the sorting bowl asset.
        sorting_scale_cfg: The configuration for the sorting scale asset.
        pose_range: The dictionary of pose ranges for the objects. Keys are
                    ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.
    """
    # extract the used quantities (to enable type-hinting)
    sorting_beaker = env.scene[sorting_beaker_cfg.name]
    factory_nut = env.scene[factory_nut_cfg.name]
    sorting_bowl = env.scene[sorting_bowl_cfg.name]
    sorting_scale = env.scene[sorting_scale_cfg.name]

    # get default root state
    sorting_beaker_root_states = sorting_beaker.data.default_root_state[env_ids].clone()
    factory_nut_root_states = factory_nut.data.default_root_state[env_ids].clone()
    sorting_bowl_root_states = sorting_bowl.data.default_root_state[env_ids].clone()
    sorting_scale_root_states = sorting_scale.data.default_root_state[env_ids].clone()

    # get pose ranges
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=sorting_beaker.device)

    # randomize sorting beaker and factory nut together
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=sorting_beaker.device
    )
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    positions_sorting_beaker = (
        sorting_beaker_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    )
    positions_factory_nut = factory_nut_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_sorting_beaker = math_utils.quat_mul(sorting_beaker_root_states[:, 3:7], orientations_delta)
    orientations_factory_nut = math_utils.quat_mul(factory_nut_root_states[:, 3:7], orientations_delta)

    # randomize sorting bowl
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=sorting_beaker.device
    )
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    positions_sorting_bowl = sorting_bowl_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_sorting_bowl = math_utils.quat_mul(sorting_bowl_root_states[:, 3:7], orientations_delta)

    # randomize scorting scale
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=sorting_beaker.device
    )
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    positions_sorting_scale = sorting_scale_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_sorting_scale = math_utils.quat_mul(sorting_scale_root_states[:, 3:7], orientations_delta)

    # set into the physics simulation
    sorting_beaker.write_root_pose_to_sim(
        torch.cat([positions_sorting_beaker, orientations_sorting_beaker], dim=-1), env_ids=env_ids
    )
    factory_nut.write_root_pose_to_sim(
        torch.cat([positions_factory_nut, orientations_factory_nut], dim=-1), env_ids=env_ids
    )
    sorting_bowl.write_root_pose_to_sim(
        torch.cat([positions_sorting_bowl, orientations_sorting_bowl], dim=-1), env_ids=env_ids
    )
    sorting_scale.write_root_pose_to_sim(
        torch.cat([positions_sorting_scale, orientations_sorting_scale], dim=-1), env_ids=env_ids
    )



def reset_object_in_hand(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    reset environments
    object must be in robot's left hand.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
 




OBJECT_ASSET_CFG = SceneEntityCfg("Object") # 예시: 장면에 정의된 물체 이름

# 로봇의 DOF 인덱스 (USD 또는 Articulation 애셋에서 확인 필요)
RIGHT_ARM_DOF_INDICES = [10, 11, 12, 13, 14, 15, 16] # 예시 인덱스
RIGHT_GRIPPER_DOF_INDICES = [17, 18] # 예시 인덱스

# 그리퍼가 닫혔을 때의 DOF 값
GRIPPER_CLOSED_DOF_VALUE = 1.0 # 예시 값

# -------------------------------------------------

def reset_robot_and_grasp_object_with_ik(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]], # 손의 타겟 포즈 범위를 정의
    velocity_range: dict[str, tuple[float, float]], # 루트 속도 범위
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    IK를 사용하여 로봇의 오른쪽 손을 랜덤 포즈로 설정하고,
    해당 위치에 물체를 쥔 상태로 초기화합니다.
    """
    # -- 1. 애셋 가져오기 --
    robot: Articulation = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[OBJECT_ASSET_CFG.name]
    num_resets = len(env_ids)

    # -- 2. 로봇 루트 상태 재설정 (선택 사항, 기존 코드 활용) --
    # (기존 코드와 동일... 루트 상태를 랜덤화하고 write_root_pose_to_sim 호출)
    # ... (생략) ...
    # 이 예시에서는 루트는 기본값으로 둔다고 가정합니다.
    root_states = robot.data.default_root_state[env_ids].clone()
    root_states[:, 0:3] += env.scene.env_origins[env_ids]
    robot.write_root_pose_to_sim(root_states[:, 0:7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_states[:, 7:13], env_ids=env_ids)


    # -- 3. 랜덤 핸드 타겟 포즈 생성 --
    # pose_range를 사용하여 손의 '타겟' 위치와 방향 샘플링
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=robot.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_resets, 6), device=robot.device)

    # 타겟 포즈 (환경 원점 기준)
    target_positions = rand_samples[:, 0:3] + env.scene.env_origins[env_ids]
    target_orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    # 기본 손 방향이 있다면 곱해줄 수 있습니다. 여기서는 단순화.
    target_orientations = target_orientations_delta 

    # -- 4. IK 계산 (이 부분은 IK 솔버 설정에 따라 달라짐) --
    # *******************************************************************
    # * 중요: 이 부분은 'env'에 IK 솔버가 설정되어 있다고 가정한 의사 코드입니다.
    # * 실제로는 env.ik_solver.set_targets(...) 와 env.ik_solver.compute() 등을
    # * 사용하여 목표 DOF를 계산해야 합니다.
    # *
    # * (의사 코드)
    # * target_dof_pos = env.ik_solver.compute_inverse_kinematics(
    # * target_positions=target_positions,
    # * target_orientations=target_orientations,
    # * env_ids=env_ids
    # * )
    # *******************************************************************
    
    # (임시: IK가 없다고 가정하고, 대신 랜덤 '관절' 각도를 설정하는 방식)
    # 이 방식은 부정확하지만, IK 설정 전 테스트용으로 사용할 수 있습니다.
    # pose_range를 관절 각도 범위로 해석해야 합니다.
    # ... 이 방식은 물체 위치를 정확히 잡기 어려워 권장하지 않습니다.
    
    # IK가 계산되었다고 가정하고 다음 단계 진행

    # -- 5. 로봇 DOF 상태 설정 --
    dof_pos = robot.data.default_dof_pos[env_ids].clone()
    dof_vel = robot.data.default_dof_vel[env_ids].clone() # 0으로 설정

    # (의사 코드) 계산된 IK 결과를 오른팔 인덱스에 적용
    # dof_pos[:, RIGHT_ARM_DOF_INDICES] = target_dof_pos 
    
    # (임시) IK 대신 랜덤 조인트 각도 설정 (pose_range를 조인트 범위로 사용)
    # 이 경우 target_positions/orientations는 부정확해집니다.
    # 여기서는 IK를 사용했다는 가정 하에 계속 진행합니다.
    # ... (IK 계산 결과가 dof_pos에 적용되었다고 가정) ...


    # 그리퍼를 '닫힘' 상태로 설정
    dof_pos[:, RIGHT_GRIPPER_DOF_INDICES] = GRIPPER_CLOSED_DOF_VALUE
    
    # DOF 상태 쓰기
    robot.write_dof_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

    # -- 6. 물체 텔레포트 --
    # 물체를 IK 타겟 포즈(target_positions, target_orientations)로 이동
    obj_root_state = obj.data.default_root_state[env_ids].clone()
    obj_root_state[:, 0:3] = target_positions
    obj_root_state[:, 3:7] = target_orientations
    obj_root_state[:, 7:13] = 0.0 # 속도는 0

    obj.write_root_pose_to_sim(obj_root_state[:, 0:7], env_ids=env_ids)
    obj.write_root_velocity_to_sim(obj_root_state[:, 7:13], env_ids=env_ids)








    return 0