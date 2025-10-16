# hand_to_hand_env_cfg.py
from __future__ import annotations
from dataclasses import dataclass
from isaaclab.envs import ManagerBasedRLEnvCfg

@dataclass
class HandToHandEnvCfg(ManagerBasedRLEnvCfg):
    # 로봇/오브젝트 스폰, 시뮬 파라미터 등 공통 설정들…
    episode_length_s: float = 8.0
    decimation: int = 2

    def __post_init__(self):
        # === Observations ===
        self.observations.policy.enable = True
        self.observations.policy.append_term(
            name="lh_obj_rel",
            func="groot.tasks.manipulation.hand_to_hand.mdp.observations:obs_lh_obj_rel",
            clip=[-3.0, 3.0], scale=1.0
        )
        self.observations.policy.append_term(
            name="rh_obj_rel",
            func="groot.tasks.manipulation.hand_to_hand.mdp.observations:obs_rh_obj_rel",
            clip=[-3.0, 3.0], scale=1.0
        )
        self.observations.policy.append_term(
            name="hands_rel",
            func="groot.tasks.manipulation.hand_to_hand.mdp.observations:obs_hands_rel",
            clip=[-3.0, 3.0], scale=1.0
        )
        self.observations.policy.append_term(
            name="grasp_flags",
            func="groot.tasks.manipulation.hand_to_hand.mdp.observations:obs_grasp_flags",
            clip=[0.0, 1.0], scale=1.0
        )
        self.observations.privileged.enable = True
        self.observations.privileged.append_term(
            name="obj_vel",
            func="groot.tasks.manipulation.hand_to_hand.mdp.observations:obs_obj_linvel",
            clip=[-5.0, 5.0], scale=1.0
        )

        # === Rewards ===
        self.rewards.append_term(
            name="left_approach",
            func="groot.tasks.manipulation.hand_to_hand.mdp.rewards:rew_left_approach",
            scale=1.0
        )
        self.rewards.append_term(
            name="left_grasp",
            func="groot.tasks.manipulation.hand_to_hand.mdp.rewards:rew_left_grasp",
            scale=1.0
        )
        self.rewards.append_term(
            name="handover_align",
            func="groot.tasks.manipulation.hand_to_hand.mdp.rewards:rew_handover_align",
            scale=1.0
        )
        self.rewards.append_term(
            name="transfer",
            func="groot.tasks.manipulation.hand_to_hand.mdp.rewards:rew_transfer",
            scale=1.0
        )
        self.rewards.append_term(
            name="stability",
            func="groot.tasks.manipulation.hand_to_hand.mdp.rewards:rew_stability",
            scale=0.2
        )
        self.rewards.append_term(
            name="energy_penalty",
            func="groot.tasks.manipulation.hand_to_hand.mdp.rewards:rew_energy_penalty",
            scale=1.0
        )

        # === Terminations ===
        self.terminations.append_term(
            name="timeout",
            func="groot.tasks.manipulation.hand_to_hand.mdp.terminations:term_timeout"
        )
        self.terminations.append_term(
            name="dropped",
            func="groot.tasks.manipulation.hand_to_hand.mdp.terminations:term_dropped"
        )
        self.terminations.append_term(
            name="success",
            func="groot.tasks.manipulation.hand_to_hand.mdp.terminations:term_success"
        )
