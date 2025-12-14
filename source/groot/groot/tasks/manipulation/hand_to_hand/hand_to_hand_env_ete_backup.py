from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils

from .mdp.observations import get_left_eef_pos, get_right_eef_pos


class GR1T2HandToHandEnv(ManagerBasedRLEnv):
    """Debug IK-follow env.

    - ref는 오른손만 사용한다.
    - debug_ik_follow=True 이면 RL actions를 무시하고
      오른손 ref를 추종하는 delta를 만들고,
      왼손은 그 delta를 Y축만 반전해 대칭으로 추종한다.
    - action layout:
        [0:3] left pos delta
        [3:6] right pos delta
    """

    def step(self, actions):
        # if self.cfg.debug_ik_follow:
        #     act_dim = self.action_manager.action.shape[1]
        #     new_actions = torch.zeros((self.num_envs, act_dim), device=self.device)

        #     ref_r = self.extras.get("ref_right_ee_pos", None)
        #     step_counter = self.extras.get("step_counter", None)

        #     if getattr(self.cfg, "log_traj_csv", False):
        #         # Fallback inline call to ensure per-step logging even if event ordering skips
        #         from . import mdp
        #         mdp.log_traj_step_ref_cur(self)

        #     if ref_r is None or step_counter is None:
        #         return super().step(new_actions)

        #     T = ref_r.shape[1]
        #     idx = torch.clamp(step_counter.long(), min=0, max=T - 1)
        #     arng = torch.arange(self.num_envs, device=self.device)

        #     cur_r = get_right_eef_pos(self)
        #     cur_l = get_left_eef_pos(self)
        #     root_quat = self.scene["robot"].data.root_quat_w

        #     tgt_r = ref_r[arng, idx]

        #     err_world = tgt_r - cur_r
        #     err_root = math_utils.quat_rotate_inverse(root_quat, err_world)
        #     delta_r = torch.clamp(err_root * 3.0, min=-1.0, max=1.0)

        #     delta_l = delta_r.clone()
        #     delta_l[:, 1] *= -1.0

        #     new_actions[:, 0:3] = delta_l
        #     new_actions[:, 3:6] = delta_r

        #     return super().step(new_actions)

        return super().step(actions)
