from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils

from .mdp.actions import SymmetricDualIKAction
from .mdp.observations import (
    get_left_eef_pos,
    get_left_eef_quat,
    get_right_eef_pos,
    get_right_eef_quat,
)


class GR1T2HandToHandEnv(ManagerBasedRLEnv):
    """Debug IK-follow env.

    - ref는 오른손만 사용한다.
    - debug_ik_follow=True 이면 RL actions를 무시하고
      오른손 ref를 추종하는 delta를 만들고,
      왼손은 그 delta를 Y축만 반전해 대칭으로 추종한다.
    - action layout:
        [0:3] left pos delta
        [3:6] left rot delta
        [6:9] right pos delta
        [9:12] right rot delta
    """

    def step(self, actions):
        if self.cfg.debug_ik_follow:
            act_dim = self.action_manager.action.shape[1]
            new_actions = torch.zeros((self.num_envs, act_dim), device=self.device)
            if act_dim < 12:
                raise RuntimeError(f"[debug_ik_follow] Expected action_dim>=12 (pos+rot per hand) but got {act_dim}.")

            ref_r = self.extras.get("ref_right_ee_pos", None)
            ref_l = self.extras.get("ref_left_ee_pos", None)
            step_counter = self.extras.get("step_counter", None)
            ref_qr = self.extras.get("ref_right_ee_quat", None)

            if ref_r is None or step_counter is None:
                return super().step(new_actions)

            T = ref_r.shape[1]
            idx = torch.clamp(step_counter.long(), min=0, max=T - 1)
            arng = torch.arange(self.num_envs, device=self.device)

            cur_r = get_right_eef_pos(self)
            cur_l = get_left_eef_pos(self)
            cur_qr = get_right_eef_quat(self)
            cur_ql = get_left_eef_quat(self)
            root_pos_env = self.scene["robot"].data.root_pos_w - self.scene.env_origins
            root_quat = self.scene["robot"].data.root_quat_w

            tgt_r = ref_r[arng, idx]
            tgt_qr = cur_qr if ref_qr is None else ref_qr[arng, idx]

            T_ref = int(self.extras.get("T_ref", 0))
            phase_a = max(2, int(0.5 * T_ref))
            phase_b = max(2, T_ref - phase_a)
            t_hold_start = phase_a + phase_b

            cur_r_root = math_utils.quat_rotate_inverse(root_quat, cur_r - root_pos_env)
            cur_l_root = math_utils.quat_rotate_inverse(root_quat, cur_l - root_pos_env)

            plane_valid = self.extras.get("sagittal_plane_valid", None)
            plane_p0 = self.extras.get("sagittal_plane_p0_root", None)
            plane_n = self.extras.get("mirror_plane_n_root", None)
            if plane_valid is None or plane_valid.shape[0] != self.num_envs:
                plane_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
                self.extras["sagittal_plane_valid"] = plane_valid
            if plane_p0 is None or plane_p0.shape != (self.num_envs, 3):
                plane_p0 = torch.zeros((self.num_envs, 3), device=self.device)
                self.extras["sagittal_plane_p0_root"] = plane_p0
            if plane_n is None or plane_n.shape != (self.num_envs, 3):
                plane_n = torch.zeros((self.num_envs, 3), device=self.device)
                self.extras["mirror_plane_n_root"] = plane_n

            need_plane = ~plane_valid
            if torch.any(need_plane):
                robot = self.scene["robot"]
                names = robot.data.body_names

                pairs = [
                    ("right_hip_link", "left_hip_link"),
                    ("right_thigh_link", "left_thigh_link"),
                    ("right_upper_leg_link", "left_upper_leg_link"),
                    ("right_clavicle_link", "left_clavicle_link"),
                    ("right_shoulder_link", "left_shoulder_link"),
                    ("right_shoulder_pitch_link", "left_shoulder_pitch_link"),
                ]

                def _find_pair():
                    for r_name, l_name in pairs:
                        if r_name in names and l_name in names:
                            return names.index(r_name), names.index(l_name), (r_name, l_name)
                    return None, None, None

                idxR, idxL, used_pair = _find_pair()
                if idxR is None or idxL is None:
                    p0_root = torch.zeros_like(cur_r_root)
                    n_root = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, -1)
                    anchor_mode = "axis_fallback"
                else:
                    posR_w = robot.data.body_pos_w[:, idxR] - self.scene.env_origins
                    posL_w = robot.data.body_pos_w[:, idxL] - self.scene.env_origins
                    posR_root = math_utils.quat_rotate_inverse(root_quat, posR_w - root_pos_env)
                    posL_root = math_utils.quat_rotate_inverse(root_quat, posL_w - root_pos_env)

                    p0_root = 0.5 * (posL_root + posR_root)
                    up_world = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1)
                    up_root = math_utils.quat_apply_inverse(root_quat, up_world)
                    up_root = torch.nn.functional.normalize(up_root, dim=-1)
                    lr = posL_root - posR_root
                    lr_h = lr - torch.sum(lr * up_root, dim=-1, keepdim=True) * up_root
                    norm_h = torch.norm(lr_h, dim=-1, keepdim=True)
                    fallback = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand_as(lr_h)
                    lr_h = torch.where(norm_h < 1e-6, fallback, lr_h)
                    lr_h = torch.nn.functional.normalize(lr_h, dim=-1)
                    n_root = lr_h
                    anchor_mode = "paired_links"

                plane_p0[need_plane] = p0_root[need_plane]
                plane_n[need_plane] = n_root[need_plane]
                plane_valid[need_plane] = True

                if not self.extras.get("printed_axis_debug", False) and self.num_envs > 0:
                    env0 = 0
                    pair_str = used_pair if used_pair is not None else ("None", "None")
                    if idxR is None or idxL is None:
                        dR = dL = 0.0
                        sum_dl = 0.0
                        diff_abs = 0.0
                    else:
                        posR_w = robot.data.body_pos_w[:, idxR] - self.scene.env_origins
                        posL_w = robot.data.body_pos_w[:, idxL] - self.scene.env_origins
                        posR_root_dbg = math_utils.quat_rotate_inverse(root_quat, posR_w - root_pos_env)
                        posL_root_dbg = math_utils.quat_rotate_inverse(root_quat, posL_w - root_pos_env)
                        dR = float(torch.sum((posR_root_dbg[env0] - plane_p0[env0]) * plane_n[env0]).item())
                        dL = float(torch.sum((posL_root_dbg[env0] - plane_p0[env0]) * plane_n[env0]).item())
                        sum_dl = dR + dL
                        diff_abs = abs(dR) - abs(dL)
                    print(
                        f"[sagittal_axis_debug] env0 mode={anchor_mode} pair={pair_str} "
                        f"dR={dR:.6f} dL={dL:.6f} dR+dL={sum_dl:.6f} "
                        f"|dR|-|dL|={diff_abs:.6f} p0={plane_p0[env0].tolist()} n={plane_n[env0].tolist()}"
                    )
                    self.extras["printed_axis_debug"] = True

            tgt_r_root = math_utils.quat_rotate_inverse(root_quat, tgt_r - root_pos_env)
            bias = float(getattr(self.cfg, "debug_lateral_bias", 0.0))
            n_root = self.extras["mirror_plane_n_root"]
            p0_root = self.extras["sagittal_plane_p0_root"]
            tgt_r_root_biased = tgt_r_root - bias * n_root[arng]
            dist_tr = torch.sum((tgt_r_root_biased - p0_root[arng]) * n_root[arng], dim=-1, keepdim=True)
            tgt_l_root = tgt_r_root_biased - 2.0 * dist_tr * n_root[arng] + bias * n_root[arng]

            def _quat_from_matrix(R: torch.Tensor) -> torch.Tensor:
                m00 = R[..., 0, 0]
                m11 = R[..., 1, 1]
                m22 = R[..., 2, 2]
                trace = m00 + m11 + m22
                qw = torch.sqrt(torch.clamp(trace + 1.0, min=1e-6)) * 0.5
                qx = torch.zeros_like(qw)
                qy = torch.zeros_like(qw)
                qz = torch.zeros_like(qw)

                cond = trace > 0.0
                qx = torch.where(cond, (R[..., 2, 1] - R[..., 1, 2]) / (4.0 * qw), qx)
                qy = torch.where(cond, (R[..., 0, 2] - R[..., 2, 0]) / (4.0 * qw), qy)
                qz = torch.where(cond, (R[..., 1, 0] - R[..., 0, 1]) / (4.0 * qw), qz)

                cond1 = (~cond) & (m00 > m11) & (m00 > m22)
                qx = torch.where(cond1, torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-6)) * 0.5, qx)
                qy = torch.where(
                    cond1, (R[..., 0, 1] + R[..., 1, 0]) / (4.0 * torch.clamp(qx, min=1e-6)), qy
                )
                qz = torch.where(
                    cond1, (R[..., 0, 2] + R[..., 2, 0]) / (4.0 * torch.clamp(qx, min=1e-6)), qz
                )
                qw = torch.where(
                    cond1, (R[..., 2, 1] - R[..., 1, 2]) / (4.0 * torch.clamp(qx, min=1e-6)), qw
                )

                cond2 = (~cond) & (~cond1) & (m11 > m22)
                qy = torch.where(cond2, torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=1e-6)) * 0.5, qy)
                qx = torch.where(
                    cond2, (R[..., 0, 1] + R[..., 1, 0]) / (4.0 * torch.clamp(qy, min=1e-6)), qx
                )
                qz = torch.where(
                    cond2, (R[..., 1, 2] + R[..., 2, 1]) / (4.0 * torch.clamp(qy, min=1e-6)), qz
                )
                qw = torch.where(
                    cond2, (R[..., 0, 2] - R[..., 2, 0]) / (4.0 * torch.clamp(qy, min=1e-6)), qw
                )

                cond3 = (~cond) & (~cond1) & (~cond2)
                qz = torch.where(cond3, torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=1e-6)) * 0.5, qz)
                qx = torch.where(
                    cond3, (R[..., 0, 2] + R[..., 2, 0]) / (4.0 * torch.clamp(qz, min=1e-6)), qx
                )
                qy = torch.where(
                    cond3, (R[..., 1, 2] + R[..., 2, 1]) / (4.0 * torch.clamp(qz, min=1e-6)), qy
                )
                qw = torch.where(
                    cond3, (R[..., 1, 0] - R[..., 0, 1]) / (4.0 * torch.clamp(qz, min=1e-6)), qw
                )
                quat = torch.stack([qw, qx, qy, qz], dim=-1)
                return torch.nn.functional.normalize(quat, dim=-1)

            n_norm = n_root[arng] / torch.clamp(torch.norm(n_root[arng], dim=-1, keepdim=True), min=1e-6)
            M = torch.eye(3, device=self.device).expand(self.num_envs, 3, 3) - 2.0 * n_norm.unsqueeze(-1) * n_norm.unsqueeze(-2)
            qr_root = math_utils.quat_mul(
                math_utils.quat_conjugate(root_quat.unsqueeze(1)),
                tgt_qr.unsqueeze(1)
            ).squeeze(1)
            Rr = math_utils.matrix_from_quat(qr_root)
            Rl = torch.matmul(M, torch.matmul(Rr, M))
            ql_root = _quat_from_matrix(Rl)
            tgt_ql = math_utils.quat_mul(root_quat, ql_root)

            if not self.extras.get("printed_sym_debug", False) and self.num_envs > 0:
                dist_r = torch.sum((cur_r_root - p0_root[arng]) * n_root[arng], dim=-1, keepdim=True)
                dist_l = torch.sum((cur_l_root - p0_root[arng]) * n_root[arng], dim=-1, keepdim=True)
                dist_tl = torch.sum((tgt_l_root - p0_root[arng]) * n_root[arng], dim=-1, keepdim=True)
                env0 = 0
                print(
                    f"[sagittal_debug] env0 dist_r={float(dist_r[env0]):.6f} "
                    f"dist_l={float(dist_l[env0]):.6f} "
                    f"dist_r+dist_l={float((dist_r+dist_l)[env0]):.6f} "
                    f"dist_t_r={float(dist_tr[env0]):.6f} dist_t_l={float(dist_tl[env0]):.6f} "
                    f"n={n_root[env0].tolist()} p0={p0_root[env0].tolist()}"
                )
                self.extras["printed_sym_debug"] = True

            err_root = tgt_r_root - cur_r_root
            err_l_root = tgt_l_root - cur_l_root
            delta_r = torch.clamp(err_root * 3.0, min=-1.0, max=1.0)
            delta_l = torch.clamp(err_l_root * 3.0, min=-1.0, max=1.0)

            q_err_r = math_utils.quat_mul(tgt_qr, math_utils.quat_conjugate(cur_qr))
            q_err_l = math_utils.quat_mul(tgt_ql, math_utils.quat_conjugate(cur_ql))
            rotvec_r_world = math_utils.axis_angle_from_quat(q_err_r)
            rotvec_l_world = math_utils.axis_angle_from_quat(q_err_l)
            rotvec_r_root = math_utils.quat_apply_inverse(root_quat, rotvec_r_world)
            rotvec_l_root = math_utils.quat_apply_inverse(root_quat, rotvec_l_world)
            rot_gain = 3.0
            delta_rot_r = torch.clamp(rotvec_r_root * rot_gain, min=-1.0, max=1.0)
            delta_rot_l = torch.clamp(rotvec_l_root * rot_gain, min=-1.0, max=1.0)

            tol_pos = float(getattr(self.cfg, "debug_hold_pos_tol", 0.015))
            tol_rot = float(getattr(self.cfg, "debug_hold_rot_tol", 0.15))
            time_hold = (idx >= t_hold_start)
            pos_err_r = torch.norm(err_root, dim=-1)
            pos_err_l = torch.norm(err_l_root, dim=-1)
            rot_err_r = torch.norm(rotvec_r_root, dim=-1)
            rot_err_l = torch.norm(rotvec_l_root, dim=-1)
            done_r = (pos_err_r < tol_pos) & (rot_err_r < tol_rot)
            done_l = (pos_err_l < tol_pos) & (rot_err_l < tol_rot)
            if getattr(self.cfg, "debug_hold_use_error_gate", True):
                hold_mask_r = (time_hold & done_r).unsqueeze(-1)
                hold_mask_l = (time_hold & done_l).unsqueeze(-1)
            else:
                hold_mask_r = time_hold.unsqueeze(-1)
                hold_mask_l = time_hold.unsqueeze(-1)

            delta_r = torch.where(hold_mask_r, torch.zeros_like(delta_r), delta_r)
            delta_l = torch.where(hold_mask_l, torch.zeros_like(delta_l), delta_l)
            delta_rot_r = torch.where(hold_mask_r, torch.zeros_like(delta_rot_r), delta_rot_r)
            delta_rot_l = torch.where(hold_mask_l, torch.zeros_like(delta_rot_l), delta_rot_l)

            new_actions[:, 0:3] = delta_l
            new_actions[:, 3:6] = delta_rot_l
            new_actions[:, 6:9] = delta_r
            new_actions[:, 9:12] = delta_rot_r

            term = None
            cand_sources = (
                getattr(self.action_manager, "_action_terms", None),
                getattr(self.action_manager, "action_terms", None),
                getattr(self.action_manager, "_terms", None),
            )
            for cand_list in cand_sources:
                if term is not None or cand_list is None:
                    continue
                try:
                    iterable = cand_list.values() if isinstance(cand_list, dict) else cand_list
                    for cand in iterable:
                        if isinstance(cand, SymmetricDualIKAction):
                            term = cand
                            break
                except TypeError:
                    continue

            ref_left_grip = self.extras.get("ref_left_grip", None)
            ref_right_grip = self.extras.get("ref_right_grip", None)
            lg_idx = None
            rg_idx = None
            if term is not None:
                lg_idx = term.left_grasp_action_index
                rg_idx = term.right_grasp_action_index
            elif act_dim >= 2:
                lg_idx = act_dim - 2
                rg_idx = act_dim - 1
                if not self.extras.get("warn_missing_term_once", False):
                    term_list = []
                    for cand_list in cand_sources:
                        if cand_list is None:
                            continue
                        try:
                            iterable = cand_list.values() if isinstance(cand_list, dict) else cand_list
                            term_list.extend([type(c).__name__ for c in iterable])
                        except TypeError:
                            pass
                    print(
                        f"[debug_ik_follow] SymmetricDualIKAction not found, falling back to last two action dims "
                        f"(act_dim={act_dim}, terms={term_list})"
                    )
                    self.extras["warn_missing_term_once"] = True

            if lg_idx is not None and rg_idx is not None and isinstance(ref_left_grip, torch.Tensor) and isinstance(ref_right_grip, torch.Tensor):
                lg_idx = int(lg_idx)
                rg_idx = int(rg_idx)
                if lg_idx < new_actions.shape[1] and rg_idx < new_actions.shape[1]:
                    new_actions[:, lg_idx] = ref_left_grip[arng, idx]
                    new_actions[:, rg_idx] = ref_right_grip[arng, idx]
                    self.extras.setdefault("last_applied_grip_l", torch.zeros(self.num_envs, device=self.device))
                    self.extras.setdefault("last_applied_grip_r", torch.zeros(self.num_envs, device=self.device))
                    self.extras["last_applied_grip_l"][arng] = new_actions[:, lg_idx]
                    self.extras["last_applied_grip_r"][arng] = new_actions[:, rg_idx]

            return super().step(new_actions)

        return super().step(actions)
