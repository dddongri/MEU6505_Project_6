from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import os
import csv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

REF_RATIO = 0.6

def reset_episode_extras(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """Reset per-episode buffers to minimal defaults."""
    device = env.device
    n = env.num_envs
    ids = torch.arange(n, device=device) if env_ids is None else torch.as_tensor(env_ids, device=device)

    # step counter (optional)
    if "step_counter" not in env.extras or env.extras["step_counter"].shape[0] != n:
        env.extras["step_counter"] = torch.zeros(n, device=device, dtype=torch.long)
    else:
        env.extras["step_counter"][ids] = 0

    # allocate reference buffers to the full episode horizon (T is episode horizon)
    dt = float(env.cfg.sim.dt)
    decim = int(env.cfg.decimation)
    T = max(10, int(env.cfg.episode_length_s / (dt * decim)))

    ref_pos = env.extras.get("ref_right_ee_pos")
    if ref_pos is None or ref_pos.shape != (n, T, 3):
        env.extras["ref_right_ee_pos"] = torch.zeros((n, T, 3), device=device)
    else:
        env.extras["ref_right_ee_pos"][ids].zero_()

    ref_quat = env.extras.get("ref_right_ee_quat")
    if ref_quat is None or ref_quat.shape != (n, T, 4):
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        env.extras["ref_right_ee_quat"] = identity.expand(n, T, 4).clone()
    else:
        env.extras["ref_right_ee_quat"][ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(ids.numel(), T, 1)

    ref_pos_l = env.extras.get("ref_left_ee_pos")
    if ref_pos_l is None or ref_pos_l.shape != (n, T, 3):
        env.extras["ref_left_ee_pos"] = torch.zeros((n, T, 3), device=device)
    else:
        env.extras["ref_left_ee_pos"][ids].zero_()

    ref_quat_l = env.extras.get("ref_left_ee_quat")
    if ref_quat_l is None or ref_quat_l.shape != (n, T, 4):
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        env.extras["ref_left_ee_quat"] = identity.expand(n, T, 4).clone()
    else:
        env.extras["ref_left_ee_quat"][ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(ids.numel(), T, 1)

    # grip reference buffers (float)
    ref_left_grip = env.extras.get("ref_left_grip")
    if ref_left_grip is None or ref_left_grip.shape != (n, T):
        env.extras["ref_left_grip"] = torch.zeros((n, T), device=device, dtype=torch.float)
    else:
        env.extras["ref_left_grip"][ids].zero_()

    ref_right_grip = env.extras.get("ref_right_grip")
    if ref_right_grip is None or ref_right_grip.shape != (n, T):
        env.extras["ref_right_grip"] = torch.ones((n, T), device=device, dtype=torch.float)
    else:
        env.extras["ref_right_grip"][ids] = 1.0

    # mirror anchors for relative left/right reference mirroring
    if "mirror_anchor_r_root" not in env.extras or env.extras["mirror_anchor_r_root"].shape != (n, 3):
        env.extras["mirror_anchor_r_root"] = torch.zeros((n, 3), device=device)
    if "mirror_anchor_l_root" not in env.extras or env.extras["mirror_anchor_l_root"].shape != (n, 3):
        env.extras["mirror_anchor_l_root"] = torch.zeros((n, 3), device=device)
    if "mirror_anchor_valid" not in env.extras or env.extras["mirror_anchor_valid"].shape[0] != n:
        env.extras["mirror_anchor_valid"] = torch.zeros(n, device=device, dtype=torch.bool)
    else:
        env.extras["mirror_anchor_valid"][ids] = False
    env.extras["mirror_anchor_r_root"][ids] = 0.0
    env.extras["mirror_anchor_l_root"][ids] = 0.0
    if "mirror_plane_n_root" not in env.extras or env.extras["mirror_plane_n_root"].shape != (n, 3):
        env.extras["mirror_plane_n_root"] = torch.zeros((n, 3), device=device)
    if "mirror_plane_valid" not in env.extras or env.extras["mirror_plane_valid"].shape[0] != n:
        env.extras["mirror_plane_valid"] = torch.zeros(n, device=device, dtype=torch.bool)
    else:
        env.extras["mirror_plane_valid"][ids] = False
    env.extras["mirror_plane_n_root"][ids] = 0.0
    if "sagittal_axis_idx" not in env.extras or env.extras["sagittal_axis_idx"].shape[0] != n:
        env.extras["sagittal_axis_idx"] = torch.zeros(n, device=device, dtype=torch.long)
    if "sagittal_axis_valid" not in env.extras or env.extras["sagittal_axis_valid"].shape[0] != n:
        env.extras["sagittal_axis_valid"] = torch.zeros(n, device=device, dtype=torch.bool)
    else:
        env.extras["sagittal_axis_valid"][ids] = False
    if "sagittal_lat_axis_idx" not in env.extras or env.extras["sagittal_lat_axis_idx"].shape[0] != n:
        env.extras["sagittal_lat_axis_idx"] = torch.zeros(n, device=device, dtype=torch.long)
    else:
        env.extras["sagittal_lat_axis_idx"][ids] = 0
    if "sagittal_plane_p0_root" not in env.extras or env.extras["sagittal_plane_p0_root"].shape != (n, 3):
        env.extras["sagittal_plane_p0_root"] = torch.zeros((n, 3), device=device)
    else:
        env.extras["sagittal_plane_p0_root"][ids] = 0.0
    if "sagittal_plane_valid" not in env.extras or env.extras["sagittal_plane_valid"].shape[0] != n:
        env.extras["sagittal_plane_valid"] = torch.zeros(n, device=device, dtype=torch.bool)
    else:
        env.extras["sagittal_plane_valid"][ids] = False

    # logging counters (DeepMimic export)
    if "mimic_write_t" not in env.extras or env.extras["mimic_write_t"].shape[0] != n:
        env.extras["mimic_write_t"] = torch.zeros(n, device=device, dtype=torch.long)
    else:
        env.extras["mimic_write_t"][ids] = 0
    env.extras.setdefault("mimic_ep", -1)
    env.extras["mimic_task_path"] = None
    env.extras["mimic_joint_path"] = None
    env.extras["last_applied_grip_l"] = torch.zeros(n, device=device)
    env.extras["last_applied_grip_r"] = torch.zeros(n, device=device)


def inc_step_counter(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """Increment per-env step counter every sim step."""
    device = env.device
    n = env.num_envs
    if "step_counter" not in env.extras or env.extras["step_counter"].shape[0] != n:
        env.extras["step_counter"] = torch.zeros(n, device=device, dtype=torch.long)
    ids = torch.arange(n, device=device) if env_ids is None else torch.as_tensor(env_ids, device=device)
    env.extras["step_counter"][ids] += 1


def mimic_csv_init_episode(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """Initialize DeepMimic CSV files for the selected env at episode reset."""
    if not getattr(env.cfg, "export_mimic_csv", False):
        return

    dbg_id = int(getattr(env.cfg, "export_mimic_env_id", 0))
    if dbg_id < 0 or dbg_id >= env.num_envs:
        return
    # If called from a reset event, only initialize when the selected env is actually being reset.
    if env_ids is not None and not isinstance(env_ids, slice):
        if isinstance(env_ids, torch.Tensor):
            if not bool((env_ids == dbg_id).any().item()):
                return
        else:
            if dbg_id not in env_ids:
                return

    device = env.device
    export_dir = getattr(env.cfg, "export_mimic_dir", "logs/hand2hand_mimic")
    os.makedirs(export_dir, exist_ok=True)

    robot = env.scene["robot"]

    env.extras.setdefault("mimic_ep", -1)
    env.extras["mimic_ep"] += 1
    ep = int(env.extras["mimic_ep"])

    task_path = os.path.join(export_dir, f"mimic_task_env{dbg_id}_ep{ep}.csv")
    joint_path = os.path.join(export_dir, f"mimic_joint_env{dbg_id}_ep{ep}.csv")
    env.extras["mimic_task_path"] = task_path
    env.extras["mimic_joint_path"] = joint_path

    if "mimic_write_t" not in env.extras or env.extras["mimic_write_t"].shape[0] != env.num_envs:
        env.extras["mimic_write_t"] = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    env.extras["mimic_write_t"][dbg_id] = 0

    task_header = [
        "t",
        "phase",
        "root_px",
        "root_py",
        "root_pz",
        "root_qw",
        "root_qx",
        "root_qy",
        "root_qz",
        "ref_r_px",
        "ref_r_py",
        "ref_r_pz",
        "ref_r_qw",
        "ref_r_qx",
        "ref_r_qy",
        "ref_r_qz",
        "ref_l_px",
        "ref_l_py",
        "ref_l_pz",
        "ref_l_qw",
        "ref_l_qx",
        "ref_l_qy",
        "ref_l_qz",
        "cur_r_px",
        "cur_r_py",
        "cur_r_pz",
        "cur_r_qw",
        "cur_r_qx",
        "cur_r_qy",
        "cur_r_qz",
        "cur_l_px",
        "cur_l_py",
        "cur_l_pz",
        "cur_l_qw",
        "cur_l_qx",
        "cur_l_qy",
        "cur_l_qz",
        "ref_grip_l",
        "ref_grip_r",
        "act_grip_l",
        "act_grip_r",
    ]
    with open(task_path, "w", newline="") as f_task:
        csv.writer(f_task).writerow(task_header)

    qpos_sample = robot.data.joint_pos[dbg_id]
    qvel_sample = robot.data.joint_vel[dbg_id] if hasattr(robot.data, "joint_vel") else None
    joint_names = getattr(robot.data, "joint_names", None)
    if joint_names is None or len(joint_names) != qpos_sample.shape[-1]:
        joint_names = [str(i) for i in range(qpos_sample.shape[-1])]

    joint_header = [
        "t",
        "phase",
        "root_px",
        "root_py",
        "root_pz",
        "root_qw",
        "root_qx",
        "root_qy",
        "root_qz",
    ]
    root_lin_vel = getattr(robot.data, "root_lin_vel_w", None)
    root_ang_vel = getattr(robot.data, "root_ang_vel_w", None)
    if root_lin_vel is not None:
        joint_header += ["root_vx", "root_vy", "root_vz"]
    if root_ang_vel is not None:
        joint_header += ["root_avx", "root_avy", "root_avz"]
    for name in joint_names:
        joint_header.append(f"qpos_{name}")
    if qvel_sample is not None:
        for name in joint_names:
            joint_header.append(f"qvel_{name}")
    joint_header += ["ref_grip_l", "ref_grip_r", "act_grip_l", "act_grip_r"]

    with open(joint_path, "w", newline="") as f_joint:
        csv.writer(f_joint).writerow(joint_header)


def mimic_csv_log_step(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """Append DeepMimic-style task/joint CSV rows for the selected env."""
    if not getattr(env.cfg, "export_mimic_csv", False):
        return

    dbg_id = int(getattr(env.cfg, "export_mimic_env_id", 0))
    if dbg_id < 0 or dbg_id >= env.num_envs:
        return

    if env.extras.get("mimic_task_path") is None or env.extras.get("mimic_joint_path") is None:
        mimic_csv_init_episode(env)

    task_path = env.extras.get("mimic_task_path")
    joint_path = env.extras.get("mimic_joint_path")
    if task_path is None or joint_path is None:
        return

    device = env.device
    robot = env.scene["robot"]
    if "mimic_write_t" not in env.extras or env.extras["mimic_write_t"].shape[0] != env.num_envs:
        env.extras["mimic_write_t"] = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    t_write = int(env.extras["mimic_write_t"][dbg_id].item())
    env.extras["mimic_write_t"][dbg_id] += 1

    write_every = max(1, int(getattr(env.cfg, "export_mimic_write_every", 1)))
    if t_write % write_every != 0:
        return

    ref_r = env.extras.get("ref_right_ee_pos", None)
    ref_l = env.extras.get("ref_left_ee_pos", None)
    ref_qr = env.extras.get("ref_right_ee_quat", None)
    ref_ql = env.extras.get("ref_left_ee_quat", None)
    ref_gl = env.extras.get("ref_left_grip", None)
    ref_gr = env.extras.get("ref_right_grip", None)
    if ref_r is None or ref_l is None or ref_qr is None or ref_ql is None:
        return
    T = ref_r.shape[1]
    idx = max(0, min(t_write, T - 1))

    T_ref_val = env.extras.get("T_ref", 0)
    try:
        T_ref = int(T_ref_val) if not isinstance(T_ref_val, torch.Tensor) else int(T_ref_val.item())
    except Exception:
        T_ref = 0
    phase_a = int(env.extras.get("phase_a", max(2, int(0.5 * T_ref))))
    phase_b = int(env.extras.get("phase_b", max(2, T_ref - phase_a)))
    phase_hold_default = max(2, T_ref - (phase_a + phase_b + max(2, T_ref - (phase_a + phase_b))))
    phase_hold = int(env.extras.get("phase_hold", phase_hold_default))
    phase_sep = int(env.extras.get("phase_sep", max(2, T_ref - (phase_a + phase_b + phase_hold))))
    if T_ref == 0:
        phase = "unknown"
    elif t_write < phase_a:
        phase = "lift"
    elif t_write < phase_a + phase_b:
        phase = "translate"
    elif t_write < phase_a + phase_b + phase_hold:
        phase = "hold"
    elif t_write < phase_a + phase_b + phase_hold + phase_sep:
        phase = "separate"
    else:
        phase = "hold_final"

    from .observations import (
        get_left_eef_pos,
        get_left_eef_quat,
        get_right_eef_pos,
        get_right_eef_quat,
    )

    root_pos_env = robot.data.root_pos_w[dbg_id] - env.scene.env_origins[dbg_id]
    root_quat_w = robot.data.root_quat_w[dbg_id]

    cur_r_pos = get_right_eef_pos(env)[dbg_id]
    cur_l_pos = get_left_eef_pos(env)[dbg_id]
    cur_r_quat = get_right_eef_quat(env)[dbg_id]
    cur_l_quat = get_left_eef_quat(env)[dbg_id]

    ref_r_p = ref_r[dbg_id, idx]
    ref_l_p = ref_l[dbg_id, idx]
    ref_r_q = ref_qr[dbg_id, idx]
    ref_l_q = ref_ql[dbg_id, idx]
    ref_gl_val = float(ref_gl[dbg_id, idx]) if isinstance(ref_gl, torch.Tensor) else 0.0
    ref_gr_val = float(ref_gr[dbg_id, idx]) if isinstance(ref_gr, torch.Tensor) else 0.0

    last_gl = env.extras.get("last_applied_grip_l", None)
    last_gr = env.extras.get("last_applied_grip_r", None)
    act_gl = float(last_gl[dbg_id]) if isinstance(last_gl, torch.Tensor) and last_gl.shape[0] > dbg_id else 0.0
    act_gr = float(last_gr[dbg_id]) if isinstance(last_gr, torch.Tensor) and last_gr.shape[0] > dbg_id else 0.0

    task_row = [
        t_write,
        phase,
        float(root_pos_env[0]),
        float(root_pos_env[1]),
        float(root_pos_env[2]),
        float(root_quat_w[0]),
        float(root_quat_w[1]),
        float(root_quat_w[2]),
        float(root_quat_w[3]),
        float(ref_r_p[0]),
        float(ref_r_p[1]),
        float(ref_r_p[2]),
        float(ref_r_q[0]),
        float(ref_r_q[1]),
        float(ref_r_q[2]),
        float(ref_r_q[3]),
        float(ref_l_p[0]),
        float(ref_l_p[1]),
        float(ref_l_p[2]),
        float(ref_l_q[0]),
        float(ref_l_q[1]),
        float(ref_l_q[2]),
        float(ref_l_q[3]),
        float(cur_r_pos[0]),
        float(cur_r_pos[1]),
        float(cur_r_pos[2]),
        float(cur_r_quat[0]),
        float(cur_r_quat[1]),
        float(cur_r_quat[2]),
        float(cur_r_quat[3]),
        float(cur_l_pos[0]),
        float(cur_l_pos[1]),
        float(cur_l_pos[2]),
        float(cur_l_quat[0]),
        float(cur_l_quat[1]),
        float(cur_l_quat[2]),
        float(cur_l_quat[3]),
        ref_gl_val,
        ref_gr_val,
        act_gl,
        act_gr,
    ]
    with open(task_path, "a", newline="") as f_task:
        csv.writer(f_task).writerow(task_row)

    qpos = robot.data.joint_pos[dbg_id].detach().cpu()
    qvel_data = robot.data.joint_vel if hasattr(robot.data, "joint_vel") else None
    qvel = qvel_data[dbg_id].detach().cpu() if qvel_data is not None else None
    joint_names = getattr(robot.data, "joint_names", None)
    if joint_names is None or len(joint_names) != qpos.shape[-1]:
        joint_names = [str(i) for i in range(qpos.shape[-1])]

    joint_row = [
        t_write,
        phase,
        float(root_pos_env[0]),
        float(root_pos_env[1]),
        float(root_pos_env[2]),
        float(root_quat_w[0]),
        float(root_quat_w[1]),
        float(root_quat_w[2]),
        float(root_quat_w[3]),
    ]
    root_lin_vel = getattr(robot.data, "root_lin_vel_w", None)
    root_ang_vel = getattr(robot.data, "root_ang_vel_w", None)
    if root_lin_vel is not None:
        rv = root_lin_vel[dbg_id]
        joint_row += [float(rv[0]), float(rv[1]), float(rv[2])]
    if root_ang_vel is not None:
        ra = root_ang_vel[dbg_id]
        joint_row += [float(ra[0]), float(ra[1]), float(ra[2])]
    for i in range(qpos.shape[-1]):
        joint_row.append(float(qpos[i]))
    if qvel is not None:
        for i in range(qvel.shape[-1]):
            joint_row.append(float(qvel[i]))
    joint_row += [ref_gl_val, ref_gr_val, act_gl, act_gr]
    with open(joint_path, "a", newline="") as f_joint:
        csv.writer(f_joint).writerow(joint_row)

def build_right_arm_reference_trajectory_legacy(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """
    Build a smooth right-arm end-effector reference for the full episode horizon.

    Stores per-env tensors in env.extras:
        - ref_right_ee_pos: [N, T, 3] in env-origin frame
        - ref_right_ee_quat: [N, T, 4] world orientation (kept constant)
    """
    device = env.device
    n = env.num_envs
    ids = torch.arange(n, device=device) if env_ids is None else torch.as_tensor(env_ids, device=device)
    if ids.numel() == 0:
        return

    dt = float(env.cfg.sim.dt)
    decim = int(env.cfg.decimation)
    T_episode = max(10, int(env.cfg.episode_length_s / (dt * decim)))
    T_ref = max(5, int(REF_RATIO * T_episode))

    ref_pos = env.extras.get("ref_right_ee_pos")
    if ref_pos is None or ref_pos.shape != (n, T_episode, 3):
        ref_pos = torch.zeros((n, T_episode, 3), device=device)
    else:
        ref_pos = ref_pos.clone()

    ref_pos_l = env.extras.get("ref_left_ee_pos")
    if ref_pos_l is None or ref_pos_l.shape != (n, T_episode, 3):
        ref_pos_l = torch.zeros((n, T_episode, 3), device=device)
    else:
        ref_pos_l = ref_pos_l.clone()

    ref_quat = env.extras.get("ref_right_ee_quat")
    if ref_quat is None or ref_quat.shape != (n, T_episode, 4):
        ref_quat = torch.zeros((n, T_episode, 4), device=device)
    else:
        ref_quat = ref_quat.clone()

    robot = env.scene["robot"]
    names = robot.data.body_names
    rh_idx = names.index("right_hand_pitch_link")
    lh_idx = names.index("left_hand_pitch_link")

    root_pos = robot.data.root_pos_w - env.scene.env_origins
    root_quat = robot.data.root_quat_w

    curr_pos = robot.data.body_pos_w[:, rh_idx] - env.scene.env_origins
    curr_quat = robot.data.body_quat_w[:, rh_idx]
    curr_quat_l = robot.data.body_quat_w[:, lh_idx]

    phase_a = max(2, int(0.3 * T_ref))  # lift
    phase_b = max(2, int(0.4 * T_ref))  # move forward
    phase_c = max(1, T_ref - (phase_a + phase_b))  # hold
    total = phase_a + phase_b + phase_c
    if total < T_ref:
        phase_c += T_ref - total

    lift_target = curr_pos[ids].clone()
    lift_target[:, 2] = torch.maximum(
        lift_target[:, 2],
        torch.full((ids.numel(),), 1.02, device=device)
    )

    exchange_offset = torch.tensor([0.25, -0.05, 0.10], device=device).expand(ids.numel(), -1)
    exchange_target = root_pos[ids] + math_utils.quat_apply(root_quat[ids], exchange_offset)

    hold_target = exchange_target

    pos_a = _min_jerk_interp(curr_pos[ids], lift_target, phase_a, device)
    pos_b = _min_jerk_interp(lift_target, exchange_target, phase_b, device)
    pos_c = hold_target.unsqueeze(1).repeat(1, phase_c, 1)

    pos_seq = torch.cat([pos_a, pos_b, pos_c], dim=1)[:, :T_ref, :]
    last = pos_seq[:, -1:, :]
    hold_len = T_episode - T_ref
    if hold_len > 0:
        pos_full = torch.cat([pos_seq, last.repeat(1, hold_len, 1)], dim=1)
    else:
        pos_full = pos_seq[:, :T_episode, :]

    ref_pos[ids] = pos_full

    q_full = curr_quat[ids].unsqueeze(1).repeat(1, T_episode, 1)
    ref_quat[ids] = q_full

    env.extras["ref_right_ee_pos"] = ref_pos
    env.extras["ref_right_ee_quat"] = ref_quat
    env.extras["ref_ratio"] = REF_RATIO
    env.extras["T_ref"] = T_ref


def build_right_arm_reference_trajectory(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """
    NEW (simple L-shape) right-arm end-effector reference for DeepMimic warm-start.

    Shape concept:
      - Phase A: lift vertically to chest height (keep x,y)
      - Phase B: translate horizontally to sternum target (keep z)
      - Then hold for rest of episode

    Stores:
      - ref_right_ee_pos: [N, T, 3] in env-origin frame
      - ref_right_ee_quat / ref_left_ee_quat: [N, T, 4] (right fixed; left mirrored across sagittal plane)
    """
    device = env.device
    n = env.num_envs
    ids = torch.arange(n, device=device) if env_ids is None else torch.as_tensor(env_ids, device=device)
    if ids.numel() == 0:
        return

    # episode horizon
    dt = float(env.cfg.sim.dt)
    decim = int(env.cfg.decimation)
    T_episode = max(10, int(env.cfg.episode_length_s / (dt * decim)))
    T_ref = max(5, int(REF_RATIO * T_episode))

    # allocate / clone buffers
    ref_pos = env.extras.get("ref_right_ee_pos")
    if ref_pos is None or ref_pos.shape != (n, T_episode, 3):
        ref_pos = torch.zeros((n, T_episode, 3), device=device)
    else:
        ref_pos = ref_pos.clone()

    ref_quat = env.extras.get("ref_right_ee_quat")
    if ref_quat is None or ref_quat.shape != (n, T_episode, 4):
        ref_quat = torch.zeros((n, T_episode, 4), device=device)
    else:
        ref_quat = ref_quat.clone()

    ref_pos_l = env.extras.get("ref_left_ee_pos")
    if ref_pos_l is None or ref_pos_l.shape != (n, T_episode, 3):
        ref_pos_l = torch.zeros((n, T_episode, 3), device=device)
    else:
        ref_pos_l = ref_pos_l.clone()

    ref_quat_l = env.extras.get("ref_left_ee_quat")
    if ref_quat_l is None or ref_quat_l.shape != (n, T_episode, 4):
        ref_quat_l = torch.zeros((n, T_episode, 4), device=device)
    else:
        ref_quat_l = ref_quat_l.clone()

    left_grip = env.extras.get("ref_left_grip")
    if left_grip is None or left_grip.shape != (n, T_episode):
        left_grip = torch.zeros((n, T_episode), device=device, dtype=torch.float)
    else:
        left_grip = left_grip.clone()

    right_grip = env.extras.get("ref_right_grip")
    if right_grip is None or right_grip.shape != (n, T_episode):
        right_grip = torch.ones((n, T_episode), device=device, dtype=torch.float)
    else:
        right_grip = right_grip.clone()

    # robot data
    robot = env.scene["robot"]
    names = robot.data.body_names
    rh_idx = names.index("right_hand_pitch_link")
    lh_idx = names.index("left_hand_pitch_link")

    root_pos = robot.data.root_pos_w - env.scene.env_origins
    root_quat = robot.data.root_quat_w

    # current right-hand pose in env-origin frame
    curr_pos = robot.data.body_pos_w[:, rh_idx] - env.scene.env_origins
    curr_quat = robot.data.body_quat_w[:, rh_idx]
    curr_quat_l = robot.data.body_quat_w[:, lh_idx]
    curr_pos_l = robot.data.body_pos_w[:, lh_idx] - env.scene.env_origins

    # --- targets ---
    # Chest height in env-origin frame (simple constant)
    CHEST_Z_LOCAL = 1.20

    # Phase A target: vertical lift (same x,y)
    lift_target = curr_pos[ids].clone()
    lift_target[:, 2] = CHEST_Z_LOCAL
    lift_target_l = curr_pos_l[ids].clone()
    lift_target_l[:, 2] = CHEST_Z_LOCAL

    # Phase B target: sternum vicinity relative to robot root
    # (front centerline)
    sternum_offset = torch.tensor([0.20, -0.06, 0.12], device=device).expand(ids.numel(), -1)
    sternum_target = root_pos[ids] + math_utils.quat_apply(root_quat[ids], sternum_offset)
    sternum_target[:, 2] = CHEST_Z_LOCAL
    sternum_offset_l = sternum_offset.clone()
    sternum_offset_l[:, 1] *= -1.0
    sternum_target_l = root_pos[ids] + math_utils.quat_apply(root_quat[ids], sternum_offset_l)
    sternum_target_l[:, 2] = CHEST_Z_LOCAL

    sep_m = float(getattr(env.cfg, "debug_post_grasp_sep_m", 0.15))
    sep_ratio = float(getattr(env.cfg, "debug_post_grasp_sep_ratio", 0.25))
    grip_wait_steps = 4
    post_release_hold_steps = int(getattr(env.cfg, "debug_post_release_hold_steps", 6))
    phase_hold_min = grip_wait_steps + post_release_hold_steps
    phase_sep = max(2, int(sep_ratio * T_ref))
    phase_hold = max(2, phase_hold_min)
    remaining = T_ref - phase_sep - phase_hold
    if remaining < 4:
        deficit = 4 - remaining
        phase_sep = max(2, phase_sep - deficit)
        remaining = T_ref - phase_sep - phase_hold
    remaining = max(0, remaining)

    phase_a = max(2, int(0.45 * remaining))                  # lift
    phase_b = max(2, remaining - phase_a)                    # translate

    total = phase_a + phase_b + phase_hold + phase_sep
    if total != T_ref:
        phase_b = max(2, T_ref - (phase_a + phase_hold + phase_sep))
        total = phase_a + phase_b + phase_hold + phase_sep
        if total < T_ref:
            phase_hold += (T_ref - total)
            total = phase_a + phase_b + phase_hold + phase_sep
        elif total > T_ref:
            reduce = total - T_ref
            reduce_hold = min(reduce, max(0, phase_hold - phase_hold_min))
            phase_hold = max(phase_hold_min, phase_hold - reduce_hold)
            reduce -= reduce_hold
            if reduce > 0:
                phase_b = max(2, phase_b - reduce)
            total = phase_a + phase_b + phase_hold + phase_sep

    def _recompute_times(pa: int, pb: int, ph: int, ps: int):
        t_h = pa + pb
        t_sep_s = t_h + ph
        t_sep_e = t_sep_s + ps
        return t_h, t_sep_s, t_sep_e

    t_handover, t_sep_start, t_sep_end = _recompute_times(phase_a, phase_b, phase_hold, phase_sep)
    handover_delay_steps = 6
    hold_len = T_episode - T_ref

    # --- desired orientations (mirror right-hand across sagittal plane) ---
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
        qx = torch.where(cond, (R[..., 2, 1] - R[..., 1, 2]) / (4.0 * qw.clamp_min(1e-6)), qx)
        qy = torch.where(cond, (R[..., 0, 2] - R[..., 2, 0]) / (4.0 * qw.clamp_min(1e-6)), qy)
        qz = torch.where(cond, (R[..., 1, 0] - R[..., 0, 1]) / (4.0 * qw.clamp_min(1e-6)), qz)

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

    pairs = [
        ("right_hip_link", "left_hip_link"),
        ("right_thigh_link", "left_thigh_link"),
        ("right_upper_leg_link", "left_upper_leg_link"),
        ("right_clavicle_link", "left_clavicle_link"),
        ("right_shoulder_link", "left_shoulder_link"),
        ("right_shoulder_pitch_link", "left_shoulder_pitch_link"),
    ]

    idxR, idxL = None, None
    for r_name, l_name in pairs:
        if r_name in names and l_name in names:
            idxR = names.index(r_name)
            idxL = names.index(l_name)
            break

    if idxR is None or idxL is None:
        p0_root = torch.zeros((ids.numel(), 3), device=device)
        n_root = torch.tensor([0.0, 1.0, 0.0], device=device).expand(ids.numel(), -1)
    else:
        posR_w = robot.data.body_pos_w[:, idxR] - env.scene.env_origins
        posL_w = robot.data.body_pos_w[:, idxL] - env.scene.env_origins
        posR_root = math_utils.quat_rotate_inverse(root_quat, posR_w - root_pos)
        posL_root = math_utils.quat_rotate_inverse(root_quat, posL_w - root_pos)

        p0_root_all = 0.5 * (posL_root + posR_root)
        up_world = torch.tensor([0.0, 0.0, 1.0], device=device).expand(root_quat.shape[0], -1)
        up_root_all = math_utils.quat_apply_inverse(root_quat, up_world)
        up_root_all = torch.nn.functional.normalize(up_root_all, dim=-1)
        lr = posL_root - posR_root
        lr_h = lr - torch.sum(lr * up_root_all, dim=-1, keepdim=True) * up_root_all
        fallback = torch.tensor([0.0, 1.0, 0.0], device=device).expand_as(lr_h)
        lr_norm = torch.norm(lr_h, dim=-1, keepdim=True)
        lr_h = torch.where(lr_norm < 1e-6, fallback, lr_h)
        n_root_all = torch.nn.functional.normalize(lr_h, dim=-1)

        p0_root = p0_root_all[ids]
        n_root = n_root_all[ids]

    q_full_r = curr_quat[ids].unsqueeze(1).repeat(1, T_episode, 1)
    ref_quat[ids] = q_full_r

    root_quat_full = root_quat[ids].unsqueeze(1).expand(-1, T_episode, -1)
    M = torch.eye(3, device=device).expand(ids.numel(), 3, 3) - 2.0 * n_root.unsqueeze(-1) * n_root.unsqueeze(-2)
    qr_root = math_utils.quat_mul(
        math_utils.quat_conjugate(root_quat_full),
        q_full_r,
    )
    Rr = math_utils.matrix_from_quat(qr_root)
    Rl = torch.matmul(M.unsqueeze(1), torch.matmul(Rr, M.unsqueeze(1)))
    ql_root = _quat_from_matrix(Rl)
    q_full_l = math_utils.quat_mul(root_quat_full, ql_root)
    ref_quat_l[ids] = q_full_l

    plane_n = env.extras.get("mirror_plane_n_root")
    plane_p0 = env.extras.get("sagittal_plane_p0_root")
    plane_valid = env.extras.get("sagittal_plane_valid")
    if plane_n is None or plane_n.shape != (n, 3):
        plane_n = torch.zeros((n, 3), device=device)
    if plane_p0 is None or plane_p0.shape != (n, 3):
        plane_p0 = torch.zeros((n, 3), device=device)
    if plane_valid is None or plane_valid.shape[0] != n:
        plane_valid = torch.zeros(n, device=device, dtype=torch.bool)
    plane_n[ids] = n_root
    plane_p0[ids] = p0_root
    plane_valid[ids] = True
    env.extras["mirror_plane_n_root"] = plane_n
    env.extras["sagittal_plane_p0_root"] = plane_p0
    env.extras["sagittal_plane_valid"] = plane_valid

    sep_off_r_w = math_utils.quat_apply(root_quat[ids], (-sep_m) * n_root)
    sep_off_l_w = math_utils.quat_apply(root_quat[ids], (sep_m) * n_root)
    sep_target_r = sternum_target + sep_off_r_w
    sep_target_l = sternum_target_l + sep_off_l_w
    sep_target_r[:, 2] = CHEST_Z_LOCAL
    sep_target_l[:, 2] = CHEST_Z_LOCAL

    for _ in range(2):
        t_handover, t_sep_start, t_sep_end = _recompute_times(phase_a, phase_b, phase_hold, phase_sep)
        t_grip_on = min(t_handover + grip_wait_steps, T_episode)
        t_release = t_grip_on
        desired_hold = max(phase_hold, (t_release - t_handover) + post_release_hold_steps)
        if t_release <= t_sep_start and desired_hold <= phase_hold:
            break
        phase_hold = desired_hold
        total = phase_a + phase_b + phase_hold + phase_sep
        if total > T_ref:
            surplus = total - T_ref
            reduce_sep = min(surplus, max(0, phase_sep - 2))
            phase_sep -= reduce_sep
            surplus -= reduce_sep
            if surplus > 0:
                reduce_b = min(surplus, max(0, phase_b - 2))
                phase_b -= reduce_b
                surplus -= reduce_b
            if surplus > 0:
                reduce_a = min(surplus, max(0, phase_a - 2))
                phase_a -= reduce_a
                surplus -= reduce_a
            if surplus > 0:
                phase_hold = max(2, phase_hold - surplus)
        elif total < T_ref:
            phase_b += (T_ref - total)

    t_handover, t_sep_start, t_sep_end = _recompute_times(phase_a, phase_b, phase_hold, phase_sep)
    t_grip_on = min(t_handover + grip_wait_steps, T_episode)
    t_release = t_grip_on

    total = phase_a + phase_b + phase_hold + phase_sep
    if total != T_ref:
        diff = T_ref - total
        if diff > 0:
            phase_b += diff
        else:
            reduce = -diff
            reduce_b = min(reduce, max(0, phase_b - 2))
            phase_b -= reduce_b
            reduce -= reduce_b
            if reduce > 0:
                reduce_hold = min(reduce, max(0, phase_hold - phase_hold_min))
                phase_hold = max(phase_hold_min, phase_hold - reduce_hold)
                reduce -= reduce_hold
            if reduce > 0:
                reduce_sep = min(reduce, max(0, phase_sep - 2))
                phase_sep -= reduce_sep
        t_handover, t_sep_start, t_sep_end = _recompute_times(phase_a, phase_b, phase_hold, phase_sep)
        t_grip_on = min(t_handover + grip_wait_steps, T_episode)
        t_release = t_grip_on

    desired_hold = max(phase_hold, (t_release - t_handover) + post_release_hold_steps)
    if t_release > t_sep_start or desired_hold > phase_hold:
        phase_hold = desired_hold
        total = phase_a + phase_b + phase_hold + phase_sep
        if total > T_ref:
            surplus = total - T_ref
            reduce_sep = min(surplus, max(0, phase_sep - 2))
            phase_sep -= reduce_sep
            surplus -= reduce_sep
            if surplus > 0:
                reduce_b = min(surplus, max(0, phase_b - 2))
                phase_b -= reduce_b
                surplus -= reduce_b
            if surplus > 0:
                reduce_a = min(surplus, max(0, phase_a - 2))
                phase_a -= reduce_a
                surplus -= reduce_a
            if surplus > 0:
                phase_hold = max(2, phase_hold - surplus)
        elif total < T_ref:
            phase_b += (T_ref - total)
        t_handover, t_sep_start, t_sep_end = _recompute_times(phase_a, phase_b, phase_hold, phase_sep)
        t_grip_on = min(t_handover + grip_wait_steps, T_episode)
        t_release = t_grip_on

    total = phase_a + phase_b + phase_hold + phase_sep
    if total != T_ref:
        phase_b = max(2, T_ref - (phase_a + phase_hold + phase_sep))
        total = phase_a + phase_b + phase_hold + phase_sep
        if total < T_ref:
            phase_hold += (T_ref - total)
            total = phase_a + phase_b + phase_hold + phase_sep
        elif total > T_ref:
            reduce = total - T_ref
            reduce_hold = min(reduce, max(0, phase_hold - phase_hold_min))
            phase_hold = max(phase_hold_min, phase_hold - reduce_hold)
            reduce -= reduce_hold
            if reduce > 0:
                phase_b = max(2, phase_b - reduce)
        t_handover, t_sep_start, t_sep_end = _recompute_times(phase_a, phase_b, phase_hold, phase_sep)
        t_grip_on = min(t_handover + grip_wait_steps, T_episode)
        t_release = t_grip_on

    if t_release > t_sep_start:
        phase_hold = max(phase_hold, (t_release - t_handover) + post_release_hold_steps)
        total = phase_a + phase_b + phase_hold + phase_sep
        if total > T_ref:
            surplus = total - T_ref
            reduce_sep = min(surplus, max(0, phase_sep - 2))
            phase_sep -= reduce_sep
            surplus -= reduce_sep
            if surplus > 0:
                reduce_b = min(surplus, max(0, phase_b - 2))
                phase_b -= reduce_b
                surplus -= reduce_b
            if surplus > 0:
                reduce_a = min(surplus, max(0, phase_a - 2))
                phase_a -= reduce_a
                surplus -= reduce_a
            if surplus > 0:
                phase_hold = max(2, phase_hold - surplus)
        elif total < T_ref:
            phase_b += (T_ref - total)
        t_handover, t_sep_start, t_sep_end = _recompute_times(phase_a, phase_b, phase_hold, phase_sep)
        t_grip_on = min(t_handover + grip_wait_steps, T_episode)
        t_release = t_grip_on

    # --- interpolation ---
    pos_a = _min_jerk_interp(curr_pos[ids], lift_target, phase_a, device)
    pos_b = _min_jerk_interp(lift_target, sternum_target, phase_b, device)
    pos_h = sternum_target.unsqueeze(1).repeat(1, phase_hold, 1)
    pos_s = _min_jerk_interp(sternum_target, sep_target_r, phase_sep, device)
    pos_a_l = _min_jerk_interp(curr_pos_l[ids], lift_target_l, phase_a, device)
    pos_b_l = _min_jerk_interp(lift_target_l, sternum_target_l, phase_b, device)
    pos_h_l = sternum_target_l.unsqueeze(1).repeat(1, phase_hold, 1)
    pos_s_l = _min_jerk_interp(sternum_target_l, sep_target_l, phase_sep, device)

    pos_seq = torch.cat([pos_a, pos_b, pos_h, pos_s], dim=1)[:, :T_ref, :]
    pos_seq_l = torch.cat([pos_a_l, pos_b_l, pos_h_l, pos_s_l], dim=1)[:, :T_ref, :]

    # extend to full episode by holding last reference
    last = pos_seq[:, -1:, :]
    last_l = pos_seq_l[:, -1:, :]
    if hold_len > 0:
        pos_full = torch.cat([pos_seq, last.repeat(1, hold_len, 1)], dim=1)
        pos_full_l = torch.cat([pos_seq_l, last_l.repeat(1, hold_len, 1)], dim=1)
    else:
        pos_full = pos_seq[:, :T_episode, :]
        pos_full_l = pos_seq_l[:, :T_episode, :]

    ref_pos[ids] = pos_full
    ref_pos_l[ids] = pos_full_l
    ref_quat[ids] = q_full_r
    ref_quat_l[ids] = q_full_l
    env.extras["ref_left_ee_pos"] = ref_pos_l

    left_grip[ids].zero_()
    right_grip[ids].fill_(1.0)

    t_grip_on = min(t_handover + grip_wait_steps, T_episode)
    t_release = t_grip_on
    left_grip[ids, t_grip_on:] = 1.0
    right_grip[ids, :t_release] = 1.0
    right_grip[ids, t_release:] = 0.0

    env.extras["ref_right_ee_pos"] = ref_pos
    env.extras["ref_right_ee_quat"] = ref_quat
    env.extras["ref_left_ee_quat"] = ref_quat_l
    env.extras["ref_left_grip"] = left_grip
    env.extras["ref_right_grip"] = right_grip
    env.extras["ref_ratio"] = REF_RATIO
    env.extras["T_ref"] = T_ref
    env.extras["t_handover"] = int(t_handover)
    env.extras["phase_a"] = int(phase_a)
    env.extras["phase_b"] = int(phase_b)
    env.extras["phase_hold"] = int(phase_hold)
    env.extras["phase_sep"] = int(phase_sep)
    env.extras["t_sep_start"] = int(t_sep_start)
    env.extras["t_sep_end"] = int(t_sep_end)
    env.extras["t_grip_on"] = int(t_grip_on)
    env.extras["t_release"] = int(t_release)
    env.extras["handover_delay_steps"] = int(handover_delay_steps)
    env.extras["debug_post_grasp_sep_m"] = float(sep_m)


def _min_jerk_interp(start: torch.Tensor, end: torch.Tensor, steps: int, device) -> torch.Tensor:
    """Minimum-jerk interpolation between start/end with shape [M, 3]."""
    if steps <= 1:
        return end.unsqueeze(1)
    t = torch.linspace(0.0, 1.0, steps, device=device)
    tau = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    return (1 - tau)[None, :, None] * start[:, None, :] + tau[None, :, None] * end[:, None, :]
