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

    # reset logging handles/flags
    env.extras["traj_csv_done"] = False
    if env.extras.get("traj_csv_f") is not None:
        try:
            env.extras["traj_csv_f"].close()
        except Exception:
            pass
    env.extras["traj_csv_f"] = None
    env.extras["traj_csv_w"] = None


def inc_step_counter(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """Increment per-env step counter every sim step."""
    device = env.device
    n = env.num_envs
    if "step_counter" not in env.extras or env.extras["step_counter"].shape[0] != n:
        env.extras["step_counter"] = torch.zeros(n, device=device, dtype=torch.long)
    ids = torch.arange(n, device=device) if env_ids is None else torch.as_tensor(env_ids, device=device)
    env.extras["step_counter"][ids] += 1


def _init_traj_handles(env: ManagerBasedRLEnv):
    """Ensure CSV handle slots exist in extras (works even if env has __slots__)."""
    extras = env.extras
    if "traj_csv_f" not in extras:
        extras["traj_csv_f"] = None
    if "traj_csv_w" not in extras:
        extras["traj_csv_w"] = None
    if "traj_csv_ep" not in extras:
        extras["traj_csv_ep"] = -1
    if "traj_csv_done" not in extras:
        extras["traj_csv_done"] = False


def log_traj_csv_step(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """Per-step CSV logging of right-hand ref/cur trajectory for a single env."""
    if not getattr(env.cfg, "log_traj_csv", False):
        return

    _init_traj_handles(env)

    step_counter = env.extras.get("step_counter")
    ref_r = env.extras.get("ref_right_ee_pos")
    if step_counter is None or ref_r is None:
        return

    dbg_id = int(getattr(env.cfg, "log_traj_env_id", 0))
    if dbg_id < 0 or dbg_id >= ref_r.shape[0] or dbg_id >= step_counter.shape[0]:
        return

    # 에피소드마다 한 번만 덤프
    if env.extras.get("traj_csv_done", False):
        return

    if env.extras["traj_csv_f"] is not None:
        env.extras["traj_csv_f"].close()
        env.extras["traj_csv_f"] = None
        env.extras["traj_csv_w"] = None

    log_dir = getattr(env.cfg, "log_traj_dir", "logs/hand2hand_traj")
    os.makedirs(log_dir, exist_ok=True)

    env.extras["traj_csv_ep"] += 1
    path = os.path.join(log_dir, f"traj_env{dbg_id}_ep{env.extras['traj_csv_ep']}.csv")

    env.extras["traj_csv_f"] = open(path, "w", newline="")
    env.extras["traj_csv_w"] = csv.writer(env.extras["traj_csv_f"])
    env.extras["traj_csv_w"].writerow(
        [
            "t",
            "phase",
            "ref_x",
            "ref_y",
            "ref_z",
            "T_episode",
            "T_ref",
            "phase_a",
            "phase_b",
        ]
    )

    T_ref_val = env.extras.get("T_ref", 0)
    if isinstance(T_ref_val, torch.Tensor):
        try:
            T_ref = int(T_ref_val.item())
        except Exception:
            T_ref = 0
    else:
        try:
            T_ref = int(T_ref_val)
        except Exception:
            T_ref = 0

    phase_a = max(2, int(0.5 * T_ref))
    phase_b = max(2, T_ref - phase_a)

    ref_env = ref_r[dbg_id]
    T_episode = ref_env.shape[0]

    for ti in range(int(T_episode)):
        ref_p = ref_env[ti]
        if T_ref == 0:
            phase = "unknown"
        elif ti < phase_a:
            phase = "lift"
        elif ti < phase_a + phase_b:
            phase = "translate"
        else:
            phase = "hold"

        env.extras["traj_csv_w"].writerow(
            [
                ti,
                phase,
                float(ref_p[0]),
                float(ref_p[1]),
                float(ref_p[2]),
                int(T_episode),
                int(T_ref),
                int(phase_a),
                int(phase_b),
            ]
        )

    env.extras["traj_csv_f"].flush()
    env.extras["traj_csv_f"].close()
    env.extras["traj_csv_f"] = None
    env.extras["traj_csv_w"] = None
    env.extras["traj_csv_done"] = True


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

    ref_quat = env.extras.get("ref_right_ee_quat")
    if ref_quat is None or ref_quat.shape != (n, T_episode, 4):
        ref_quat = torch.zeros((n, T_episode, 4), device=device)
    else:
        ref_quat = ref_quat.clone()

    robot = env.scene["robot"]
    names = robot.data.body_names
    rh_idx = names.index("right_hand_pitch_link")

    root_pos = robot.data.root_pos_w - env.scene.env_origins
    root_quat = robot.data.root_quat_w

    curr_pos = robot.data.body_pos_w[:, rh_idx] - env.scene.env_origins
    curr_quat = robot.data.body_quat_w[:, rh_idx]

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

    exchange_offset = torch.tensor([0.25, -0.15, 0.10], device=device).expand(ids.numel(), -1)
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
      - ref_right_ee_quat: [N, T, 4] (constant from current hand orientation)
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

    # robot data
    robot = env.scene["robot"]
    names = robot.data.body_names
    rh_idx = names.index("right_hand_pitch_link")

    root_pos = robot.data.root_pos_w - env.scene.env_origins
    root_quat = robot.data.root_quat_w

    # current right-hand pose in env-origin frame
    curr_pos = robot.data.body_pos_w[:, rh_idx] - env.scene.env_origins
    curr_quat = robot.data.body_quat_w[:, rh_idx]

    # --- targets ---
    # Chest height in env-origin frame (simple constant)
    CHEST_Z_LOCAL = 1.20

    # Phase A target: vertical lift (same x,y)
    lift_target = curr_pos[ids].clone()
    lift_target[:, 2] = CHEST_Z_LOCAL

    # Phase B target: sternum vicinity relative to robot root
    # (front centerline)
    sternum_offset = torch.tensor([0.10, 0.0, 0.12], device=device).expand(ids.numel(), -1)
    sternum_target = root_pos[ids] + math_utils.quat_apply(root_quat[ids], sternum_offset)
    sternum_target[:, 2] = CHEST_Z_LOCAL

    # --- timing ---
    # Two-segment L-shape
    phase_a = max(2, int(0.5 * T_ref))  # lift
    phase_b = max(2, T_ref - phase_a)   # translate to sternum
    total = phase_a + phase_b
    if total < T_ref:
        phase_b += (T_ref - total)

    # --- interpolation ---
    pos_a = _min_jerk_interp(curr_pos[ids], lift_target, phase_a, device)
    pos_b = _min_jerk_interp(lift_target, sternum_target, phase_b, device)

    pos_seq = torch.cat([pos_a, pos_b], dim=1)[:, :T_ref, :]

    # extend to full episode by holding last reference
    last = pos_seq[:, -1:, :]
    hold_len = T_episode - T_ref
    if hold_len > 0:
        pos_full = torch.cat([pos_seq, last.repeat(1, hold_len, 1)], dim=1)
    else:
        pos_full = pos_seq[:, :T_episode, :]

    ref_pos[ids] = pos_full

    # keep orientation constant (current right hand quat)
    q_full = curr_quat[ids].unsqueeze(1).repeat(1, T_episode, 1)
    ref_quat[ids] = q_full

    env.extras["ref_right_ee_pos"] = ref_pos
    env.extras["ref_right_ee_quat"] = ref_quat
    env.extras["ref_ratio"] = REF_RATIO
    env.extras["T_ref"] = T_ref


def _min_jerk_interp(start: torch.Tensor, end: torch.Tensor, steps: int, device) -> torch.Tensor:
    """Minimum-jerk interpolation between start/end with shape [M, 3]."""
    if steps <= 1:
        return end.unsqueeze(1)
    t = torch.linspace(0.0, 1.0, steps, device=device)
    tau = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    return (1 - tau)[None, :, None] * start[:, None, :] + tau[None, :, None] * end[:, None, :]
