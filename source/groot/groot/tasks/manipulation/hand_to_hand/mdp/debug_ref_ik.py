from __future__ import annotations

import csv
from typing import Dict

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .observations import get_right_eef_pos


def compute_ref_follow_action(env: ManagerBasedRLEnv, t: int) -> torch.Tensor:
    """
    Return action tensor that makes ONLY right arm follow ref_right_ee_pos at index t.
    """
    act_shape = env.action_manager.action.shape[1]
    actions = torch.zeros((env.num_envs, act_shape), device=env.device)

    ref = env.extras.get("ref_right_ee_pos", None)
    if ref is None or ref.shape[0] != env.num_envs:
        return actions

    T = ref.shape[1]
    idx = min(max(t, 0), T - 1)
    cur_pos = get_right_eef_pos(env)
    delta = ref[:, idx] - cur_pos
    delta = torch.clamp(delta * 3.0, min=-1.0, max=1.0)

    # action layout: right IK pos slice starts at index 6
    POS_SLICE = slice(6, 9)
    actions[:, POS_SLICE] = delta
    return actions


def debug_run_right_ref_ik(env: ManagerBasedRLEnv, num_steps: int | None = None):
    """
    Run a short rollout without RL to follow the right-arm reference.
    """
    env.reset()
    if "T_ref" in env.extras:
        T_ref = int(env.extras["T_ref"])
    else:
        ref = env.extras.get("ref_right_ee_pos", None)
        T_ref = ref.shape[1] if ref is not None else 0
    steps = T_ref if num_steps is None else min(num_steps, T_ref)
    for t in range(steps):
        actions = compute_ref_follow_action(env, t)
        env.step(actions)
        ref_pos = env.extras["ref_right_ee_pos"][0, t]
        cur = get_right_eef_pos(env)[0]
        err = torch.norm(ref_pos - cur).item()
        print(f"[IK-FOLLOW] t={t} err={err:.4f}")


def grip_to_finger_targets(grip_l: float, grip_r: float, closed: float = 1.0, open_: float = 0.0) -> Dict[str, float]:
    """
    Simple mapping from binary grip signals to per-finger targets (open/closed).
    """
    finger_names = ["index", "middle", "ring", "pinky", "thumb"]
    targets: Dict[str, float] = {}
    for prefix, grip in (("L", grip_l), ("R", grip_r)):
        val = closed if grip > 0.5 else open_
        for name in finger_names:
            targets[f"{prefix}_{name}"] = val
    return targets


def load_traj_csv_with_grip(path: str):
    """
    Load a saved trajectory CSV (traj_env*.csv) and print pose + grip info per timestep.
    If finger targets are used in a preview, use grip_to_finger_targets() to map grip bits.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("[TRAJ] empty csv")
            return
        has_gl = "ref_grip_l" in reader.fieldnames
        has_gr = "ref_grip_r" in reader.fieldnames
        for row in reader:
            if not row:
                continue
            t = row.get("t", "0")
            ref_pos = (
                float(row.get("ref_x", 0.0)),
                float(row.get("ref_y", 0.0)),
                float(row.get("ref_z", 0.0)),
            )
            grip_l = float(row["ref_grip_l"]) if has_gl and row.get("ref_grip_l") not in (None, "") else 0.0
            grip_r = float(row["ref_grip_r"]) if has_gr and row.get("ref_grip_r") not in (None, "") else 0.0
            targets = grip_to_finger_targets(grip_l, grip_r)
            print(
                f"[TRAJ] t={t} ref_pos={ref_pos} grip_l={grip_l:.2f} "
                f"grip_r={grip_r:.2f} finger_targets={targets}"
            )
