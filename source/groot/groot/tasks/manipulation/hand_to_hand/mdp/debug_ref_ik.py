from __future__ import annotations

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

    # action layout: left IK pos slice (first 3)
    POS_SLICE = slice(0, 3)
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
