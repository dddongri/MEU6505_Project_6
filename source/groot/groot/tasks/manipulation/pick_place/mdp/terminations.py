from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from .rewards import reward_place_success_upright

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def task_done_place_upright(
    env: "ManagerBasedEnv",
    target_pos_rel: tuple[float, float, float] = (0.0, 0.55, 0.86),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when success condition is met."""
    return reward_place_success_upright(
        env,
        target_pos_rel=target_pos_rel,
        require_released=True,
        object_cfg=object_cfg,
    ).bool()
