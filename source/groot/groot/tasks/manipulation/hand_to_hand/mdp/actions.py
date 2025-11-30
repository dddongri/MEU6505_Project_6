from __future__ import annotations

import torch
from dataclasses import MISSING, field

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    BinaryJointPositionActionCfg,
)
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class SymmetricDualIKAction(ActionTerm):
    """Task-space IK for both hands with enforced point symmetry and binary grasp commands.

    Action layout:
        [0:6] : left hand differential IK command (position/orientation velocity)
        [6]   : left grasp (binary after sigmoid)
        [7]   : right grasp (binary after sigmoid)

    The right hand command is mirrored from the left by negating translational/rotational
    components, which reduces the controllable DOF while keeping hands symmetric about the
    object center.
    """

    cfg: "SymmetricDualIKActionCfg"

    def __init__(self, cfg: "SymmetricDualIKActionCfg", env):
        super().__init__(cfg, env)
        # Build internal IK terms for each hand using the standard action implementation.
        self._left_term = DifferentialInverseKinematicsAction(
            DifferentialInverseKinematicsActionCfg(
                asset_name=cfg.asset_name,
                joint_names=cfg.left_joint_names,
                body_name=cfg.left_body_name,
                body_offset=cfg.body_offset,
                scale=cfg.scale,
                controller=cfg.controller,
            ),
            env,
        )
        self._right_term = DifferentialInverseKinematicsAction(
            DifferentialInverseKinematicsActionCfg(
                asset_name=cfg.asset_name,
                joint_names=cfg.right_joint_names,
                body_name=cfg.right_body_name,
                body_offset=cfg.body_offset,
                scale=cfg.scale,
                controller=cfg.controller,
            ),
            env,
        )
        self._task_dim = self._left_term.action_dim
        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._grasp = torch.zeros((self.num_envs, 2), device=self.device)
        self._step_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._handover_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._left_body_id = self._asset.data.body_names.index(cfg.left_body_name)
        self._right_body_id = self._asset.data.body_names.index(cfg.right_body_name)
        # optionally freeze waist joints to keep torso steady
        self._freeze_waist = cfg.freeze_waist
        self._waist_ids = []
        self._waist_default = None
        if self._freeze_waist:
            ids, _ = self._asset.find_joints("waist_.*")
            if ids:
                self._waist_ids = ids
                self._waist_default = self._asset.data.default_joint_pos[:, ids]
        # optionally freeze arm joints at defaults
        self._disable_arms = cfg.disable_arms
        self._arm_ids = []
        self._arm_default = None
        if self._disable_arms:
            arm_ids_left, _ = self._asset.find_joints(cfg.left_joint_names)
            arm_ids_right, _ = self._asset.find_joints(cfg.right_joint_names)
            self._arm_ids = sorted(list(set(arm_ids_left + arm_ids_right)))
            if self._arm_ids:
                self._arm_default = self._asset.data.default_joint_pos[:, self._arm_ids]
        # gripper (binary) terms
        self._left_grip = BinaryJointPositionAction(
            BinaryJointPositionActionCfg(
                asset_name=cfg.asset_name,
                joint_names=cfg.left_gripper_joint_names,
                open_command_expr=cfg.left_gripper_open_command,
                close_command_expr=cfg.left_gripper_close_command,
            ),
            env,
        ) if cfg.left_gripper_joint_names else None
        # auto-generate right command dicts if not explicitly provided
        right_open = cfg.right_gripper_open_command or {
            k.replace("L_", "R_", 1): v for k, v in cfg.left_gripper_open_command.items()
        }
        right_close = cfg.right_gripper_close_command or {
            k.replace("L_", "R_", 1): v for k, v in cfg.left_gripper_close_command.items()
        }
        self._right_grip = BinaryJointPositionAction(
            BinaryJointPositionActionCfg(
                asset_name=cfg.asset_name,
                joint_names=cfg.right_gripper_joint_names or [name.replace("L_", "R_", 1) for name in cfg.left_gripper_joint_names],
                open_command_expr=right_open,
                close_command_expr=right_close,
            ),
            env,
        ) if cfg.left_gripper_joint_names else None

    @property
    def action_dim(self) -> int:
        # left IK (task_dim) + 2 grasp bits
        return self._task_dim + (2 if self.cfg.include_grasp else 0)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        # Concatenate left/right processed actions for logging if needed.
        parts = [self._left_term.processed_actions, self._right_term.processed_actions, self._grasp]
        if self._left_grip is not None and self._right_grip is not None:
            parts.extend([self._left_grip.processed_actions, self._right_grip.processed_actions])
        return torch.cat(parts, dim=1)

    def _update_handover_mask(self):
        """Sticky flag once left closes and right opens near the object."""
        if not self.cfg.decouple_after_handover:
            self._handover_mask[:] = False
            self._env.extras["handover_mask"] = self._handover_mask
            return

        if not self.cfg.include_grasp:
            self._handover_mask[:] = False
            self._env.extras["handover_mask"] = self._handover_mask
            return

        left_cmd = self._grasp[:, 0] > 0.5
        right_cmd = self._grasp[:, 1] > 0.5
        obj_pos = self._env.scene["object"].data.root_pos_w - self._env.scene.env_origins
        body_pos = self._asset.data.body_pos_w - self._env.scene.env_origins[:, None, :]
        left = body_pos[:, self._left_body_id]

        dist_left = torch.norm(obj_pos - left, dim=-1)
        warm_ok = self._step_counter >= self.cfg.warmup_steps
        # Consider handover done when left closed near object AND right opened (or at least not closing).
        # This avoids triggering post-handover logic while both hands are still squeezing the object.
        ready = warm_ok & left_cmd & (~right_cmd) & (dist_left < self.cfg.handover_obj_tol)
        prev_mask = self._handover_mask.clone()
        new_ready = ready & (~prev_mask)
        self._handover_mask = prev_mask | ready
        self._env.extras["handover_mask"] = self._handover_mask
        # mark step when handover happened for downstream rewards/terminations
        if "handover_step" not in self._env.extras:
            self._env.extras["handover_step"] = torch.full_like(self._step_counter, -1)
        if torch.any(new_ready):
            handover_step = self._env.extras["handover_step"]
            handover_step[new_ready] = self._step_counter[new_ready]
            self._env.extras["handover_step"] = handover_step

    def _compute_center(self):
        """Use current hand positions to define symmetry center; clamp Z above table."""
        body_pos_w = self._asset.data.body_pos_w - self._env.scene.env_origins[:, None, :]
        left = body_pos_w[:, self._asset.data.body_names.index(self.cfg.left_body_name)]
        right = body_pos_w[:, self._asset.data.body_names.index(self.cfg.right_body_name)]
        center = 0.5 * (left + right)
        center[:, 2] = torch.clamp(center[:, 2], min=self.cfg.center_z_floor)
        return center

    def process_actions(self, actions: torch.Tensor):
        # store raw
        self._raw_actions[:] = actions
        # split
        task_actions = actions[:, : self._task_dim]
        if self.cfg.disable_arms:
            task_actions = torch.zeros_like(task_actions)
        # grasp (optional)
        if self.cfg.include_grasp:
            grasp_raw = actions[:, self._task_dim : self._task_dim + 2]
            self._grasp[:] = (torch.sigmoid(grasp_raw) > 0.5).float()
            # warmup: force right closed, left open for initial steps
            warm_mask = (self._step_counter < self.cfg.warmup_steps).unsqueeze(-1)
            self._grasp = torch.where(warm_mask, torch.tensor([0.0, 1.0], device=self.device), self._grasp)
        # latch handover state once left closed and right opened near object
        self._update_handover_mask()
        handover_mask = self._handover_mask.unsqueeze(-1)
        if self.cfg.decouple_after_handover and self.cfg.post_handover_left_scale < 1.0:
            task_actions = torch.where(
                handover_mask, task_actions * self.cfg.post_handover_left_scale, task_actions
            )
        # left processes directly
        self._left_term.process_actions(task_actions)
        # symmetry center from current hands
        center = self._compute_center()
        # mirror for right hand: negate pos/rot deltas about center
        right_actions = task_actions.clone()
        right_actions[:, :3] = -right_actions[:, :3]
        right_actions[:, 3:] = -right_actions[:, 3:]
        if self.cfg.decouple_after_handover:
            right_actions = torch.where(
                handover_mask, right_actions * self.cfg.post_handover_right_scale, right_actions
            )
            if self.cfg.force_right_open_after_handover and self.cfg.include_grasp:
                self._grasp[:, 1] = torch.where(
                    handover_mask.squeeze(-1), torch.zeros_like(self._grasp[:, 1]), self._grasp[:, 1]
                )
        self._right_term.process_actions(right_actions)
        # drive binary grippers: open=+1, close=-1
        if self._left_grip is not None and self._right_grip is not None and self.cfg.include_grasp:
            if not self.cfg.disable_grasp:
                left_bin = torch.where(self._grasp[:, 0:1] > 0.5, -torch.ones_like(self._grasp[:, 0:1]), torch.ones_like(self._grasp[:, 0:1]))
                right_bin = torch.where(self._grasp[:, 1:2] > 0.5, -torch.ones_like(self._grasp[:, 1:2]), torch.ones_like(self._grasp[:, 1:2]))
                self._left_grip.process_actions(left_bin)
                self._right_grip.process_actions(right_bin)
        # expose grasp command to extras for downstream logging/reward if desired
        self._env.extras["grasp_cmd"] = self._grasp

    def apply_actions(self):
        if self._disable_arms:
            if self._arm_ids:
                zeros = torch.zeros_like(self._arm_default)
                self._asset.set_joint_position_target(self._arm_default, joint_ids=self._arm_ids)
                self._asset.set_joint_velocity_target(zeros, joint_ids=self._arm_ids)
                self._asset.set_joint_effort_target(zeros, joint_ids=self._arm_ids)
        else:
            self._left_term.apply_actions()
            self._right_term.apply_actions()
        if self._freeze_waist and self._waist_ids:
            zeros = torch.zeros_like(self._waist_default)
            # aggressively hold waist at default
            self._asset.set_joint_position_target(self._waist_default, joint_ids=self._waist_ids)
            self._asset.set_joint_velocity_target(zeros, joint_ids=self._waist_ids)
            self._asset.set_joint_effort_target(zeros, joint_ids=self._waist_ids)
        if self._left_grip is not None:
            self._left_grip.apply_actions()
        if self._right_grip is not None:
            self._right_grip.apply_actions()
        # step counter increments each env step
        self._step_counter += 1
        # expose step counter for term/reward gating
        self._env.extras["step_counter"] = self._step_counter.clone()

    def reset(self, env_ids=None):
        self._left_term.reset(env_ids)
        self._right_term.reset(env_ids)
        if self._left_grip is not None:
            self._left_grip.reset(env_ids)
        if self._right_grip is not None:
            self._right_grip.reset(env_ids)
        # reset step counter
        if env_ids is None:
            self._step_counter[:] = 0
            self._handover_mask[:] = False
        else:
            self._step_counter[env_ids] = 0
            self._handover_mask[env_ids] = False
        self._env.extras["handover_mask"] = self._handover_mask
        # reset episode-scoped counters used by terminations
        zeros_long = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._env.extras["success_counter"] = zeros_long.clone()
        self._env.extras["both_off_counter"] = zeros_long.clone()
        self._env.extras["stuck_counter"] = zeros_long.clone()
        self._env.extras["clamp_counter"] = zeros_long.clone()
        self._env.extras["pre_crowd_counter"] = zeros_long.clone()
        self._env.extras["handover_step"] = torch.full_like(zeros_long, -1)
        if "prev_obj_z" in self._env.extras:
            del self._env.extras["prev_obj_z"]
        if "prev_obj_z_clamp" in self._env.extras:
            del self._env.extras["prev_obj_z_clamp"]
        if "prev_hand_sep" in self._env.extras:
            del self._env.extras["prev_hand_sep"]
        # cache spawn height for stuck detection
        try:
            obj = self._env.scene["object"]
        except KeyError:
            obj = None
        if obj is not None:
            self._env.extras["obj_spawn_z"] = (
                obj.data.default_root_state[:, 2] - self._env.scene.env_origins[:, 2]
            ).clone()
        # default: left open, right closed so object stays in right hand
        open_act = torch.ones((self.num_envs, 1), device=self.device)
        close_act = -torch.ones((self.num_envs, 1), device=self.device)
        if self._left_grip is not None:
            self._left_grip.process_actions(open_act)
        if self._right_grip is not None:
            self._right_grip.process_actions(close_act)
        # grasp bits default: left open (0), right closed (1)
        self._grasp[:] = 0.0
        self._grasp[:, 1] = 1.0


@configclass
class SymmetricDualIKActionCfg(ActionTermCfg):
    """Config for symmetric dual-hand IK with optional binary grasp commands."""

    class_type = SymmetricDualIKAction

    # IK settings
    asset_name: str = "robot"
    left_joint_names: list[str] = MISSING
    right_joint_names: list[str] = MISSING
    left_body_name: str = "left_hand_pitch_link"
    right_body_name: str = "right_hand_pitch_link"
    body_offset: DifferentialInverseKinematicsActionCfg.OffsetCfg | None = None
    scale: float = 0.25
    controller: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=True, ik_method="dls"
    )
    include_grasp: bool = True
    freeze_waist: bool = True
    disable_arms: bool =  False
    disable_grasp: bool = False
    center_z_floor: float = 0.65  # keep symmetry plane above the table
    warmup_steps: int = 10   # force right hand closed for initial steps
    decouple_after_handover: bool = True  # stop mirroring once left hand takes over
    post_handover_left_scale: float = 0.2  # damp left-hand commands after handover
    post_handover_right_scale: float = 0.5  # keep some motion on right hand after handover
    force_right_open_after_handover: bool = True
    handover_obj_tol: float = 0.14
    left_gripper_joint_names: list[str] = (
        ["L_index_.*", "L_middle_.*", "L_pinky_.*", "L_ring_.*", "L_thumb_.*"]
    )
    left_gripper_open_command: dict[str, float] = field(
        default_factory=lambda: {
            "L_index_.*": 0.0,
            "L_middle_.*": 0.0,
            "L_pinky_.*": 0.0,
            "L_ring_.*": 0.0,
            "L_thumb_.*": 0.0,
        }
    )
    left_gripper_close_command: dict[str, float] = field(
        default_factory=lambda: {
            "L_index_.*": -1.0,
            "L_middle_.*": -1.0,
            "L_pinky_.*": -1.0,
            "L_ring_.*": -1.0,
            "L_thumb_proximal_yaw_joint": -1.7,
            "L_thumb_proximal_pitch_joint": 0.35,
            "L_thumb_distal_joint": 1.0,
        }
    )
    right_gripper_joint_names: list[str] | None = None
    right_gripper_open_command: dict[str, float] | None = None
    right_gripper_close_command: dict[str, float] | None = None
