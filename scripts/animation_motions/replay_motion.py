"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --motion_file source/groot/groot/assets/g1/motions/lafan_walk_short.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--playback", type=float, default=None, help="Playback speed. If not set, runs in real-time.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from groot.robots.fourier import GR1T2_CFG
from groot.motions import GR1T2_MOTION_DIR
from groot.tasks.manager_based.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = GR1T2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, motion_file: str):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    print("motion time_step_total: ", motion.time_step_total)

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def run_simulator_playback(sim: sim_utils.SimulationContext, scene: InteractiveScene, motion_file: str):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )

    N = scene.num_envs
    device = sim.device
    T = motion.time_step_total
    print("motion time_step_total: ", motion.time_step_total)

    motion_dt = 1.0 / 30.0
    t_idx_f = torch.zeros(N, dtype=torch.float32, device=device)
    increment = torch.full((N,), args_cli.playback * (sim_dt / motion_dt), device=device)

    def quat_slerp(q0, q1, a, eps=1e-8):
        dot = (q0 * q1).sum(dim=-1)
        q1 = torch.where(dot.unsqueeze(-1) < 0, -q1, q1)
        dot = torch.abs(dot)
        close = dot > (1.0 - 1e-6)
        lerp = torch.nn.functional.normalize((1 - a).unsqueeze(-1) * q0 + a.unsqueeze(-1) * q1, dim=-1)
        theta = torch.acos(dot.clamp(-1 + eps, 1 - eps))
        sin_theta = torch.sin(theta)
        w0 = torch.sin((1 - a) * theta) / (sin_theta + eps)
        w1 = torch.sin(a * theta)       / (sin_theta + eps)
        slerp = torch.nn.functional.normalize(w0.unsqueeze(-1) * q0 + w1.unsqueeze(-1) * q1, dim=-1)
        return torch.where(close.unsqueeze(-1), lerp, slerp)

    while simulation_app.is_running():
        t_idx_f = torch.remainder(t_idx_f + increment, T - 1.0)  # T-1
        i0 = torch.floor(t_idx_f).to(torch.long)                 # [N]
        i1 = (i0 + 1) % T
        a  = (t_idx_f - i0.float()).clamp(0.0, 1.0)              # [N]

        root_pos0  = motion.body_pos_w[i0][:, 0, :]              # [N,3]
        root_pos1  = motion.body_pos_w[i1][:, 0, :]
        root_pos   = (1 - a).unsqueeze(-1) * root_pos0 + a.unsqueeze(-1) * root_pos1
        root_pos   = root_pos + scene.env_origins[:, None, :].squeeze(1)

        root_quat0 = motion.body_quat_w[i0][:, 0, :]             # [N,4]
        root_quat1 = motion.body_quat_w[i1][:, 0, :]
        root_quat  = quat_slerp(root_quat0, root_quat1, a)       # [N,4]

        root_lin0  = motion.body_lin_vel_w[i0][:, 0, :]          # [N,3]
        root_lin1  = motion.body_lin_vel_w[i1][:, 0, :]
        root_lin   = (1 - a).unsqueeze(-1) * root_lin0 + a.unsqueeze(-1) * root_lin1

        root_ang0  = motion.body_ang_vel_w[i0][:, 0, :]
        root_ang1  = motion.body_ang_vel_w[i1][:, 0, :]
        root_ang   = (1 - a).unsqueeze(-1) * root_ang0 + a.unsqueeze(-1) * root_ang1

        q0  = motion.joint_pos[i0]                                # [N, n_j]
        q1  = motion.joint_pos[i1]
        q   = (1 - a).unsqueeze(-1) * q0 + a.unsqueeze(-1) * q1

        qd0 = motion.joint_vel[i0]                                # [N, n_j]
        qd1 = motion.joint_vel[i1]
        qd  = (1 - a).unsqueeze(-1) * qd0 + a.unsqueeze(-1) * qd1

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3]   = root_pos
        root_states[:, 3:7]  = root_quat
        root_states[:, 7:10] = root_lin
        root_states[:, 10:]  = root_ang

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(q, qd)
        scene.write_data_to_sim()

        sim.render()
        scene.update(sim_dt)

        pos_lookat = root_pos[0, :].detach().cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Get motion file path
    motion_file = str(f"{GR1T2_MOTION_DIR}/animation/groot_b/groot_b.npz")

    # Run the simulator
    if args_cli.playback is not None:
        run_simulator_playback(sim, scene, motion_file)
    else:
        run_simulator(sim, scene, motion_file)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
