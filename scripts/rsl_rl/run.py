"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import time
import torch
import gymnasium as gym

import isaacsim
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import groot.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config



class PolicyPlayer:
    def __init__(self, env, agent_cfg, device):
        print("PolicyPlayer initialized")
        self.env = env
        self.agent_cfg = agent_cfg
        self.device = device
        

        self.models = {
            "pick_left_hand": "/home/qkd/Desktop/MEU6505_Project_6/scripts/rsl_rl/skills/skill_1.pt",
            "place_right_hand": "/home/qkd/Desktop/MEU6505_Project_6/scripts/rsl_rl/skills/skill_1.pt",
            "hand_over_left_to_right": "/home/qkd/Desktop/MEU6505_Project_6/scripts/rsl_rl/skills/skill_1.pt"
        }
        
        self.policy = None          
        self.is_playing = False     
        self.current_step = 0       
        self.max_steps = 200          #n스텝동안 실행하게 함(MCP에서 넘겨주는 인수)

    def rollout_skill(self, skill_name: str, max_steps: int = 200) -> dict:
        print(f"[PolicyPlayer] Request received: Executing skill '{skill_name}' for {max_steps} steps.")
        
        if skill_name not in self.models:
            print(f"[Error] Unknown skill: {skill_name}")
            return {"status": "error", "message": f"Unknown skill: {skill_name}"}
            
        file_path = self.models[skill_name]
        
        if not os.path.exists(file_path):
            print(f"[Error] File not found at: {file_path}")
            return {"status": "error", "message": "Checkpoint file not found"}

        try:
            log_dir = os.path.dirname(file_path)
            
            if self.agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(self.env, self.agent_cfg.to_dict(), log_dir=None, device=self.agent_cfg.device)
            elif self.agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(self.env, self.agent_cfg.to_dict(), log_dir=None, device=self.agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {self.agent_cfg.class_name}")
            
            runner.load(file_path)
            
            self.policy = runner.get_inference_policy(device=self.env.unwrapped.device)
            
            self.is_playing = True
            self.current_step = 0
            self.max_steps = max_steps
            
            print(f"[Success] Skill '{skill_name}' loaded and started.")
            return {"status": "success", "message": f"Skill '{skill_name}' started!"}

        except Exception as e:
            print(f"[Error] Failed to load checkpoint or create runner: {e}")
            self.is_playing = False
            return {"status": "error", "message": str(e)}

    # 루프마다 호출하는 콜백함수(action = policy(obs)를 받아옴)
    def get_action(self, obs):

        if self.is_playing and self.policy is not None:
            action = self.policy(obs)
            self.current_step += 1
            
            if self.current_step >= self.max_steps:
                print(f"[PolicyPlayer] Skill execution finished ({self.max_steps} steps). Returning to idle.")
                self.is_playing = False
                
            return action
        
        num_envs = obs.shape[0]

        if hasattr(self.env, "num_actions"):
            action_dim = self.env.num_actions
        else:
            try:
                action_dim = self.env.unwrapped.action_space.shape[0]
            except:
                action_dim = 12
                
        return torch.zeros((num_envs, action_dim), device=self.device)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    dt = env.unwrapped.step_dt

    policy_player = PolicyPlayer(env, agent_cfg, device=env.unwrapped.device)
    
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.policy_player = policy_player 

    obs = env.get_observations()
    
    print("---------------------------------------------------------")
    print("[INFO] Simulator Ready. Waiting for 'policy_player.rollout_skill()' call...")
    print("---------------------------------------------------------")


    while simulation_app.is_running():
        start_time = time.time()
        
        with torch.inference_mode():
            actions = policy_player.get_action(obs)
            obs, _, _, _ = env.step(actions)

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()

if __name__ == "__main__":
    main()