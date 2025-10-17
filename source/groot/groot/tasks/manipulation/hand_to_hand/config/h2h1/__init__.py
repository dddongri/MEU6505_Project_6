# agents/__init__.py  (for hand_to_hand)
import gymnasium as gym
from . import agents, basic_env_cfg

##
# Register Gym environments for Hand-to-Hand task.
##

gym.register(
    id="H2H1-Basic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":  basic_env_cfg.H2H1BasicEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H2H1BasicPPORunnerCfg",
    },
)

gym.register(
    id="H2H1-Basic-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":  basic_env_cfg.H2H1BasicEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H2H1BasicPPORunnerCfg",
    },
)
