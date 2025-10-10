import gymnasium as gym

from . import agents, basic_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="GR1T2-Basic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": basic_env_cfg.GR1T2BasicEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GR1T2BasicPPORunnerCfg",
    },
)

gym.register(
    id="GR1T2-Basic-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": basic_env_cfg.GR1T2BasicEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GR1T2BasicPPORunnerCfg",
    },
)

