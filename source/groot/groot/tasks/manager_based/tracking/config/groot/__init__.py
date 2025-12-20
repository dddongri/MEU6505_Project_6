import gymnasium as gym

from . import agents, flat_anim_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="GROOT-Mimic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_anim_env_cfg.GR1T2EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GR1T2PPORunnerCfg",
    },
)

gym.register(
    id="GROOT-Mimic-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_anim_env_cfg.GR1T2EnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GR1T2PPORunnerCfg",
    },
)
