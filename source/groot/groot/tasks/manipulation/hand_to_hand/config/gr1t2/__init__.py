import gymnasium as gym
from . import agents, basic_env_cfg
from ...hand_to_hand_env import GR1T2HandToHandEnv

gym.register(
    id="GR1T2-HandToHand",
    entry_point="groot.tasks.manipulation.hand_to_hand.hand_to_hand_env:GR1T2HandToHandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": basic_env_cfg.GR1T2BasicEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GR1T2BasicPPORunnerCfg",
    },
)

gym.register(
    id="GR1T2-HandToHand-Play",
    entry_point="groot.tasks.manipulation.hand_to_hand.hand_to_hand_env:GR1T2HandToHandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": basic_env_cfg.GR1T2BasicEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GR1T2BasicPPORunnerCfg",
    },
)
