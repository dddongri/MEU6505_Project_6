"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

# --- Force-register GR1T2 tasks with Gymnasium (robust) ---
try:
    import isaaclab_assets.robots.fourier  # 자산 의존성 임포트(일부 cfg에서 필요)
    from gymnasium.envs.registration import register
    from .tasks.manipulation.pick_place.config.gr1t2 import basic_env_cfg as pp_basic_env_cfg, agents as pp_agents

    register(
        id="GR1T2-Basic",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": pp_basic_env_cfg.GR1T2BasicEnvCfg,
            "rsl_rl_cfg_entry_point": f"{pp_agents.__name__}.rsl_rl_ppo_cfg:GR1T2BasicPPORunnerCfg",
        },
    )
    register(
        id="GR1T2-Basic-Play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": pp_basic_env_cfg.GR1T2BasicEnvCfg_PLAY,
            "rsl_rl_cfg_entry_point": f"{pp_agents.__name__}.rsl_rl_ppo_cfg:GR1T2BasicPPORunnerCfg",
        },
    )
    print("[GR1T2] gymnasium.register done")
except Exception as e:
    print("[GR1T2] explicit register failed:", e)


# --- Force-register GR1T2 Hand-to-Hand tasks with Gymnasium ---
try:
    from gymnasium.envs.registration import register
    from .tasks.manipulation.hand_to_hand.config.gr1t2 import basic_env_cfg as h2h_basic_env_cfg
    from .tasks.manipulation.pick_place.config.gr1t2 import agents as h2h_agents
    register(
        id="GR1T2-HandToHand",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": h2h_basic_env_cfg.GR1T2BasicEnvCfg,
            "rsl_rl_cfg_entry_point": f"{h2h_agents.__name__}.rsl_rl_ppo_cfg:GR1T2BasicPPORunnerCfg",
        },
    )
    register(
        id="GR1T2-HandToHand-Play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": h2h_basic_env_cfg.GR1T2BasicEnvCfg_PLAY,
            "rsl_rl_cfg_entry_point": f"{h2h_agents.__name__}.rsl_rl_ppo_cfg:GR1T2BasicPPORunnerCfg",
        },
    )
    print("[GR1T2-HandToHand] gymnasium.register done")
except Exception as e:
    print("[GR1T2-HandToHand] explicit register failed:", e)
