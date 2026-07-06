"""Abstract ground-intercept DirectRLEnv registration."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Stage3-Intercept-Direct-v0",
    entry_point=f"{__name__}.intercept_env:Stage3InterceptEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.intercept_env:Stage3InterceptEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Stage3InterceptPPORunnerCfg",
    },
)
