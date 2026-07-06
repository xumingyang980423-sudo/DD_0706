from __future__ import annotations

import numpy as np

from intercept_core import InterceptConfig, InterceptScenario

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


if gym is not None:
    _BaseEnv = gym.Env
else:
    _BaseEnv = object


STAGE2_SCENARIOS = [
    "head_on",
    "overfly_tail_chase",
    "crossing_left_to_right",
    "crossing_right_to_left",
    "climb_escape",
    "dive_escape",
    "s_turn_evasion",
    "double_evasion",
    "late_launch",
    "high_speed_pass",
    "low_altitude_pass",
    "far_tail_chase",
    "fighter_weave_chase",
    "maneuver_follow_chase",
]


class MissileInterceptEnv(_BaseEnv):
    """Gymnasium-style RL environment for the guidance phase.

    The boost phase is rule-controlled. RL actions are applied only after
    the scenario enters GUIDANCE.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: InterceptConfig | None = None,
        randomize: bool = True,
        scenarios: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.cfg = cfg or InterceptConfig()
        self.randomize = randomize
        self.scenarios = list(scenarios or STAGE2_SCENARIOS)
        self.scenario = InterceptScenario(cfg=self.cfg)
        self._last_obs = self.scenario.observation()

        if spaces is not None:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.scenario = InterceptScenario(cfg=self.cfg, seed=seed)
        randomize = self.randomize
        if options is not None and "randomize" in options:
            randomize = bool(options["randomize"])
        if options is not None and "scenario" in options:
            scenario_type = str(options["scenario"])
        else:
            scenario_type = str(self.scenario.rng.choice(self.scenarios))
        self._last_obs = self.scenario.reset(randomize=randomize, scenario_type=scenario_type)
        info = {
            "phase": self.scenario.phase.value,
            "scenario": self.scenario.scenario_type,
            "time": self.scenario.t,
            "closest_distance": self.scenario.closest_distance,
        }
        return self._last_obs, info

    def step(self, action):
        action = np.asarray(action, dtype=float)
        obs, reward, terminated, truncated, info = self.scenario.step(action=action, use_baseline=False)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        return None
