"""CPU baseline eval using intercept_core (no Isaac Sim)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

CORE_ROOT = Path(__file__).resolve().parents[2] / "core"
sys.path.insert(0, str(CORE_ROOT))

from intercept_core import InterceptScenario  # noqa: E402

BASIC14 = [
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


def eval_scenario(name: str, episodes: int = 20, randomize: bool = True) -> dict:
    hits = 0
    closest = []
    for _ in range(episodes):
        env = InterceptScenario()
        env.reset(randomize=randomize, scenario_type=name)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(use_baseline=True)
            done = term or trunc
        if info.get("status") == "hit":
            hits += 1
        closest.append(env.closest_distance)
    return {
        "scenario": name,
        "hit_rate": hits / episodes,
        "mean_closest": float(np.mean(closest)),
    }


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    results = [eval_scenario(name) for name in BASIC14]
    overall_hit = float(np.mean([r["hit_rate"] for r in results]))
    overall_closest = float(np.mean([r["mean_closest"] for r in results]))
    print(f"[CPU Baseline] overall hit_rate={overall_hit:.3f} mean_closest={overall_closest:.1f}")
    for r in sorted(results, key=lambda x: x["hit_rate"]):
        print(f"  {r['scenario']:28s} hit={r['hit_rate']:.2f} closest={r['mean_closest']:.1f}")


if __name__ == "__main__":
    main()
