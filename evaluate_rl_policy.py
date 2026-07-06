from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rl_env import MissileInterceptEnv, STAGE2_SCENARIOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy on the 14 stage-2 scenarios.")
    parser.add_argument("--model", required=True, help="Path to a Stable-Baselines3 PPO .zip model.")
    parser.add_argument("--episodes-per-scenario", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--output", default="logs/rl/rl_policy_eval.csv")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = PPO.load(args.model)
    rows: list[dict] = []

    for scenario in STAGE2_SCENARIOS:
        hits = 0
        closest_distances = []
        times = []
        for episode in range(args.episodes_per_scenario):
            env = MissileInterceptEnv(randomize=True, scenarios=[scenario])
            obs, info = env.reset(seed=args.seed + episode, options={"randomize": True, "scenario": scenario})
            done = False
            steps = 0
            last_info = info
            total_reward = 0.0
            while not done and steps < 3000:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, last_info = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
                steps += 1
            status = last_info.get("status", "unknown")
            hit = status == "hit"
            hits += int(hit)
            closest_distances.append(float(last_info.get("closest_distance", np.nan)))
            times.append(float(last_info.get("time", np.nan)))
            rows.append(
                {
                    "scenario": scenario,
                    "episode": episode,
                    "status": status,
                    "hit": int(hit),
                    "closest_distance": closest_distances[-1],
                    "time": times[-1],
                    "steps": steps,
                    "total_reward": total_reward,
                }
            )
            env.close()
        hit_rate = hits / max(args.episodes_per_scenario, 1)
        print(
            f"{scenario:28s} hit_rate={hit_rate:.3f} "
            f"mean_closest={np.nanmean(closest_distances):.3f} "
            f"mean_time={np.nanmean(times):.3f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Evaluation written to: {output}")


if __name__ == "__main__":
    main()
