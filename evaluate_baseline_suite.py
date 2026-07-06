from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from intercept_core import InterceptConfig, InterceptScenario


@dataclass(frozen=True)
class ScenarioCase:
    name: str
    scenario_type: str
    episodes: int
    randomize: bool


SCENARIO_SUITE = [
    ScenarioCase("head_on_frontal", "head_on", 60, True),
    ScenarioCase("overfly_tail_chase", "overfly_tail_chase", 60, True),
    ScenarioCase("crossing_left_to_right", "crossing_left_to_right", 60, True),
    ScenarioCase("crossing_right_to_left", "crossing_right_to_left", 60, True),
    ScenarioCase("climb_escape", "climb_escape", 60, True),
    ScenarioCase("dive_escape", "dive_escape", 60, True),
    ScenarioCase("s_turn_evasion", "s_turn_evasion", 70, True),
    ScenarioCase("double_evasion", "double_evasion", 70, True),
    ScenarioCase("late_launch", "late_launch", 60, True),
    ScenarioCase("high_speed_pass", "high_speed_pass", 60, True),
    ScenarioCase("low_altitude_pass", "low_altitude_pass", 60, True),
    ScenarioCase("far_tail_chase", "far_tail_chase", 60, True),
    ScenarioCase("fighter_weave_chase", "fighter_weave_chase", 60, True),
    ScenarioCase("maneuver_follow_chase", "maneuver_follow_chase", 60, True),
    ScenarioCase("long_weave_tail_chase", "long_weave_tail_chase", 60, True),
    ScenarioCase("extended_maneuver_follow", "extended_maneuver_follow", 60, True),
    ScenarioCase("climb_dive_weave_chase", "climb_dive_weave_chase", 60, True),
    ScenarioCase("delayed_sustained_evasion", "delayed_sustained_evasion", 60, True),
    ScenarioCase("cobra_pop_up_chase", "cobra_pop_up_chase", 60, True),
    ScenarioCase("circle_turn_chase", "circle_turn_chase", 60, True),
    ScenarioCase("spiral_climb_chase", "spiral_climb_chase", 60, True),
    ScenarioCase("hard_reversal_chase", "hard_reversal_chase", 60, True),
    ScenarioCase("wide_snake_chase", "wide_snake_chase", 60, True),
    ScenarioCase("super_combo_chase", "super_combo_chase", 60, True),
]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[idx]


def evaluate(output_dir: Path, seed: int, quick: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = InterceptConfig()
    scenario = InterceptScenario(cfg=cfg, seed=seed)

    episode_rows = []
    failure_rows = []

    for case in SCENARIO_SUITE:
        episodes = min(case.episodes, 10) if quick else case.episodes
        for episode_idx in range(episodes):
            result = scenario.run_baseline_episode(randomize=case.randomize, scenario_type=case.scenario_type)
            status = str(result["status"])
            row = {
                "scenario": case.name,
                "scenario_type": case.scenario_type,
                "episode": episode_idx,
                "status": status,
                "hit": status == "hit",
                "closest_distance": float(result["closest_distance"]),
                "time": float(result["time"]),
                "records": int(result["records"]),
                "final_phase": scenario.phase.value,
                "max_missile_altitude": max(record.missile_z for record in scenario.records),
                "max_missile_speed": max(record.missile_speed for record in scenario.records),
                "max_abs_yaw": max(abs(record.action_yaw) for record in scenario.records),
                "max_abs_pitch": max(abs(record.action_pitch) for record in scenario.records),
            }
            episode_rows.append(row)

            if status != "hit":
                last = scenario.records[-1]
                failure_rows.append(
                    {
                        **row,
                        "final_distance": last.distance,
                        "final_aircraft_x": last.aircraft_x,
                        "final_aircraft_y": last.aircraft_y,
                        "final_aircraft_z": last.aircraft_z,
                        "final_missile_x": last.missile_x,
                        "final_missile_y": last.missile_y,
                        "final_missile_z": last.missile_z,
                    }
                )

    summary_rows = []
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for row in episode_rows:
        by_scenario[row["scenario"]].append(row)

    for scenario_name, rows in by_scenario.items():
        statuses = Counter(str(row["status"]) for row in rows)
        hits = statuses.get("hit", 0)
        closest = [float(row["closest_distance"]) for row in rows]
        times = [float(row["time"]) for row in rows]
        altitudes = [float(row["max_missile_altitude"]) for row in rows]
        speeds = [float(row["max_missile_speed"]) for row in rows]
        summary_rows.append(
            {
                "scenario": scenario_name,
                "episodes": len(rows),
                "hits": hits,
                "hit_rate": hits / len(rows),
                "avg_closest_distance": mean(closest),
                "p95_closest_distance": percentile(closest, 0.95),
                "max_closest_distance": max(closest),
                "avg_time": mean(times),
                "max_time": max(times),
                "avg_max_missile_altitude": mean(altitudes),
                "max_missile_altitude": max(altitudes),
                "avg_max_missile_speed": mean(speeds),
                "max_missile_speed": max(speeds),
                "status_counts": ";".join(f"{key}:{value}" for key, value in sorted(statuses.items())),
            }
        )

    suite_summary_path = output_dir / "baseline_suite_summary.csv"
    episode_path = output_dir / "baseline_suite_episodes.csv"
    failure_path = output_dir / "baseline_suite_failures.csv"

    with episode_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    with suite_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with failure_path.open("w", newline="", encoding="utf-8") as handle:
        if failure_rows:
            writer = csv.DictWriter(handle, fieldnames=list(failure_rows[0].keys()))
            writer.writeheader()
            writer.writerows(failure_rows)
        else:
            handle.write("scenario,episode,status\n")

    total = len(episode_rows)
    hits = sum(1 for row in episode_rows if row["hit"])
    print(f"episodes={total}")
    print(f"hits={hits}")
    print(f"hit_rate={hits / total:.3f}")
    print(f"summary_csv={suite_summary_path}")
    print(f"episodes_csv={episode_path}")
    print(f"failures_csv={failure_path}")

    print("\nScenario summary:")
    for row in summary_rows:
        print(
            f"{row['scenario']}: hit_rate={row['hit_rate']:.3f}, "
            f"avg_closest={row['avg_closest_distance']:.2f}, "
            f"p95_closest={row['p95_closest_distance']:.2f}, "
            f"avg_time={row['avg_time']:.2f}, statuses={row['status_counts']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="logs", help="Directory for evaluation CSV files.")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--quick", action="store_true", help="Run 10 episodes per scenario for a fast smoke test.")
    args = parser.parse_args()
    evaluate(Path(args.output_dir), args.seed, args.quick)


if __name__ == "__main__":
    main()
