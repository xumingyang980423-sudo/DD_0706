from __future__ import annotations

import csv
from pathlib import Path

from intercept_core import InterceptConfig, InterceptScenario


def main() -> None:
    cfg = InterceptConfig()
    output_dir = Path("logs")
    output_dir.mkdir(exist_ok=True)

    rows = []
    scenario = InterceptScenario(cfg=cfg, seed=7)

    fixed_result = scenario.run_baseline_episode(randomize=False)
    scenario.write_records_csv(output_dir / "baseline_fixed_episode.csv")
    rows.append({"episode": 0, "randomize": False, **fixed_result})

    for episode in range(1, 21):
        result = scenario.run_baseline_episode(randomize=True)
        rows.append({"episode": episode, "randomize": True, **result})

    summary_path = output_dir / "baseline_batch_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "randomize", "status", "closest_distance", "time", "records"])
        writer.writeheader()
        writer.writerows(rows)

    hits = sum(1 for row in rows if row["status"] == "hit")
    avg_closest = sum(float(row["closest_distance"]) for row in rows) / len(rows)
    print(f"episodes={len(rows)}")
    print(f"hits={hits}")
    print(f"hit_rate={hits / len(rows):.3f}")
    print(f"avg_closest_distance={avg_closest:.3f}")
    print(f"summary_csv={summary_path}")
    print(f"fixed_episode_csv={output_dir / 'baseline_fixed_episode.csv'}")


if __name__ == "__main__":
    main()
