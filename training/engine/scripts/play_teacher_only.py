"""Visual teacher-only rollout (Phase 0 Gate verification, no checkpoint)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "training" / "engine"
sys.path.insert(0, str(ENGINE_ROOT / "source" / "isaaclab_stage3"))

import isaaclab_stage3  # noqa: F401
from isaaclab.app import AppLauncher  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", default="basic14")
    parser.add_argument("--randomization", default="eval", choices=["train", "eval", "stress"])
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    os.environ["STAGE3_SCENARIO_CURRICULUM"] = args.curriculum
    os.environ["STAGE3_RANDOMIZATION_MODE"] = args.randomization
    os.environ["STAGE3_TEACHER_MODE"] = "only"
    os.environ["STAGE3_REWARD_STAGE"] = "A"

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym  # noqa: E402
    from isaaclab_stage3.tasks.intercept.intercept_env import Stage3InterceptEnvCfg  # noqa: E402

    env_cfg = Stage3InterceptEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device if hasattr(args, "device") and args.device else env_cfg.sim.device

    env = gym.make("Isaac-Stage3-Intercept-Direct-v0", cfg=env_cfg)
    unwrapped = env.unwrapped
    env.reset(seed=args.seed)

    print(
        f"[TeacherPlay] curriculum={args.curriculum} randomization={args.randomization} "
        f"teacher_mode=only (no .pth checkpoint)",
        flush=True,
    )

    for step in range(args.steps):
        teacher = unwrapped.get_teacher_action_for_bc()
        env.step(teacher)
        if step % 200 == 0:
            guidance = unwrapped._flight_t() >= unwrapped.boost_duration
            if guidance.any():
                hit, contact, center, _, _ = unwrapped._evaluate_intercept()
                print(
                    f"[TeacherPlay] step={step} contact={contact.mean().item():.1f} "
                    f"center={center.mean().item():.1f} hit={hit.float().mean().item():.3f} "
                    f"ep_hit={unwrapped.episode_hit.mean().item():.3f}",
                    flush=True,
                )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
