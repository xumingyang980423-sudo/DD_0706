"""Save sample IR frames to PNG for visual debug (Phase 6A)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "training" / "engine"
import sys

sys.path.insert(0, str(ENGINE_ROOT / "source" / "isaaclab_stage3"))

import isaaclab_stage3  # noqa: F401
from isaaclab.app import AppLauncher  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", default="basic14")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--fixed_sid", type=int, default=-1)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    os.environ["STAGE3_IR_ENABLE"] = "1"
    os.environ["STAGE3_SCENARIO_CURRICULUM"] = args.curriculum
    os.environ["STAGE3_TEACHER_MODE"] = "only"
    os.environ["STAGE3_RANDOMIZATION_MODE"] = "eval"
    if args.fixed_sid >= 0:
        os.environ["STAGE3_FIXED_SCENARIO_ID"] = str(args.fixed_sid)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym  # noqa: E402
    from isaaclab_stage3.tasks.intercept.intercept_env import Stage3InterceptEnvCfg  # noqa: E402

    try:
        from PIL import Image
    except ImportError:
        Image = None

    env_cfg = Stage3InterceptEnvCfg()
    env_cfg.scene.num_envs = 1
    env = gym.make("Isaac-Stage3-Intercept-Direct-v0", cfg=env_cfg)
    unwrapped = env.unwrapped
    env.reset()

    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "data" / "ir" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for step in range(args.steps):
        teacher = unwrapped.get_teacher_action_for_bc()
        env.step(teacher)
        guidance = unwrapped._flight_t() >= unwrapped.boost_duration
        if not guidance[0]:
            continue
        ir = unwrapped.get_ir_outputs()
        if not bool(ir["locked"][0].item()):
            continue
        if Image is None:
            print("[IR Debug] PIL not installed; skip PNG save", flush=True)
            break
        frame = (ir["frame"][0].detach().cpu().numpy() * 255.0).astype("uint8")
        u, v = ir["track"][0, 1].item(), ir["track"][0, 2].item()
        path = out_dir / f"ir_step{step:05d}_u{u:+.2f}_v{v:+.2f}.png"
        Image.fromarray(frame, mode="L").save(path)
        saved += 1
        if saved >= 12:
            break

    print(f"[IR Debug] Saved {saved} PNG(s) -> {out_dir}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
