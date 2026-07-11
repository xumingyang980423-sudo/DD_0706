"""Collect IR frames + GT track labels for TrackNet (Phase 6B, sharded)."""

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


def _flush_shard(
    frames: list,
    tracks: list,
    sids: list,
    shard_path: Path,
) -> int:
    import torch

    if not frames:
        return 0
    frame_u8 = (torch.cat(frames, dim=0).clamp(0.0, 1.0) * 255.0).to(torch.uint8).unsqueeze(1)
    data = {
        "frame": frame_u8,
        "track": torch.cat(tracks, dim=0).float(),
        "scenario_id": torch.cat(sids, dim=0).long(),
    }
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, shard_path)
    n = int(data["frame"].shape[0])
    frames.clear()
    tracks.clear()
    sids.clear()
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", default="basic14")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--flush_steps", type=int, default=200, help="Write shard to disk every N sim steps")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    os.environ["STAGE3_IR_ENABLE"] = "1"
    os.environ["STAGE3_SCENARIO_CURRICULUM"] = args.curriculum
    os.environ["STAGE3_TEACHER_MODE"] = "only"
    os.environ["STAGE3_RANDOMIZATION_MODE"] = "train"
    os.environ["STAGE3_REWARD_STAGE"] = "A"

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym  # noqa: E402
    import torch  # noqa: E402
    from isaaclab_stage3.tasks.intercept.intercept_env import Stage3InterceptEnvCfg  # noqa: E402

    out = Path(args.output) if args.output else PROJECT_ROOT / "data" / "ir" / args.curriculum / "ir_frames.pt"
    shard_dir = out.parent / f"{out.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = Stage3InterceptEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env = gym.make("Isaac-Stage3-Intercept-Direct-v0", cfg=env_cfg)
    unwrapped = env.unwrapped
    env.reset(seed=args.seed)

    frames: list[torch.Tensor] = []
    tracks: list[torch.Tensor] = []
    sids: list[torch.Tensor] = []
    shard_paths: list[str] = []
    total_samples = 0
    locked_sum = 0.0
    shard_idx = 0
    steps_since_flush = 0

    for step in range(args.steps):
        teacher = unwrapped.get_teacher_action_for_bc()
        env.step(teacher)
        guidance = unwrapped._flight_t() >= unwrapped.boost_duration
        if guidance.any():
            ir = unwrapped.get_ir_outputs()
            g = guidance
            chunk_track = ir["track"][g].detach().cpu()
            frames.append(ir["frame"][g].detach().cpu())
            tracks.append(chunk_track)
            sids.append(unwrapped.scenario_id[g].detach().cpu())
            locked_sum += float(chunk_track[:, 0].sum().item())

        steps_since_flush += 1
        if steps_since_flush >= args.flush_steps:
            shard_file = shard_dir / f"shard_{shard_idx:04d}.pt"
            n = _flush_shard(frames, tracks, sids, shard_file)
            if n > 0:
                shard_paths.append(str(shard_file.relative_to(out.parent)))
                total_samples += n
                print(f"[IR Collect] shard={shard_idx:04d} samples={n} total={total_samples}", flush=True)
                shard_idx += 1
            steps_since_flush = 0
        elif step % 500 == 0 and step > 0:
            print(f"[IR Collect] step={step} total_samples={total_samples}", flush=True)

    if frames:
        shard_file = shard_dir / f"shard_{shard_idx:04d}.pt"
        n = _flush_shard(frames, tracks, sids, shard_file)
        if n > 0:
            shard_paths.append(str(shard_file.relative_to(out.parent)))
            total_samples += n
            shard_idx += 1

    if total_samples <= 0:
        raise RuntimeError("No IR samples collected. Check guidance phase / STAGE3_IR_ENABLE.")

    manifest = {
        "format": "sharded_uint8",
        "curriculum": args.curriculum,
        "track_fields": ["locked", "u", "v", "u_dot", "v_dot", "confidence"],
        "frame_dtype": "uint8",
        "frame_scale": 255.0,
        "num_samples": total_samples,
        "num_shards": len(shard_paths),
        "shards": shard_paths,
        "shard_dir": str(shard_dir.relative_to(out.parent)),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, out)
    locked_rate = locked_sum / total_samples
    print(
        f"[IR Collect] Saved manifest {total_samples} samples in {len(shard_paths)} shards -> {out}\n"
        f"[IR Collect] locked_rate={locked_rate:.3f} shards_dir={shard_dir}",
        flush=True,
    )
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
