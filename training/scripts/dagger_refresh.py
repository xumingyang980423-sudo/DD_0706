"""DAgger refresh: collect teacher labels on policy rollout divergences and append to BC dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "training" / "engine"
if not ENGINE_ROOT.exists():
    ENGINE_ROOT = PROJECT_ROOT / "sandbox" / "isaaclab_gpu"

import sys

sys.path.insert(0, str(ENGINE_ROOT / "source" / "isaaclab_stage3"))

import isaaclab_stage3  # noqa: F401
from isaaclab.app import AppLauncher  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", default="tail4_warmup_residual")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--divergence", type=float, default=0.35)
    parser.add_argument("--dataset", default="")
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    os.environ["STAGE3_SCENARIO_CURRICULUM"] = args.curriculum
    os.environ["STAGE3_TEACHER_MODE"] = "full"
    os.environ["STAGE3_RESIDUAL_SCHEDULE"] = "manual"
    os.environ["STAGE3_RESIDUAL_ALPHA"] = "0.5"
    os.environ["STAGE3_RESIDUAL_BETA"] = "0.5"

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym  # noqa: E402

    env = gym.make("Isaac-Stage3-Intercept-Direct-v0", num_envs=args.num_envs)
    unwrapped = env.unwrapped
    # Policy actions come from checkpoint via play script normally; here use teacher residual comparison
    obs, _ = env.reset()
    new_obs, new_act, new_sid = [], [], []

    for _ in range(args.steps):
        guidance = unwrapped._flight_t() >= unwrapped.boost_duration
        teacher = unwrapped.get_teacher_action_for_bc()
        policy = unwrapped.actions
        diff = torch.linalg.norm(teacher - policy, dim=1)
        mask = guidance & (diff > args.divergence)
        if mask.any():
            new_obs.append(obs["policy"][mask].detach().cpu())
            new_act.append(teacher[mask].detach().cpu())
            new_sid.append(unwrapped.scenario_id[mask].detach().cpu())
        obs, _, _, _, _ = env.step(policy)
        policy = unwrapped.actions  # applied after step

    ds_path = Path(args.dataset) if args.dataset else PROJECT_ROOT / "data" / "bc" / args.curriculum / "transitions.pt"
    if new_obs:
        fresh = {
            "obs": torch.cat(new_obs),
            "action": torch.cat(new_act),
            "scenario_id": torch.cat(new_sid),
        }
        if ds_path.exists():
            old = torch.load(ds_path, map_location="cpu", weights_only=False)
            merged = {
                "obs": torch.cat([old["obs"], fresh["obs"]]),
                "action": torch.cat([old["action"], fresh["action"]]),
                "scenario_id": torch.cat([old["scenario_id"], fresh["scenario_id"]]),
                "curriculum": args.curriculum,
            }
        else:
            merged = {**fresh, "curriculum": args.curriculum}
        ds_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(merged, ds_path)
        print(f"[DAgger] Appended {fresh['obs'].shape[0]} samples -> {ds_path}", flush=True)
    else:
        print("[DAgger] No divergent samples collected.", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
