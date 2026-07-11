"""Collect teacher (ir_track obs, action) transitions for IR BC (Phase 6C)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "training" / "engine"
import sys

sys.path.insert(0, str(ENGINE_ROOT / "source" / "isaaclab_stage3"))

import isaaclab_stage3  # noqa: E402,F401
from isaaclab.app import AppLauncher  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", default="basic14")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--tracknet_ckpt", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    os.environ["STAGE3_SCENARIO_CURRICULUM"] = args.curriculum
    os.environ["STAGE3_TEACHER_MODE"] = "only"
    os.environ["STAGE3_RANDOMIZATION_MODE"] = "train"
    os.environ["STAGE3_REWARD_STAGE"] = "A"
    os.environ["STAGE3_OBS_MODE"] = "ir_track"
    os.environ["STAGE3_IR_ENABLE"] = "1"
    os.environ["STAGE3_TRACKNET_CKPT"] = str(Path(args.tracknet_ckpt).resolve())

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym  # noqa: E402
    from isaaclab_stage3.tasks.intercept.intercept_env import Stage3InterceptEnvCfg  # noqa: E402

    env_cfg = Stage3InterceptEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make("Isaac-Stage3-Intercept-Direct-v0", cfg=env_cfg)
    obs_list: list[torch.Tensor] = []
    act_list: list[torch.Tensor] = []
    sid_list: list[torch.Tensor] = []

    obs, _ = env.reset()
    policy_obs = obs["policy"]
    unwrapped = env.unwrapped

    for _ in range(args.steps):
        guidance = unwrapped._flight_t() >= unwrapped.boost_duration
        teacher = unwrapped.get_teacher_action_for_bc()
        mask = guidance
        if mask.any():
            obs_list.append(policy_obs[mask].detach().cpu())
            act_list.append(teacher[mask].detach().cpu())
            sid_list.append(unwrapped.scenario_id[mask].detach().cpu())

        action = teacher.unsqueeze(0) if teacher.ndim == 1 else teacher
        obs, _, _, _, _ = env.step(action)
        policy_obs = obs["policy"]

    out = Path(args.output) if args.output else PROJECT_ROOT / "data" / "bc" / f"ir_{args.curriculum}" / "transitions.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "obs": torch.cat(obs_list, dim=0),
        "action": torch.cat(act_list, dim=0),
        "scenario_id": torch.cat(sid_list, dim=0),
        "curriculum": args.curriculum,
        "obs_mode": "ir_track",
        "obs_dim": 9,
        "tracknet_ckpt": os.environ["STAGE3_TRACKNET_CKPT"],
    }
    torch.save(data, out)
    print(
        f"[IR BC Collect] Saved {data['obs'].shape[0]} transitions obs_dim={data['obs_dim']} -> {out}",
        flush=True,
    )
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
