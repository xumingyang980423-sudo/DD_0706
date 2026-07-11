"""Lightweight teacher-only rollout eval (no PPO training loop)."""

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
    parser.add_argument("--randomization", default="train", choices=["train", "eval", "stress"])
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1500)
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
    import torch  # noqa: E402
    from isaaclab_stage3.tasks.intercept.intercept_env import Stage3InterceptEnvCfg  # noqa: E402

    env_cfg = Stage3InterceptEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device if hasattr(args, "device") and args.device else env_cfg.sim.device

    env = gym.make("Isaac-Stage3-Intercept-Direct-v0", cfg=env_cfg)
    unwrapped = env.unwrapped
    obs, _ = env.reset(seed=args.seed)

    # Guidance-phase step metrics (diagnostic; not used for Gate closest).
    hit_step_sum = 0.0
    ep_hit_step_sum = 0.0
    contact_sum = 0.0
    center_sum = 0.0
    guidance_closest_sum = 0.0
    guidance_closest_center_sum = 0.0
    ahead_sum = 0.0
    guidance_env_steps = 0

    # Episode-level metrics (Gate uses these; aligned with CPU baseline).
    episode_total = 0
    episode_hits = 0
    episode_closest_sum = 0.0
    episode_closest_center_sum = 0.0
    hit_term_total = 0
    ground_term_total = 0
    bounds_term_total = 0
    timeout_total = 0

    hit_event_steps = 0
    min_episode_closest = float("inf")

    def _log_progress(step: int, final: bool = False) -> None:
        if episode_total <= 0 and guidance_env_steps <= 0:
            return
        completed_ep_hit_rate = (episode_hits / episode_total) if episode_total else 0.0
        episode_closest_avg = (episode_closest_sum / episode_total) if episode_total else float("inf")
        episode_closest_center_avg = (
            (episode_closest_center_sum / episode_total) if episode_total else float("inf")
        )
        gate_ep = completed_ep_hit_rate > 0.30
        gate_closest = episode_closest_avg < 30.0
        gate = "PASS" if gate_ep and gate_closest else "FAIL"
        prefix = "[TeacherEval] FINAL" if final else f"[TeacherEval] step={step}"

        hit_rate = hit_step_sum / guidance_env_steps if guidance_env_steps else 0.0
        ep_hit_rate = ep_hit_step_sum / guidance_env_steps if guidance_env_steps else 0.0
        guidance_closest_avg = guidance_closest_sum / guidance_env_steps if guidance_env_steps else float("inf")

        print(
            f"{prefix} hit={hit_rate:.4f} ep_hit={ep_hit_rate:.4f} "
            f"completed_ep_hit_rate={completed_ep_hit_rate:.4f} "
            f"episode_closest={episode_closest_avg:.1f} episode_closest_c={episode_closest_center_avg:.1f} "
            f"guidance_closest_avg={guidance_closest_avg:.1f} "
            f"contact={contact_sum/guidance_env_steps:.1f} center={center_sum/guidance_env_steps:.1f} "
            f"ahead={ahead_sum/guidance_env_steps:.1f} "
            f"episodes={episode_total} hit_events={hit_event_steps} "
            f"min_episode_closest={min_episode_closest:.1f} "
            f"gate={gate} (need completed_ep_hit_rate>0.30 episode_closest<30m)",
            flush=True,
        )
        if final:
            print(
                f"[TeacherEval] terminals hit={hit_term_total} ground={ground_term_total} "
                f"bounds={bounds_term_total} timeout={timeout_total}",
                flush=True,
            )
            print(
                f"[TeacherEval] config randomization={args.randomization} curriculum={args.curriculum} "
                f"num_envs={args.num_envs} steps={args.steps}",
                flush=True,
            )

    for step in range(args.steps):
        pre_guidance = unwrapped._flight_t() >= unwrapped.boost_duration
        teacher = unwrapped.get_teacher_action_for_bc()
        obs, reward, terminated, truncated, _ = env.step(teacher)
        done = terminated | truncated

        if done.any():
            done_ids = done
            n_done = int(done.sum().item())
            episode_total += n_done
            episode_hits += int(unwrapped._last_episode_hit_snapshot[done_ids].sum().item())
            ep_closest = unwrapped._last_step_closest[done_ids]
            ep_closest_c = unwrapped._last_step_closest_center[done_ids]
            episode_closest_sum += float(ep_closest.sum().item())
            episode_closest_center_sum += float(ep_closest_c.sum().item())
            min_episode_closest = min(min_episode_closest, float(ep_closest.min().item()))
            hit_term_total += int((terminated & unwrapped._last_step_hit)[done_ids].sum().item())
            ground_term_total += int(
                (terminated & unwrapped._last_step_ground & ~unwrapped._last_step_hit)[done_ids].sum().item()
            )
            bounds_term_total += int(
                (terminated & unwrapped._last_step_bounds & ~unwrapped._last_step_hit)[done_ids].sum().item()
            )
            timeout_total += int(truncated[done_ids].sum().item())

        hit_event_steps += int(unwrapped._last_step_hit.sum().item())

        if pre_guidance.any():
            g = pre_guidance
            n = int(g.sum().item())
            guidance_env_steps += n
            hit_step_sum += unwrapped._last_step_hit[g].float().sum().item()
            ep_hit_step_sum += unwrapped._last_episode_hit_snapshot[g].sum().item()
            contact_sum += unwrapped._last_step_contact[g].sum().item()
            center_sum += unwrapped._last_step_center[g].sum().item()
            guidance_closest_sum += unwrapped._last_step_closest[g].sum().item()
            guidance_closest_center_sum += unwrapped._last_step_closest_center[g].sum().item()
            post_guidance = pre_guidance & ~done
            if post_guidance.any():
                target_fwd = unwrapped._unit(
                    unwrapped.aircraft_vel, torch.tensor([1.0, 0.0, 0.0], device=unwrapped.device)
                )
                along = torch.sum((unwrapped.missile_pos - unwrapped.aircraft_pos) * target_fwd, dim=1)
                ahead_sum += torch.clamp(along, min=0.0)[post_guidance].sum().item()

        if step % 100 == 0 and (guidance_env_steps > 0 or episode_total > 0):
            _log_progress(step)

    if guidance_env_steps or episode_total:
        _log_progress(args.steps - 1, final=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
