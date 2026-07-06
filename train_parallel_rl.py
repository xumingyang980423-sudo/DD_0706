from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl_env import MissileInterceptEnv, STAGE2_SCENARIOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel PPO training for the guidance-phase intercept RL env.")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--run-name", default="ppo_stage3_parallel")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--scenarios",
        default=",".join(STAGE2_SCENARIOS[:8]),
        help="Comma-separated scenario names used for training. Defaults to the easier first 8 stage-2 scenarios.",
    )
    return parser.parse_args()


def make_env(training_scenarios: list[str]):
    def _factory() -> MissileInterceptEnv:
        return MissileInterceptEnv(randomize=True, scenarios=training_scenarios)

    return _factory


def main() -> None:
    args = parse_args()
    training_scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    if not training_scenarios:
        raise ValueError("At least one scenario is required for training.")

    output_dir = Path("logs") / "rl" / args.run_name
    model_dir = output_dir / "models"
    tensorboard_dir = output_dir / "tb"
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(
        make_env(training_scenarios),
        n_envs=args.num_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )
    checkpoint = CheckpointCallback(
        save_freq=max(args.total_timesteps // max(args.num_envs, 1) // 10, 1_000),
        save_path=str(model_dir),
        name_prefix="ppo_checkpoint",
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(tensorboard_dir),
        n_steps=1024,
        batch_size=2048,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
    )
    print(f"Training scenarios: {training_scenarios}")
    print(f"Output directory: {output_dir.resolve()}")
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint, tb_log_name=args.run_name)

    final_model_path = model_dir / "ppo_final"
    model.save(str(final_model_path))
    env.close()
    print(f"Saved final model: {final_model_path}.zip")


if __name__ == "__main__":
    main()
