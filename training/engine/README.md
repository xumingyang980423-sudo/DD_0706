# Isaac Lab GPU Training Engine

Isaac Lab `DirectRLEnv` + RL-Games PPO integration.

Entry scripts:
- `scripts/train_isaaclab_intercept_rlgames.py`
- `scripts/play_isaaclab_intercept_rlgames.py`

Core env:
- `source/isaaclab_stage3/isaaclab_stage3/tasks/intercept/intercept_env.py`
- `source/isaaclab_stage3/isaaclab_stage3/tasks/intercept/teacher_guidance.py`

Logs: `logs/rl_games/stage3_intercept_direct`

Launch training via `training/curriculum/*/run.ps1`.
