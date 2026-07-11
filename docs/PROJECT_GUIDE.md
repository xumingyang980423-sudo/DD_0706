# Project Guide / 项目指南

## Scope / 范围

Abstract intercept-training prototype — not a real missile model.  
抽象拦截训练原型，非真实武器模型。

Maintained workflows 维护的工作流:
- Isaac Lab GPU RL training (`training/engine`)
- Isaac Sim visual demos (`demos/`)
- BC pre-training pipeline (`training/scripts`)

## Directory Map / 目录说明

| Path | Purpose |
|------|---------|
| `core/intercept_core.py` | CPU 单环境仿真 + demo baseline |
| `training/engine/` | GPU DirectRLEnv + RL-Games PPO |
| `training/curriculum/` | 课程训练启动器 Curriculum launchers |
| `training/scripts/` | BC 采集、训练、DAgger |
| `demos/` | Isaac Sim 三维可视化 |
| `data/bc/` | BC 数据集 |
| `checkpoints/` | 模型检查点 |
| `docs/OPTIMIZATION_PLAN.md` | 完整优化方案 |

## Core Environment Files

```text
training/engine/source/isaaclab_stage3/isaaclab_stage3/tasks/intercept/
  intercept_env.py       # GPU 训练环境
  teacher_guidance.py    # Full teacher (Phase 0)
  agents/rl_games_ppo_cfg.yaml
```

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STAGE3_SCENARIO_CURRICULUM` | basic14 | 场景采样课程 |
| `STAGE3_TEACHER_MODE` | full | full/simple/none/only |
| `STAGE3_REWARD_STAGE` | auto | A/B/C/auto 分阶段奖励 |
| `STAGE3_RESIDUAL_SCHEDULE` | auto | 残差退火 auto/manual |
| `STAGE3_RANDOMIZATION_MODE` | train | train/eval/stress |
| `STAGE3_BASELINE_*` | see plan | Teacher 参数 |

## Curriculum Tasks / 课程任务

| Folder | Scenario IDs | Notes |
|--------|-------------|-------|
| `easy4` | 0,2,3,6 | 子课程 Gate 40% |
| `mid6` | 1,4,5,8,11 | Gate 25% |
| `hard4` | 7,9,10,12,13 | Gate 15% |
| `basic14` | 0–13 | 基础课程 |
| `tail4_warmup_residual` | 14–17 | 残差 + warmup |
| `tail4_warmup` | 14–17 | 纯 PPO warmup |
| `tail4` | 14–17 | 完整 tail4 |
| `hard6` | 18–23 | 高机动 |
| `tail10` | 14–23 | 全部 tail |
| `mix24` | 60/40 mix | 混合微调 |
| `full24` | 0–23 | 最终泛化 |

## Metrics / 评估指标

Prefer over raw reward 优先于 reward:
- `hit_rate` — step-level
- `episode_hit_rate` — episode-level（新增）
- `mean_closest_distance`
- `mean_ahead_distance`
- `selection_score` — checkpoint 选择

## Common Commands

Train basic14:
```powershell
cd training\curriculum\basic14
.\run.ps1
```

Teacher-only eval:
```powershell
.\run_teacher_eval.ps1
```

BC pipeline:
```powershell
cd training\scripts
.\collect_teacher_trajectories.ps1 -Curriculum basic14
.\train_bc_policy.ps1 -Dataset ..\..\data\bc\basic14\transitions.pt
```

Visual demo:
```powershell
.\demos\launchers\run_visual_scenario_menu.bat
```

## External Paths

- Isaac Sim: `E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64`
- Isaac Lab: `E:\Issac_sim\IsaacLab`
