# DD_0706 Isaac Lab Intercept Training

Abstract intercept-training prototype（抽象拦截制导 RL 原型）.

## Project Layout / 目录结构

```text
isaac_0703/
├── core/                    仿真核心 CPU simulation core
│   └── intercept_core.py
├── demos/                   可视化 Visual demos (Isaac Sim)
│   ├── ground_intercept_demo.py
│   └── launchers/           场景启动脚本 Scenario launchers
├── training/
│   ├── engine/              Isaac Lab + RL-Games GPU 训练引擎
│   ├── curriculum/          课程训练入口 Curriculum tasks
│   └── scripts/             BC / DAgger / 评估工具
├── data/bc/                 BC 数据集 BC datasets
├── checkpoints/             模型检查点 Model checkpoints
├── assets/converted/        USD 视觉资源 Visual assets
└── docs/                    文档 Documentation
```

## Quick Start / 快速开始

### 1. Teacher 验证 Verify teacher (Phase 0)

```powershell
cd D:\Rocket\Missle\isaac_0703\training\curriculum\basic14
.\run_teacher_eval.ps1
```

### 2. BC 预训练 BC pre-training (Phase 1)

```powershell
cd D:\Rocket\Missle\isaac_0703\training\scripts
.\collect_teacher_trajectories.ps1 -Curriculum basic14
.\train_bc_policy.ps1 -Dataset ..\..\data\bc\basic14\transitions.pt
.\init_rlgames_from_bc.ps1 -BcCheckpoint ..\..\data\bc\basic14\bc_policy.pth -Output ..\..\checkpoints\ppo_bc_init.pth
```

### 3. PPO 课程训练 PPO curriculum (Phase 5)

```powershell
cd D:\Rocket\Missle\isaac_0703\training\curriculum\easy4
.\run.ps1 -Checkpoint "D:\Rocket\Missle\isaac_0703\checkpoints\ppo_bc_init.pth"
```

Recommended order 推荐顺序:

```text
teacher eval → BC → easy4 → mid6 → hard4 → basic14
→ tail4_warmup_residual → tail4_warmup → tail4 → hard6 → tail10 → mix24 → full24
```

## Main Training Entry

```powershell
cd D:\Rocket\Missle\isaac_0703\training\curriculum\basic14
.\run.ps1
```

Logs: `training\engine\logs\rl_games\stage3_intercept_direct`

## Visual Validate Checkpoint / 可视化验证检查点

项目重组后 play 脚本路径为 `training\engine\scripts\`（旧路径 `sandbox\isaaclab_gpu\` 已废弃）。

```powershell
cd D:\Rocket\Missle\isaac_0703\training\engine\scripts

$ckpt = "D:\Rocket\Missle\isaac_0703\training\engine\logs\rl_games\stage3_intercept_direct\2026-07-05_22-19-58\nn\stage3_intercept_direct.pth"

.\play_checkpoint.ps1 -Checkpoint $ckpt -NumEnvs 1 -Device cuda:0 -Video
```

或手动调用：

```powershell
$env:TORCH_COMPILE_DISABLE = "1"
$env:PYTHONPATH = "D:\Rocket\Missle\isaac_0703\training\engine\source\isaaclab_stage3;$env:PYTHONPATH"
$env:STAGE3_SCENARIO_CURRICULUM = "basic14"
$env:STAGE3_RANDOMIZATION_MODE = "eval"

$ckpt = "D:\Rocket\Missle\isaac_0703\training\engine\logs\rl_games\stage3_intercept_direct\2026-07-05_22-19-58\nn\stage3_intercept_direct.pth"

E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat `
  D:\Rocket\Missle\isaac_0703\training\engine\scripts\play_isaaclab_intercept_rlgames.py `
  --task Isaac-Stage3-Intercept-Direct-v0 `
  --checkpoint $ckpt `
  --num_envs 1 `
  --device cuda:0 `
  --video
```


```powershell
cd D:\Rocket\Missle\isaac_0703
.\demos\launchers\run_visual_scenario_menu.bat
```

## Documentation

- [PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) — 项目指南
- [OPTIMIZATION_PLAN.md](docs/OPTIMIZATION_PLAN.md) — 优化方案 Phase 0–5

## Environment

- Isaac Sim: `E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64`
- Isaac Lab: `E:\Issac_sim\IsaacLab`
