# Intercept RL Optimization Plan / 拦截制导 RL 优化方案

> Project: `D:\Rocket\Missle\isaac_0703`  
> Status: Implemented (Phase 0–5)  
> Language: 中英混合 Mixed CN/EN

---

## 1. Background / 背景

当前训练 pipeline 使用 Isaac Lab GPU + RL-Games PPO，但历史 run 中 **hit_rate 最高仅 ~6%**，tail 场景几乎无法命中。

根因 Root causes:
- `intercept_core.py`（可视化 demo）与 `intercept_env.py`（RL 训练）的 **teacher 能力不一致**
- tail 场景 (ID 14–23) 在 demo 中使用 `long_follow_velocity()`，训练 env 仅用简化 baseline
- 奖励项过多，credit assignment 困难
- 未做 BC 预训练，PPO 从零探索稀疏命中事件

---

## 2. Target Architecture / 目标架构

```text
core/                    仿真核心 Simulation core (CPU reference)
training/
  engine/                Isaac Lab + RL-Games GPU engine
  curriculum/            课程训练入口 Curriculum launchers
  scripts/               BC 采集 / 训练 / 评估 BC & eval scripts
demos/                   Isaac Sim 可视化 Visual demos
data/bc/                 BC 数据集 BC datasets
docs/                    文档 Documentation
assets/converted/        USD 模型 Visual assets
```

---

## 3. Phase 0 — Unified Teacher / 统一 Teacher

**Goal:** 训练 env 与 demo 使用同一套 full teacher。

| Env Var | Values | Description |
|---------|--------|-------------|
| `STAGE3_TEACHER_MODE` | `full` / `simple` / `none` / `only` | Teacher 模式 |
| `STAGE3_BASELINE_TAIL_GAIN` | float | 基线增益 |
| `STAGE3_BASELINE_TAIL_OFFSET` | float | 尾追瞄准偏移 |
| `STAGE3_BASELINE_ACTION_ALPHA` | float | 动作平滑系数 |
| `STAGE3_BASELINE_CLOSEOUT_RANGE` | float | 终端 closeout 距离 |
| `STAGE3_BASELINE_TANGENT_BLEND_MAX` | float | 切向混合上限 |

Implementation file: `training/engine/source/isaaclab_stage3/.../teacher_guidance.py`

- **basic14 (ID 0–13):** lead-point PN + smoothing（移植自 `intercept_core.baseline_guidance_action`）
- **tail (ID 14–23):** history-based `long_follow_velocity`（与 demo 一致）
- **`only` mode:** 忽略 policy，纯 teacher rollout（用于 BC 采集与 teacher 验证）

**Gate 门槛:** teacher-only basic14 eval `completed_ep_hit_rate > 30%`，且 **episode_closest < 30m**（每局结束时的 `closest_distance` 再取平均，与 CPU baseline 口径一致）

---

## 4. Phase 1 — BC Pre-training / 行为克隆预训练

### 4.1 Data Collection / 数据采集

```powershell
cd training\curriculum\basic14
..\..\scripts\collect_teacher_trajectories.ps1 -Curriculum basic14 -Episodes 200
```

Output: `data/bc/{curriculum}/transitions.pt`

Fields: `obs (N,18)`, `action (N,2)`, `scenario_id`, `residual_label (optional)`

### 4.2 BC Training / BC 训练

```powershell
..\..\scripts\train_bc_policy.ps1 -Dataset data\bc\basic14\transitions.pt -Output checkpoints\bc_basic14.pth
```

Network: `[128,128,64]` ELU — **same as PPO actor**

### 4.3 Init PPO from BC / BC 初始化 PPO

```powershell
..\..\scripts\init_rlgames_from_bc.ps1 -BcCheckpoint checkpoints\bc_basic14.pth -Output checkpoints\ppo_bc_init.pth
```

---

## 5. Phase 2 — Staged Reward / 分阶段奖励

| Stage | Env Var | Reward Components |
|-------|---------|-------------------|
| **A** | `STAGE3_REWARD_STAGE=A` | closing + hit(+50) + ground/timeout |
| **B** | `STAGE3_REWARD_STAGE=B` | A + heading + rear_alignment |
| **C** | `STAGE3_REWARD_STAGE=C` | Full shaping (tail, near, control, ahead) |

Auto mode: `STAGE3_REWARD_STAGE=auto` — switches A→B→C by `common_step_counter`.

RL-Games `reward_shaper.scale_value` changed from **0.1 → 1.0**.

---

## 6. Phase 3 — Residual Annealing + DAgger / 残差退火

### Residual schedule (by training step)

```text
step 0–50k:    α=1.0, β=0.0   (pure teacher)
step 50k–150k: α=0.8, β=0.2
step 150k–300k: α=0.5, β=0.5
step 300k–500k: α=0.2, β=0.8
step 500k+:    α=0.0, β=1.0   (pure policy)
```

Override via `STAGE3_RESIDUAL_SCHEDULE=auto|manual`.

### DAgger hook

Every N PPO iterations, run `training/scripts/dagger_refresh.ps1`:
1. Rollout current policy
2. Label divergent steps with teacher action
3. Append to BC dataset and fine-tune 1 epoch

---

## 7. Phase 4 — Evaluation / 评估体系

### Metrics 指标

| Metric | Description |
|--------|-------------|
| `hit_rate` | Step-level 瞬时命中率 |
| `episode_hit_rate` | Episode 级命中率（新增 New） |
| `mean_closest_distance` | 最近接近距离 |
| `mean_ahead_distance` | 超前距离（尾追质量） |
| `selection_score` | Checkpoint 选择综合分（恢复 Restored） |

```python
selection_score = hit_rate * 100 - mean_closest_distance * 0.5 - mean_ahead_distance * 0.3
```

### Eval protocol 评估协议

| Set | Randomization | Pass criteria |
|-----|---------------|---------------|
| train_eval | train | completed_ep_hit_rate > 20%, **episode_closest** < 30m |
| holdout_eval | eval | completed_ep_hit_rate > 10%, **episode_closest** < 40m |
| stress_eval | stress | completed_ep_hit_rate > 5%, **episode_closest** < 50m |

---

## 8. Phase 5 — Revised Curriculum Pipeline / 修订训练流程

```text
Phase 0 verify teacher
  ↓
BC basic14 → PPO init
  ↓
easy4 (ID 0,2,3,6)     Gate: eval hit > 40%
  ↓
mid6  (ID 1,4,5,8,11)  Gate: eval hit > 25%
  ↓
hard4 (ID 7,9,10,12,13) Gate: eval hit > 15%
  ↓
basic14 full            Gate: eval hit > 20%
  ↓
BC tail4_residual → tail4_warmup_residual (DAgger)
  ↓
tail4_warmup → tail4 → hard6 → tail10 → mix24 → full24
```

### New curriculum folders 新增课程目录

```text
training/curriculum/
  easy4/
  mid6/
  hard4/
  basic14/
  tail4_warmup_residual/
  ...
```

---

## 9. PPO Hyperparameter Changes / PPO 超参调整

| Parameter | Old | New |
|-----------|-----|-----|
| `fixed_sigma` | True | **False** |
| `entropy_coef` | 0.008 | **0.015** (anneal to 0.005) |
| `horizon_length` | 64 | **128** |
| `reward_shaper.scale_value` | 0.1 | **1.0** |
| `learning_rate` (post-BC) | 3e-4 | **1e-4** |

---

## 10. Quick Start / 快速开始

```powershell
# 1. Verify teacher
cd D:\Rocket\Missle\isaac_0703\training\curriculum\basic14
.\run_teacher_eval.ps1

# 2. Collect BC data + train BC
..\..\scripts\collect_teacher_trajectories.ps1 -Curriculum basic14
..\..\scripts\train_bc_policy.ps1 -Dataset ..\..\data\bc\basic14\transitions.pt

# 3. Init PPO from BC and train
.\run.ps1 -Checkpoint ..\..\checkpoints\ppo_bc_init.pth -MaxIterations 800
```

---

## 11. Next Steps After Implementation / 实施后下一步

1. Run teacher eval on basic14 — confirm hit_rate > 30%
2. Collect BC dataset (~200 episodes × 4096 envs)
3. Train BC → init PPO → run easy4 curriculum
4. Monitor `selection_score` and **do not advance curriculum until gate passes**
5. Visual validate best checkpoint with `play_isaaclab_intercept_rlgames.py --video`
