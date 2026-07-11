# IR Seeker + TrackNet Plan / 红外导引头与感知网络方案

> Project: `D:\Rocket\Missle\isaac_0703`  
> Status: Phase 6A/6B implemented (2026-07-09)  
> Language: 中英混合 Mixed CN/EN  
> **Decision:** 图像级 IR 输出 + **TrackNet 感知** + 现有 **lateral guidance**；**不做端到端** image→action。

---

## 1. Goals / 目标

| Goal | Description |
|------|-------------|
| **IR 图像** | _seeker 输出 H×W 单通道热像（非 oracle 相对坐标） |
| **TrackNet** | CNN 从 IR 帧预测 track state + 目标类别 |
| **Guidance** | 现有 2D lateral action / PN / PPO policy，输入改为 track state |
| **Decoy** | 后续阶段：飞机释放诱饵，TrackNet 判别真目标 |
| **Keep** | Boost + 轴向定速 + 横向加速度 动力学框架不变 |

---

## 2. Architecture / 总体架构

```text
仿真真值 (internal, privileged)
  aircraft_pos/vel, decoy states, aspect, plume intensity
        │
        ▼
IRSeekerModel.render_frame()
  弹体系投影 + 热辐射合成 + 噪声/模糊/延迟
        │
        ▼
  IR frame [1, H, W]   (e.g. 64×64 float32)
        │
        ▼
TrackNet (CNN, trainable)
  locked, u, v, u_dot, v_dot, confidence, target_class
        │
        ▼
Guidance policy / Teacher-PN (existing action space)
  2D lateral action → intercept_env 动力学
```

**不做：** `IR image → CNN → lateral action` 端到端（样本效率低、诱饵扩展难）。

**要做：** Privileged learning — Teacher 仍可用 oracle 产 BC 标签；部署路径用 IR→TrackNet→policy。

---

## 3. IR Sensor Model / 红外传感器模型

### 3.1 输出

| Output | Shape | Description |
|--------|-------|-------------|
| `frame` | `[H, W]` | 归一化灰度热像，默认 H=W=64 |
| `meta.locked` | bool | 是否有有效目标在 FOV 内 |
| `meta.blobs_gt` | list | 训练用真值（不喂 policy） |

### 3.2 合成流程（几何热像，非光追）

1. **Seeker 坐标系：** `forward = unit(missile_vel)`，定义 right/up，FOV 锥形视场。
2. **热源：**
   - 飞机：机身 + 尾喷口（aspect 决定强度，尾追 > 迎头）
   - 诱饵（Phase 7+）：点源，亮度 `B(t)=B0*exp(-t/τ)`，独立漂移速度
3. **投影：** 各源在 seeker 平面上的角度 → 像素坐标 + 高斯斑点
4. **退化：** PSF 模糊、读出噪声、饱和、可选 1–2 帧延迟

### 3.3 环境变量（计划）

| Env Var | Default | Description |
|---------|---------|-------------|
| `STAGE3_OBS_MODE` | `oracle` | `oracle` / `ir_track` / `ir_image` |
| `STAGE3_IR_RES` | `64` | 图像边长 |
| `STAGE3_IR_FOV_DEG` | `4.0` | 视场角（度） |
| `STAGE3_IR_NOISE` | `0.05` | 噪声强度 |
| `STAGE3_DECOY_COUNT` | `0` | 诱饵数量（Phase 7 启用） |

### 3.4 实现位置（计划）

```text
core/ir_seeker_model.py              CPU 参考 + 单元测试
training/engine/.../ir_seeker.py     GPU batch 渲染（Torch）
training/engine/.../intercept_env.py  obs 模式切换
```

---

## 4. TrackNet / 感知网络

### 4.1 输入

| Input | Shape | Notes |
|-------|-------|-------|
| `frame` | `[B, 1, H, W]` | 单帧；可选 stack 2–3 帧 `[B, T, H, W]` |
| `phase_flag` | `[B, 1]` | boost=0 / guidance=1（可选拼接到 FC） |

### 4.2 输出（track state，供 Guidance 使用）

| Output | Dim | Description |
|--------|-----|-------------|
| `locked` | 1 | sigmoid，是否锁定 |
| `u, v` | 2 | 目标质心，图像归一化坐标 [-1, 1] |
| `u_dot, v_dot` | 2 | 质心角速率（≈ LOS rate 代理） |
| `confidence` | 1 | 跟踪置信度 |
| `target_class` | 2 | Phase 7：`[p_aircraft, p_decoy]` 或 multi-blob ID |

**总维度（Phase 6）：** 7（无分类）或 9（含 class logits）

### 4.3 网络结构（建议）

```text
Conv 32→64→64→128 (3×3, stride 2) → GlobalAvgPool → FC 128 → heads
```

- 参数量 ~200K–500K，64×64 输入下 BC 训练快
- 与 PPO actor **分离**；Guidance actor 仍为 MLP `[128,128,64]`

### 4.4 损失函数

```python
L = w1 * MSE(u,v | locked) + w2 * MSE(u_dot,v_dot | locked)
  + w3 * BCE(locked) + w4 * CE(target_class)   # Phase 7
```

| 项 | 权重建议 |
|----|----------|
| 质心回归 | 1.0 |
| 角速率回归 | 0.5 |
| lock 分类 | 0.3 |
| 诱饵分类 | 0.5（Phase 7） |

### 4.5 Gate（TrackNet 单训）

| Metric | Phase 6 (no decoy) | Phase 7 (decoy) |
|--------|--------------------|-----------------|
| 质心误差（归一化） | < 0.05 | < 0.08 |
| `locked` accuracy | > 95% | > 90% |
| 真目标分类 accuracy | — | > 85% |

---

## 5. Guidance Layer / 制导层（非端到端）

Track state 进入 **现有** guidance，不改 action 语义：

| 模式 | 输入 | 输出 |
|------|------|------|
| `oracle`（当前） | 18D rel_pos/vel/los/... | lateral [2] |
| `ir_track`（目标） | TrackNet 7D + phase + altitude | lateral [2] |
| Teacher | oracle → PN | lateral [2]（BC 标签） |
| PPO | ir_track obs | lateral [2] |

**Teacher 迁移（Phase 6C）：** 可选 `seeker_pn(u_dot, v_dot)` 替代 rel_pos PN，与 TrackNet 输出对齐。

---

## 6. Decoy Model / 诱饵（Phase 7+）

### 6.1 仿真

- 飞机在 `t_release` 释放 `N` 个 decoy
- 初速 ≈ 飞机速度 + 横向扩散；亮度衰减快于飞机尾焰
- IR 图像：多 blob，诱饵常更亮、寿命更短、轨迹不连续

### 6.2 决策状态机

```text
SEARCH → ACQUIRE → TRACK → DISCRIM (multi-blob) → GUIDANCE
                              ↓ 选错诱饵
                            LOST → SEARCH
```

TrackNet 的 `target_class` + 多帧一致性用于 DISCRIM；Guidance 仍只跟 **一个** track point。

### 6.3 课程

| Stage | Decoys | Gate |
|-------|--------|------|
| IR-0 | 0 | TrackNet 质心误差达标 |
| IR-1 | 1 | 分类 acc > 85%，hit > 30% |
| IR-2 | 2–4 + 机动 | hit > 25%，误跟率 < 10% |
| IR-3 | 迎头 + decoy | hit > 15% |

---

## 7. Training Pipeline / 训练流程

### 7.1 与 Oracle PPO 课程的关系

```text
Mainline (继续):  easy4 ✓ → mid6 → hard4 → basic14 → tail...
IR branch (并行): Phase 6A → 6B → 6C → Phase 7 decoy
```

**建议：** Oracle PPO 先推进到 **basic14 full**；IR-0/6A–6B 可与 mid6 **并行**（不占 PPO 关键路径）。

### 7.2 Phase 6 步骤

#### 6A — IR 合成 + 可视化

- 实现 `IRSeekerModel`
- `play_teacher_only.ps1` 或 debug 脚本显示 IR 小窗
- **Gate：** 单目标 FOV 内 blob 与真值质心对齐（人工目视）

#### 6B — TrackNet 数据采集 + 训练

```powershell
cd training\scripts
# 计划脚本（待实现）
.\collect_ir_track_dataset.ps1 -Curriculum basic14 -NumEnvs 512 -Steps 5000
.\train_tracknet.ps1 -Dataset ..\..\data\ir\basic14\ir_frames.pt -Epochs 30
```

Dataset fields:

```python
{
  "frame": [N, 1, 64, 64],
  "locked": [N],
  "u, v, u_dot, v_dot": [N],
  "target_class": [N],      # Phase 7
  "scenario_id": [N],
}
```

#### 6C — Guidance 切换 obs

- `STAGE3_OBS_MODE=ir_track`
- 冻结 TrackNet，重采 BC（track→action）或 distillation
- PPO 微调，Gate：hit_rate ≥ oracle 同场景的 **80%**

### 7.3 数据量建议

| Dataset | 规模 |
|---------|------|
| IR TrackNet（无诱饵） | ≥ 500K 帧 |
| IR + decoy | ≥ 1M 帧，诱饵场景过采样 |

---

## 8. File Plan / 计划新增文件

```text
docs/IR_SEEKER_PLAN.md                 本文档
core/ir_seeker_model.py                CPU IR 合成 ✓
training/engine/.../ir_seeker.py       GPU batch IR ✓
training/engine/.../intercept_env.py   STAGE3_IR_ENABLE hook ✓
training/scripts/collect_ir_track_dataset.py   ✓
training/scripts/collect_ir_track_dataset.ps1  ✓
training/scripts/train_tracknet.py             ✓
training/scripts/train_tracknet.ps1            ✓
training/engine/scripts/play_ir_seeker_debug.py  ✓
training/engine/scripts/play_ir_seeker_debug.ps1 ✓
data/ir/basic14/ir_frames.pt           (run collect)
checkpoints/tracknet_basic14.pth       (run train)
```

---

## 9. Observation Mode Summary / 观测模式对照

| Mode | Policy 输入 | 用途 |
|------|-------------|------|
| `oracle` | 18D 全知 | 当前 BC/PPO/Teacher |
| `ir_image` | 64×64 raw | 仅 TrackNet 训练/debug |
| `ir_track` | 7–9D track state | 部署 + PPO 微调 |

---

## 10. Risks / 风险

| Risk | Mitigation |
|------|------------|
| IR 合成与真传感器差距大 | 后期加 noise/domain randomization |
| TrackNet 误差传导到 miss | Teacher residual + lock 丢失时惯导 |
| 诱饵误跟 | 独立 DISCRIM 头 + 课程 |
| 重训成本 | 分层训练，不端到端 |

---

## 11. Quick Reference / 快速索引

- 动力学：`intercept_env.py`（不改 boost/guidance 分解）
- Oracle 课程：`docs/OPTIMIZATION_PLAN.md` Phase 5
- Teacher play：`training/engine/scripts/play_teacher_only.ps1`
- PPO play：`training/engine/scripts/play_checkpoint.ps1`

---

## 12. Revision Log

| Date | Note |
|------|------|
| 2026-07-09 | Initial plan: IR image + TrackNet, no end-to-end, decoy Phase 7 |
