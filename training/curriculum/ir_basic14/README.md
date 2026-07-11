# ir_basic14 — Phase 6C IR mainline

Policy obs: **9D ir_track** (frozen TrackNet + phase + altitude + time)  
Action: lateral [2] (same as oracle basic14)

## Pipeline

```powershell
# 0) TrackNet already trained -> tracknet_best.pth

# 1) Collect IR-BC (teacher labels, ir_track obs)
cd D:\Rocket\Missle\isaac_0703\training\scripts
.\collect_ir_bc_trajectories.ps1 -Curriculum basic14 -NumEnvs 512 -Steps 2000

# 2) Train BC (9D actor)
.\train_bc_policy.ps1 -Dataset ..\..\data\bc\ir_basic14\transitions.pt -Epochs 50

# 3) Init RL-Games checkpoint from BC
.\init_rlgames_from_bc.ps1 `
  -BcCheckpoint "..\..\data\bc\ir_basic14\bc_policy.pth" `
  -Output "..\..\checkpoints\ppo_ir_bc_init.pth"

# 4) PPO fine-tune (ir_track obs, teacher residual full)
cd ..\curriculum\ir_basic14
.\run.ps1 -MaxIterations 400 -NumEnvs 4096

# 5) Play / Gate (need completed_ep_hit >= 80% of oracle basic14)
cd ..\..\engine\scripts
.\play_checkpoint.ps1 `
  -Checkpoint "<your_ir_ppo.pth>" `
  -Curriculum basic14 `
  -Randomization eval `
  -ObsMode ir_track `
  -TrackNetCkpt "D:\Rocket\Missle\isaac_0703\data\ir\basic14\tracknet_best.pth" `
  -NumEnvs 1
```

## Gate

| Metric | Target |
|--------|--------|
| `completed_ep_hit` | ≥ **80%** of frozen oracle `basic14_ep601` hit rate |
| TrackNet | frozen `tracknet_best.pth` |

## Env vars

| Var | ir_track value |
|-----|----------------|
| `STAGE3_OBS_MODE` | `ir_track` |
| `STAGE3_IR_ENABLE` | `1` |
| `STAGE3_TRACKNET_CKPT` | path to `tracknet_best.pth` |
