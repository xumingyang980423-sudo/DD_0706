# basic14

Scenario IDs: `0-13`

Purpose: main early curriculum for stable guidance behavior.

Run PPO:

```powershell
.\run.ps1
```

Teacher-only eval (Phase 0):

```powershell
.\run_teacher_eval.ps1
```

Continue from BC/PPO checkpoint:

```powershell
.\run.ps1 -Checkpoint "D:\Rocket\Missle\isaac_0703\checkpoints\ppo_bc_init.pth" -MaxIterations 800
```
