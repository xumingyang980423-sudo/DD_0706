# tail4_warmup_residual

Scenario IDs: `14-17`

Purpose: easier tail4 randomization with residual teacher guidance.

Default residual settings:

```text
alpha=0.3
beta=1.0
baseline gain=3.6
baseline offset=2.5
```

Run from checkpoint:

```powershell
.\run.ps1 -Checkpoint "...\stage3_intercept_direct.pth" -MaxIterations 800
```
