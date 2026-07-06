# Isaac Lab GPU Stage-3 Sandbox

This sandbox contains the Isaac Lab/RL-Games version of the stage-3 abstract intercept training task.

It is separate from the older `sb3_cpu_baseline` path:

- `sb3_cpu_baseline`: Stable-Baselines3, custom NumPy/Gymnasium environment, CPU vectorized environment.
- `isaaclab_gpu`: Isaac Lab `DirectRLEnv`, RL-Games PPO, vectorized torch tensors on `cuda:0`.

Run a quick smoke test:

```powershell
cd D:\Rocket\Missle\isaac_0703
.\bat_scripts\training\run_isaaclab_gpu_smoke.bat
```

Run training:

```powershell
cd D:\Rocket\Missle\isaac_0703
.\bat_scripts\training\run_isaaclab_gpu_train.bat
```

Default training command:

```text
task: Isaac-Stage3-Intercept-Direct-v0
num_envs: 4096
device: cuda:0
headless: true
max_iterations: 300
STAGE3_SCENARIO_CURRICULUM: basic14
STAGE3_RANDOMIZATION_MODE: train
```

Equivalent explicit command:

```powershell
cd D:\Rocket\Missle\isaac_0703\sandbox\isaaclab_gpu

$env:TORCH_COMPILE_DISABLE="1"
$env:PYTHONPATH="D:\Rocket\Missle\isaac_0703\sandbox\isaaclab_gpu\source\isaaclab_stage3;$env:PYTHONPATH"
$env:STAGE3_SCENARIO_CURRICULUM="basic14"
$env:STAGE3_RANDOMIZATION_MODE="train"
Remove-Item Env:\STAGE3_FIXED_SCENARIO_ID -ErrorAction SilentlyContinue

E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat `
  D:\Rocket\Missle\isaac_0703\sandbox\isaaclab_gpu\scripts\train_isaaclab_intercept_rlgames.py `
  --task Isaac-Stage3-Intercept-Direct-v0 `
  --num_envs 4096 `
  --device cuda:0 `
  --headless `
  --max_iterations 300
```

Logs are written by Isaac Lab/RL-Games under:

```text
D:\Rocket\Missle\isaac_0703\sandbox\isaaclab_gpu\logs\rl_games\stage3_intercept_direct
```

The environment is an abstract algorithm-training model. It keeps the stage-2 structure:

- BOOST is rule-controlled.
- GUIDANCE is controlled by the RL policy.
- Observation size is 18.
- Action size is 2.
- The default curriculum samples scenario IDs 0-13 (`basic14`).

Useful environment variables:

```text
STAGE3_SCENARIO_CURRICULUM=easy8   scenario IDs 0-7
STAGE3_SCENARIO_CURRICULUM=basic14 scenario IDs 0-13
STAGE3_SCENARIO_CURRICULUM=tail4_warmup scenario IDs 14-17, reduced speed/maneuver/delay
STAGE3_SCENARIO_CURRICULUM=tail4   scenario IDs 14-17
STAGE3_SCENARIO_CURRICULUM=tail4_residual scenario IDs 14-17, baseline guidance + PPO residual
STAGE3_SCENARIO_CURRICULUM=hard6   scenario IDs 18-23
STAGE3_SCENARIO_CURRICULUM=tail10  scenario IDs 14-23
STAGE3_SCENARIO_CURRICULUM=mix24   60% IDs 0-13, 40% IDs 14-23
STAGE3_SCENARIO_CURRICULUM=full24  scenario IDs 0-23

STAGE3_RANDOMIZATION_MODE=train    target speed 20-28 m/s, mild randomization
STAGE3_RANDOMIZATION_MODE=eval     target speed 18-32 m/s, wider randomization
STAGE3_RANDOMIZATION_MODE=stress   target speed 30-36 m/s, stress randomization

STAGE3_FIXED_SCENARIO_ID=0..23     fixed single scenario for visualization/evaluation

STAGE3_RESIDUAL_ALPHA=1.0          baseline guidance weight for residual curricula
STAGE3_RESIDUAL_BETA=0.25          PPO residual weight for residual curricula
STAGE3_BASELINE_TAIL_GAIN=2.2      baseline tail guidance gain
STAGE3_BASELINE_TAIL_OFFSET=8.0    tail aim point behind target, meters
```
