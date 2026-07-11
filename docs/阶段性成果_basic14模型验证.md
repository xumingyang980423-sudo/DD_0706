# 阶段性成果：basic14 模型验证

## 验证目标

验证 `basic14` 课程训练得到的 Isaac Lab / RL-Games PPO 模型。

模型路径：

```text
D:\Rocket\Missle\isaac_0703\sandbox\isaaclab_gpu\logs\rl_games\stage3_intercept_direct\2026-07-05_22-19-58\nn\stage3_intercept_direct.pth
```

验证设置：

```text
STAGE3_SCENARIO_CURRICULUM=basic14
STAGE3_RANDOMIZATION_MODE=eval
num_envs=16
device=cuda:0
video=true
video_length=2700
```

## 验证指令

```powershell
cd D:\Rocket\Missle\isaac_0703\training\engine

$env:TORCH_COMPILE_DISABLE="1"
$env:PYTHONPATH="D:\Rocket\Missle\isaac_0703\training\engine\source\isaaclab_stage3;$env:PYTHONPATH"
$env:STAGE3_SCENARIO_CURRICULUM="basic14"
$env:STAGE3_RANDOMIZATION_MODE="eval"
$env:STAGE3_TEACHER_MODE="only"
Remove-Item Env:\STAGE3_FIXED_SCENARIO_ID -ErrorAction SilentlyContinue

$ckpt="D:\Rocket\Missle\isaac_0703\training\engine\logs\rl_games\stage3_intercept_direct\2026-07-05_22-19-58\nn\stage3_intercept_direct.pth"

E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat `
  D:\Rocket\Missle\isaac_0703\training\engine\scripts\play_isaaclab_intercept_rlgames.py `
  --task Isaac-Stage3-Intercept-Direct-v0 `
  --checkpoint $ckpt `
  --num_envs 16 `
  --device cuda:0 `
  --video `
  --video_length 2700
```
