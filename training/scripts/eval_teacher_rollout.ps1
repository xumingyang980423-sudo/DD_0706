param(
    [string]$Curriculum = "basic14",
    [string]$Randomization = "train",
    [int]$NumEnvs = 512,
    [int]$Steps = 1500,
    [string]$BaselineTailGain = "4.8",
    [string]$BaselineTailOffset = "4.5",
    [string]$BaselineActionAlpha = "0.42",
    [string]$BaselineCloseoutRange = "22.0",
    [string]$BaselineTangentBlendMax = "0.45"
)

$ErrorActionPreference = "Stop"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$engineRoot = Join-Path $projectRoot "training\engine"
$python = "E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat"
$script = Join-Path $PSScriptRoot "eval_teacher_rollout.py"

$env:TORCH_COMPILE_DISABLE = "1"
$env:PYTHONPATH = "$engineRoot\source\isaaclab_stage3;$env:PYTHONPATH"
$env:STAGE3_BASELINE_TAIL_GAIN = $BaselineTailGain
$env:STAGE3_BASELINE_TAIL_OFFSET = $BaselineTailOffset
$env:STAGE3_BASELINE_ACTION_ALPHA = $BaselineActionAlpha
$env:STAGE3_BASELINE_CLOSEOUT_RANGE = $BaselineCloseoutRange
$env:STAGE3_BASELINE_TANGENT_BLEND_MAX = $BaselineTangentBlendMax

Write-Host "Teacher eval: curriculum=$Curriculum randomization=$Randomization"
Write-Host "Baseline gain=$BaselineTailGain offset=$BaselineTailOffset alpha=$BaselineActionAlpha"

& $python $script --curriculum $Curriculum --randomization $Randomization --num_envs $NumEnvs --steps $Steps --headless
