# Phase 0 teacher visual validation — no .pth checkpoint (rule-based teacher).
param(
    [string]$Curriculum = "basic14",
    [string]$Randomization = "eval",
    [int]$NumEnvs = 1,
    [int]$Steps = 3000,
    [string]$Device = "cuda:0",
    [string]$FixedScenarioId = ""
)

$ErrorActionPreference = "Stop"
$engineRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = "E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat"
$script = Join-Path $PSScriptRoot "play_teacher_only.py"

$env:TORCH_COMPILE_DISABLE = "1"
$env:PYTHONPATH = "$engineRoot\source\isaaclab_stage3;$env:PYTHONPATH"
$env:STAGE3_TEACHER_MODE = "only"
$env:STAGE3_SCENARIO_CURRICULUM = $Curriculum
$env:STAGE3_RANDOMIZATION_MODE = $Randomization
if ($FixedScenarioId) {
    $env:STAGE3_FIXED_SCENARIO_ID = $FixedScenarioId
} else {
    Remove-Item Env:\STAGE3_FIXED_SCENARIO_ID -ErrorAction SilentlyContinue
}

Write-Host "Teacher-only play (Phase 0): no checkpoint, curriculum=$Curriculum randomization=$Randomization"

& $python $script --curriculum $Curriculum --randomization $Randomization `
    --num_envs $NumEnvs --steps $Steps --device $Device
