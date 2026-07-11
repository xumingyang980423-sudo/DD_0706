param(
    [Parameter(Mandatory = $true)]
    [string]$Curriculum,

    [string]$RandomizationMode = "train",
    [int]$NumEnvs = 4096,
    [string]$Device = "cuda:0",
    [int]$MaxIterations = 500,
    [string]$Checkpoint = "",

    [string]$TeacherMode = "full",
    [string]$RewardStage = "auto",
    [string]$ResidualSchedule = "auto",
    [string]$ResidualAlpha = "",
    [string]$ResidualBeta = "",
    [string]$BaselineTailGain = "",
    [string]$BaselineTailOffset = "",
    [string]$BaselineActionAlpha = "",
    [string]$BaselineCloseoutRange = "",
    [string]$BaselineTangentBlendMax = "",
    [string]$RewardTailOffset = "",
    [string]$FixedScenarioId = "",
    [string]$ObsMode = "",
    [string]$TrackNetCkpt = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$engineRoot = Join-Path $projectRoot "training\engine"
$python = "E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat"
$trainScript = Join-Path $engineRoot "scripts\train_isaaclab_intercept_rlgames.py"

if (!(Test-Path -LiteralPath $python)) {
    throw "Isaac Sim python not found: $python"
}
if (!(Test-Path -LiteralPath $trainScript)) {
    throw "Training script not found: $trainScript"
}

$env:TORCH_COMPILE_DISABLE = "1"
$env:PYTHONPATH = "$engineRoot\source\isaaclab_stage3;$env:PYTHONPATH"
$env:STAGE3_SCENARIO_CURRICULUM = $Curriculum
$env:STAGE3_RANDOMIZATION_MODE = $RandomizationMode
$env:STAGE3_TEACHER_MODE = $TeacherMode
$env:STAGE3_REWARD_STAGE = $RewardStage
$env:STAGE3_RESIDUAL_SCHEDULE = $ResidualSchedule

if ($FixedScenarioId) {
    $env:STAGE3_FIXED_SCENARIO_ID = $FixedScenarioId
} else {
    Remove-Item Env:\STAGE3_FIXED_SCENARIO_ID -ErrorAction SilentlyContinue
}

if ($ObsMode) {
    $env:STAGE3_OBS_MODE = $ObsMode
    if ($ObsMode -eq "ir_track") {
        $env:STAGE3_IR_ENABLE = "1"
    }
} else {
    Remove-Item Env:\STAGE3_OBS_MODE -ErrorAction SilentlyContinue
}

if ($TrackNetCkpt) {
    if (!(Test-Path -LiteralPath $TrackNetCkpt)) {
        throw "TrackNet checkpoint not found: $TrackNetCkpt"
    }
    $env:STAGE3_TRACKNET_CKPT = (Resolve-Path $TrackNetCkpt).Path
    $env:STAGE3_IR_ENABLE = "1"
} else {
    Remove-Item Env:\STAGE3_TRACKNET_CKPT -ErrorAction SilentlyContinue
}

$optionalEnv = @{
    STAGE3_RESIDUAL_ALPHA = $ResidualAlpha
    STAGE3_RESIDUAL_BETA = $ResidualBeta
    STAGE3_BASELINE_TAIL_GAIN = $BaselineTailGain
    STAGE3_BASELINE_TAIL_OFFSET = $BaselineTailOffset
    STAGE3_BASELINE_ACTION_ALPHA = $BaselineActionAlpha
    STAGE3_BASELINE_CLOSEOUT_RANGE = $BaselineCloseoutRange
    STAGE3_BASELINE_TANGENT_BLEND_MAX = $BaselineTangentBlendMax
    STAGE3_REWARD_TAIL_OFFSET = $RewardTailOffset
}

foreach ($item in $optionalEnv.GetEnumerator()) {
    if ($item.Value) {
        [Environment]::SetEnvironmentVariable($item.Key, $item.Value, "Process")
    } else {
        Remove-Item "Env:\$($item.Key)" -ErrorAction SilentlyContinue
    }
}

$argsList = @(
    $trainScript,
    "--task", "Isaac-Stage3-Intercept-Direct-v0",
    "--num_envs", "$NumEnvs",
    "--device", $Device,
    "--headless",
    "--max_iterations", "$MaxIterations"
)

if ($Checkpoint) {
    if (!(Test-Path -LiteralPath $Checkpoint)) {
        throw "Checkpoint not found: $Checkpoint"
    }
    $argsList += @("--checkpoint", $Checkpoint)
}

Write-Host "Project: $projectRoot"
Write-Host "Curriculum: $Curriculum"
Write-Host "Teacher mode: $TeacherMode"
Write-Host "Reward stage: $RewardStage"
Write-Host "Randomization: $RandomizationMode"
Write-Host "Num envs: $NumEnvs"
Write-Host "Max iterations: $MaxIterations"
if ($ObsMode) { Write-Host "Obs mode: $ObsMode" }
if ($TrackNetCkpt) { Write-Host "TrackNet: $TrackNetCkpt" }
if ($Checkpoint) { Write-Host "Checkpoint: $Checkpoint" }

Push-Location $engineRoot
try {
    & $python @argsList
} finally {
    Pop-Location
}
