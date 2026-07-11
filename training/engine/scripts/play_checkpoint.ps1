param(
    [string]$Checkpoint = "D:\Rocket\Missle\isaac_0703\training\engine\logs\rl_games\stage3_intercept_direct\2026-07-05_22-19-58\nn\stage3_intercept_direct.pth",
    [string]$Curriculum = "basic14",
    [string]$Randomization = "eval",
    [int]$NumEnvs = 1,
    [string]$Device = "cuda:0",
    [string]$ObsMode = "oracle",
    [string]$TrackNetCkpt = "D:\Rocket\Missle\isaac_0703\data\ir\basic14\tracknet_best.pth",
    [switch]$Video,
    [int]$VideoLength = 2700
)

$ErrorActionPreference = "Stop"
$engineRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$projectRoot = Resolve-Path (Join-Path $engineRoot "..")
$python = "E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat"
$script = Join-Path $PSScriptRoot "play_isaaclab_intercept_rlgames.py"

$env:TORCH_COMPILE_DISABLE = "1"
$env:PYTHONPATH = "$engineRoot\source\isaaclab_stage3;$env:PYTHONPATH"
$env:STAGE3_SCENARIO_CURRICULUM = $Curriculum
$env:STAGE3_RANDOMIZATION_MODE = $Randomization
Remove-Item Env:\STAGE3_TEACHER_MODE -ErrorAction SilentlyContinue
Remove-Item Env:\STAGE3_FIXED_SCENARIO_ID -ErrorAction SilentlyContinue

if ($ObsMode) {
    $env:STAGE3_OBS_MODE = $ObsMode
} else {
    Remove-Item Env:\STAGE3_OBS_MODE -ErrorAction SilentlyContinue
}

if ($ObsMode -eq "ir_track") {
    $env:STAGE3_IR_ENABLE = "1"
    if (!(Test-Path -LiteralPath $TrackNetCkpt)) {
        throw "TrackNet checkpoint not found: $TrackNetCkpt"
    }
    $env:STAGE3_TRACKNET_CKPT = (Resolve-Path $TrackNetCkpt).Path
} else {
    Remove-Item Env:\STAGE3_TRACKNET_CKPT -ErrorAction SilentlyContinue
    Remove-Item Env:\STAGE3_IR_ENABLE -ErrorAction SilentlyContinue
}

$argsList = @(
    $script,
    "--task", "Isaac-Stage3-Intercept-Direct-v0",
    "--checkpoint", $Checkpoint,
    "--num_envs", $NumEnvs,
    "--device", $Device
)
if ($Video) {
    $argsList += @("--video", "--video_length", $VideoLength)
}

Write-Host "Play checkpoint: $Checkpoint"
Write-Host "Curriculum=$Curriculum randomization=$Randomization num_envs=$NumEnvs device=$Device obs_mode=$ObsMode"
if ($ObsMode -eq "ir_track") { Write-Host "TrackNet: $TrackNetCkpt" }

& $python @argsList
