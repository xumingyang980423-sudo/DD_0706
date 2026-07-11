# Save IR seeker PNG samples (Phase 6A visual check). Opens Isaac viewport + writes PNGs.
param(
    [string]$Curriculum = "basic14",
    [int]$Steps = 800,
    [int]$FixedSid = -1
)

$ErrorActionPreference = "Stop"
$engineRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$projectRoot = Resolve-Path (Join-Path $engineRoot "..")
$python = "E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat"
$script = Join-Path $PSScriptRoot "play_ir_seeker_debug.py"

$env:TORCH_COMPILE_DISABLE = "1"
$env:PYTHONPATH = "$engineRoot\source\isaaclab_stage3;$env:PYTHONPATH"
$env:STAGE3_IR_ENABLE = "1"

$argsList = @($script, "--curriculum", $Curriculum, "--steps", "$Steps")
if ($FixedSid -ge 0) { $argsList += @("--fixed_sid", "$FixedSid") }

Write-Host "IR seeker debug -> data\ir\debug\"
& $python @argsList --device cuda:0
