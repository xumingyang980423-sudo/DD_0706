param(
    [string]$Curriculum = "basic14",
    [int]$NumEnvs = 512,
    [int]$Steps = 5000,
    [int]$FlushSteps = 200,
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$engineRoot = Join-Path $projectRoot "training\engine"
$python = "E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat"
$script = Join-Path $PSScriptRoot "collect_ir_track_dataset.py"
$out = if ($Output) { $Output } else { Join-Path $projectRoot "data\ir\$Curriculum\ir_frames.pt" }

$env:TORCH_COMPILE_DISABLE = "1"
$env:PYTHONPATH = "$engineRoot\source\isaaclab_stage3;$env:PYTHONPATH"
$env:STAGE3_IR_ENABLE = "1"

Write-Host "Python: $python"
Write-Host "IR TrackNet dataset: curriculum=$Curriculum steps=$Steps flush_steps=$FlushSteps"
Write-Host "Manifest -> $out  (frames saved as uint8 shards, low RAM)"

& $python $script --curriculum $Curriculum --num_envs $NumEnvs --steps $Steps `
    --flush_steps $FlushSteps --output $out --headless
