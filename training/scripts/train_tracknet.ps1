param(
    [string]$Dataset = "D:\Rocket\Missle\isaac_0703\data\ir\basic14\ir_frames.pt",
    [string]$Output = "",
    [int]$Epochs = 40,
    [float]$UvWeight = 20.0,
    [int]$Patience = 12,
    [string]$Resume = "",
    [string]$Device = "cuda:0",
    [int]$BatchSize = 0,
    [string]$Python = "E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat",
    [switch]$Preload
)

$ErrorActionPreference = "Stop"
Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
$env:TORCH_COMPILE_DISABLE = "1"
$script = Join-Path $PSScriptRoot "train_tracknet.py"
$argsList = @(
    $script,
    "--dataset", $Dataset,
    "--epochs", "$Epochs",
    "--uv_weight", "$UvWeight",
    "--patience", "$Patience",
    "--device", $Device
)
if ($BatchSize -gt 0) { $argsList += @("--batch_size", "$BatchSize") }
if ($Output) { $argsList += @("--output", $Output) }
if ($Resume) { $argsList += @("--resume", $Resume) }
if ($Preload) { $argsList += @("--preload") }

Write-Host "Python: $Python"
Write-Host "TrackNet train: device=$Device epochs=$Epochs uv_weight=$UvWeight preload=$Preload"
& $Python @argsList
