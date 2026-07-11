param(
    [string]$Dataset = "D:\Rocket\Missle\isaac_0703\data\ir\basic14\ir_frames.pt",
    [string]$Checkpoint = "D:\Rocket\Missle\isaac_0703\data\ir\basic14\tracknet_best.pth",
    [string]$Device = "cuda:0",
    [int]$SavePng = 8
)

$ErrorActionPreference = "Stop"
Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
$env:TORCH_COMPILE_DISABLE = "1"
$python = "E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat"
$script = Join-Path $PSScriptRoot "eval_tracknet.py"

Write-Host "TrackNet eval: $Checkpoint"
& $python $script --dataset $Dataset --checkpoint $Checkpoint --device $Device --save_png $SavePng
