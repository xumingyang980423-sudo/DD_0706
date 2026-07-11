param(
    [string]$Curriculum = "basic14",
    [int]$NumEnvs = 512,
    [int]$Steps = 2000,
    [string]$Output = "",
    [string]$TrackNetCkpt = "D:\Rocket\Missle\isaac_0703\data\ir\basic14\tracknet_best.pth"
)

$ErrorActionPreference = "Stop"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$engineRoot = Join-Path $projectRoot "training\engine"
$python = "E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat"
$script = Join-Path $PSScriptRoot "collect_ir_bc_trajectories.py"

$out = if ($Output) { $Output } else { Join-Path $projectRoot "data\bc\ir_$Curriculum\transitions.pt" }

$env:TORCH_COMPILE_DISABLE = "1"
$env:PYTHONPATH = "$engineRoot\source\isaaclab_stage3;$env:PYTHONPATH"

Write-Host "IR BC collect: curriculum=$Curriculum tracknet=$TrackNetCkpt"
Write-Host "Output -> $out"

& $python $script --curriculum $Curriculum --num_envs $NumEnvs --steps $Steps `
    --output $out --tracknet_ckpt $TrackNetCkpt --headless
