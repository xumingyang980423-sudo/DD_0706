param(
    [string]$TrackNetCkpt = "D:\Rocket\Missle\isaac_0703\data\ir\basic14\tracknet_best.pth",
    [switch]$SkipCollect,
    [switch]$SkipBcTrain,
    [switch]$SkipInit,
    [switch]$RunPpo
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$scripts = Join-Path $root "training\scripts"
$bcData = Join-Path $root "data\bc\ir_basic14\transitions.pt"
$bcPolicy = Join-Path $root "data\bc\ir_basic14\bc_policy.pth"
$ppoInit = Join-Path $root "checkpoints\ppo_ir_bc_init.pth"

if (-not $SkipCollect) {
    Write-Host "=== Phase 6C Step 1: IR BC collect ==="
    & (Join-Path $scripts "collect_ir_bc_trajectories.ps1") -TrackNetCkpt $TrackNetCkpt
}

if (-not $SkipBcTrain) {
    Write-Host "=== Phase 6C Step 2: BC train ==="
    & (Join-Path $scripts "train_bc_policy.ps1") -Dataset $bcData -Epochs 50
}

if (-not $SkipInit) {
    Write-Host "=== Phase 6C Step 3: RL-Games init from BC ==="
    & (Join-Path $scripts "init_rlgames_from_bc.ps1") -BcCheckpoint $bcPolicy -Output $ppoInit
}

if ($RunPpo) {
    Write-Host "=== Phase 6C Step 4: PPO fine-tune ==="
    & (Join-Path $root "training\curriculum\ir_basic14\run.ps1") -TrackNetCkpt $TrackNetCkpt -Checkpoint $ppoInit
}

Write-Host "Phase 6C prep done. PPO init -> $ppoInit"
Write-Host "Run PPO: cd training\curriculum\ir_basic14; .\run.ps1"
