param(
    [Parameter(Mandatory = $true)]
    [string]$BcCheckpoint,
    [string]$RlCheckpoint = "D:\Rocket\Missle\isaac_0703\checkpoints\basic14_ep601.pth",
    [Parameter(Mandatory = $true)]
    [string]$Output
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $BcCheckpoint)) {
    throw "BC checkpoint not found: $BcCheckpoint`nRun train_bc_policy.ps1 first."
}

if (-not (Test-Path $RlCheckpoint)) {
    $logs = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..")) "training\engine\logs\rl_games"
    $candidate = Get-ChildItem $logs -Recurse -Filter "*.pth" -ErrorAction SilentlyContinue |
        Where-Object { $_.Length -gt 100000 } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($candidate) {
        $RlCheckpoint = $candidate.FullName
        Write-Host "Auto template RL checkpoint: $RlCheckpoint"
    } else {
        Write-Warning "No RL template checkpoint found; output may not work with play_checkpoint.ps1"
    }
}

$python = "E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat"
$script = Join-Path $PSScriptRoot "init_rlgames_from_bc.py"
$argsList = @($script, "--bc_checkpoint", $BcCheckpoint, "--output", $Output)
if ($RlCheckpoint) { $argsList += @("--rl_checkpoint", $RlCheckpoint) }
& $python @argsList

if (Test-Path $Output) {
    Write-Host "Created: $Output"
} else {
    throw "Failed to create output checkpoint: $Output"
}
