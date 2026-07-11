# Inspect BC dataset (transitions.pt). No Isaac Sim.
param(
    [string]$Dataset = "D:\Rocket\Missle\isaac_0703\data\bc\basic14\transitions.pt"
)

$ErrorActionPreference = "Stop"
Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
& py -3.12 (Join-Path $PSScriptRoot "inspect_bc_dataset.py") --dataset $Dataset
