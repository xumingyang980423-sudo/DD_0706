param(
    [string]$Curriculum = "tail4_warmup_residual",
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [int]$Steps = 500
)

$ErrorActionPreference = "Stop"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$engineRoot = Join-Path $projectRoot "training\engine"
$python = "E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat"
$script = Join-Path $PSScriptRoot "dagger_refresh.py"

$env:TORCH_COMPILE_DISABLE = "1"
$env:PYTHONPATH = "$engineRoot\source\isaaclab_stage3;$env:PYTHONPATH"

& $python $script --curriculum $Curriculum --checkpoint $Checkpoint --steps $Steps --headless
