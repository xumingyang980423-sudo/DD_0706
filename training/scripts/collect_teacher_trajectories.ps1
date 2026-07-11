param(
    [string]$Curriculum = "basic14",
    [int]$NumEnvs = 512,
    [int]$Steps = 2000,
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$engineRoot = Join-Path $projectRoot "training\engine"
$python = "E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat"
$script = Join-Path $PSScriptRoot "collect_teacher_trajectories.py"

$out = if ($Output) { $Output } else { Join-Path $projectRoot "data\bc\$Curriculum\transitions.pt" }

$env:TORCH_COMPILE_DISABLE = "1"
$env:PYTHONPATH = "$engineRoot\source\isaaclab_stage3;$env:PYTHONPATH"

& $python $script --curriculum $Curriculum --num_envs $NumEnvs --steps $Steps --output $out --headless
