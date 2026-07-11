# CPU teacher baseline (no Isaac Sim). Uses system Python 3.12 + clean PYTHONPATH.

$ErrorActionPreference = "Stop"
$script = Join-Path $PSScriptRoot "eval_teacher_cpu_baseline.py"

# Isaac Lab / Isaac Sim PYTHONPATH breaks numpy when using bundled python.
Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue

Write-Host "CPU teacher baseline (no GPU, no Isaac Sim)"
py -3.12 $script
