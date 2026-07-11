param(
    [string]$Checkpoint = "",
    [int]$MaxIterations = 700,
    [int]$NumEnvs = 4096
)

& (Join-Path $PSScriptRoot "..\run_task.ps1") `
  -Curriculum "mix24" `
  -MaxIterations $MaxIterations `
  -NumEnvs $NumEnvs `
  -Checkpoint $Checkpoint
