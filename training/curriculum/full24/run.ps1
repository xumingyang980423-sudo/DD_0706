param(
    [string]$Checkpoint = "",
    [int]$MaxIterations = 800,
    [int]$NumEnvs = 4096
)

& (Join-Path $PSScriptRoot "..\run_task.ps1") `
  -Curriculum "full24" `
  -MaxIterations $MaxIterations `
  -NumEnvs $NumEnvs `
  -Checkpoint $Checkpoint
