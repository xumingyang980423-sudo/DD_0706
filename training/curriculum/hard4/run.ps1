param(
    [string]$Checkpoint = "",
    [int]$MaxIterations = 600,
    [int]$NumEnvs = 4096
)

& (Join-Path $PSScriptRoot "..\run_task.ps1") `
  -Curriculum "hard4" `
  -MaxIterations $MaxIterations `
  -NumEnvs $NumEnvs `
  -TeacherMode "full" `
  -RewardStage "auto" `
  -Checkpoint $Checkpoint
