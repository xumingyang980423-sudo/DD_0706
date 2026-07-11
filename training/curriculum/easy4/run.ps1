param(
    [string]$Checkpoint = "",
    [int]$MaxIterations = 400,
    [int]$NumEnvs = 4096
)

& (Join-Path $PSScriptRoot "..\run_task.ps1") `
  -Curriculum "easy4" `
  -MaxIterations $MaxIterations `
  -NumEnvs $NumEnvs `
  -TeacherMode "full" `
  -RewardStage "A" `
  -Checkpoint $Checkpoint
