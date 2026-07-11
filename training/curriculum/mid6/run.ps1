param(
    [string]$Checkpoint = "",
    [int]$MaxIterations = 500,
    [int]$NumEnvs = 4096
)

& (Join-Path $PSScriptRoot "..\run_task.ps1") `
  -Curriculum "mid6" `
  -MaxIterations $MaxIterations `
  -NumEnvs $NumEnvs `
  -TeacherMode "full" `
  -RewardStage "auto" `
  -Checkpoint $Checkpoint
