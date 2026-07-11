param(
    [string]$Checkpoint = "",
    [int]$MaxIterations = 800,
    [int]$NumEnvs = 4096
)

& (Join-Path $PSScriptRoot "..\run_task.ps1") `
  -Curriculum "tail4_warmup_residual" `
  -MaxIterations $MaxIterations `
  -NumEnvs $NumEnvs `
  -Checkpoint $Checkpoint `
  -TeacherMode "full" `
  -RewardStage "auto" `
  -ResidualSchedule "auto" `
  -BaselineTailGain "3.6" `
  -BaselineTailOffset "2.5" `
  -BaselineActionAlpha "0.32" `
  -BaselineCloseoutRange "18.0" `
  -BaselineTangentBlendMax "0.55" `
  -RewardTailOffset "4.2"
