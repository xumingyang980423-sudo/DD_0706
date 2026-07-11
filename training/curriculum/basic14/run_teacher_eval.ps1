param(
    [int]$NumEnvs = 512,
    [int]$Steps = 1500,
    [string]$Randomization = "train"
)

& (Join-Path $PSScriptRoot "..\..\scripts\eval_teacher_rollout.ps1") `
  -Curriculum "basic14" `
  -Randomization $Randomization `
  -NumEnvs $NumEnvs `
  -Steps $Steps `
  -BaselineTailGain "4.8" `
  -BaselineTailOffset "4.5" `
  -BaselineActionAlpha "0.42" `
  -BaselineCloseoutRange "22.0" `
  -BaselineTangentBlendMax "0.45"
