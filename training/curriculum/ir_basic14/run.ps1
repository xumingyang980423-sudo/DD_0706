param(
    [string]$Checkpoint = "",
    [string]$TrackNetCkpt = "D:\Rocket\Missle\isaac_0703\data\ir\basic14\tracknet_best.pth",
    [int]$MaxIterations = 400,
    [int]$NumEnvs = 4096
)

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$bcInit = Join-Path $projectRoot "checkpoints\ppo_ir_bc_init.pth"

& (Join-Path $PSScriptRoot "..\run_task.ps1") `
  -Curriculum "basic14" `
  -ObsMode "ir_track" `
  -TrackNetCkpt $TrackNetCkpt `
  -MaxIterations $MaxIterations `
  -NumEnvs $NumEnvs `
  -TeacherMode "full" `
  -RewardStage "auto" `
  -Checkpoint $(if ($Checkpoint) { $Checkpoint } else { $bcInit })
