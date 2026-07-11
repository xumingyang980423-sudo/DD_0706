$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$python = "E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat"
$aircraftModel = "C:\Users\81042\Desktop\F22.fbx"
$missileModel = "C:\Users\81042\Desktop\HQ9DD.fbx"
$aircraftModelScale = "1,1,1"
$missileModelScale = "1,1,1"
$aircraftModelRotation = "0,0,0"
$missileModelRotation = "0,0,0"
$aircraftModelForward = "0,0,-1"
$aircraftModelUp = "0,1,0"
$missileModelForward = "-1,0,0"
$missileModelUp = "0,0,1"
$aircraftModelColor = "0.05,0.25,1.0"
$missileModelColor = "1.0,0.08,0.02"
$scenarios = @(
    @{ Id = "1"; Name = "head_on"; Label = "Head-on frontal intercept" },
    @{ Id = "2"; Name = "overfly_tail_chase"; Label = "Aircraft overflies launcher, then tail chase" },
    @{ Id = "3"; Name = "crossing_left_to_right"; Label = "Left-to-right crossing target" },
    @{ Id = "4"; Name = "crossing_right_to_left"; Label = "Right-to-left crossing target" },
    @{ Id = "5"; Name = "climb_escape"; Label = "Target climbs after launch" },
    @{ Id = "6"; Name = "dive_escape"; Label = "Target dives after launch" },
    @{ Id = "7"; Name = "s_turn_evasion"; Label = "S-turn evasion" },
    @{ Id = "8"; Name = "double_evasion"; Label = "Two evasive turns before hit" },
    @{ Id = "9"; Name = "late_launch"; Label = "Delayed crossing shot after target has passed the launcher" },
    @{ Id = "10"; Name = "high_speed_pass"; Label = "Fast diagonal pass, high-speed tail pursuit" },
    @{ Id = "11"; Name = "low_altitude_pass"; Label = "Low-altitude terrain-hugging pass, damped pursuit" },
    @{ Id = "12"; Name = "far_tail_chase"; Label = "Far, high-altitude oblique tail chase" },
    @{ Id = "13"; Name = "fighter_weave_chase"; Label = "Fast fighter weave, missile follows IR target motion" },
    @{ Id = "14"; Name = "maneuver_follow_chase"; Label = "Target evades, missile follows the maneuver path and catches it" },
    @{ Id = "15"; Name = "long_weave_tail_chase"; Label = "Overfly first, long wide S-turn tail chase" },
    @{ Id = "16"; Name = "extended_maneuver_follow"; Label = "Overfly first, fast multi-break trail-follow chase" },
    @{ Id = "17"; Name = "climb_dive_weave_chase"; Label = "Overfly first, climb/dive/loop-like trail chase" },
    @{ Id = "18"; Name = "delayed_sustained_evasion"; Label = "Overfly first, delayed high-speed sustained evasion" },
    @{ Id = "19"; Name = "cobra_pop_up_chase"; Label = "Overfly first, sudden nose-up pop-up evasion" },
    @{ Id = "20"; Name = "circle_turn_chase"; Label = "Overfly first, aircraft wide circle, missile follows" },
    @{ Id = "21"; Name = "spiral_climb_chase"; Label = "Overfly first, spiral climbing evasion" },
    @{ Id = "22"; Name = "hard_reversal_chase"; Label = "Overfly first, hard reversal and trail chase" },
    @{ Id = "23"; Name = "wide_snake_chase"; Label = "Overfly first, large-amplitude snake evasion" },
    @{ Id = "24"; Name = "super_combo_chase"; Label = "Overfly first, combined pop-up break and turn" }
)

Write-Host ""
Write-Host "Isaac Sim Stage-2 Scenario Demo"
Write-Host "-------------------------------"
foreach ($scenario in $scenarios) {
    Write-Host "$($scenario.Id). $($scenario.Name) - $($scenario.Label)"
}
Write-Host ""
$choice = Read-Host "Input scenario number"
$selected = $scenarios | Where-Object { $_.Id -eq $choice } | Select-Object -First 1
if ($null -eq $selected) {
    throw "Invalid scenario number: $choice"
}

Write-Host "Starting scenario: $($selected.Name) - $($selected.Label)"
$useExternalModels = Read-Host "Use F22/HQ9DD FBX models? y/N"
$modelArgs = @()
if ($useExternalModels -match "^(y|Y)$") {
    $modelArgs = @(
        "--aircraft-model", $aircraftModel,
        "--missile-model", $missileModel,
        "--aircraft-model-scale=$aircraftModelScale",
        "--missile-model-scale=$missileModelScale",
        "--aircraft-model-rotation=$aircraftModelRotation",
        "--missile-model-rotation=$missileModelRotation",
        "--aircraft-model-forward=$aircraftModelForward",
        "--aircraft-model-up=$aircraftModelUp",
        "--missile-model-forward=$missileModelForward",
        "--missile-model-up=$missileModelUp",
        "--aircraft-model-color=$aircraftModelColor",
        "--missile-model-color=$missileModelColor"
    )
    Write-Host "Using external FBX models."
} else {
    Write-Host "Using built-in simple models."
}

& $python (Join-Path $projectRoot "demos\ground_intercept_demo.py") --scenario $selected.Name --randomize --hold-frames 7200 @modelArgs
