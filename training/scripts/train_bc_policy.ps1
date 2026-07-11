param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset,
    [string]$Output = "",
    [int]$Epochs = 50
)

$ErrorActionPreference = "Stop"
$python = "E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat"
$script = Join-Path $PSScriptRoot "train_bc_policy.py"
$argsList = @($script, "--dataset", $Dataset, "--epochs", "$Epochs")
if ($Output) { $argsList += @("--output", $Output) }
& $python @argsList
