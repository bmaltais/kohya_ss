# extract_loha.ps1 — wrapper for extract_loha_from_tuned_model.py
# Usage: .\extract_loha.ps1 -TunedModel <path> [-OriginalModel <path>] [-Output <path>] [-Rank 8] [-Iterations 500]
#
# Example:
#   .\extract_loha.ps1 -TunedModel E:/models/sdxl/dreamshaperXL_alpha2Xl10.safetensors
#   .\extract_loha.ps1 -TunedModel E:/models/sdxl/myModel.safetensors -Rank 16 -Iterations 1000

param(
    [Parameter(Mandatory=$true)]
    [string]$TunedModel,

    [string]$BaseModel = "E:/models/sdxl/base/sd_xl_base_1.0_0.9vae.safetensors",

    [string]$Output = "",

    [int]$Rank = 8,
    [int]$Iterations = 500,
    [float]$Lr = 0.001,
    [float]$InitialAlpha = 8.0,
    [string]$Device = "cuda"
)

if (-not $Output) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($TunedModel)
    $Output = "E:/lora/sdxl/${name}_loha.safetensors"
}

Write-Host "Extracting LoHa: $TunedModel -> $Output (rank=$Rank, iter=$Iterations)"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }
$scriptPath = Join-Path $PSScriptRoot "extract_loha_from_tuned_model.py"

& $pythonExe $scriptPath `
    $BaseModel `
    $TunedModel `
    $Output `
    --rank $Rank `
    --iterations $Iterations `
    --lr $Lr `
    --initial_alpha $InitialAlpha `
    --device $Device `
    --verbose `
    --verbose_layer
