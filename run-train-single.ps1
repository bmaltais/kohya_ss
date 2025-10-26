# ===================================
# SDXL LoRA 단일 학습 (PowerShell → Docker)
# ===================================

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

if ($Arguments.Count -eq 0) {
    Write-Host "Usage: run-train-single.ps1 --folder FOLDER [OPTIONS]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\run-train-single.ps1 --folder ../dataset/training/01_alice"
    Write-Host "  .\run-train-single.ps1 --folder ../dataset/training/01_alice --epochs 25"
    Write-Host "  .\run-train-single.ps1 --folder ../dataset/training/01_alice --lr 0.0002 --dim 64"
    exit 1
}

# 타임스탬프
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# Arguments를 문자열로 결합
$argsString = $Arguments -join ' '

# 작은따옴표 이스케이프
$argsString = $argsString -replace "'", "'\\''"

Write-Host "===================================" -ForegroundColor Green
Write-Host "Starting SDXL LoRA Training" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green
Write-Host "Arguments: $argsString" -ForegroundColor Cyan
Write-Host "Log file: train_$timestamp.log" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Green
Write-Host ""

# Docker 명령어
$dockerCmd = "cd /app/sdxl_train_captioner/sd-scripts && python run-train-single.py $argsString 2>&1 | tee /app/sdxl_train_captioner/logs/train_$timestamp.log"

# 실행
docker exec -it sdxl_train_captioner bash -c $dockerCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "===================================" -ForegroundColor Green
    Write-Host "Training completed successfully!" -ForegroundColor Green
    Write-Host "===================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "===================================" -ForegroundColor Red
    Write-Host "Training failed with error code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "===================================" -ForegroundColor Red
}

pause