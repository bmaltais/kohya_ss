@echo off
setlocal enabledelayedexpansion

REM ===================================
REM SDXL LoRA 단일 학습 (Windows → Docker)
REM ===================================

REM 도움말
if "%1"=="" (
    echo Usage: run-train-single.cmd --folder FOLDER [OPTIONS]
    echo.
    echo Examples:
    echo   run-train-single.cmd --folder ../dataset/training/01_alice
    echo   run-train-single.cmd --folder ../dataset/training/01_alice --epochs 25
    echo   run-train-single.cmd --folder ../dataset/training/01_alice --lr 0.0002 --dim 64
    echo.
    echo All arguments are passed to Python script inside container.
    exit /b 1
)

REM 현재 시간 (로그 파일명용)
for /f "tokens=1-6 delims=/:. " %%a in ("%date% %time%") do (
    set timestamp=%%a%%b%%c_%%d%%e%%f
)
set timestamp=%timestamp: =0%

REM 모든 arguments를 하나의 문자열로 결합
set args=%*

REM 작은따옴표 이스케이프 (Bash에서 안전하게)
set args=%args:'='\''%

echo ===================================
echo Starting SDXL LoRA Training
echo ===================================
echo Arguments: %args%
echo Log file: train_%timestamp%.log
echo ===================================
echo.

REM Docker에서 실행
docker exec -it sdxl_train_captioner bash -c "cd /app/sdxl_train_captioner/sd-scripts && python run-train-single.py %args% 2>&1 | tee /app/sdxl_train_captioner/logs/train_%timestamp%.log"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===================================
    echo Training completed successfully!
    echo ===================================
) else (
    echo.
    echo ===================================
    echo Training failed with error code: %ERRORLEVEL%
    echo ===================================
)

endlocal

pause