@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"

echo ============================================================
echo Road anomaly GPU dependency installer
echo Project: %CD%
echo ============================================================

if not exist "%PYTHON_EXE%" (
    echo.
    echo [ERROR] Virtual environment Python was not found:
    echo %PYTHON_EXE%
    echo.
    echo Make sure this file exists:
    echo .venv\Scripts\python.exe
    pause
    exit /b 1
)

echo.
echo [INFO] Python executable:
"%PYTHON_EXE%" -c "import sys; print(sys.executable); print(sys.version)"
if errorlevel 1 goto :failed

echo.
echo [1/3] Upgrade pip tools...
"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :failed

echo.
echo [2/3] Install PyTorch CUDA 12.8...
"%PYTHON_EXE%" -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :failed

echo.
echo [3/3] Verify PyTorch CUDA...
"%PYTHON_EXE%" -c "import torch,sys; print('torch=',torch.__version__); print('torch_cuda=',torch.version.cuda); print('cuda_available=',torch.cuda.is_available()); print('gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); sys.exit(0 if torch.cuda.is_available() else 2)"
if errorlevel 1 goto :failed

echo.
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [WARN] ffmpeg was not found.
    echo Video saving will use OpenCV CPU encoding.
) else (
    echo [INFO] Checking NVENC support...
    ffmpeg -hide_banner -encoders 2>nul | findstr /i "h264_nvenc" >nul
    if errorlevel 1 (
        echo [WARN] h264_nvenc was not found.
        echo Video saving will use OpenCV CPU encoding.
    ) else (
        echo [OK] h264_nvenc is available.
    )
)

echo.
echo ============================================================
echo [SUCCESS] GPU dependencies are ready.
echo ============================================================
pause
exit /b 0

:failed
echo.
echo ============================================================
echo [ERROR] Installation or CUDA verification failed.
echo Check the error messages above.
echo ============================================================
pause
exit /b 1
