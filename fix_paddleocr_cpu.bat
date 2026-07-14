@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"

echo ============================================================
echo PaddleOCR CPU repair V4
echo ONNX and road anomaly will continue using NVIDIA GPU.
echo ============================================================

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Missing virtual environment:
    echo %PYTHON_EXE%
    pause
    exit /b 1
)

echo.
echo [IMPORTANT] Close Electron and the backend before continuing.
echo Press Ctrl+C now if the project is still running.
pause

echo.
echo [1/4] Remove conflicting Paddle packages...
"%PYTHON_EXE%" -m pip uninstall -y paddlepaddle-gpu paddlepaddle
if errorlevel 1 goto :failed

echo.
echo [2/4] Install PaddlePaddle CPU 3.3.0...
"%PYTHON_EXE%" -m pip install --upgrade --no-cache-dir "paddlepaddle==3.3.0" -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
if errorlevel 1 goto :failed

echo.
echo [3/4] Keep PaddleOCR dependencies available...
"%PYTHON_EXE%" -m pip install --upgrade --no-cache-dir "paddleocr>=3.3.0,<4"
if errorlevel 1 goto :failed

echo.
echo [4/4] Load the real PP-OCRv6 tiny model on CPU...
"%PYTHON_EXE%" ".\tools\test_paddleocr_cpu.py"
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo [SUCCESS] PaddleOCR CPU model is ready.
echo Start the project with npm start.
echo ============================================================
pause
exit /b 0

:failed
echo.
echo ============================================================
echo [ERROR] PaddleOCR CPU repair failed.
echo Read the error above.
echo ============================================================
pause
exit /b 1
