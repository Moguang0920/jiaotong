@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "MAIN_PY=%CD%\.venv\Scripts\python.exe"
set "OCR_VENV=%CD%\.venv_ocr"
set "OCR_PY=%OCR_VENV%\Scripts\python.exe"

echo ============================================================
echo PaddleOCR GPU subprocess setup V6
echo Main .venv: ONNX Runtime CUDA
echo OCR .venv_ocr: PaddlePaddle GPU
echo ============================================================

if not exist "%MAIN_PY%" (
    echo [ERROR] Missing main virtual environment:
    echo %MAIN_PY%
    goto :failed
)

echo.
echo [1/6] Recreate dedicated OCR virtual environment...
if exist "%OCR_VENV%" rmdir /s /q "%OCR_VENV%"
"%MAIN_PY%" -m venv "%OCR_VENV%"
if errorlevel 1 goto :failed

echo.
echo [2/6] Upgrade OCR environment tools...
"%OCR_PY%" -m pip install --upgrade pip wheel "setuptools<82"
if errorlevel 1 goto :failed

echo.
echo [3/6] Install PaddlePaddle GPU CUDA 12.6...
"%OCR_PY%" -m pip install --no-cache-dir "paddlepaddle-gpu==3.3.0" -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
if errorlevel 1 goto :failed

echo.
echo [4/6] Install PaddleOCR worker dependencies...
"%OCR_PY%" -m pip install --no-cache-dir "numpy==1.26.4" "opencv-python-headless==4.10.0.84" "paddleocr>=3.3.0,<4"
if errorlevel 1 goto :failed

echo.
echo [5/6] Verify Paddle GPU inside OCR process...
"%OCR_PY%" -c "import paddle,sys; print('paddle=',paddle.__version__); print('cuda=',paddle.device.is_compiled_with_cuda()); print('gpu_count=',paddle.device.cuda.device_count()); paddle.set_device('gpu:0'); print('device=',paddle.get_device()); sys.exit(0 if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count()>0 else 2)"
if errorlevel 1 goto :failed

echo.
echo [6/6] Verify ONNX CUDA and PaddleOCR GPU in separate processes...
"%MAIN_PY%" ".\tools\verify_dual_gpu_processes.py"
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo [SUCCESS] Dual-process GPU environment is ready.
echo Run: npm start
echo ============================================================
pause
exit /b 0

:failed
echo.
echo ============================================================
echo [ERROR] Setup failed at the step shown above.
echo ============================================================
pause
exit /b 1
