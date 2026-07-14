@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Missing .venv\Scripts\python.exe
    pause
    exit /b 1
)

if not exist "multi_reference_road_anomaly_demo_GPU_V3.py" (
    echo [ERROR] Missing multi_reference_road_anomaly_demo_GPU_V3.py
    pause
    exit /b 1
)

if not exist "road_roi_vehicle_mask_demo_GPU_V3.py" (
    echo [ERROR] Missing road_roi_vehicle_mask_demo_GPU_V3.py
    pause
    exit /b 1
)

copy /y "multi_reference_road_anomaly_demo.py" "multi_reference_road_anomaly_demo_before_GPU_V3.py.bak" >nul 2>nul
copy /y "road_roi_vehicle_mask_demo.py" "road_roi_vehicle_mask_demo_before_GPU_V3.py.bak" >nul 2>nul

copy /y "multi_reference_road_anomaly_demo_GPU_V3.py" "multi_reference_road_anomaly_demo.py" >nul
if errorlevel 1 goto :failed

copy /y "road_roi_vehicle_mask_demo_GPU_V3.py" "road_roi_vehicle_mask_demo.py" >nul
if errorlevel 1 goto :failed

echo ============================================================
echo [OK] GPU V3 files installed.
echo ============================================================
echo.
echo Testing import order...
".venv\Scripts\python.exe" -c "import torch; x=torch.zeros(1,device='cuda'); import onnxruntime as ort; ort.preload_dlls(directory=None); print('torch=',torch.__version__); print('gpu=',torch.cuda.get_device_name(0)); print('providers=',ort.get_available_providers())"
if errorlevel 1 goto :failed

echo.
echo [SUCCESS]
pause
exit /b 0

:failed
echo.
echo [ERROR] File replacement or CUDA test failed.
pause
exit /b 1
